from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wan22 import config
from wan22.infer import generate
from wan22.infer.foley import FoleyError, add_audio
from wan22.infer.upscale import (
    UpscaleError, UpscaleOutOfMemory, release_gpu as release_upscale_gpu,
    stop as stop_upscale, upscale_video,
)
from wan22.log import get_logger
from wan22.media import download, status, webhook
from wan22.media.upload import upload_video
from wan22.net.urlguard import UrlError

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# 单卡只有一个执行位：consumer 抢到位才开跑，抢不到就让消息退回队列。不排队、不重试。
_slot = threading.Lock()
_current: str | None = None


def current() -> str | None:
    with _slot:
        return _current


def reserve(task_id: str) -> bool:
    global _current
    with _slot:
        if _current is not None:
            return False
        _current = task_id
        return True


def release(task_id: str) -> None:
    global _current
    with _slot:
        if _current == task_id:
            _current = None


def spawn(task: dict[str, Any]) -> None:
    """占位成功后调用。推理在后台线程里跑完，终态写 S3 状态文件并发 webhook。"""
    thread = threading.Thread(
        target=_run_guarded,
        args=(task,),
        name=f"wan22-task-{task['id'][:8]}",
        daemon=True,
    )
    thread.start()


def startup() -> None:
    # 无情生成机器：没有本地留档，重启后上一单的事务全靠 SQS 重投恢复。
    if config.PRELOAD and not config.DRY_RUN:
        threading.Thread(target=_preload, name="wan22-preload", daemon=True).start()


def _preload() -> None:
    try:
        generate.load_pipe()
    except Exception:
        logger.exception("pipeline preload failed")


def _run_guarded(task: dict[str, Any]) -> None:
    task_id = task["id"]
    started = time.monotonic()
    outcome: dict[str, Any]
    try:
        outcome = _run(task)
    except Exception:
        logger.exception("task crashed task=%s", task_id)
        outcome = {"status": "failed", "error": "generate_failed"}
    finally:
        # 显存搬回来、临时帧删完才放开执行位，否则下一单会撞上收尾中的卡。
        release(task_id)
    timing = outcome.pop("timing", {})
    timing.setdefault("started_at", _now_iso())
    timing["finished_at"] = _now_iso()
    timing.setdefault("duration_s", round(time.monotonic() - started, 3))
    outcome["timing"] = timing
    _finish(task, outcome)


def _run(task: dict[str, Any]) -> dict[str, Any]:
    """跑完一条流水线，返回终态。状态文件和 webhook 由 _finish 统一写。"""
    task_id = task["id"]
    output = str(config.OUTPUT_DIR / f"{task_id}.mp4")
    started = time.monotonic()
    started_at = _now_iso()
    logger.info(
        "running task=%s duration=%s resolution=%s steps=%s audio=%s",
        task_id,
        task.get("duration"),
        task.get("resolution"),
        task.get("steps"),
        bool(task.get("audio")),
    )
    first_path: str | None = None
    last_path: str | None = None
    try:
        first_path = _ensure_image(task, "first")
        last_path = _ensure_image(task, "last") if task.get("last_image_url") else None
        generate_s = foley_s = upscale_s = upload_s = 0.0
        foley_ok = 0
        upscale_ok = 0
        used_seed = task.get("seed")
        generation_metrics: dict = {}
        gen_started = time.monotonic()
        status.write(task_id, "generating")
        try:
            used_seed = generate.generate_video(
                prompt=task["prompt"],
                output_path=output,
                first_frame_path=first_path,
                last_frame_path=last_path,
                duration=float(task["duration"] or 5),
                seed=task.get("seed"),
                steps=task.get("steps"),
                negative_prompt=task.get("negative_prompt"),
                quality=task.get("quality"),
                resolution=task.get("resolution"),
                metrics=generation_metrics,
            )
        except Exception:
            logger.exception("generate failed task=%s", task_id)
            _log_timing(
                task,
                status="failed",
                error="generate_failed",
                seed=used_seed,
                total_s=time.monotonic() - gen_started,
                generate_s=time.monotonic() - gen_started,
            )
            return {"status": "failed", "error": "generate_failed"}
        generate_s = time.monotonic() - gen_started

        needs_upscale = (task.get("resolution") or "").lower() == "1080p" and not config.DRY_RUN
        needs_foley = config.FOLEY_ENABLE and not config.DRY_RUN and bool(task.get("audio"))
        if needs_upscale or needs_foley:
            transfer_s = 0.0
            def move_wan(action):
                nonlocal transfer_s
                moved = time.monotonic()
                try:
                    action()
                finally:
                    transfer_s += time.monotonic() - moved

            shared_gpu = needs_upscale and not needs_foley and generate.can_share_gpu_for_upscale(task.get("resolution"))
            paused = False
            upscale_released = False
            if not shared_gpu:
                move_wan(generate.pause_gpu)
                paused = True
            else:
                logger.info("1080p keep Wan on GPU during upscale task=%s", task_id)
            try:
                if needs_upscale:
                    upscale_t = time.monotonic()
                    try:
                        try:
                            upscale_video(output)
                        except UpscaleOutOfMemory:
                            if not shared_gpu:
                                raise
                            logger.warning("1080p shared upscale CUDA OOM; retry with Wan on CPU task=%s", task_id)
                            stop_upscale()
                            move_wan(generate.pause_gpu)
                            paused = True
                            upscale_video(output)
                        upscale_ok = 1
                    except UpscaleError:
                        logger.exception("upscale failed task=%s", task_id)
                        upscale_s = time.monotonic() - upscale_t
                        _log_timing(
                            task,
                            status="failed",
                            error="upscale_failed",
                            seed=used_seed,
                            total_s=time.monotonic() - gen_started,
                            generate_s=generate_s,
                            upscale_s=upscale_s,
                            upscale_ok=0,
                        )
                        return {"status": "failed", "error": "upscale_failed"}
                    upscale_s = time.monotonic() - upscale_t
                    # Release SR weights before Foley starts, including the non-sharing path.
                    if needs_foley:
                        release_upscale_gpu()
                        upscale_released = True
                if needs_foley:
                    foley_failed = False
                    foley_t = time.monotonic()
                    try:
                        foley_ok = int(bool(add_audio(output)))
                    except FoleyError:
                        foley_failed = True
                        logger.exception("foley required and failed task=%s", task_id)
                    finally:
                        foley_s = time.monotonic() - foley_t
                    if foley_failed:
                        _log_timing(
                            task,
                            status="failed",
                            error="foley_failed",
                            seed=used_seed,
                            total_s=time.monotonic() - gen_started,
                            generate_s=generate_s,
                            upscale_s=upscale_s,
                            upscale_ok=upscale_ok,
                            foley_s=foley_s,
                            foley_ok=0,
                        )
                        return {"status": "failed", "error": "foley_failed"}
            finally:
                if needs_upscale and not upscale_released:
                    # Same lifecycle for shared and non-shared paths. Keep CPU weights warm.
                    release_upscale_gpu()
                if paused:
                    move_wan(generate.resume_gpu)
        else:
            transfer_s = 0.0

        upload_t = time.monotonic()
        if config.DRY_RUN:
            video_url = f"http://127.0.0.1/dry-run/{task_id}.mp4"
        else:
            try:
                video_url = upload_video(output, object_name=f"{task_id}.mp4")
            except Exception:
                logger.exception("upload failed task=%s local=%s", task_id, output)
                upload_s = time.monotonic() - upload_t
                _log_timing(
                    task,
                    status="failed",
                    error="upload_failed",
                    seed=used_seed,
                    total_s=time.monotonic() - gen_started,
                    generate_s=generate_s,
                    upscale_s=upscale_s,
                    upscale_ok=upscale_ok,
                    foley_s=foley_s,
                    foley_ok=foley_ok,
                    upload_s=upload_s,
                )
                return {"status": "failed", "error": "upload_failed"}
        upload_s = time.monotonic() - upload_t
        total_s = time.monotonic() - gen_started
        wall_s = time.monotonic() - started

        _log_timing(
            task,
            status="succeeded",
            seed=used_seed,
            total_s=total_s,
            generate_s=generate_s,
            upscale_s=upscale_s,
            upscale_ok=upscale_ok,
            foley_s=foley_s,
            upload_s=upload_s,
            foley_ok=foley_ok,
        )
        logger.info(
            "succeeded task=%s seed=%s elapsed=%.1fs video=%s",
            task_id,
            used_seed,
            wall_s,
            video_url,
        )
        return {
            "status": "succeeded",
            "seed": used_seed,
            "video_url": video_url,
            # 计时收进 timing 子对象，事后可查每单花在哪。
            "timing": {
                "started_at": started_at,
                "finished_at": _now_iso(),
                "duration_s": round(total_s, 3),
                "generate_s": round(generate_s, 3),
                "upscale_s": round(upscale_s, 3),
                "foley_s": round(foley_s, 3),
                "upload_s": round(upload_s, 3),
                "transfer_s": round(transfer_s, 3),
                "generation": generation_metrics,
            },
        }
    except UrlError:
        logger.exception(
            "download failed task=%s image=%s last=%s",
            task_id,
            task.get("image_url"),
            task.get("last_image_url"),
        )
        _log_timing(
            task,
            status="failed",
            error="download_failed",
            seed=task.get("seed"),
            total_s=time.monotonic() - started,
        )
        return {"status": "failed", "error": "download_failed"}
    finally:
        _cleanup_frames(task, first_path, last_path)


def _ensure_image(task: dict, kind: str) -> str:
    url_key = "image_url" if kind == "first" else "last_image_url"
    url = task.get(url_key)
    if not url:
        raise UrlError("missing image")
    local = Path(url)
    if local.is_file():
        return str(local.resolve())
    logger.info("download task=%s kind=%s url=%s", task["id"], kind, url)
    return download.download_image(url, config.UPLOAD_DIR / f"{task['id']}_{kind}")


def _log_timing(
    task: dict,
    *,
    status: str,
    seed,
    total_s: float,
    generate_s: float = 0.0,
    upscale_s: float = 0.0,
    foley_s: float = 0.0,
    upload_s: float = 0.0,
    upscale_ok: int = 0,
    foley_ok: int = 0,
    error: str | None = None,
) -> None:
    """固定 key=value，方便后期 grep / 收集。total_s = 开跑 I2V 到 S3 传完。"""
    steps = task.get("steps") if task.get("steps") is not None else config.NUM_STEPS
    quality = task.get("quality") if task.get("quality") is not None else config.VIDEO_QUALITY
    foley_on = (
        1
        if config.FOLEY_ENABLE and not config.DRY_RUN and bool(task.get("audio"))
        else 0
    )
    prompt = task.get("prompt") or ""
    fields = [
        f"task={task['id']}",
        f"status={status}",
        f"total_s={total_s:.3f}",
        f"generate_s={generate_s:.3f}",
        f"upscale_s={upscale_s:.3f}",
        f"foley_s={foley_s:.3f}",
        f"upload_s={upload_s:.3f}",
        f"duration={task.get('duration')}",
        f"resolution={task.get('resolution') or '-'}",
        f"steps={steps}",
        f"quality={quality}",
        f"seed={seed if seed is not None else '-'}",
        f"last_frame={1 if task.get('last_image_url') else 0}",
        f"prompt_chars={len(prompt)}",
        f"foley={foley_on}",
        f"foley_ok={foley_ok}",
        f"foley_steps={config.FOLEY_STEPS if foley_on else 0}",
        f"foley_size={config.FOLEY_SIZE if foley_on else '-'}",
        f"upscale={1 if (task.get('resolution') or '').lower() == '1080p' and not config.DRY_RUN else 0}",
        f"upscale_ok={upscale_ok}",
    ]
    if error:
        fields.append(f"error={error}")
    logger.info("timing %s", " ".join(fields))


def _finish(task: dict[str, Any], outcome: dict[str, Any]) -> None:
    """写 S3 终态 → 回调 Java。失败不重投，短码直接给出去。

    状态文件必须在 webhook 之前写：Java 在回调里立刻查 S3 也要能看到终态。
    task["request"]（lambda 入队 JSON 原样）和 outcome["timing"] 一起进状态文件，
    保持 lambda 建档时的格式：request 字段不丢，计时收在 timing 里。
    """
    task_id = task["id"]
    fields: dict[str, Any] = {"timing": outcome.get("timing", {})}
    if task.get("request") is not None:
        fields["request"] = task["request"]
    if outcome["status"] == "succeeded":
        fields["seed"] = outcome.get("seed")
        fields["video_url"] = outcome.get("video_url")
        status.write(task_id, "succeeded", **fields)
    else:
        error = outcome.get("error") or "generate_failed"
        fields["error"] = error
        status.write(task_id, "failed", **fields)
        logger.warning("failed task=%s error=%s", task_id, error)
    try:
        webhook.notify({**task, **outcome})
    except Exception:
        logger.exception("webhook failed task=%s", task_id)


def _cleanup_frames(task: dict, *paths: str | None) -> None:
    originals = {
        str(Path(value).resolve())
        for value in (task.get("image_url"), task.get("last_image_url"))
        if value and Path(str(value)).is_file()
    }
    for path in paths:
        if not path:
            continue
        resolved = str(Path(path).resolve()) if Path(path).exists() else path
        if resolved in originals:
            continue
        Path(path).unlink(missing_ok=True)
