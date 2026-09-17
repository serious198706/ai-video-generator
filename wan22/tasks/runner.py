from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from wan22 import config
from wan22.infer import generate
from wan22.infer.foley import FoleyError, add_audio
from wan22.infer.upscale import UpscaleError, upscale_video
from wan22.log import get_logger
from wan22.media import download, webhook
from wan22.media.upload import upload_video
from wan22.net.urlguard import UrlError
from wan22.tasks import store

logger = get_logger(__name__)

# 单卡只有一个执行位：接单即开跑，跑不动就让接口回 429，不排队、不重试。
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


def spawn(task_id: str) -> None:
    """占位成功后调用。HTTP 已经 202 返回，推理在后台线程里跑。"""
    thread = threading.Thread(
        target=_run_guarded,
        args=(task_id,),
        name=f"wan22-task-{task_id[:8]}",
        daemon=True,
    )
    thread.start()


def startup() -> None:
    stale = store.fail_running("interrupted")
    for task in stale:
        logger.warning("fail interrupted task=%s process restarted", task["id"])
    if stale:
        threading.Thread(
            target=_notify_all,
            args=(stale,),
            name="wan22-interrupted",
            daemon=True,
        ).start()
    if config.PRELOAD and not config.DRY_RUN:
        threading.Thread(target=_preload, name="wan22-preload", daemon=True).start()


def _preload() -> None:
    try:
        generate.load_pipe()
    except Exception:
        logger.exception("pipeline preload failed")


def _notify_all(tasks: list[dict[str, Any]]) -> None:
    for task in tasks:
        try:
            webhook.notify(task)
        except Exception:
            logger.exception("webhook failed task=%s", task.get("id"))


def _run_guarded(task_id: str) -> None:
    outcome: dict[str, Any] | None = None
    try:
        outcome = _run(task_id)
    except Exception:
        logger.exception("task crashed task=%s", task_id)
        outcome = {"status": "failed", "error": "generate_failed"}
    finally:
        # 显存搬回来、临时帧删完才放开执行位，否则下一单会撞上收尾中的卡。
        release(task_id)
    if outcome:
        _finish(task_id, outcome)


def _run(task_id: str) -> dict[str, Any] | None:
    """跑完一条流水线，返回终态。落库和 webhook 由调用方在释放执行位后做。"""
    task = store.get_task(task_id)
    if not task:
        logger.warning("missing task=%s", task_id)
        return None

    output = str(config.OUTPUT_DIR / f"{task_id}.mp4")
    first_path = task.get("first_frame_path")
    last_path = task.get("last_frame_path")
    started = time.monotonic()
    logger.info(
        "running task=%s duration=%s resolution=%s steps=%s audio=%s",
        task_id,
        task.get("duration"),
        task.get("resolution"),
        task.get("steps"),
        bool(task.get("audio")),
    )
    try:
        first_path = _ensure_image(task, "first")
        last_path = _ensure_image(task, "last") if task.get("last_image_url") else last_path
        generate_s = foley_s = upscale_s = upload_s = 0.0
        foley_ok = 0
        upscale_ok = 0
        used_seed = task.get("seed")
        gen_started = time.monotonic()
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
            generate.pause_gpu()
            try:
                if needs_upscale:
                    upscale_t = time.monotonic()
                    try:
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
                generate.resume_gpu()

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
        return {"status": "succeeded", "seed": used_seed, "video_url": video_url}
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
    path_key = "first_frame_path" if kind == "first" else "last_frame_path"
    url_key = "image_url" if kind == "first" else "last_image_url"
    path = task.get(path_key)
    if path and Path(path).is_file():
        return path
    url = task.get(url_key)
    if not url:
        raise UrlError("missing image")
    local = Path(url)
    if local.is_file():
        resolved = str(local.resolve())
        store.update_task(task["id"], **{path_key: resolved})
        return resolved
    logger.info("download task=%s kind=%s url=%s", task["id"], kind, url)
    saved = download.download_image(url, config.UPLOAD_DIR / f"{task['id']}_{kind}")
    store.update_task(task["id"], **{path_key: saved})
    return saved


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


def _finish(task_id: str, outcome: dict[str, Any]) -> None:
    """落终态再回调 Java。失败不重投，短码直接给出去。"""
    if outcome["status"] == "succeeded":
        store.update_task(
            task_id,
            status="succeeded",
            error=None,
            seed=outcome.get("seed"),
            video_url=outcome.get("video_url"),
        )
    else:
        error = outcome.get("error") or "generate_failed"
        store.update_task(task_id, status="failed", error=error, video_url=None)
        logger.warning("failed task=%s error=%s", task_id, error)
    task = store.get_task(task_id)
    if task:
        webhook.notify(task)


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
