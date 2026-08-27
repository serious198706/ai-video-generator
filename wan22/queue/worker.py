from __future__ import annotations

import threading
import time
from pathlib import Path

from wan22 import config
from wan22.infer import generate
from wan22.log import get_logger
from wan22.media import download, webhook
from wan22.media.upload import upload_video
from wan22.net.urlguard import UrlError
from wan22.queue import store

logger = get_logger(__name__)

_started = False
_lock = threading.Lock()


def start() -> None:
    global _started
    with _lock:
        if _started:
            return
        _recover_interrupted()
        thread = threading.Thread(target=_loop, name="wan22-worker", daemon=True)
        thread.start()
        _started = True


def _recover_interrupted() -> None:
    task_id = store.take_interrupted()
    if not task_id:
        return
    task = store.get_task(task_id)
    if not task or task.get("status") != "running":
        logger.info("cleared stale running key task=%s status=%s", task_id, (task or {}).get("status"))
        return
    store.update_task(task_id, status="failed", error="interrupted")
    updated = store.get_task(task_id)
    if updated:
        webhook.notify(updated)
    logger.warning("interrupted running task=%s", task_id)


def _loop() -> None:
    if config.PRELOAD and not config.DRY_RUN:
        try:
            generate.load_pipe()
        except Exception:
            logger.exception("pipeline preload failed")
    while True:
        try:
            task_id = store.pop_task(timeout=5)
        except Exception:
            logger.exception("queue pop failed")
            store.reset_client()
            time.sleep(1)
            continue
        if not task_id:
            continue
        try:
            _run(task_id)
        except Exception:
            logger.exception("worker crashed task=%s", task_id)
            _finish(task_id, status="failed", error="generate_failed")
            store.clear_running(task_id)


def _run(task_id: str) -> None:
    task = store.get_task(task_id)
    if not task:
        logger.warning("missing task=%s after pop", task_id)
        return

    output = str(config.OUTPUT_DIR / f"{task_id}.mp4")
    first_path = task.get("first_frame_path")
    last_path = task.get("last_frame_path")
    started = time.monotonic()
    store.set_running(task_id)
    store.update_task(task_id, status="running", error=None)
    logger.info(
        "running task=%s duration=%s resolution=%s steps=%s",
        task_id,
        task.get("duration"),
        task.get("resolution"),
        task.get("steps"),
    )
    try:
        first_path = _ensure_image(task, "first")
        last_path = _ensure_image(task, "last") if task.get("last_image_url") else last_path
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
            )
        except Exception:
            logger.exception("generate failed task=%s", task_id)
            _finish(task_id, status="failed", error="generate_failed")
            return

        try:
            video_url = (
                f"http://127.0.0.1/dry-run/{task_id}.mp4"
                if config.DRY_RUN
                else upload_video(output, object_name=f"{task_id}.mp4")
            )
        except Exception:
            logger.exception("upload failed task=%s", task_id)
            _finish(task_id, status="failed", error="upload_failed")
            return

        elapsed = time.monotonic() - started
        logger.info(
            "succeeded task=%s seed=%s elapsed=%.1fs video=%s",
            task_id,
            used_seed,
            elapsed,
            video_url,
        )
        _finish(
            task_id,
            status="succeeded",
            error=None,
            seed=used_seed,
            video_url=video_url,
        )
    except UrlError:
        logger.exception("download failed task=%s", task_id)
        _finish(task_id, status="failed", error="download_failed")
    finally:
        _remove(first_path, last_path, output)
        store.clear_running(task_id)


def _ensure_image(task: dict, kind: str) -> str:
    path_key = "first_frame_path" if kind == "first" else "last_frame_path"
    url_key = "image_url" if kind == "first" else "last_image_url"
    path = task.get(path_key)
    if path and Path(path).is_file():
        return path
    url = task.get(url_key)
    if not url:
        raise UrlError("missing image")
    logger.info("re-download task=%s kind=%s", task["id"], kind)
    saved = download.download_image(url, config.UPLOAD_DIR / f"{task['id']}_{kind}")
    store.update_task(task["id"], **{path_key: saved})
    return saved


def _finish(task_id: str, **fields) -> None:
    store.update_task(task_id, **fields)
    task = store.get_task(task_id)
    if task:
        webhook.notify(task)


def _remove(*paths: str | None) -> None:
    for path in paths:
        if path:
            Path(path).unlink(missing_ok=True)
