from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from wan22 import config

_OPTIONAL = (
    "negative_prompt",
    "image_url",
    "last_image_url",
    "first_frame_path",
    "last_frame_path",
    "resolution",
    "webhook_url",
    "seed",
    "steps",
    "quality",
    "video_url",
    "error",
    "worker_id",
)

_lock = threading.Lock()
_tasks: dict[str, dict[str, Any]] = {}
_queue: deque[str] = deque()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker_id() -> str:
    return getattr(config, "WORKER_ID", None) or "gpu"


def _running_file() -> Path:
    safe = "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in _worker_id())
    return config.ROOT / f"worker-running.{safe}"


def _normalize(task: dict[str, Any]) -> dict[str, Any]:
    out = dict(task)
    if out.get("duration") is not None:
        out["duration"] = float(out["duration"])
    for key in ("seed", "steps", "quality", "attempts"):
        if out.get(key) is not None and out.get(key) != "":
            out[key] = int(out[key])
        elif key == "attempts":
            out[key] = 0
    out["audio"] = bool(out.get("audio"))
    for key in _OPTIONAL:
        if out.get(key) in ("",):
            out[key] = None
    return out


def ping() -> None:
    return None


def reset_client() -> None:
    return None


def create_task(task_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    now = _now()
    payload = _normalize(
        {
            "id": task_id,
            "status": "queued",
            "attempts": 0,
            "created_at": now,
            "updated_at": now,
            "worker_id": None,
            "prompt": fields.get("prompt"),
            "negative_prompt": fields.get("negative_prompt"),
            "image_url": fields.get("image_url"),
            "last_image_url": fields.get("last_image_url"),
            "first_frame_path": fields.get("first_frame_path"),
            "last_frame_path": fields.get("last_frame_path"),
            "duration": fields.get("duration"),
            "resolution": fields.get("resolution"),
            "webhook_url": fields.get("webhook_url"),
            "seed": fields.get("seed"),
            "steps": fields.get("steps"),
            "quality": fields.get("quality"),
            "audio": bool(fields.get("audio")),
            "video_url": None,
            "error": None,
        }
    )
    with _lock:
        _tasks[task_id] = payload
        _queue.append(task_id)
    return get_task(task_id)  # type: ignore[return-value]


def update_task(task_id: str, **fields: Any) -> None:
    if not fields:
        return
    with _lock:
        task = _tasks.get(task_id)
        if not task:
            return
        for key, value in fields.items():
            if value == "":
                value = None
            task[key] = value
        task["updated_at"] = _now()
        _tasks[task_id] = _normalize(task)


def get_task(task_id: str) -> dict[str, Any] | None:
    with _lock:
        task = _tasks.get(task_id)
        return _normalize(task) if task else None


def pending() -> int:
    with _lock:
        return len(_queue)


def pop_task(timeout: int = 5) -> str | None:
    deadline = time.monotonic() + max(timeout, 0)
    while True:
        with _lock:
            if _queue:
                return _queue.popleft()
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


def set_running(task_id: str) -> None:
    _running_file().write_text(task_id, encoding="utf-8")
    update_task(task_id, status="running", error=None, worker_id=_worker_id())


def clear_running(task_id: str | None = None) -> None:
    path = _running_file()
    if not path.is_file():
        return
    current = path.read_text(encoding="utf-8").strip()
    if task_id is None or current == task_id:
        path.unlink(missing_ok=True)


def take_interrupted() -> str | None:
    path = _running_file()
    if not path.is_file():
        return None
    task_id = path.read_text(encoding="utf-8").strip()
    path.unlink(missing_ok=True)
    return task_id or None


def requeue(task_id: str, *, attempts: int, error: str | None = None) -> None:
    with _lock:
        task = _tasks.get(task_id)
        if not task:
            return
        task["status"] = "queued"
        task["attempts"] = attempts
        task["error"] = error
        task["worker_id"] = None
        task["updated_at"] = _now()
        _tasks[task_id] = _normalize(task)
        _queue.append(task_id)
