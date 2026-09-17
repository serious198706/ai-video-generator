from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from wan22 import config

# SQLite 只做留档：任务从 running 开始，终态 succeeded / failed，进程重启后仍可查。
_UPDATABLE = frozenset(
    {
        "status",
        "prompt",
        "negative_prompt",
        "image_url",
        "last_image_url",
        "first_frame_path",
        "last_frame_path",
        "duration",
        "resolution",
        "webhook_url",
        "seed",
        "steps",
        "quality",
        "audio",
        "video_url",
        "error",
        "worker_id",
    }
)

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def client() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        path = config.TASK_DB
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(
            str(path),
            timeout=30,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                prompt TEXT,
                negative_prompt TEXT,
                image_url TEXT,
                last_image_url TEXT,
                first_frame_path TEXT,
                last_frame_path TEXT,
                duration REAL,
                resolution TEXT,
                webhook_url TEXT,
                seed INTEGER,
                steps INTEGER,
                quality INTEGER,
                audio INTEGER NOT NULL DEFAULT 0,
                video_url TEXT,
                error TEXT,
                worker_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at)")
        _conn = conn
    return _conn


def reset_client() -> None:
    global _conn
    old = _conn
    _conn = None
    if old is None:
        return
    try:
        old.close()
    except Exception:
        pass


def ping() -> None:
    with _lock:
        client().execute("SELECT 1").fetchone()


def _dump(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    return value


def _load(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    task: dict[str, Any] = dict(row)
    if task.get("duration") is not None:
        task["duration"] = float(task["duration"])
    for key in ("seed", "steps", "quality"):
        if task.get(key) is not None:
            task[key] = int(task[key])
    task["audio"] = bool(task.get("audio"))
    return task


def create_task(task_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    """建档即 running：接单就开跑，不存在 queued 这个中间态。"""
    now = _now()
    with _lock:
        client().execute(
            """
            INSERT INTO tasks (
                id, status, prompt, negative_prompt, image_url, last_image_url,
                first_frame_path, last_frame_path, duration, resolution, webhook_url,
                seed, steps, quality, audio, video_url, error, worker_id,
                created_at, updated_at
            ) VALUES (
                ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?
            )
            """,
            (
                task_id,
                _dump(fields.get("prompt")),
                _dump(fields.get("negative_prompt")),
                _dump(fields.get("image_url")),
                _dump(fields.get("last_image_url")),
                _dump(fields.get("first_frame_path")),
                _dump(fields.get("last_frame_path")),
                _dump(fields.get("duration")),
                _dump(fields.get("resolution")),
                _dump(fields.get("webhook_url")),
                _dump(fields.get("seed")),
                _dump(fields.get("steps")),
                _dump(fields.get("quality")),
                int(bool(fields.get("audio"))),
                _dump(fields.get("worker_id") or config.WORKER_ID),
                now,
                now,
            ),
        )
    return get_task(task_id)  # type: ignore[return-value]


def update_task(task_id: str, **fields: Any) -> None:
    payload = {key: _dump(value) for key, value in fields.items() if key in _UPDATABLE}
    if not payload:
        return
    payload["updated_at"] = _now()
    assignments = ", ".join(f"{key}=?" for key in payload)
    values = [*payload.values(), task_id]
    with _lock:
        client().execute(f"UPDATE tasks SET {assignments} WHERE id=?", values)


def get_task(task_id: str) -> dict[str, Any] | None:
    with _lock:
        row = client().execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
    return _load(row)


def running() -> list[dict[str, Any]]:
    with _lock:
        rows = client().execute(
            "SELECT * FROM tasks WHERE status='running' ORDER BY created_at, rowid"
        ).fetchall()
    return [task for task in (_load(row) for row in rows) if task]


def fail_running(error: str) -> list[dict[str, Any]]:
    """进程重启后把上次没跑完的 running 一次性判死，返回改过的任务用于回调。"""
    stale = running()
    if not stale:
        return []
    now = _now()
    with _lock:
        client().execute(
            "UPDATE tasks SET status='failed', error=?, updated_at=? WHERE status='running'",
            (error, now),
        )
    for task in stale:
        task["status"] = "failed"
        task["error"] = error
        task["updated_at"] = now
    return stale
