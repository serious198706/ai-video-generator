from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timezone
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
)

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
        "video_url",
        "error",
        "in_queue",
    }
)

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def client() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        path = config.QUEUE_DB
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
        conn.execute("PRAGMA foreign_keys=ON")
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
                video_url TEXT,
                error TEXT,
                in_queue INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_queue "
            "ON tasks(created_at) WHERE in_queue=1"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)"
        )
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
    return value


def _load(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    task: dict[str, Any] = {key: row[key] for key in row.keys() if key != "in_queue"}
    if task.get("duration") is not None:
        task["duration"] = float(task["duration"])
    for key in ("seed", "steps", "quality"):
        if task.get(key) is not None:
            task[key] = int(task[key])
    for key in _OPTIONAL:
        if task.get(key) == "":
            task[key] = None
    return task


def create_task(task_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    now = _now()
    with _lock:
        client().execute(
            """
            INSERT INTO tasks (
                id, status, prompt, negative_prompt, image_url, last_image_url,
                first_frame_path, last_frame_path, duration, resolution, webhook_url,
                seed, steps, quality, video_url, error, in_queue, created_at, updated_at
            ) VALUES (
                ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, 1, ?, ?
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
    values = list(payload.values())
    values.append(task_id)
    with _lock:
        client().execute(
            f"UPDATE tasks SET {assignments} WHERE id=?",
            values,
        )


def get_task(task_id: str) -> dict[str, Any] | None:
    with _lock:
        row = client().execute(
            "SELECT * FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
    return _load(row)


def pending() -> int:
    with _lock:
        row = client().execute(
            """
            SELECT COUNT(*) AS n FROM tasks
            WHERE (status='queued' AND in_queue=1) OR status='running'
            """
        ).fetchone()
    return int(row["n"] if row else 0)


def pop_task(timeout: int = 5) -> str | None:
    deadline = time.monotonic() + max(timeout, 0)
    while True:
        with _lock:
            conn = client()
            row = conn.execute(
                """
                SELECT id FROM tasks
                WHERE status='queued' AND in_queue=1
                ORDER BY created_at, rowid
                LIMIT 1
                """
            ).fetchone()
            if row is not None:
                task_id = row["id"]
                updated = conn.execute(
                    "UPDATE tasks SET in_queue=0 WHERE id=? AND in_queue=1",
                    (task_id,),
                )
                if updated.rowcount == 1:
                    return task_id
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


def set_running(task_id: str) -> None:
    update_task(task_id, status="running", error=None)


def clear_running(task_id: str | None = None) -> None:
    # 状态已在 succeeded/failed 里落库；SQLite 不另存 running key。
    return


def take_interrupted() -> str | None:
    """启动时取走残留 running。"""
    with _lock:
        row = client().execute(
            """
            SELECT id FROM tasks
            WHERE status='running'
            ORDER BY updated_at, rowid
            LIMIT 1
            """
        ).fetchone()
    return row["id"] if row else None
