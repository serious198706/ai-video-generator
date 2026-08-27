from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import redis

from wan22 import config

# ElastiCache Serverless 按 Cluster slot 校验 MULTI。{wan22} 让 queue / running / task 同槽。
QUEUE_KEY = "{wan22}:queue"
RUNNING_KEY = "{wan22}:running"
TASK_PREFIX = "{wan22}:task:"

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

_client: redis.Redis | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def client() -> redis.Redis:
    global _client
    if _client is None:
        # Serverless + TLS 会掐空闲/阻塞读。不要设 socket_timeout，也不要用 BRPOP。
        _client = redis.Redis.from_url(
            config.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_keepalive=True,
            health_check_interval=30,
            retry_on_timeout=True,
        )
    return _client


def reset_client() -> None:
    global _client
    old = _client
    _client = None
    if old is None:
        return
    try:
        old.close()
    except Exception:
        pass


def ping() -> None:
    client().ping()


def task_key(task_id: str) -> str:
    return f"{TASK_PREFIX}{task_id}"


def _dump(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _load(raw: dict[str, str]) -> dict[str, Any]:
    task: dict[str, Any] = dict(raw)
    for key in ("duration",):
        if task.get(key) not in (None, ""):
            task[key] = float(task[key])
    for key in ("seed", "steps", "quality"):
        if task.get(key) in (None, ""):
            task[key] = None
        else:
            task[key] = int(task[key])
    for key in _OPTIONAL:
        if task.get(key) in (None, ""):
            task[key] = None
    return task


def create_task(task_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    now = _now()
    payload = {
        "id": task_id,
        "status": "queued",
        "created_at": now,
        "updated_at": now,
        **{key: _dump(value) for key, value in fields.items()},
    }
    pipe = client().pipeline()
    pipe.hset(task_key(task_id), mapping=payload)
    pipe.rpush(QUEUE_KEY, task_id)
    pipe.execute()
    return get_task(task_id)  # type: ignore[return-value]


def update_task(task_id: str, **fields: Any) -> None:
    if not fields:
        return
    fields["updated_at"] = _now()
    client().hset(task_key(task_id), mapping={key: _dump(value) for key, value in fields.items()})


def get_task(task_id: str) -> dict[str, Any] | None:
    raw = client().hgetall(task_key(task_id))
    if not raw:
        return None
    return _load(raw)


def pending() -> int:
    count = int(client().llen(QUEUE_KEY))
    if client().get(RUNNING_KEY):
        count += 1
    return count


def pop_task(timeout: int = 5) -> str | None:
    """RPUSH + LPOP。Serverless 上 BRPOP 会被 TLS 读超时打死。"""
    deadline = time.monotonic() + max(timeout, 0)
    while True:
        try:
            item = client().lpop(QUEUE_KEY)
        except (redis.TimeoutError, redis.ConnectionError, OSError):
            reset_client()
            item = None
        if item is not None:
            return item
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


def set_running(task_id: str) -> None:
    client().set(RUNNING_KEY, task_id)


def clear_running(task_id: str | None = None) -> None:
    current = client().get(RUNNING_KEY)
    if task_id is None or current == task_id:
        client().delete(RUNNING_KEY)


def take_interrupted() -> str | None:
    """启动时取走残留 running。"""
    pipe = client().pipeline()
    pipe.get(RUNNING_KEY)
    pipe.delete(RUNNING_KEY)
    current, _ = pipe.execute()
    return current
