from __future__ import annotations

import json
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis

from wan22 import config

# ElastiCache Serverless 按 Cluster slot 校验 MULTI。{wan22} 让 queue / task 同槽。
QUEUE_KEY = "{wan22}:queue"
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
    "worker_id",
)

_client: redis.Redis | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _worker_id() -> str:
    return getattr(config, "WORKER_ID", None) or f"{socket.gethostname()}"


def _running_file() -> Path:
    safe = "".join(ch if ch.isalnum() or ch in ".-" else "_" for ch in _worker_id())
    return config.ROOT / f"worker-running.{safe}"


def client() -> redis.Redis:
    global _client
    if _client is None:
        # Serverless TLS 会掐空闲连接。LPOP 非阻塞，必须设 socket_timeout，
        # 否则成片几十秒没人碰 Redis，下一次 GET/SET 会永远卡住。
        _client = redis.Redis.from_url(
            config.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            socket_keepalive=True,
            health_check_interval=30,
            # 单次命令不要内部再重试：5s×2 会吃光 Lambda 10s，日志只剩 timeout。
            retry_on_timeout=False,
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
    """只 ping 一次。失败原样抛出，避免卡到 Lambda 超时却没有任何错误日志。"""
    client().ping()


def _retry(op):
    try:
        return op()
    except (redis.TimeoutError, redis.ConnectionError, OSError):
        reset_client()
        return op()


def task_key(task_id: str) -> str:
    return f"{TASK_PREFIX}{task_id}"


def _load(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    task = json.loads(raw)
    if task.get("duration") is not None:
        task["duration"] = float(task["duration"])
    for key in ("seed", "steps", "quality", "attempts"):
        if task.get(key) is not None and task.get(key) != "":
            task[key] = int(task[key])
        elif key == "attempts":
            task[key] = 0
    if "audio" in task:
        task["audio"] = True if task["audio"] is None else bool(task["audio"])
    else:
        task["audio"] = True
    for key in _OPTIONAL:
        if task.get(key) in ("",):
            task[key] = None
    return task


def _dump_task(task: dict[str, Any]) -> str:
    return json.dumps(task, ensure_ascii=False, default=str)


def create_task(task_id: str, fields: dict[str, Any]) -> dict[str, Any]:
    now = _now()
    payload = {
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
        "audio": True if fields.get("audio") is None else bool(fields.get("audio")),
        "video_url": None,
        "error": None,
    }

    def _create() -> None:
        pipe = client().pipeline()
        pipe.set(task_key(task_id), _dump_task(payload))
        pipe.rpush(QUEUE_KEY, task_id)
        pipe.execute()

    _retry(_create)
    return get_task(task_id)  # type: ignore[return-value]


def update_task(task_id: str, **fields: Any) -> None:
    if not fields:
        return

    def _update() -> None:
        raw = client().get(task_key(task_id))
        task = _load(raw)
        if not task:
            return
        for key, value in fields.items():
            if value == "":
                value = None
            task[key] = value
        task["updated_at"] = _now()
        client().set(task_key(task_id), _dump_task(task))

    _retry(_update)


def get_task(task_id: str) -> dict[str, Any] | None:
    raw = _retry(lambda: client().get(task_key(task_id)))
    return _load(raw)


def pending() -> int:
    return int(_retry(lambda: client().llen(QUEUE_KEY)) or 0)


def pop_task(timeout: int = 5) -> str | None:
    """RPUSH + LPOP。Serverless 上 BRPOP 会被 TLS 读超时打死。LPOP 即抢锁。"""
    deadline = time.monotonic() + max(timeout, 0)
    while True:
        try:
            item = client().lpop(QUEUE_KEY)
        except (redis.TimeoutError, redis.ConnectionError, OSError):
            reset_client()
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.25)
            continue
        if item is not None:
            return item
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.25)


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
    """本机上次没跑完的 running。不抢其它机器的任务。"""
    path = _running_file()
    if not path.is_file():
        return None
    task_id = path.read_text(encoding="utf-8").strip()
    path.unlink(missing_ok=True)
    return task_id or None


def requeue(task_id: str, *, attempts: int, error: str | None = None) -> None:
    """失败未超限：写回 queued 并 RPUSH。"""

    def _requeue() -> None:
        raw = client().get(task_key(task_id))
        task = _load(raw)
        if not task:
            return
        task["status"] = "queued"
        task["attempts"] = attempts
        task["error"] = error
        task["worker_id"] = None
        task["updated_at"] = _now()
        pipe = client().pipeline()
        pipe.set(task_key(task_id), _dump_task(task))
        pipe.rpush(QUEUE_KEY, task_id)
        pipe.execute()

    _retry(_requeue)
