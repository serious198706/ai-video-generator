from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from wan22 import config
from wan22.log import get_logger
from wan22.media.upload import client as s3_client

logger = get_logger(__name__)

# 单 key 覆盖写：{prefix}status/{task_id}.json。S3 没有 rename，独立文件名
# （.pending/.generating）切换要 copy+delete 两步，中途会查到 0 或 2 个文件。
# 覆盖 PutObject 是原子的，永远只读到一个对象，内容还能带时间戳/错误码等属性。
STATUSES = ("queued", "pending", "generating", "succeeded", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def key(task_id: str) -> str:
    prefix = config.STATUS_PREFIX.strip("/")
    name = f"status/{task_id}.json"
    return f"{prefix}/{name}" if prefix else name


def write(task_id: str, status: str, **fields: Any) -> None:
    """写状态文件。失败只记日志：状态是旁路观测，不能影响生成/回调主流程。"""
    if not config.S3_BUCKET:
        return
    if status not in STATUSES:
        raise ValueError(f"unknown status: {status}")
    payload = {
        "task_id": task_id,
        "status": status,
        "updated_at": _now(),
        "worker_id": config.WORKER_ID,
    }
    payload.update({k: v for k, v in fields.items() if v is not None})
    try:
        s3_client().put_object(
            Bucket=config.S3_BUCKET,
            Key=key(task_id),
            Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
            CacheControl="no-store",
        )
        logger.info("status %s task=%s", status, task_id)
    except Exception:
        # 状态文件是旁路观测：网络/权限/凭证任何问题都不能让生成或回调失败。
        logger.exception("status write failed status=%s task=%s", status, task_id)
