from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
S3_BUCKET = os.environ.get("S3_BUCKET", "").strip()
STATUS_PREFIX = os.environ.get("STATUS_PREFIX", "video/").strip().strip("/")
# 本地/自建 S3 兼容端点（可选）。线上留空走默认 AWS 端点。
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "").strip()
# generating 超过这个秒数没有更新就报 stalled：GPU 挂了/重启了/卡住。
# 要大于单次生成最坏耗时（1080p + Foley）。
STALL_SECONDS = int(os.environ.get("STATUS_STALL_SECONDS", "2700"))

_s3 = boto3.client(
    "s3",
    region_name=AWS_REGION,
    **({"endpoint_url": S3_ENDPOINT} if S3_ENDPOINT else {}),
)



def key(task_id: str) -> str:
    name = f"status/{task_id}.json"
    return f"{STATUS_PREFIX}/{name}" if STATUS_PREFIX else name


def read(task_id: str) -> dict[str, Any] | None:
    """读状态文件。没有对象返回 None；S3 抖动抛 RuntimeError（HTTP 503），不要伪装成 unknown。"""
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET is not configured")
    try:
        obj = _s3.get_object(Bucket=S3_BUCKET, Key=key(task_id))
        body = obj["Body"].read()
    except ClientError as exc:
        code = (exc.response.get("Error") or {}).get("Code", "")
        if code in {"NoSuchKey", "404", "NotFound"}:
            return None
        logger.exception("status read failed task=%s", task_id)
        raise RuntimeError("status read failed") from exc
    except BotoCoreError as exc:
        logger.exception("status read failed task=%s", task_id)
        raise RuntimeError("status read failed") from exc
    try:
        data = json.loads(body)
    except ValueError:
        logger.warning("status file not json task=%s", task_id)
        return None
    return data if isinstance(data, dict) else None


def write_queued(task_id: str, request: dict[str, Any]) -> None:
    """入队时写 queued，让查询端能区分"排着队"和"没这单"。失败不影响入队。"""
    if not S3_BUCKET:
        return
    payload = {
        "task_id": task_id,
        "status": "queued",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "request": request,
    }
    try:
        _s3.put_object(
            Bucket=S3_BUCKET,
            Key=key(task_id),
            Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            ContentType="application/json",
            CacheControl="no-store",
        )
    except (BotoCoreError, ClientError):
        logger.exception("status write failed task=%s", task_id)


def describe(task_id: str) -> dict[str, Any]:
    """对外响应。succeeded/failed 原样带 video_url/error；generating 超时报 stalled。"""
    data = read(task_id)
    if data is None:
        return {"id": task_id, "task_id": task_id, "status": "unknown"}
    result = dict(data)
    result["id"] = task_id
    result["task_id"] = task_id
    status = result.get("status")
    if status == "generating" and _stalled(result.get("updated_at")):
        result["status"] = "stalled"
        result["error"] = "worker_unresponsive"
    return result


def _stalled(updated_at: Any) -> bool:
    if STALL_SECONDS <= 0 or not isinstance(updated_at, str):
        return False
    try:
        seen = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
    except ValueError:
        return False
    if seen.tzinfo is None:
        seen = seen.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - seen).total_seconds()
    return age > STALL_SECONDS
