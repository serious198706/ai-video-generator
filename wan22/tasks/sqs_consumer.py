from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from wan22 import config
from wan22.api.schemas import GenerateRequest
from wan22.log import get_logger
from wan22.media import status, webhook
from wan22.net.urlguard import UrlError, assert_image_source
from wan22.tasks import runner, store

logger = get_logger(__name__)

# vast.ai 上没有 IAM Role：凭证走 AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY 环境变量，
# 权限只需要目标队列的 ReceiveMessage / DeleteMessage / ChangeMessageVisibility /
# GetQueueAttributes。进程不设队列：HTTP POST 和 SQS 共用 runner 的单执行位，
# 抢不到位的消息 ChangeMessageVisibility 退避，绝不本地堆积。
_stop = threading.Event()
_thread: threading.Thread | None = None
_client = None
_client_lock = threading.Lock()


def enabled() -> bool:
    return bool(config.SQS_QUEUE_URL)


def start() -> None:
    """lifespan 里调用。未配 SQS_QUEUE_URL 则整个跳过，HTTP 直打不受影响。"""
    global _thread
    if not enabled():
        logger.info("sqs consumer disabled: SQS_QUEUE_URL is empty")
        return
    if config.DRY_RUN:
        # dry-run 常在本机起，没有真实队列时不至于刷错误；配了队列也允许跑（会真的消费）。
        logger.info("sqs consumer starting in dry-run queue=%s", config.SQS_QUEUE_URL)
    if _thread is not None:
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="wan22-sqs", daemon=True)
    _thread.start()
    logger.info("sqs consumer started queue=%s", config.SQS_QUEUE_URL)


def stop() -> None:
    _stop.set()
    thread = _thread
    if thread is not None:
        thread.join(timeout=5)


def _sqs():
    global _client
    with _client_lock:
        if _client is None:
            kwargs: dict[str, Any] = {
                "config": BotoConfig(retries={"max_attempts": 5, "mode": "standard"}),
            }
            if config.SQS_REGION:
                kwargs["region_name"] = config.SQS_REGION
            _client = boto3.client("sqs", **kwargs)
        return _client


def _loop() -> None:
    idle_backoff = 1.0
    while not _stop.is_set():
        # 执行位空着才去取消息；HTTP POST 正在跑时轮询只会白拉。
        if runner.current() is not None:
            _stop.wait(2)
            continue
        try:
            response = _sqs().receive_message(
                QueueUrl=config.SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=config.SQS_WAIT_SECONDS,
                VisibilityTimeout=config.SQS_VISIBILITY_TIMEOUT,
                AttributeNames=["ApproximateReceiveCount"],
            )
            idle_backoff = 1.0
        except (BotoCoreError, ClientError) as exc:
            logger.warning("sqs receive failed: %s; backoff %.0fs", exc, idle_backoff)
            _stop.wait(idle_backoff)
            idle_backoff = min(idle_backoff * 2, 30)
            continue
        for message in response.get("Messages", []):
            _handle(message)


def _handle(message: dict[str, Any]) -> None:
    receipt = message["ReceiptHandle"]
    body_raw = message.get("Body") or ""
    receive_count = int((message.get("Attributes") or {}).get("ApproximateReceiveCount") or "1")
    task_id, request, error = _parse(body_raw)

    if error is not None:
        # lambda 侧已按同一套规则校验过；走到这里基本是坏消息，按 DLQ 前的次数处理。
        if task_id and receive_count >= config.SQS_MAX_RECEIVE:
            _fail_and_delete(task_id, receipt, body_raw, "invalid_request")
        elif receive_count >= config.SQS_MAX_RECEIVE:
            logger.error("drop poison message after %s receives: %s", receive_count, error)
            _delete(receipt)
        else:
            logger.warning("reject message (receive=%s): %s", receive_count, error)
            _requeue(receipt, config.SQS_BUSY_BACKOFF)
        return

    assert task_id is not None and request is not None
    if receive_count >= config.SQS_MAX_RECEIVE:
        logger.error("task=%s hit max receive %s, fail and delete", task_id, receive_count)
        _fail_and_delete(task_id, receipt, body_raw, "poisoned")
        return

    # 执行位被 HTTP POST 抢走：消息退回队列稍后重投，不排队。
    if not runner.reserve(task_id):
        logger.info("slot busy, requeue task=%s", task_id)
        _requeue(receipt, config.SQS_BUSY_BACKOFF)
        return

    try:
        _create_task(task_id, request)
    except sqlite3.IntegrityError:
        # 重复投递（上次进程崩在跑完前）。已在跑就退回等终态；已有终态直接删。
        runner.release(task_id)
        existing = store.get_task(task_id)
        if existing and existing.get("status") != "running":
            logger.info("duplicate delivery, task=%s already %s", task_id, existing["status"])
            _delete(receipt)
        else:
            _requeue(receipt, config.SQS_BUSY_BACKOFF)
        return
    except Exception:
        runner.release(task_id)
        logger.exception("failed to create task=%s", task_id)
        _requeue(receipt, config.SQS_BUSY_BACKOFF)
        return

    logger.info("dequeued task=%s receive=%s", task_id, receive_count)
    # 状态文件带上入参，事后能直接看出这台机器收到了什么。
    status.write(
        task_id,
        "pending",
        receive_count=receive_count,
        request=request.model_dump(by_alias=True, exclude_none=True),
    )
    # 后台线程跑流水线；看门线程按心跳续 visibility，终态后删消息。
    runner.spawn(task_id)
    threading.Thread(
        target=_watch_until_done,
        args=(task_id, receipt),
        name=f"wan22-sqs-ack-{task_id[:8]}",
        daemon=True,
    ).start()


def _parse(body_raw: str) -> tuple[str | None, GenerateRequest | None, str | None]:
    """SQS body 是 lambda_api 的 JSON：task_id + image_url / webhook_url 等蛇形字段。"""
    try:
        data = json.loads(body_raw)
    except (TypeError, ValueError):
        return None, None, "body is not json"
    if not isinstance(data, dict):
        return None, None, "body is not an object"
    task_id = str(data.get("task_id") or "").strip() or None
    try:
        request = GenerateRequest(
            image=data.get("image_url") or "",
            prompt=data.get("prompt"),
            negativePrompt=data.get("negative_prompt"),
            duration=data.get("duration") or 5,
            resolution=data.get("resolution"),
            webhookUrl=data.get("webhook_url"),
            steps=data.get("steps"),
            quality=data.get("quality"),
            seed=data.get("seed"),
            lastImage=data.get("last_image_url"),
            audio=bool(data.get("audio")),
        )
    except ValueError as exc:
        return task_id, None, f"invalid fields: {exc}"
    try:
        assert_image_source(request.image, config.IMAGE_HOSTS, kind="image")
        if request.last_image:
            assert_image_source(request.last_image, config.IMAGE_HOSTS, kind="image")
    except UrlError as exc:
        return task_id, None, f"image url rejected: {exc}"
    if (
        (request.resolution or "").lower() == "1080p"
        and not config.DRY_RUN
        and not config.UPSCALE_ENABLE
    ):
        return task_id, None, "1080p requires upscale"
    return task_id, request, None


def _create_task(task_id: str, request: GenerateRequest) -> None:
    store.create_task(
        task_id,
        {
            "prompt": (request.prompt or "").strip() or config.DEFAULT_PROMPT,
            "negative_prompt": (request.negative_prompt or "").strip() or config.NEGATIVE_PROMPT,
            "image_url": request.image,
            "last_image_url": request.last_image,
            "first_frame_path": None,
            "last_frame_path": None,
            "duration": request.duration,
            "resolution": request.resolution,
            "webhook_url": request.webhook_url,
            "seed": request.seed,
            "steps": request.steps,
            "quality": request.quality,
            "audio": bool(request.audio),
        },
    )


def _fail_and_delete(task_id: str, receipt: str, body_raw: str, error: str) -> None:
    """坏消息也要让 Java 收到一次失败回调：能建档就建档后立刻判死。"""
    try:
        if store.get_task(task_id) is None:
            data = json.loads(body_raw) if body_raw else {}
            fields = data if isinstance(data, dict) else {}
            store.create_task(
                task_id,
                {
                    "prompt": fields.get("prompt"),
                    "negative_prompt": fields.get("negative_prompt"),
                    "image_url": fields.get("image_url"),
                    "last_image_url": fields.get("last_image_url"),
                    "duration": fields.get("duration") or 5,
                    "resolution": fields.get("resolution"),
                    "webhook_url": fields.get("webhook_url"),
                    "seed": fields.get("seed"),
                    "steps": fields.get("steps"),
                    "quality": fields.get("quality"),
                    "audio": bool(fields.get("audio")),
                },
            )
        store.update_task(task_id, status="failed", error=error, video_url=None)
        task = store.get_task(task_id)
        if task:
            status.write(task_id, "failed", error=error)
            webhook.notify(task)
    except Exception:
        logger.exception("failed to record poison task=%s", task_id)
    _delete(receipt)


def _watch_until_done(task_id: str, receipt: str) -> None:
    """生成可能跑十几分钟：每 SQS_HEARTBEAT_SECONDS 续一次 visibility，终态后删除消息。"""
    while not _stop.is_set():
        _stop.wait(config.SQS_HEARTBEAT_SECONDS)
        task = store.get_task(task_id)
        if task is None or task.get("status") != "running":
            _delete(receipt)
            return
        try:
            _sqs().change_message_visibility(
                QueueUrl=config.SQS_QUEUE_URL,
                ReceiptHandle=receipt,
                VisibilityTimeout=config.SQS_VISIBILITY_TIMEOUT,
            )
        except (BotoCoreError, ClientError):
            logger.exception("visibility heartbeat failed task=%s", task_id)
    logger.warning("consumer stopping, task=%s message not acked; will redeliver", task_id)


def _delete(receipt: str) -> None:
    try:
        _sqs().delete_message(QueueUrl=config.SQS_QUEUE_URL, ReceiptHandle=receipt)
    except (BotoCoreError, ClientError):
        logger.exception("delete message failed")


def _requeue(receipt: str, delay: int) -> None:
    try:
        _sqs().change_message_visibility(
            QueueUrl=config.SQS_QUEUE_URL,
            ReceiptHandle=receipt,
            VisibilityTimeout=delay,
        )
    except (BotoCoreError, ClientError):
        logger.exception("requeue failed")
