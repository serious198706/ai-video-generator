from __future__ import annotations

import json
import threading
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import BotoCoreError, ClientError

from wan22 import config
from wan22.api.schemas import GenerateRequest
from wan22.log import get_logger
from wan22.media import status, webhook
from wan22.net.urlguard import UrlError, assert_image_source
from wan22.tasks import runner

logger = get_logger(__name__)

# vast.ai 上没有 IAM Role：凭证走 AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY 环境变量，
# 权限只需要目标队列的 ReceiveMessage / DeleteMessage / ChangeMessageVisibility /
# GetQueueAttributes。进程不排队：执行位被占的消息 ChangeMessageVisibility 退回。
# 无本地留档（SQLite 已删）：幂等看 S3 状态文件，崩溃恢复靠 SQS 重投。
_stop = threading.Event()
_thread: threading.Thread | None = None
_client = None
_client_lock = threading.Lock()


def enabled() -> bool:
    return bool(config.SQS_QUEUE_URL)


def start() -> None:
    """lifespan 里调用。未配 SQS_QUEUE_URL 则整个跳过。"""
    global _thread
    if not enabled():
        logger.info("sqs consumer disabled: SQS_QUEUE_URL is empty")
        return
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
        # 执行位空着才去取消息；上一单还在跑时轮询只会白拉。
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
    task_id, request, error, data = _parse(body_raw)

    if error is not None:
        # lambda 侧已按同一套规则校验过；走到这里基本是坏消息，到次数就判死删除。
        if receive_count >= config.SQS_MAX_RECEIVE:
            if task_id:
                _fail_and_delete(task_id, receipt, error, data)
            else:
                logger.error("drop poison message after %s receives: %s", receive_count, error)
                _delete(receipt)
        else:
            logger.warning("reject message (receive=%s): %s", receive_count, error)
            _requeue(receipt, config.SQS_BUSY_BACKOFF)
        return

    assert task_id is not None and request is not None
    if receive_count >= config.SQS_MAX_RECEIVE:
        logger.error("task=%s hit max receive %s, fail and delete", task_id, receive_count)
        _fail_and_delete(task_id, receipt, "poisoned", data)
        return

    # 幂等：重复投递时 S3 状态文件已有终态就删消息；进行中/不存在则继续走。
    if status.is_terminal(task_id):
        logger.info("duplicate delivery, task=%s already terminal", task_id)
        _delete(receipt)
        return

    # 执行位被占：消息退回队列稍后重投，进程内不排队。
    if not runner.reserve(task_id):
        logger.info("slot busy, requeue task=%s", task_id)
        _requeue(receipt, config.SQS_BUSY_BACKOFF)
        return

    task = _build_task(task_id, request)
    # lambda 入队时已把这份 dict 写成状态文件里的 request；终态原样带回，不丢格式。
    task["request"] = data
    logger.info("dequeued task=%s receive=%s", task_id, receive_count)
    # request 用消息原文，与 lambda queued 文件里的完全一致。
    status.write(
        task_id,
        "pending",
        receive_count=receive_count,
        request=data,
    )
    runner.spawn(task)
    threading.Thread(
        target=_watch_until_done,
        args=(task_id, receipt),
        name=f"wan22-sqs-ack-{task_id[:8]}",
        daemon=True,
    ).start()


def _parse(body_raw: str) -> tuple[str | None, GenerateRequest | None, str | None, dict | None]:
    """SQS body 是 lambda_api 的 JSON：task_id + image_url / webhook_url 等蛇形字段。

    第四个返回值是原始 dict（即 lambda 写进 queued 状态文件的 request），原样透传。
    """
    try:
        data = json.loads(body_raw)
    except (TypeError, ValueError):
        return None, None, "body is not json", None
    if not isinstance(data, dict):
        return None, None, "body is not an object", None
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
        return task_id, None, f"invalid fields: {exc}", data
    try:
        assert_image_source(request.image, config.IMAGE_HOSTS, kind="image")
        if request.last_image:
            assert_image_source(request.last_image, config.IMAGE_HOSTS, kind="image")
    except UrlError as exc:
        return task_id, None, f"image url rejected: {exc}", data
    if (
        (request.resolution or "").lower() == "1080p"
        and not config.DRY_RUN
        and not config.UPSCALE_ENABLE
    ):
        return task_id, None, "1080p requires upscale", data
    return task_id, request, None, data


def _build_task(task_id: str, request: GenerateRequest) -> dict[str, Any]:
    """内存里的任务字典：从消息直接来，不留档。"""
    return {
        "id": task_id,
        "prompt": (request.prompt or "").strip() or config.DEFAULT_PROMPT,
        "negative_prompt": (request.negative_prompt or "").strip() or config.NEGATIVE_PROMPT,
        "image_url": request.image,
        "last_image_url": request.last_image,
        "duration": request.duration,
        "resolution": request.resolution,
        "webhook_url": request.webhook_url,
        "seed": request.seed,
        "steps": request.steps,
        "quality": request.quality,
        "audio": bool(request.audio),
    }


def _fail_and_delete(task_id: str, receipt: str, error: str, data: dict | None) -> None:
    """坏消息也要让 Java 收到一次失败：写 S3 failed（带短码）+ webhook + 删消息。"""
    logger.error("failing task=%s error=%s", task_id, error)
    fields = {"error": error}
    if data is not None:
        fields["request"] = data
    status.write(task_id, "failed", **fields)
    webhook.notify({"id": task_id, "error": error, "status": "failed"})
    _delete(receipt)


def _watch_until_done(task_id: str, receipt: str) -> None:
    """生成可能跑十几分钟：每 SQS_HEARTBEAT_SECONDS 续一次 visibility，终态后删消息。

    终态由 _finish 先写 S3 状态文件，这里用轮询事件标志判断（不做本地留档查询）。
    """
    while not _stop.is_set():
        _stop.wait(config.SQS_HEARTBEAT_SECONDS)
        if runner.current() != task_id:
            # 执行位已释放 = 流水线跑完（_finish 已写终态）。
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
