from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager

import boto3
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from wan22.api.schemas import GenerateRequest
from wan22.net.urlguard import UrlError, assert_https_url, assert_image_source

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
QUEUE_URL = os.environ.get("SQS_QUEUE_URL", "").strip()
SQS_MAX_QUEUE = int(os.environ.get("SQS_MAX_QUEUE", "0"))
# 显式传 region：vpc 内解析不到 region 时 boto3 会去访问 IMDS，
# 而 lambda 在 vpc 里访问 169.254.169.254 会一直挂到函数超时。
AWS_REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
_sqs = boto3.client("sqs", region_name=AWS_REGION)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("lambda queue api starting configured=%s region=%s", bool(QUEUE_URL), AWS_REGION)
    yield


app = FastAPI(
    title="Wan 2.2 I2V Queue API",
    version="0.4.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.post("/v1/generate")
def enqueue_generation(body: GenerateRequest):
    if not QUEUE_URL:
        logger.error("SQS_QUEUE_URL is not configured")
        raise HTTPException(503, "queue is not configured")

    if body.webhook_url:
        try:
            assert_https_url(body.webhook_url, (), kind="webhook", allow_private=True)
        except UrlError as exc:
            raise HTTPException(400, str(exc)) from exc

    try:
        assert_image_source(body.image, (), kind="image")
        if body.last_image:
            assert_image_source(body.last_image, (), kind="image")
    except UrlError as exc:
        raise HTTPException(400, str(exc)) from exc

    if SQS_MAX_QUEUE > 0:
        try:
            logger.info("checking queue size limit=%s", SQS_MAX_QUEUE)
            attributes = _sqs.get_queue_attributes(
                QueueUrl=QUEUE_URL,
                AttributeNames=["ApproximateNumberOfMessages"],
            ).get("Attributes", {})
            queue_size = int(attributes.get("ApproximateNumberOfMessages", "0"))
            logger.info("queue size=%s", queue_size)
        except Exception as exc:
            logger.exception("failed to inspect queue before enqueue")
            raise HTTPException(503, "failed to inspect queue") from exc

        if queue_size >= SQS_MAX_QUEUE:
            logger.warning(
                "queue full size=%s limit=%s",
                queue_size,
                SQS_MAX_QUEUE,
            )
            raise HTTPException(429, "queue is full")

    task_id = uuid.uuid4().hex
    task = {
        "task_id": task_id,
        "prompt": (body.prompt or "").strip(),
        "negative_prompt": (body.negative_prompt or "").strip(),
        "image_url": body.image,
        "last_image_url": body.last_image,
        "duration": body.duration,
        "resolution": body.resolution,
        "webhook_url": body.webhook_url,
        "seed": body.seed,
        "steps": body.steps,
        "quality": body.quality,
        "audio": bool(body.audio),
    }

    try:
        logger.info("sending task=%s", task_id)
        result = _sqs.send_message(
            QueueUrl=QUEUE_URL,
            MessageBody=json.dumps(task, ensure_ascii=False),
        )
    except Exception as exc:
        logger.exception("failed to enqueue task=%s", task_id)
        raise HTTPException(503, "failed to enqueue task") from exc

    logger.info("enqueued task=%s message=%s", task_id, result.get("MessageId"))
    return JSONResponse(
        {"id": task_id, "task_id": task_id, "status": "queued"},
        status_code=202,
    )


@app.get("/health")
def health():
    return {"ok": True, "queue_configured": bool(QUEUE_URL)}
