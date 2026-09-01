from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from wan22 import config
from wan22.api.schemas import GenerateRequest
from wan22.log import get_logger, setup_logging
from wan22.net.urlguard import UrlError, assert_https_url
from wan22.queue import store

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging(force=True)
    parsed = urlparse(config.REDIS_URL)
    logger.info(
        "lambda api starting queue_max=%s redis=%s://%s:%s",
        config.QUEUE_MAX,
        parsed.scheme,
        parsed.hostname,
        parsed.port or 6379,
    )
    # 不在 lifespan 里 ping：连不上时会卡满超时，CloudWatch 里看不到异常。
    yield


app = FastAPI(
    title="Wan 2.2 I2V API",
    version="0.3.0",
    docs_url="/docs" if config.ENABLE_DOCS else None,
    redoc_url="/redoc" if config.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if config.ENABLE_DOCS else None,
    lifespan=lifespan,
)


def _public_task(task: dict) -> dict:
    return {
        "id": task["id"],
        "task_id": task["id"],
        "status": task["status"],
        "prompt": task["prompt"],
        "duration": task["duration"],
        "resolution": task.get("resolution"),
        "seed": task.get("seed"),
        "video_url": task.get("video_url"),
        "error": task.get("error"),
        "audio": bool(task.get("audio")),
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
    }


@app.post("/v1/generate")
def create_generation(body: GenerateRequest):
    try:
        pending = store.pending()
        if pending >= config.QUEUE_MAX:
            logger.warning("queue full pending=%s max=%s", pending, config.QUEUE_MAX)
            raise HTTPException(429, "queue full")
    except HTTPException:
        raise
    except Exception:
        logger.exception("queue unavailable on enqueue")
        raise HTTPException(503, "queue unavailable")

    if body.webhook_url:
        try:
            assert_https_url(
                body.webhook_url,
                config.WEBHOOK_HOSTS,
                kind="webhook",
                allow_private=True,
            )
        except UrlError as exc:
            logger.warning("reject webhookUrl: %s", exc)
            raise HTTPException(400, str(exc)) from exc

    try:
        assert_https_url(body.image, config.IMAGE_HOSTS, kind="image")
        if body.last_image:
            assert_https_url(body.last_image, config.IMAGE_HOSTS, kind="image")
    except UrlError as exc:
        logger.warning("reject image url: %s", exc)
        raise HTTPException(400, str(exc)) from exc

    task_id = uuid.uuid4().hex
    prompt = (body.prompt or "").strip() or config.DEFAULT_PROMPT
    negative = (body.negative_prompt or "").strip() or config.NEGATIVE_PROMPT
    try:
        store.create_task(
            task_id,
            {
                "prompt": prompt,
                "negative_prompt": negative,
                "image_url": body.image,
                "last_image_url": body.last_image,
                "first_frame_path": None,
                "last_frame_path": None,
                "duration": body.duration,
                "resolution": body.resolution,
                "webhook_url": body.webhook_url,
                "seed": body.seed,
                "steps": body.steps,
                "quality": body.quality,
                "audio": bool(body.audio),
            },
        )
    except Exception:
        logger.exception("task=%s enqueue failed", task_id)
        raise HTTPException(503, "queue unavailable") from None

    logger.info(
        "queued task=%s duration=%s resolution=%s steps=%s quality=%s audio=%s pending=%s webhook=%s",
        task_id,
        body.duration,
        body.resolution,
        body.steps,
        body.quality,
        bool(body.audio),
        pending + 1,
        bool(body.webhook_url),
    )
    return JSONResponse({"id": task_id, "task_id": task_id, "status": "queued"}, status_code=202)


@app.get("/v1/tasks/{task_id}")
def get_task(task_id: str):
    try:
        task = store.get_task(task_id)
    except Exception:
        logger.exception("queue unavailable on get task=%s", task_id)
        raise HTTPException(503, "queue unavailable") from None
    if not task:
        raise HTTPException(404, "task 不存在")
    return _public_task(task)


@app.get("/health")
def health():
    parsed = urlparse(config.REDIS_URL)
    logger.info("redis ping host=%s", parsed.hostname)
    try:
        store.ping()
    except Exception:
        logger.exception("redis ping failed host=%s", parsed.hostname)
        raise HTTPException(503, "queue unavailable") from None
    return {"ok": True}
