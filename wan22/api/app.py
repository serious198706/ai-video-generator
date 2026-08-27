from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from wan22 import config
from wan22.api.schemas import GenerateRequest
from wan22.infer import generate
from wan22.log import get_logger, setup_logging
from wan22.media import download
from wan22.media.upload import assert_configured as assert_s3
from wan22.net.urlguard import UrlError, assert_https_url
from wan22.queue import store, worker

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging(force=True)
    logger.info("service starting dry_run=%s docs=%s", config.DRY_RUN, config.ENABLE_DOCS)
    store.ping()
    logger.info("redis ok")
    if not config.DRY_RUN:
        assert_s3()
        logger.info(
            "s3 bucket=%s region=%s prefix=%s",
            config.S3_BUCKET,
            config.S3_REGION or "-",
            config.S3_PREFIX,
        )
    worker.start()
    logger.info("worker started")
    yield
    logger.info("service stopping")


app = FastAPI(
    title="Wan 2.2 I2V",
    version="0.2.0",
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
        "seed": task["seed"],
        "video_url": task["video_url"],
        "error": task["error"],
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
    }


@app.post("/v1/generate")
def create_generation(body: GenerateRequest):
    if not generate.is_ready():
        logger.warning("reject generate: model not ready")
        raise HTTPException(503, generate.load_error() or "model not ready")
    try:
        pending = store.pending()
        if pending >= config.QUEUE_MAX:
            logger.warning("queue full pending=%s max=%s", pending, config.QUEUE_MAX)
            raise HTTPException(429, "queue full")
    except HTTPException:
        raise
    except Exception:
        logger.exception("redis unavailable on enqueue")
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
    first_path = None
    last_path = None
    try:
        first_path = download.download_image(
            body.image,
            config.UPLOAD_DIR / f"{task_id}_first",
        )
        if body.last_image:
            last_path = download.download_image(
                body.last_image,
                config.UPLOAD_DIR / f"{task_id}_last",
            )
    except UrlError as exc:
        logger.warning("task=%s download failed: %s", task_id, exc)
        _remove(first_path, last_path)
        raise HTTPException(400, str(exc)) from exc

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
                "first_frame_path": first_path,
                "last_frame_path": last_path,
                "duration": body.duration,
                "resolution": body.resolution,
                "webhook_url": body.webhook_url,
                "seed": body.seed,
                "steps": body.steps,
                "quality": body.quality,
            },
        )
    except Exception:
        logger.exception("task=%s enqueue failed", task_id)
        _remove(first_path, last_path)
        raise HTTPException(503, "queue unavailable") from None

    logger.info(
        "queued task=%s duration=%s resolution=%s steps=%s quality=%s pending=%s webhook=%s",
        task_id,
        body.duration,
        body.resolution,
        body.steps,
        body.quality,
        pending + 1,
        bool(body.webhook_url),
    )
    return JSONResponse({"id": task_id, "task_id": task_id, "status": "queued"}, status_code=202)


@app.get("/v1/tasks/{task_id}")
def get_task(task_id: str):
    try:
        task = store.get_task(task_id)
    except Exception:
        logger.exception("redis unavailable on get task=%s", task_id)
        raise HTTPException(503, "queue unavailable") from None
    if not task:
        raise HTTPException(404, "task 不存在")
    return _public_task(task)


@app.get("/health")
def health():
    return {"ok": True, "model_ready": generate.is_ready()}


@app.get("/ready")
def ready():
    if not generate.is_ready():
        raise HTTPException(503, generate.load_error() or "model not ready")
    return {"ok": True}


def _remove(*paths: str | None) -> None:
    for path in paths:
        if path:
            Path(path).unlink(missing_ok=True)
