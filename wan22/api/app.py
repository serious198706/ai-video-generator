from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from wan22 import config
from wan22.api.schemas import GenerateRequest
from wan22.infer import generate
from wan22.log import get_logger, setup_logging
from wan22.media.upload import assert_configured as assert_s3
from wan22.net.urlguard import UrlError, assert_https_url, assert_image_source
from wan22.tasks import runner, store

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging(force=True)
    logger.info("gpu api starting dry_run=%s docs=%s", config.DRY_RUN, config.ENABLE_DOCS)
    store.ping()
    logger.info("task db=%s", config.TASK_DB)
    if not config.DRY_RUN:
        assert_s3()
        logger.info(
            "s3 bucket=%s region=%s prefix=%s",
            config.S3_BUCKET,
            config.S3_REGION or "-",
            config.S3_PREFIX,
        )
    runner.startup()
    yield
    logger.info("service stopping")
    from wan22.infer.upscale import stop as stop_upscale
    from wan22.infer.foley import stop as stop_foley

    stop_upscale()
    stop_foley()


app = FastAPI(
    title="Wan 2.2 I2V",
    version="0.4.0",
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
        assert_image_source(body.image, config.IMAGE_HOSTS, kind="image")
        if body.last_image:
            assert_image_source(body.last_image, config.IMAGE_HOSTS, kind="image")
    except UrlError as exc:
        logger.warning("reject image url: %s", exc)
        raise HTTPException(400, str(exc)) from exc

    if (
        (body.resolution or "").lower() == "1080p"
        and not config.DRY_RUN
        and not config.UPSCALE_ENABLE
    ):
        logger.warning("reject 1080p: upscale is disabled")
        raise HTTPException(503, "1080p requires upscale")

    task_id = uuid.uuid4().hex
    # 单卡一次只跑一个，没有队列：占不到执行位就让 Java 端稍后重投。
    if not runner.reserve(task_id):
        logger.warning("busy running=%s reject task=%s", runner.current(), task_id)
        raise HTTPException(429, "busy")

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
        runner.spawn(task_id)
    except Exception as exc:
        runner.release(task_id)
        logger.exception("failed to start task=%s", task_id)
        raise HTTPException(500, "failed to start task") from exc

    logger.info(
        "accepted task=%s duration=%s resolution=%s steps=%s quality=%s audio=%s webhook=%s",
        task_id,
        body.duration,
        body.resolution,
        body.steps,
        body.quality,
        bool(body.audio),
        bool(body.webhook_url),
    )
    return JSONResponse({"id": task_id, "task_id": task_id, "status": "running"}, status_code=202)


@app.get("/v1/tasks/{task_id}")
def get_task(task_id: str):
    task = store.get_task(task_id)
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
