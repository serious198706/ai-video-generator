from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from wan22 import config
from wan22.infer import generate
from wan22.log import get_logger, setup_logging
from wan22.tasks import runner, sqs_consumer

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging(force=True)
    logger.info(
        "gpu worker starting dry_run=%s docs=%s sqs=%s",
        config.DRY_RUN,
        config.ENABLE_DOCS,
        bool(config.SQS_QUEUE_URL),
    )
    runner.startup()
    sqs_consumer.start()
    yield
    logger.info("worker stopping")
    sqs_consumer.stop()
    from wan22.infer.foley import stop as stop_foley
    from wan22.infer.upscale import stop as stop_upscale

    stop_upscale()
    stop_foley()


app = FastAPI(
    title="Wan 2.2 I2V Worker",
    version="0.5.0",
    docs_url="/docs" if config.ENABLE_DOCS else None,
    redoc_url="/redoc" if config.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if config.ENABLE_DOCS else None,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return JSONResponse({"ok": True, "model_ready": generate.is_ready()})


@app.get("/ready")
def ready():
    if not generate.is_ready():
        raise HTTPException(503, generate.load_error() or "model not ready")
    return JSONResponse({"ok": True})
