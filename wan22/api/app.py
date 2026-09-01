from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from wan22 import config
from wan22.infer import generate
from wan22.log import get_logger, setup_logging
from wan22.media.upload import assert_configured as assert_s3
from wan22.queue import store, worker

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging(force=True)
    logger.info("gpu worker starting dry_run=%s docs=%s", config.DRY_RUN, config.ENABLE_DOCS)
    store.ping()
    logger.info("redis queue max=%s attempts=%s", config.QUEUE_MAX, config.MAX_ATTEMPTS)
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
    from wan22.infer.foley import stop as stop_foley

    stop_foley()


app = FastAPI(
    title="Wan 2.2 I2V GPU worker",
    version="0.3.0",
    docs_url="/docs" if config.ENABLE_DOCS else None,
    redoc_url="/redoc" if config.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if config.ENABLE_DOCS else None,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    try:
        store.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {"ok": True, "model_ready": generate.is_ready(), "redis": redis_ok}


@app.get("/ready")
def ready():
    try:
        store.ping()
    except Exception:
        raise HTTPException(503, "queue unavailable") from None
    if not generate.is_ready():
        raise HTTPException(503, generate.load_error() or "model not ready")
    return {"ok": True}
