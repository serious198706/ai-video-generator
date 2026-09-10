from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from wan22.log import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_logging(force=True)
    logger.info("lambda api parked: POST /v1/generate on GPU uvicorn instead")
    yield


app = FastAPI(
    title="Wan 2.2 I2V API (parked)",
    version="0.3.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


@app.get("/health")
def health():
    raise HTTPException(
        503,
        "lambda api parked; call GPU /v1/generate on uvicorn wan22.api.app:app",
    )
