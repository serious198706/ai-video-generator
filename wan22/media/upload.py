from __future__ import annotations

from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

from wan22 import config
from wan22.log import get_logger

logger = get_logger(__name__)

_client = None


def assert_configured() -> None:
    if not config.S3_BUCKET:
        raise RuntimeError("WAN22_S3_BUCKET is not set")


def _client_s3():
    global _client
    if _client is None:
        kwargs: dict = {
            "config": BotoConfig(retries={"max_attempts": 3, "mode": "standard"}),
        }
        if config.S3_REGION:
            kwargs["region_name"] = config.S3_REGION
        if config.S3_ENDPOINT:
            kwargs["endpoint_url"] = config.S3_ENDPOINT
        _client = boto3.client("s3", **kwargs)
    return _client


def _object_key(filename: str) -> str:
    prefix = config.S3_PREFIX.strip("/")
    name = filename.lstrip("/")
    if prefix:
        return f"{prefix}/{name}"
    return name


def public_url(key: str) -> str:
    base = config.S3_PUBLIC_BASE_URL.rstrip("/")
    if base:
        return f"{base}/{key}"
    bucket = config.S3_BUCKET
    region = config.S3_REGION
    if not region or region == "us-east-1":
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def upload_video(path: str | Path, *, object_name: str | None = None) -> str:
    """上传 mp4 到 S3，返回公网/CloudFront URL。凭证走 IAM Role 或 AWS_* 环境变量。"""
    assert_configured()
    path = Path(path)
    key = _object_key(object_name or path.name)
    extra = {"ContentType": "video/mp4"}
    _client_s3().upload_file(str(path), config.S3_BUCKET, key, ExtraArgs=extra)
    url = public_url(key)
    logger.info("uploaded s3://%s/%s -> %s", config.S3_BUCKET, key, url)
    return url
