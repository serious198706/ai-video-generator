from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

import requests

from wan22 import config
from wan22.log import get_logger
from wan22.net.urlguard import UrlError, assert_https_url
from wan22.media.image import decode_to_jpeg

logger = get_logger(__name__)

_MAX_REDIRECTS = 5


def download_image(url: str, dest: Path) -> str:
    """下载图片。Pillow 能开的格式都会转成 JPEG。有 WAN22_IMAGE_HOSTS 时按白名单；空则任意公网 https。"""
    current = assert_https_url(url, config.IMAGE_HOSTS, kind="image")
    logger.info("downloading %s", current)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        for _ in range(_MAX_REDIRECTS + 1):
            assert_https_url(current, config.IMAGE_HOSTS, kind="image")
            try:
                response = session.get(
                    current,
                    stream=True,
                    timeout=config.DOWNLOAD_TIMEOUT,
                    allow_redirects=False,
                    headers={"User-Agent": "wan22-server/1"},
                )
            except requests.RequestException as exc:
                raise UrlError(f"下载图片失败 url={current}") from exc

            if response.is_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise UrlError(f"下载图片失败 url={current}")
                current = urljoin(current, location)
                continue
            if response.status_code != 200:
                raise UrlError(f"下载图片失败 status={response.status_code} url={current}")
            payload = _read_limited(response)
            try:
                path = decode_to_jpeg(payload, dest)
            except Exception as exc:
                raise UrlError(f"无法解码图片 url={current}") from exc
            logger.info("downloaded %s bytes=%s -> %s", current, len(payload), path.name)
            return str(path)

    raise UrlError("下载图片失败")


def _read_limited(response: requests.Response) -> bytes:
    chunks = bytearray()
    try:
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            chunks.extend(chunk)
            if len(chunks) > config.UPLOAD_MAX_BYTES:
                raise UrlError("图片超过大小限制")
    finally:
        response.close()
    if not chunks:
        raise UrlError("图片为空")
    return bytes(chunks)
