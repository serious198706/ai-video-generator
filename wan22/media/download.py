from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

import requests

from wan22 import config
from wan22.log import get_logger
from wan22.net.urlguard import UrlError, assert_https_url

logger = get_logger(__name__)

_MAX_REDIRECTS = 5


def download_image(url: str, dest: Path) -> str:
    """按白名单下载 JPEG/PNG。返回最终文件路径。"""
    current = assert_https_url(url, config.IMAGE_HOSTS, kind="image")
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
                raise UrlError("下载图片失败") from exc

            if response.is_redirect:
                location = response.headers.get("Location")
                if not location:
                    raise UrlError("下载图片失败")
                current = urljoin(current, location)
                continue
            if response.status_code != 200:
                raise UrlError("下载图片失败")
            payload = _read_limited(response)
            suffix = _sniff_suffix(payload)
            path = dest.with_suffix(suffix)
            path.write_bytes(payload)
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


def _sniff_suffix(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8"):
        return ".jpg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    raise UrlError("只支持 JPEG 或 PNG")
