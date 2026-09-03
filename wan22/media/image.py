from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFile, UnidentifiedImageError

from wan22.log import get_logger

logger = get_logger(__name__)


def open_rgb(path: str | Path) -> Image.Image:
    """打开首/尾帧。容忍损坏的 PNG eXIf CRC，RGBA 铺到白底。"""
    path = Path(path)
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(path) as image:
            image.load()
            return _to_rgb(image)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def decode_to_jpeg(payload: bytes, dest: Path) -> Path:
    """任意 Pillow 能开的格式（JPEG/PNG/WebP/GIF/BMP/TIFF 等）转成 RGB JPEG。"""
    if _looks_like_html(payload):
        raise ValueError("响应是 HTML 不是图片")
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
            fmt = (image.format or "unknown").lower()
            rgb = _to_rgb(image)
    except UnidentifiedImageError as exc:
        raise ValueError("无法识别图片格式") from exc
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous

    dest.parent.mkdir(parents=True, exist_ok=True)
    out = dest.with_suffix(".jpg")
    rgb.save(out, format="JPEG", quality=95, optimize=True)
    logger.info("decoded format=%s -> %s %sx%s", fmt, out.name, rgb.width, rgb.height)
    return out


def rewrite_rgb(path: Path) -> Path:
    """落盘为无附属块的 JPEG，避免下游 Pillow/diffusers 再踩坏 PNG。"""
    rgb = open_rgb(path)
    out = path.with_suffix(".jpg")
    rgb.save(out, format="JPEG", quality=95, optimize=True)
    if out.resolve() != path.resolve():
        path.unlink(missing_ok=True)
    logger.info("normalized image %s -> %s %sx%s", path.name, out.name, rgb.width, rgb.height)
    return out


def _looks_like_html(payload: bytes) -> bool:
    start = payload.lstrip()[:64].lower()
    return start.startswith(b"<!doctype") or start.startswith(b"<html") or start.startswith(b"<head")


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[-1])
        return background
    return image.convert("RGB")
