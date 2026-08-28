from __future__ import annotations

import subprocess
from pathlib import Path

from wan22.log import get_logger

logger = get_logger(__name__)


class MuxError(RuntimeError):
    pass


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def mux_audio(video_path: str, wav_path: str) -> None:
    """把 wav 并进无声 mp4，原地替换。"""
    src = Path(video_path)
    wav = Path(wav_path)
    if not src.is_file():
        raise MuxError(f"video missing: {src}")
    if not wav.is_file():
        raise MuxError(f"wav missing: {wav}")

    tmp = src.with_suffix(".aud.mp4")
    cmd = [
        _ffmpeg(),
        "-y",
        "-i",
        str(src),
        "-i",
        str(wav),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    logger.info("mux video=%s wav=%s", src.name, wav.name)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = err[-8:] if err else ["ffmpeg failed"]
        raise MuxError(" ".join(tail))
    tmp.replace(src)
