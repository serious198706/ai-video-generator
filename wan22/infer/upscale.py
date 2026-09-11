from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from wan22 import config
from wan22.log import get_logger

logger = get_logger(__name__)

_SIDECAR = Path(__file__).resolve().parent / "upscale_sidecar.py"
_lock = threading.Lock()
_proc: subprocess.Popen | None = None


class UpscaleError(RuntimeError):
    pass


def even(value: int) -> int:
    value = max(2, int(value))
    return value - (value % 2)


def decode_json_line(raw: str) -> dict | None:
    """协议行必须是 JSON 对象。sidecar 的非 JSON 输出直接丢掉。"""
    text = (raw or "").strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def target_short_side(width: int, height: int, scale: float | None = None) -> int:
    """720p 画布 1.5x：竖/横短边 1080，方图 1440。"""
    return int(round(min(width, height) * (scale if scale is not None else config.UPSCALE_SCALE)))


def target_max_edge(width: int, height: int, scale: float | None = None) -> int:
    return int(round(max(width, height) * (scale if scale is not None else config.UPSCALE_SCALE)))


def target_size(width: int, height: int, scale: float | None = None) -> tuple[int, int]:
    factor = scale if scale is not None else config.UPSCALE_SCALE
    return even(round(width * factor)), even(round(height * factor))


def _model_path() -> Path:
    return config.UPSCALE_MODEL_DIR / config.UPSCALE_MODEL


def preflight() -> None:
    if not config.UPSCALE_ENABLE:
        return
    if config.UPSCALE_SCALE <= 1:
        raise ValueError("WAN22_UPSCALE_SCALE must be greater than 1")
    if not _SIDECAR.is_file():
        raise FileNotFoundError(f"missing upscale sidecar: {_SIDECAR}")
    python = _python()
    if config.UPSCALE_PYTHON and not Path(python).is_file():
        raise FileNotFoundError(f"WAN22_UPSCALE_PYTHON not found: {python}")
    if config.UPSCALE_TIMEOUT < 1:
        raise ValueError("WAN22_UPSCALE_TIMEOUT must be greater than 0")
    model = _model_path()
    if not model.is_file():
        raise FileNotFoundError(f"compact weights not found: {model}")
    logger.info(
        "upscale enabled python=%s model=%s scale=%s timeout=%s",
        python,
        model,
        config.UPSCALE_SCALE,
        config.UPSCALE_TIMEOUT,
    )


def upscale_video(video_path: str) -> None:
    """把 720p 无声 mp4 超到约 1080p，原地替换。"""
    if not config.UPSCALE_ENABLE:
        raise UpscaleError("upscale disabled")
    video = Path(video_path).expanduser().resolve()
    tmp = video.with_suffix(".up.mp4")
    tmp.unlink(missing_ok=True)
    try:
        if not video.is_file():
            raise UpscaleError(f"video missing: {video}")
        with _lock:
            _ensure()
            reply = _request(
                {
                    "cmd": "generate",
                    "video": str(video),
                    "output": str(tmp),
                    "scale": config.UPSCALE_SCALE,
                }
            )
        if not tmp.is_file() or tmp.stat().st_size < 64:
            raise UpscaleError("upscale output missing")
        tmp.replace(video)
        logger.info(
            "upscaled video=%s %sx%s",
            video.name,
            reply.get("width") or "-",
            reply.get("height") or "-",
        )
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        if isinstance(exc, UpscaleError):
            raise
        logger.exception("upscale failed video=%s", video)
        raise UpscaleError(str(exc)) from exc


def stop() -> None:
    global _proc
    with _lock:
        if _proc is None:
            return
        try:
            if _proc.poll() is None:
                _write({"cmd": "quit"})
                _proc.wait(timeout=10)
        except Exception:
            logger.exception("upscale sidecar quit failed")
        _kill()


def _python() -> str:
    return config.UPSCALE_PYTHON or sys.executable


def _ensure() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return
    _kill()
    cmd = [
        _python(),
        str(_SIDECAR),
        "--model",
        str(_model_path()),
        "--scale",
        str(config.UPSCALE_SCALE),
    ]
    logger.info("starting upscale sidecar: %s", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    _proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(_SIDECAR.parent),
        env=env,
    )
    threading.Thread(target=_drain_stderr, args=(_proc,), daemon=True).start()
    ready = _readline(timeout=max(config.UPSCALE_TIMEOUT, 300))
    if not ready.get("ok") or not ready.get("ready"):
        _kill()
        raise UpscaleError(f"sidecar not ready: {ready}")
    logger.info("upscale sidecar ready")


def _request(payload: dict) -> dict:
    _write(payload)
    reply = _readline(timeout=config.UPSCALE_TIMEOUT)
    if not reply.get("ok"):
        raise UpscaleError(reply.get("error") or "upscale generate failed")
    return reply


def _write(payload: dict) -> None:
    if _proc is None or _proc.stdin is None:
        raise UpscaleError("sidecar stdin unavailable")
    _proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _proc.stdin.flush()


def _readline(timeout: int) -> dict:
    proc = _proc
    if proc is None or proc.stdout is None:
        raise UpscaleError("sidecar stdout unavailable")
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill()
            raise UpscaleError(f"sidecar timeout: {timeout}s")
        line: list[str] = []

        def _read() -> None:
            line.append(proc.stdout.readline())

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()
        reader.join(remaining)
        if reader.is_alive():
            _kill()
            raise UpscaleError(f"sidecar timeout: {timeout}s")
        raw = line[0] if line else ""
        if raw == "":
            code = proc.poll()
            _kill()
            raise UpscaleError(f"sidecar exited with code={code}")
        payload = decode_json_line(raw)
        if payload is not None:
            return payload
        logger.info("upscale sidecar stdout: %s", raw.strip()[:300])


def _drain_stderr(proc: subprocess.Popen) -> None:
    if proc.stderr is None:
        return
    for row in proc.stderr:
        text = row.rstrip()
        if text:
            logger.info("upscale sidecar: %s", text)


def _kill() -> None:
    global _proc
    proc = _proc
    _proc = None
    if proc is None:
        return
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass
