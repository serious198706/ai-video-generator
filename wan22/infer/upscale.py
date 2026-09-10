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
_VAE_NAME = "ema_vae_fp16.safetensors"
_lock = threading.Lock()
_proc: subprocess.Popen | None = None


class UpscaleError(RuntimeError):
    pass


def snap_4n1(value: int) -> int:
    value = max(1, int(value))
    return ((value - 1) // 4) * 4 + 1


def decode_json_line(raw: str) -> dict | None:
    """协议行必须是 JSON 对象。SeedVR2 的 print 提示直接丢掉。"""
    text = (raw or "").strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def target_short_side(width: int, height: int) -> int:
    """720p 画布 1.5×：竖/横短边 1080，方图 1440。"""
    return int(round(min(width, height) * 1.5))


def target_max_edge(width: int, height: int) -> int:
    return int(round(max(width, height) * 1.5))


def preflight() -> None:
    if not config.UPSCALE_ENABLE:
        return
    if not _SIDECAR.is_file():
        raise FileNotFoundError(f"missing upscale sidecar: {_SIDECAR}")
    if config.UPSCALE_REPO is None or not (config.UPSCALE_REPO / "inference_cli.py").is_file():
        raise FileNotFoundError(
            f"WAN22_UPSCALE_REPO missing inference_cli.py: {config.UPSCALE_REPO}"
        )
    if not config.UPSCALE_MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"WAN22_UPSCALE_MODEL_DIR not found: {config.UPSCALE_MODEL_DIR}"
        )
    dit = config.UPSCALE_MODEL_DIR / config.UPSCALE_DIT
    vae = config.UPSCALE_MODEL_DIR / _VAE_NAME
    if not dit.is_file():
        raise FileNotFoundError(f"SeedVR2 DiT not found: {dit}")
    if not vae.is_file():
        raise FileNotFoundError(f"SeedVR2 VAE not found: {vae}")
    python = _python()
    if config.UPSCALE_PYTHON and not Path(python).is_file():
        raise FileNotFoundError(f"WAN22_UPSCALE_PYTHON not found: {python}")
    if config.UPSCALE_TIMEOUT < 1:
        raise ValueError("WAN22_UPSCALE_TIMEOUT must be greater than 0")
    logger.info(
        "upscale enabled python=%s dit=%s model=%s repo=%s timeout=%s batch=%s",
        python,
        config.UPSCALE_DIT,
        config.UPSCALE_MODEL_DIR,
        config.UPSCALE_REPO,
        config.UPSCALE_TIMEOUT,
        snap_4n1(config.UPSCALE_BATCH),
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
                    "batch": snap_4n1(config.UPSCALE_BATCH),
                    "chunk": max(0, config.UPSCALE_CHUNK),
                    "dit_model": config.UPSCALE_DIT,
                    "model_dir": str(config.UPSCALE_MODEL_DIR),
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
    if config.UPSCALE_REPO is None:
        raise UpscaleError("WAN22_UPSCALE_REPO is empty")
    cmd = [
        _python(),
        str(_SIDECAR),
        "--repo",
        str(config.UPSCALE_REPO),
        "--model-dir",
        str(config.UPSCALE_MODEL_DIR),
        "--dit-model",
        config.UPSCALE_DIT,
    ]
    logger.info("starting upscale sidecar: %s", " ".join(cmd))
    env = os.environ.copy()
    repo = str(config.UPSCALE_REPO)
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    _proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=repo,
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
