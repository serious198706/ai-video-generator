from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from wan22 import config
from wan22.log import get_logger
from wan22.media.mux import mux_audio

logger = get_logger(__name__)

_SIDECAR = Path(__file__).resolve().parent / "foley_sidecar.py"
_lock = threading.Lock()
_proc: subprocess.Popen | None = None


class FoleyError(RuntimeError):
    pass


def preflight() -> None:
    if not config.FOLEY_ENABLE:
        return
    if not _SIDECAR.is_file():
        raise FileNotFoundError(f"缺少 Foley sidecar: {_SIDECAR}")
    if not config.FOLEY_MODEL_DIR.is_dir():
        raise FileNotFoundError(f"WAN22_FOLEY_MODEL_DIR 不存在: {config.FOLEY_MODEL_DIR}")
    yaml_path = _config_path()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Foley 配置不存在: {yaml_path}")
    python = _python()
    if config.FOLEY_PYTHON and not Path(python).is_file():
        raise FileNotFoundError(f"WAN22_FOLEY_PYTHON 不存在: {python}")
    logger.info(
        "foley enabled python=%s size=%s model=%s repo=%s required=%s",
        python,
        config.FOLEY_SIZE,
        config.FOLEY_MODEL_DIR,
        config.FOLEY_REPO or "-",
        config.FOLEY_REQUIRED,
    )


def add_audio(video_path: str) -> bool:
    """给无声 mp4 配 Foley。成功返回 True；关闭或失败返回 False（REQUIRED 时抛错）。"""
    if not config.FOLEY_ENABLE:
        return False
    try:
        wav = Path(video_path).with_suffix(".wav")
        with _lock:
            _ensure()
            _request(
                {
                    "cmd": "generate",
                    "video": str(video_path),
                    "wav": str(wav),
                    "prompt": config.FOLEY_PROMPT,
                    "neg_prompt": config.FOLEY_NEG_PROMPT,
                    "steps": config.FOLEY_STEPS,
                    "guidance": config.FOLEY_GUIDANCE,
                }
            )
        mux_audio(video_path, str(wav))
        wav.unlink(missing_ok=True)
        logger.info("foley muxed video=%s", Path(video_path).name)
        return True
    except Exception as exc:
        Path(video_path).with_suffix(".wav").unlink(missing_ok=True)
        logger.exception("foley failed video=%s", video_path)
        if config.FOLEY_REQUIRED:
            raise FoleyError(str(exc)) from exc
        return False


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
            logger.exception("foley sidecar quit failed")
        _kill()


def _python() -> str:
    return config.FOLEY_PYTHON or sys.executable


def _config_path() -> Path:
    name = f"hunyuanvideo-foley-{config.FOLEY_SIZE}.yaml"
    alt = f"config_{config.FOLEY_SIZE}.yaml"
    roots: list[Path] = []
    if config.FOLEY_REPO:
        roots.append(config.FOLEY_REPO)
        roots.append(config.FOLEY_REPO / "configs")
    roots.append(config.FOLEY_MODEL_DIR)
    roots.append(config.FOLEY_MODEL_DIR / "configs")
    for root in roots:
        for filename in (name, alt):
            candidate = root / filename
            if candidate.is_file():
                return candidate
    return (config.FOLEY_REPO or config.FOLEY_MODEL_DIR) / "configs" / name


def _ensure() -> None:
    global _proc
    if _proc is not None and _proc.poll() is None:
        return
    _kill()
    cmd = [
        _python(),
        str(_SIDECAR),
        "--model-path",
        str(config.FOLEY_MODEL_DIR),
        "--config-path",
        str(_config_path()),
        "--model-size",
        config.FOLEY_SIZE,
    ]
    logger.info("starting foley sidecar: %s", " ".join(cmd))
    env = None
    if config.FOLEY_REPO:
        env = os.environ.copy()
        repo = str(config.FOLEY_REPO)
        env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    _proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(config.FOLEY_REPO) if config.FOLEY_REPO else None,
        env=env,
    )
    threading.Thread(target=_drain_stderr, args=(_proc,), daemon=True).start()
    ready = _readline(timeout=max(config.FOLEY_TIMEOUT, 300))
    if not ready.get("ok") or not ready.get("ready"):
        _kill()
        raise FoleyError(f"sidecar 未就绪: {ready}")
    logger.info("foley sidecar ready")


def _request(payload: dict) -> dict:
    _write(payload)
    reply = _readline(timeout=config.FOLEY_TIMEOUT)
    if not reply.get("ok"):
        raise FoleyError(reply.get("error") or "foley generate failed")
    return reply


def _write(payload: dict) -> None:
    if _proc is None or _proc.stdin is None:
        raise FoleyError("sidecar stdin 不可用")
    _proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _proc.stdin.flush()


def _readline(timeout: int) -> dict:
    proc = _proc
    if proc is None or proc.stdout is None:
        raise FoleyError("sidecar stdout 不可用")
    line: list[str] = []

    def _read() -> None:
        line.append(proc.stdout.readline())

    reader = threading.Thread(target=_read, daemon=True)
    reader.start()
    reader.join(timeout)
    if reader.is_alive():
        _kill()
        raise FoleyError(f"sidecar 超时 {timeout}s")
    raw = line[0] if line else ""
    if not raw:
        code = proc.poll()
        _kill()
        raise FoleyError(f"sidecar 退出 code={code}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FoleyError(f"sidecar 非 JSON: {raw[:200]!r}") from exc


def _drain_stderr(proc: subprocess.Popen) -> None:
    if proc.stderr is None:
        return
    for row in proc.stderr:
        text = row.rstrip()
        if text:
            logger.info("foley sidecar: %s", text)


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
