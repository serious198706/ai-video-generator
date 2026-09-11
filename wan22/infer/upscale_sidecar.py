"""Compact (realesr-general-x4v3) 常驻进程。必须用独立 venv 启动，不要 import wan22。

stdin/stdout 走 JSON 行：
  -> {"cmd":"generate","video":"...","output":"..."}
  <- {"ok":true,"width":1080,"height":1872} 或 {"ok":false,"error":"..."}
  -> {"cmd":"ping"} / {"cmd":"quit"}
日志只写 stderr。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np


_JSON_OUT = sys.stdout
_extra_arches = False


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _reply(payload: dict) -> None:
    _JSON_OUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _JSON_OUT.flush()


def even(value: int) -> int:
    value = max(2, int(value))
    return value - (value % 2)


def target_size(width: int, height: int, scale: float) -> tuple[int, int]:
    """720x1248 * 1.5 -> 1080x1872。两边都取偶数，方便 yuv420p。"""
    return even(round(width * scale)), even(round(height * scale))


def _ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("ffmpeg not found; install ffmpeg or imageio-ffmpeg") from exc


def _probe(path: str | Path) -> tuple[int, int, float, int]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 16)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if width < 1 or height < 1:
        raise RuntimeError(f"invalid video size {width}x{height}")
    if fps <= 1e-3:
        fps = 16.0
    return width, height, fps, max(frames, 1)


def _iter_rgb(path: str | Path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _write_rgb_mp4(path: str | Path, frames, fps: float, crf: int = 16) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for frame in frames:
            rgb = np.ascontiguousarray(frame)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise RuntimeError(f"expected HxWx3 RGB, got {rgb.shape}")
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            if writer is None:
                height, width = rgb.shape[:2]
                if width % 2 or height % 2:
                    raise RuntimeError(f"odd size {width}x{height}, yuv420p needs even")
                cmd = [
                    _ffmpeg_exe(),
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    f"{fps:.4f}",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-preset",
                    "fast",
                    "-crf",
                    str(crf),
                    "-movflags",
                    "+faststart",
                    str(path),
                ]
                _log(f"ffmpeg write {path.name} {width}x{height} fps={fps:.3f}")
                writer = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=os.environ.copy(),
                )
            if writer.stdin is None:
                raise RuntimeError("ffmpeg stdin unavailable")
            writer.stdin.write(rgb.tobytes())
        if writer is None:
            raise RuntimeError("no frames written")
        writer.stdin.close()
        err = writer.stderr.read().decode("utf-8", errors="replace") if writer.stderr else ""
        code = writer.wait()
        if code != 0 or not path.is_file() or path.stat().st_size < 64:
            tail = "\n".join(err.strip().splitlines()[-12:])
            raise RuntimeError(f"ffmpeg failed code={code}: {tail}")
    finally:
        if writer is not None and writer.poll() is None:
            writer.kill()


def _install_extra_arches() -> None:
    global _extra_arches
    if _extra_arches:
        return
    import spandrel_extra_arches

    spandrel_extra_arches.install(ignore_duplicates=True)
    _extra_arches = True


def _load_model(path: Path):
    import spandrel
    import torch

    _install_extra_arches()
    _log(f"load compact {path}")
    desc = spandrel.ModelLoader().load_from_file(str(path))
    if not isinstance(desc, spandrel.ImageModelDescriptor):
        raise RuntimeError(f"not an image SR model: {type(desc)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    desc = desc.to(device).eval()
    if device.type == "cuda" and desc.supports_half:
        desc = desc.half()
    return desc


def _sr_frame(desc, rgb: np.ndarray) -> np.ndarray:
    import torch

    device = next(desc.model.parameters()).device
    dtype = next(desc.model.parameters()).dtype
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div_(255)
    tensor = tensor.unsqueeze(0).to(device=device, dtype=dtype)
    out = desc(tensor)
    out = out.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (out * 255.0).round().astype(np.uint8)


def _run(desc, video: Path, output: Path, scale: float) -> dict:
    import cv2
    import torch

    width, height, fps, frames = _probe(video)
    tw, th = target_size(width, height, scale)
    _log(f"compact {width}x{height} frames={frames} fps={fps:.3f} -> {tw}x{th}")

    def _frames():
        count = 0
        for rgb in _iter_rgb(video):
            sr = _sr_frame(desc, rgb)
            if sr.shape[1] != tw or sr.shape[0] != th:
                sr = cv2.resize(sr, (tw, th), interpolation=cv2.INTER_LANCZOS4)
            count += 1
            if count == 1 or count % 30 == 0:
                _log(f"compact frame {count}/{frames} {sr.shape[1]}x{sr.shape[0]}")
            yield sr

    with torch.inference_mode():
        _write_rgb_mp4(output, _frames(), fps)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"width": tw, "height": th, "frames": frames}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--scale", type=float, default=1.5)
    boot = parser.parse_args()
    model = Path(boot.model).expanduser().resolve()
    scale = float(boot.scale)
    if scale <= 1:
        _reply({"ok": False, "ready": False, "error": "scale must be greater than 1"})
        return 1
    if not model.is_file():
        _reply({"ok": False, "ready": False, "error": f"compact weights missing: {model}"})
        return 1

    try:
        desc = _load_model(model)
        _ffmpeg_exe()
    except Exception as exc:
        _log(f"load failed: {exc}")
        traceback.print_exc(file=sys.stderr)
        try:
            _reply({"ok": False, "ready": False, "error": str(exc)})
        except Exception:
            pass
        os._exit(1)

    _reply({"ok": True, "ready": True})
    _log(f"compact ready model={model} scale={scale}")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd")
            if cmd == "quit":
                _reply({"ok": True, "bye": True})
                return 0
            if cmd == "ping":
                _reply({"ok": True, "pong": True})
                continue
            if cmd != "generate":
                _reply({"ok": False, "error": f"unknown cmd {cmd!r}"})
                continue

            video = Path(req["video"]).expanduser().resolve()
            output = Path(req["output"]).expanduser().resolve()
            job_scale = float(req.get("scale") or scale)
            output.unlink(missing_ok=True)
            stats = _run(desc, video, output, job_scale)
            _reply({"ok": True, **stats})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _reply({"ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
