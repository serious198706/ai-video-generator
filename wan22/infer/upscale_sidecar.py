"""SeedVR2 常驻进程。必须用独立 venv 启动，不要 import wan22。

stdin/stdout 走 JSON 行：
  -> {"cmd":"generate","video":"...","output":"...","batch":21,"chunk":0,
      "dit_model":"...","model_dir":"..."}
  <- {"ok":true,"width":1080,"height":1872} 或 {"ok":false,"error":"..."}
  -> {"cmd":"ping"} / {"cmd":"quit"}
日志只写 stderr。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
import tempfile
import traceback
from pathlib import Path


_JSON_OUT = sys.stdout


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _silence_library_stdout() -> None:
    """SeedVR2 会往 stdout 打 sageattention 提示，污染 JSON 协议。"""
    sys.stdout = sys.stderr


def _reply(payload: dict) -> None:
    _JSON_OUT.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _JSON_OUT.flush()


def _link_named_ffmpeg(exe: Path, bindir: Path) -> Path:
    """imageio-ffmpeg 的二进制不叫 ffmpeg，SeedVR2 却硬编码调用 ffmpeg。"""
    bindir.mkdir(parents=True, exist_ok=True)
    link = bindir / "ffmpeg"
    target = exe.resolve()
    if not target.is_file():
        raise FileNotFoundError(f"ffmpeg binary missing: {target}")
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == target:
            return link
        link.unlink()
    try:
        link.symlink_to(target)
    except OSError:
        link.write_text(
            f"#!/bin/sh\nexec {shlex.quote(str(target))} \"$@\"\n",
            encoding="utf-8",
        )
        link.chmod(0o755)
    return link


def _ensure_ffmpeg() -> None:
    """OpenCV mp4v 在 Linux 上经常写出纯绿片，必须用 ffmpeg/libx264。"""
    found = shutil.which("ffmpeg")
    if found:
        _log(f"ffmpeg={found}")
        return
    try:
        import imageio_ffmpeg

        exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg not found; install ffmpeg or imageio-ffmpeg in the upscale venv"
        ) from exc
    shim = _link_named_ffmpeg(exe, Path(tempfile.gettempdir()) / "wan22-upscale-ffmpeg")
    os.environ["PATH"] = str(shim.parent) + os.pathsep + os.environ.get("PATH", "")
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError(f"ffmpeg still not on PATH after linking {shim} -> {exe}")
    _log(f"ffmpeg={found} -> {exe}")


def _snap_4n1(value: int) -> int:
    value = max(1, int(value))
    return ((value - 1) // 4) * 4 + 1


def _probe(path: str) -> tuple[int, int, int]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if width < 1 or height < 1:
        raise RuntimeError(f"invalid video size {width}x{height}: {path}")
    return width, height, max(frames, 1)


def _cli_args(cli, req: dict, width: int, height: int, frames: int):
    short = int(round(min(width, height) * 1.5))
    max_edge = int(round(max(width, height) * 1.5))
    batch = _snap_4n1(min(int(req.get("batch") or 21), frames))
    chunk = max(0, int(req.get("chunk") or 0))
    argv = [
        "inference_cli.py",
        str(req["video"]),
        "--output",
        str(req["output"]),
        "--output_format",
        "mp4",
        "--model_dir",
        str(req["model_dir"]),
        "--dit_model",
        str(req["dit_model"]),
        "--resolution",
        str(short),
        "--max_resolution",
        str(max_edge),
        "--batch_size",
        str(batch),
        "--cache_dit",
        "--cache_vae",
        "--dit_offload_device",
        "cpu",
        "--vae_offload_device",
        "cpu",
        "--uniform_batch_size",
        "--video_backend",
        "ffmpeg",
        "--seed",
        "42",
    ]
    if chunk > 0:
        argv.extend(["--chunk_size", str(chunk)])
    old = sys.argv
    try:
        sys.argv = argv
        args = cli.parse_arguments()
    finally:
        sys.argv = old
    return args, short, max_edge, batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dit-model", required=True)
    boot = parser.parse_args()
    _silence_library_stdout()

    repo = Path(boot.repo).expanduser().resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ["PYTHONPATH"] = str(repo) + os.pathsep + os.environ.get("PYTHONPATH", "")

    try:
        _ensure_ffmpeg()
        import inference_cli as cli
        from src.utils.downloads import download_weight
        from src.utils.model_registry import DEFAULT_VAE

        download_weight(
            boot.dit_model,
            DEFAULT_VAE,
            str(Path(boot.model_dir).resolve()),
        )
    except Exception as exc:
        _log(f"load failed: {exc}")
        traceback.print_exc(file=sys.stderr)
        try:
            _reply({"ok": False, "ready": False, "error": str(exc)})
        except Exception:
            pass
        os._exit(1)

    runner_cache: dict = {}
    _reply({"ok": True, "ready": True})
    _log(f"upscale ready repo={repo} dit={boot.dit_model}")

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

            video = str(Path(req["video"]).expanduser().resolve())
            output = str(Path(req["output"]).expanduser().resolve())
            req["video"] = video
            req["output"] = output
            req["dit_model"] = req.get("dit_model") or boot.dit_model
            req["model_dir"] = req.get("model_dir") or str(Path(boot.model_dir).resolve())
            Path(output).unlink(missing_ok=True)

            width, height, frames = _probe(video)
            args, short, max_edge, batch = _cli_args(cli, req, width, height, frames)
            _log(
                f"upscale {width}x{height} frames={frames} -> short={short} "
                f"max={max_edge} batch={batch}"
            )
            written = cli.process_single_file(
                video,
                args,
                ["0"],
                output_path=output,
                format_auto_detected=False,
                runner_cache=runner_cache,
            )
            if written <= 0 or not Path(output).is_file():
                raise RuntimeError(f"upscale produced no frames written={written}")
            out_w, out_h, _ = _probe(output)
            if min(out_w, out_h) < short - 32:
                raise RuntimeError(
                    f"upscale too small {out_w}x{out_h}, expected short>={short}"
                )
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            _reply(
                {
                    "ok": True,
                    "width": out_w,
                    "height": out_h,
                    "frames": written,
                    "source_width": width,
                    "source_height": height,
                }
            )
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _reply({"ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
