"""Local GPU benchmark: no SQS, S3 or webhooks; configuration is recorded per run.

Run with the Wan venv after sourcing worker-env.sh. Does not source .env itself.
Each invocation is one configuration in a fresh process; output must be a new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import threading
import time
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--last-image", type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resolution", choices=("480p", "720p", "1080p"), default="720p")
    parser.add_argument("--duration", type=float, default=15)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--residency", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--attention", choices=("auto", "sdpa", "sage2", "sage3"), default="auto")
    parser.add_argument("--quant", choices=("int8wo", "none"), default="int8wo")
    parser.add_argument("--profile", choices=("on", "off"), default="off")
    parser.add_argument("--resize", choices=("cpu_lanczos", "gpu_bicubic"), default="cpu_lanczos")
    parser.add_argument("--compile", choices=("off", "default", "max-autotune"), default="off")
    args = parser.parse_args(argv)
    if not 0 < args.duration <= 15 or not 1 <= args.steps <= 50 or not 1 <= args.quality <= 10:
        parser.error("duration must be (0,15], steps 1..50, quality 1..10")
    if args.warmup < 1 or args.runs < 1:
        parser.error("at least one warmup and one measured run are required")
    if args.compile != "off" and (args.attention != "sdpa" or args.profile != "off" or args.residency != "on" or args.resolution == "1080p"):
        parser.error("compile experiments require --attention sdpa --profile off --residency on and 480p/720p (no postprocess offload)")
    for path in (args.image, args.last_image, args.prompt_file):
        if path is not None and not path.is_file():
            parser.error(f"missing input: {path}")
    return args


def versions() -> dict:
    result = {}
    for name in ("torch", "torchao", "diffusers", "transformers", "accelerate", "peft", "sageattention", "sageattn3"):
        try:
            package = distribution(name)
            entry = {"version": package.version}
            direct = json.loads(package.read_text("direct_url.json") or "{}")
            # Do not record package URLs: they can contain credentials.
            if commit := direct.get("vcs_info", {}).get("commit_id"):
                entry["commit"] = commit
            result[name] = entry
        except PackageNotFoundError:
            result[name] = None
    return result


class Telemetry:
    """nvidia-smi includes sidecar memory; torch peak stats cover only the Wan process."""
    fields = "timestamp,uuid,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.mem"

    def __init__(self):
        self.samples = []
        self.error = None
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.done.is_set():
            try:
                row = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self.fields}", "--format=csv,noheader,nounits"],
                    check=True, capture_output=True, text=True, timeout=5,
                )
                self.samples.append({"monotonic_s": time.monotonic(), "csv": row.stdout.strip()})
            except (OSError, subprocess.SubprocessError) as exc:
                self.error = str(exc)
                return
            self.done.wait(1)

    def stop(self):
        self.done.set()
        self.thread.join(timeout=6)


def save_report(path: Path, report: dict):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def inspect_video(path: Path, expected_frames: int) -> dict:
    """Decode outside the timed interval; save first/middle/last frames for visual A/B."""
    import av

    count = 0
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        size = (stream.width, stream.height)
        fps = float(stream.average_rate) if stream.average_rate is not None else None
        for index, frame in enumerate(video.decode(stream)):
            count += 1
            if index in {0, expected_frames // 2, expected_frames - 1}:
                frame.to_image().save(path.with_name(f"{path.stem}-frame-{index:04d}.png"))
        if count != expected_frames:
            raise ValueError(f"encoded frame count {count} != expected {expected_frames}")
        return {"frames": count, "size": size, "fps": fps}


def main(argv=None) -> int:
    args = arguments(argv)
    # Set overrides before any wan22 import; no credentials or .env content are copied.
    os.environ.update({
        "WAN22_DRY_RUN": "0", "WAN22_PRELOAD": "0", "WAN22_OFFLOAD": "none",
        "WAN22_EXPERT_RESIDENCY": args.residency, "WAN22_SAGE": "1",
        "WAN22_SAGE_BACKEND": args.attention, "WAN22_QUANT": args.quant,
        "WAN22_1080P_QUANT": "baseline", "WAN22_PROFILE": args.profile,
        "WAN22_UPSCALE_RESIZE": args.resize, "WAN22_FOLEY_ENABLE": "0",
        "WAN22_UPSCALE_ENABLE": "1" if args.resolution == "1080p" else "0",
        "WAN22_LOG_DIR": str(args.output.resolve() / "logs"),
    })
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required; no generation was run")
    args.output.mkdir(parents=True, exist_ok=False)
    from wan22 import config
    from wan22.infer import attention, generate, upscale

    report_path = args.output / "report.json"
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    prompt = args.prompt_file.read_text().strip()
    if not prompt:
        raise SystemExit("prompt file is empty")
    settings.update({
        "fps": config.FPS, "guidance": [config.GUIDANCE_SCALE, config.GUIDANCE_SCALE_2],
        "max_frames": config.MAX_FRAMES, "spatial_multiple": config.SPATIAL_MULTIPLE,
        "text_quant": config.TEXT_ENCODER_QUANT, "vae_tiling": config.VAE_TILING,
        "resident_headroom_gib": config.RESIDENT_HEADROOM_GIB,
        "lora_scales": [config.NSFW_HIGH_SCALE, config.NSFW_LOW_SCALE],
        "upscale_model": config.UPSCALE_MODEL, "upscale_scale": config.UPSCALE_SCALE,
        "upscale_keep_wan": config.UPSCALE_KEEP_WAN_1080P,
        "model_dir": config.MODEL_DIR,
        "negative_prompt_sha256": hashlib.sha256(config.NEGATIVE_PROMPT.encode()).hexdigest(),
    })
    for name, path in (("image", args.image), ("last_image", args.last_image), ("prompt", args.prompt_file)):
        if path is not None:
            settings[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {
        "settings": settings, "versions": versions(), "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gib": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "cuda": torch.version.cuda, "runs": [], "status": "running",
    }
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, check=True, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=Path(__file__).parent, check=True, capture_output=True, text=True)
        report["code"] = {"commit": revision.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        report["code"] = None
    save_report(report_path, report)
    telemetry = Telemetry()
    telemetry.thread.start()
    try:
        started = time.perf_counter()
        pipe = generate.load_pipe()
        torch.cuda.synchronize()
        report["load_s"] = time.perf_counter() - started
        actual_backend = attention.snapshot()["backend"]
        if args.attention != "auto" and actual_backend != args.attention:
            raise RuntimeError(f"requested {args.attention} but loaded {actual_backend}; comparison aborted")
        if args.compile != "off":
            for expert in (pipe.transformer, pipe.transformer_2):
                expert.compile(mode=args.compile)
        for index in range(args.warmup + args.runs):
            output = args.output / f"run-{index:02d}.mp4"
            metrics = {"index": index, "warmup": index < args.warmup, "output": str(output)}
            report["runs"].append(metrics)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            gen_metrics = {}
            generate.generate_video(
                prompt, str(output), str(args.image), str(args.last_image) if args.last_image else None,
                args.duration, args.seed, args.steps, quality=args.quality,
                resolution=args.resolution, metrics=gen_metrics,
            )
            torch.cuda.synchronize()
            metrics["generate_s"] = time.perf_counter() - started
            metrics["generation"] = gen_metrics
            if args.attention in {"sage2", "sage3"} and gen_metrics["attention"]["sage_calls"] == 0:
                raise RuntimeError("no successful Sage calls; inspect attention dispatch before benchmarking")
            metrics["wan_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
            metrics["wan_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
            if args.resolution == "1080p":
                post_started = time.perf_counter()
                paused = not generate.can_share_gpu_for_upscale("1080p")
                if paused:
                    generate.pause_gpu()
                try:
                    try:
                        upscale.upscale_video(str(output))
                    except upscale.UpscaleOutOfMemory:
                        if paused:
                            raise
                        upscale.stop()
                        generate.pause_gpu()
                        paused = True
                        upscale.upscale_video(str(output))
                finally:
                    upscale.release_gpu()
                    if paused:
                        generate.resume_gpu()
                metrics["postprocess_s"] = time.perf_counter() - post_started
            metrics["total_s"] = time.perf_counter() - started
            metrics["video"] = inspect_video(output, gen_metrics["num_frames"])
            metrics["status"] = "succeeded"
            save_report(report_path, report)
        measured = [row["total_s"] for row in report["runs"] if not row["warmup"]]
        report["median_total_s"] = statistics.median(measured)
        report["status"] = "succeeded"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        upscale.stop()
        telemetry.stop()
        report["telemetry"] = {"fields": telemetry.fields, "samples": telemetry.samples, "error": telemetry.error}
        save_report(report_path, report)
    print(f"Report: {report_path}; median total: {report['median_total_s']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
