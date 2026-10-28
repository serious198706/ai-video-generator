from __future__ import annotations

import inspect
import json
import math
import secrets
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from wan22 import config
from wan22.log import get_logger

logger = get_logger(__name__)

_pipe = None
_sage_on = False
_load_error: str | None = None
_pipe_lock = threading.RLock()

_QUANT_PRESETS = {
    "int8wo": "Int8WeightOnlyConfig",
    "int4wo": "Int4WeightOnlyConfig",
    "float8wo": "Float8WeightOnlyConfig",
    "float8dq": "Float8DynamicActivationFloat8WeightConfig",
    "int8dq": "Int8DynamicActivationInt8WeightConfig",
}


def is_ready() -> bool:
    return config.DRY_RUN or _pipe is not None


def load_error() -> str | None:
    return _load_error


def _log_memory(stage: str) -> None:
    try:
        import torch

        props = torch.cuda.get_device_properties(0)
        gpu = f", gpu {torch.cuda.get_device_name(0)} {props.total_memory / 1024**3:.0f}GiB"
    except Exception:
        gpu = ""

    ram = ""
    try:
        with open("/proc/meminfo") as handle:
            info = {
                line.split(":")[0]: int(line.split()[1])
                for line in handle
                if ":" in line
            }
        ram = (
            f", ram avail {info['MemAvailable'] / 1024**2:.1f}GiB"
            f"/{info['MemTotal'] / 1024**2:.1f}GiB"
        )
    except Exception:
        pass
    logger.info("%s%s%s", stage, ram, gpu)


def _ao_config(name: str):
    if name in ("", "none"):
        return None

    from torchao import quantization

    class_name = _QUANT_PRESETS.get(name, name)
    factory = getattr(quantization, class_name, None)
    if factory is None:
        supported = sorted(
            alias
            for alias, candidate in _QUANT_PRESETS.items()
            if hasattr(quantization, candidate)
        )
        raise ValueError(f"current torchao does not support quantization preset {name!r}, available: {supported}")
    return factory()


def _supports_last_image() -> bool:
    from diffusers import WanImageToVideoPipeline

    return "last_image" in inspect.signature(WanImageToVideoPipeline.__call__).parameters


def _probe_lora(path: str) -> str:
    if not Path(path).is_file():
        return "MISSING"
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt") as handle:
            keys = list(handle.keys())
    except Exception as exc:
        return f"UNREADABLE({exc})"
    if not any("lora" in key.lower() for key in keys):
        example = keys[0] if keys else "-"
        return f"SUSPECT(no lora key, {len(keys)} keys, e.g. {example})"
    return f"ok({len(keys)} keys)"


def _check_prompt_path() -> None:
    from diffusers.pipelines.wan.pipeline_wan_i2v import prompt_clean

    prompt_clean(config.DEFAULT_PROMPT)
    prompt_clean(config.NEGATIVE_PROMPT)


def _check_model_metadata() -> None:
    index_path = Path(config.MODEL_DIR, "model_index.json")
    try:
        metadata = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {index_path}: {exc}") from exc

    if metadata.get("_class_name") != "WanImageToVideoPipeline":
        raise ValueError(
            f"{index_path} is not WanImageToVideoPipeline: "
            f"{metadata.get('_class_name')!r}"
        )
    if metadata.get("boundary_ratio") is None:
        raise ValueError("model missing boundary_ratio, cannot switch high/low noise Transformer")

    scheduler_path = Path(config.MODEL_DIR, "scheduler", "scheduler_config.json")
    try:
        scheduler = json.loads(scheduler_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {scheduler_path}: {exc}") from exc
    scheduler_name = scheduler.get("_class_name")
    flow_shift = scheduler.get("flow_shift", scheduler.get("shift"))
    if scheduler_name != "UniPCMultistepScheduler" or float(flow_shift) != 3.0:
        raise ValueError(
            "WAMU reference config requires UniPCMultistepScheduler + flow_shift=3.0, "
            f"actual: {scheduler_name} + {flow_shift}"
        )
    logger.info("scheduler %s flow_shift=%s", scheduler_name, flow_shift)


def _validate_config() -> None:
    if config.QUANT_1080P not in {"auto", "baseline", "float8dq"}:
        raise ValueError("WAN22_1080P_QUANT must be auto, baseline or float8dq")
    if config.UPSCALE_KEEP_WAN_1080P not in {"auto", "on", "off"}:
        raise ValueError("WAN22_1080P_UPSCALE_KEEP_WAN must be auto, on or off")
    if config.EXPERT_RESIDENCY not in {"auto", "on", "off"}:
        raise ValueError("WAN22_EXPERT_RESIDENCY must be auto, on or off")
    if not math.isfinite(config.RESIDENT_HEADROOM_GIB) or config.RESIDENT_HEADROOM_GIB <= 0:
        raise ValueError("WAN22_RESIDENT_HEADROOM_GIB must be finite and positive")
    if config.PROFILE not in {"auto", "on", "off"}:
        raise ValueError("WAN22_PROFILE must be auto, on or off")
    if config.SAGE_BACKEND not in {"auto", "sdpa", "sage2", "sage3"}:
        raise ValueError("WAN22_SAGE_BACKEND must be auto, sdpa, sage2 or sage3")
    if config.UPSCALE_RESIZE not in {"cpu_lanczos", "gpu_bicubic"}:
        raise ValueError("WAN22_UPSCALE_RESIZE must be cpu_lanczos or gpu_bicubic")
    if config.OFFLOAD not in {"model", "sequential", "none"}:
        raise ValueError("WAN22_OFFLOAD must be model, sequential or none")
    if config.FOLEY_SIZE not in {"xl", "xxl"}:
        raise ValueError("WAN22_FOLEY_SIZE must be xl or xxl")
    if config.FOLEY_STEPS < 1:
        raise ValueError("WAN22_FOLEY_STEPS must be greater than 0")
    if config.FOLEY_TIMEOUT < 1:
        raise ValueError("WAN22_FOLEY_TIMEOUT must be greater than 0")
    if config.UPSCALE_TIMEOUT < 1:
        raise ValueError("WAN22_UPSCALE_TIMEOUT must be greater than 0")
    if config.UPSCALE_ENABLE and config.UPSCALE_SCALE <= 1:
        raise ValueError("WAN22_UPSCALE_SCALE must be greater than 1")
    if config.NUM_STEPS < 1:
        raise ValueError("WAN22_STEPS must be greater than 0")
    if config.FPS < 1:
        raise ValueError("WAN22_FPS must be greater than 0")
    if config.MAX_FRAMES < 1:
        raise ValueError("WAN22_MAX_FRAMES must be greater than 0")
    if config.SPATIAL_MULTIPLE < 1:
        raise ValueError("WAN22_SPATIAL_MULTIPLE must be greater than 0")
    if config.MIN_DIM > config.MAX_DIM:
        raise ValueError("WAN22_MIN_DIM must be less than WAN22_MAX_DIM")
    for name, value in (
        ("WAN22_MIN_DIM", config.MIN_DIM),
        ("WAN22_MAX_DIM", config.MAX_DIM),
        ("WAN22_SQUARE_DIM", config.SQUARE_DIM),
    ):
        if value < 1 or value % config.SPATIAL_MULTIPLE:
            raise ValueError(
                f"{name} must be a positive number and divisible by WAN22_SPATIAL_MULTIPLE"
            )
    if not 1 <= config.VIDEO_QUALITY <= 10:
        raise ValueError("WAN22_VIDEO_QUALITY must be between 1 and 10")
    for name, value in (
        ("WAN22_NSFW_HIGH_SCALE", config.NSFW_HIGH_SCALE),
        ("WAN22_NSFW_LOW_SCALE", config.NSFW_LOW_SCALE),
    ):
        if value < 0:
            raise ValueError(f"{name} must not be negative")
    if (config.FOLEY_ENABLE or config.UPSCALE_ENABLE) and config.OFFLOAD != "none":
        raise ValueError(
            "Foley/upscale requires WAN22_OFFLOAD=none, Wan must be moved to CPU after generation"
        )


def preflight() -> None:
    """在加载百 GB 权重之前验证目录、依赖和关键推理配置。"""
    from wan22.log import setup_logging

    setup_logging()
    _validate_config()

    model_dir = Path(config.MODEL_DIR)
    if not (model_dir / "model_index.json").is_file():
        raise FileNotFoundError(
            f"WAN22_MODEL_DIR={config.MODEL_DIR!r} does not have model_index.json"
        )
    for subfolder in ("transformer", "transformer_2", "vae", "text_encoder", "scheduler"):
        if not (model_dir / subfolder).is_dir():
            raise FileNotFoundError(f"{config.MODEL_DIR} missing subdirectory {subfolder}")

    _check_model_metadata()
    _ao_config(config.QUANT)
    _ao_config(config.TEXT_ENCODER_QUANT)
    _check_prompt_path()

    for name, path, scale in (
        ("nsfw_high", config.NSFW_HIGH, config.NSFW_HIGH_SCALE),
        ("nsfw_low", config.NSFW_LOW, config.NSFW_LOW_SCALE),
    ):
        status = _probe_lora(path)
        logger.info("lora %s: %s, scale=%s, %s", name, status, scale, path)
        if status.startswith(("MISSING", "UNREADABLE", "SUSPECT")):
            raise ValueError(f"LoRA {name} unavailable: {status} ({path})")

    if config.FOLEY_ENABLE:
        from wan22.infer import foley as foley_mod

        foley_mod.preflight()

    if config.UPSCALE_ENABLE:
        from wan22.infer import upscale as upscale_mod

        upscale_mod.preflight()

    logger.info(
        "last_image support=%s steps=%s cfg=%s/%s",
        _supports_last_image(),
        config.NUM_STEPS,
        config.GUIDANCE_SCALE,
        config.GUIDANCE_SCALE_2,
    )


def _load_transformer(subfolder: str):
    import torch
    from diffusers import WanTransformer3DModel

    return WanTransformer3DModel.from_pretrained(
        config.MODEL_DIR,
        subfolder=subfolder,
        torch_dtype=torch.bfloat16,
    )


def _fuse_nsfw_loras(pipe) -> None:
    """在量化前把 Space 同款 General NSFW Booster 固定融合进两个 expert。"""
    specs = (
        ("nsfw_high", config.NSFW_HIGH, config.NSFW_HIGH_SCALE, "transformer", False),
        ("nsfw_low", config.NSFW_LOW, config.NSFW_LOW_SCALE, "transformer_2", True),
    )

    for adapter, path, scale, component, into_second in specs:
        kwargs = {"adapter_name": adapter}
        if into_second:
            kwargs["load_into_transformer_2"] = True
        pipe.load_lora_weights(path, **kwargs)

        model = getattr(pipe, component)
        model.set_adapters([adapter], weights=[scale])
        pipe.fuse_lora(
            components=[component],
            adapter_names=[adapter],
            lora_scale=1.0,
        )
        logger.info("fused %s into %s, scale=%s", adapter, component, scale)

    pipe.unload_lora_weights()


def _quantize_pipe(pipe) -> None:
    from torchao.quantization import quantize_

    text_config = _ao_config(config.TEXT_ENCODER_QUANT)
    if text_config is not None:
        _log_memory(f"quantizing text encoder ({config.TEXT_ENCODER_QUANT})")
        quantize_(pipe.text_encoder, text_config)

    transformer_config = _ao_config(config.QUANT)
    if transformer_config is None:
        return

    _log_memory(f"quantizing high-noise transformer ({config.QUANT})")
    quantize_(pipe.transformer, transformer_config)
    _log_memory(f"quantizing low-noise transformer ({config.QUANT})")
    # 每次创建独立配置，避免量化实现持有组件级状态。
    quantize_(pipe.transformer_2, _ao_config(config.QUANT))


def _place_pipe(pipe) -> None:
    if config.OFFLOAD == "sequential":
        _log_memory("enable sequential cpu offload")
        pipe.enable_sequential_cpu_offload()
    elif config.OFFLOAD == "model":
        _log_memory("enable model cpu offload")
        pipe.enable_model_cpu_offload()
    else:
        _log_memory("placing pipeline on cuda (no offload)")
        pipe.to("cuda")


def _apply_sage() -> bool:
    from wan22.infer import attention

    global _sage_on
    _sage_on = attention.install(config.SAGE_ENABLE, config.SAGE_BACKEND)
    return _sage_on


def _prepare_initial_gpu(pipe) -> bool:
    """大显存且实际空闲充足时，在 GPU 融合/量化；小卡保持原路径。"""
    import torch

    if config.OFFLOAD != "none" or torch.cuda.get_device_properties(0).total_memory < 90 * 1024**3:
        return False
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    if free < 80 * 1024**3:
        logger.info("initial preparation on CPU: free_gib=%.2f (GPU preparation needs 80)", free / 1024**3)
        return False
    try:
        _log_memory("moving unquantized pipeline to GPU for LoRA fusion and quantization")
        pipe.to("cuda")
    except torch.cuda.OutOfMemoryError:
        pipe.to("cpu")
        torch.cuda.empty_cache()
        logger.warning("initial GPU placement OOM; continue preparation on CPU")
        return False
    return True


def _build_pipe():
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    preflight()
    _log_memory("loading WAMU high-noise transformer")
    transformer = _load_transformer("transformer")
    _log_memory("loading WAMU low-noise transformer")
    transformer_2 = _load_transformer("transformer_2")
    _log_memory("loading fp32 VAE")
    vae = AutoencoderKLWan.from_pretrained(
        config.MODEL_DIR,
        subfolder="vae",
        torch_dtype=torch.float32,
    )

    _log_memory("assembling WAMU pipeline")
    pipe = WanImageToVideoPipeline.from_pretrained(
        config.MODEL_DIR,
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        image_encoder=None,
        image_processor=None,
        torch_dtype=torch.bfloat16,
    )
    if config.VAE_TILING:
        pipe.vae.enable_tiling()

    _prepare_initial_gpu(pipe)
    _fuse_nsfw_loras(pipe)
    _quantize_pipe(pipe)
    _apply_sage()
    _place_pipe(pipe)
    _log_memory("pipeline ready")
    return pipe


def load_pipe():
    """加载并缓存唯一的推理 Pipeline。"""
    global _pipe, _load_error
    with _pipe_lock:
        if _pipe is not None:
            return _pipe
        try:
            _pipe = _build_pipe()
            _load_error = None
            return _pipe
        except Exception as exc:
            _load_error = str(exc)
            logger.exception("pipeline load failed")
            raise


def _use_fp8_1080p(resolution: str | None) -> bool:
    # auto 曾默认开启实验 FP8；发现非有限输出后，兼容旧 .env 的 auto 也恢复基线。
    if (resolution or "").lower() != "1080p" or config.QUANT_1080P != "float8dq":
        return False
    if config.OFFLOAD != "none":
        raise ValueError("1080p float8dq requires WAN22_OFFLOAD=none")
    import torch

    supported = torch.cuda.get_device_capability(0) >= (8, 9)
    if not supported:
        raise ValueError("1080p float8dq requires NVIDIA compute capability >=8.9")
    return True


def _prepare_fp8_linears(expert):
    """保留 Wan 的 FP32 例外层，仅归一化参与 FP8 的 Linear。返回同一量化过滤器。"""
    import torch

    keep_fp32 = set(getattr(expert, "_keep_in_fp32_modules", None) or [])

    def eligible(module, fqn):
        return isinstance(module, torch.nn.Linear) and not keep_fp32.intersection(fqn.split("."))

    for name, module in expert.named_modules():
        if eligible(module, name):
            module.to(dtype=torch.bfloat16)
    return eligible


def _select_transformers(pipe, resolution: str | None) -> str:
    """仅切换 expert；基线保留在 CPU，不能在 INT8 权重上再次量化。单任务执行位保护调用。"""
    import torch

    use_fp8 = _use_fp8_1080p(resolution)
    if config.QUANT_1080P == "auto" and getattr(pipe, "_wan22_fp8_disabled", False):
        use_fp8 = False
    desired = "1080p-float8dq" if use_fp8 else "baseline"
    if not hasattr(pipe, "_wan22_expert_cache"):
        pipe._wan22_expert_cache = {"baseline": (pipe.transformer, pipe.transformer_2)}
        pipe._wan22_expert_mode = "baseline"
    previous_mode = pipe._wan22_expert_mode
    if previous_mode == desired:
        return "float8dq" if use_fp8 else config.QUANT

    started = time.perf_counter()
    previous = pipe._wan22_expert_cache[previous_mode]
    creating = desired not in pipe._wan22_expert_cache
    for expert in previous:
        expert.to("cpu")
    torch.cuda.empty_cache()
    try:
        if desired not in pipe._wan22_expert_cache:
            from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_
            from torchao.quantization.granularity import PerRow

            logger.info("1080p preparing separate FP8 experts (first use only; baseline cached on CPU)")
            pipe.transformer = _load_transformer("transformer")
            pipe.transformer_2 = _load_transformer("transformer_2")
            # 基线已在 CPU；新 pair 在 GPU 融合，避免大型 LoRA 矩阵乘法耗在 CPU 上。
            fuse_started = time.perf_counter()
            for expert in (pipe.transformer, pipe.transformer_2):
                expert.to("cuda")
            _fuse_nsfw_loras(pipe)
            torch.cuda.synchronize()
            logger.info("1080p FP8 LoRA fusion on GPU fuse_s=%.3f", time.perf_counter() - fuse_started)
            quantize_started = time.perf_counter()
            for expert in (pipe.transformer, pipe.transformer_2):
                # Wan 有刻意保留 FP32 的层，LoRA 也可能提升权重精度。
                # 不能将整个 expert 强制 BF16；FP32 例外不参与 FP8。
                eligible = _prepare_fp8_linears(expert)
                quantize_(expert, Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()), filter_fn=eligible)
            torch.cuda.synchronize()
            logger.info("1080p FP8 quantized from BF16 quantize_s=%.3f", time.perf_counter() - quantize_started)
            pipe._wan22_expert_cache[desired] = (pipe.transformer, pipe.transformer_2)
        selected = pipe._wan22_expert_cache[desired]
        pipe.transformer, pipe.transformer_2 = selected
        for expert in selected:
            expert.to("cuda")
    except Exception:
        # 丢弃未完成的 pair，保证后续基线请求仍可使用原来的权重。
        for expert in (pipe.transformer, pipe.transformer_2):
            expert.to("cpu")
        pipe.transformer, pipe.transformer_2 = previous
        if creating:
            pipe._wan22_expert_cache.pop(desired, None)
        torch.cuda.empty_cache()
        for expert in previous:
            expert.to("cuda")
        if use_fp8 and config.QUANT_1080P == "auto":
            pipe._wan22_fp8_disabled = True
            logger.exception("1080p FP8 preparation failed; restored baseline, disable FP8 auto for this process")
            return config.QUANT
        raise
    pipe._wan22_expert_mode = desired
    torch.cuda.synchronize()
    logger.info("expert mode=%s switch_s=%.3f", desired, time.perf_counter() - started)
    return "float8dq" if use_fp8 else config.QUANT


def can_share_gpu_for_upscale(resolution: str | None) -> bool:
    """只用于无 Foley 的 1080p；给超分进程至少留 16 GiB 实际空闲显存。"""
    if (resolution or "").lower() != "1080p" or config.UPSCALE_KEEP_WAN_1080P == "off":
        return False
    if config.OFFLOAD != "none":
        return False
    import torch

    if config.UPSCALE_KEEP_WAN_1080P == "auto" and torch.cuda.get_device_properties(0).total_memory < 90 * 1024**3:
        return False
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    shared = free >= 16 * 1024**3
    logger.info("1080p upscale GPU sharing=%s free_gib=%.2f required_free_gib=16", int(shared), free / 1024**3)
    return shared


def pause_gpu() -> None:
    """把 Wan pipeline 挪到 CPU，把显存让给超分 / Foley。"""
    import torch

    with _pipe_lock:
        if _pipe is None:
            return
        started = time.perf_counter()
        _log_memory("wan pause gpu")
        _pipe.to("cpu")
        torch.cuda.empty_cache()
        _log_memory(f"wan on cpu transfer_s={time.perf_counter() - started:.3f}")


def resume_gpu() -> None:
    """超分 / Foley 结束后把 Wan 放回 GPU。"""
    with _pipe_lock:
        if _pipe is None:
            return
        started = time.perf_counter()
        _log_memory("wan resume gpu")
        _restore_staged_pipe(_pipe)
        import torch

        torch.cuda.synchronize()
        _log_memory(f"wan on cuda transfer_s={time.perf_counter() - started:.3f}")


def _restore_staged_pipe(pipe) -> None:
    """A failed restore must not leave a half-resident pipeline marked ready."""
    global _pipe, _load_error
    try:
        pipe.to("cuda")
    except Exception as exc:
        with _pipe_lock:
            if _pipe is pipe:
                _pipe = None
                _load_error = f"pipeline restore failed: {exc}"
        try:
            pipe.to("cpu")
        except Exception:
            logger.exception("failed to release partially restored pipeline")
        raise


def _boundary_timestep(pipe):
    ratio = getattr(pipe.config, "boundary_ratio", None)
    if ratio is None:
        return None
    return float(ratio) * float(pipe.scheduler.config.num_train_timesteps)


def _place_experts(pipe, timestep, boundary) -> str:
    """单卡 44GB：同一时刻只把当前 expert 放在 GPU。"""
    import torch

    need_high = boundary is None or float(timestep) >= float(boundary)
    active = "transformer" if need_high else "transformer_2"
    idle = "transformer_2" if need_high else "transformer"
    other = getattr(pipe, idle, None)
    if other is not None:
        other.to("cpu")
    torch.cuda.empty_cache()
    getattr(pipe, active).to("cuda")
    return active


def _resident_experts(width: int, height: int, num_frames: int) -> bool:
    """Use actual generation load, not the requested output resolution, for all jobs.

    Auto assumes both experts have been restored by load_pipe / the previous job.
    The configurable headroom is a conservative heuristic, not an OOM guarantee.
    """
    if config.OFFLOAD != "none":
        return False
    mode = config.EXPERT_RESIDENCY
    if mode != "auto":
        return mode == "on"
    import torch

    if torch.cuda.get_device_properties(0).total_memory < 90 * 1024**3:
        return False
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    load = (width * height / (720 * 1248)) * (num_frames / 81)
    required_gib = config.RESIDENT_HEADROOM_GIB * max(1.0, load)
    resident = free / 1024**3 >= required_gib
    logger.info(
        "expert residency=%s canvas=%sx%s frames=%s free_gib=%.2f budget_gib=%.2f",
        int(resident), width, height, num_frames, free / 1024**3, required_gib,
    )
    return resident


@contextmanager
def _profile_generation(pipe):
    """仅本次请求计时；临时包装方法，并在成功/异常时恢复。各阶段互不嵌套。"""
    import torch

    stats = {}
    patched = []
    missing = object()

    def wrap(owner, name, label):
        original = getattr(owner, name)
        previous = vars(owner).get(name, missing)

        def timed(*args, **kwargs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                torch.cuda.synchronize()
                stats[label] = stats.get(label, 0.0) + time.perf_counter() - started

        setattr(owner, name, timed)
        patched.append((owner, name, previous))

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        wrap(pipe, "encode_prompt", "prompt_s")
        # prepare_latents 包含首/尾图的 VAE 编码。
        wrap(pipe, "prepare_latents", "latent_prepare_s")
        wrap(pipe.vae, "decode", "vae_decode_s")
        wrap(pipe.transformer, "forward", "denoise_high_s")
        if getattr(pipe, "transformer_2", None) is not None:
            wrap(pipe.transformer_2, "forward", "denoise_low_s")
        yield stats
    finally:
        stats["denoise_s"] = stats.get("denoise_high_s", 0.0) + stats.get("denoise_low_s", 0.0)
        for owner, name, previous in reversed(patched):
            if previous is missing:
                delattr(owner, name)
            else:
                setattr(owner, name, previous)
        logger.info(
            "generation profile %s peak_allocated_gib=%.2f peak_reserved_gib=%.2f",
            " ".join(f"{key}={value:.3f}" for key, value in sorted(stats.items())),
            torch.cuda.max_memory_allocated() / 1024**3,
            torch.cuda.max_memory_reserved() / 1024**3,
        )
        stats["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        stats["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3


def _stage_experts(pipe, num_frames: int, *, resident: bool = False):
    """长视频激活显存大，高/低噪声 transformer 分时上卡。"""
    # Accelerate owns placement in offload modes; never mix manual .to() with hooks.
    if config.OFFLOAD != "none":
        return None
    if resident:
        pipe.transformer.to("cuda")
        if getattr(pipe, "transformer_2", None) is not None:
            pipe.transformer_2.to("cuda")
        logger.info("expert residency=on frames=%s (no ping-pong)", num_frames)
        return None
    if getattr(pipe, "transformer_2", None) is None:
        return None

    boundary = _boundary_timestep(pipe)
    active = _place_experts(pipe, 1e9, boundary)
    logger.info(
        "expert ping-pong start=%s boundary=%s frames=%s",
        active,
        boundary,
        num_frames,
    )

    def _on_step_end(pipeline, step_index, timestep, callback_kwargs):
        nonlocal active
        timesteps = pipeline.scheduler.timesteps
        nxt = step_index + 1
        if nxt < len(timesteps):
            need_high = boundary is None or float(timesteps[nxt]) >= boundary
            name = "transformer" if need_high else "transformer_2"
            if name != active:
                active = _place_experts(pipeline, timesteps[nxt], boundary)
                logger.info("expert swap step=%s next=%s", step_index + 1, active)
        return callback_kwargs

    return _on_step_end


def _snap_frames(seconds: float) -> int:
    """复刻参考 Space 的向下吸附规则；5 秒请求得到 77 帧。"""
    raw = max(9, round(float(seconds) * config.FPS))
    limit = max(1, ((config.MAX_FRAMES - 1) // 4) * 4 + 1)
    return min(((raw - 1) // 4) * 4 + 1, limit)


def dims_for_resolution(resolution: str | None) -> tuple[int, int, int]:
    """返回 (min_dim, max_dim, square_dim)。480p 对齐现网 480×832。"""
    key = (resolution or "480p").lower()
    mapping = {
        "480p": (480, 832, 640),
        "540p": (480, 832, 640),
        "720p": (720, 1248, 960),
        "1080p": (1080, 1872, 1440),
    }
    return mapping.get(key, mapping["480p"])


def generate_dims(resolution: str | None) -> tuple[int, int, int]:
    """I2V 画布。1080p 在官方 720p 上生成，再超分到输出尺寸。"""
    key = (resolution or "480p").lower()
    if key == "1080p":
        key = "720p"
    return dims_for_resolution(key)


def _resize_for_wan(
    image: Image.Image,
    pipe,
    *,
    min_dim: int,
    max_dim: int,
    square_dim: int,
) -> Image.Image:
    """按参考 Space 的 480×832 / 640×640 规则裁切并缩放首帧。"""
    image = image.convert("RGB")
    width, height = image.size
    if width == height:
        return image.resize((square_dim, square_dim), Image.Resampling.LANCZOS)

    aspect_ratio = width / height
    max_aspect = max_dim / min_dim
    min_aspect = min_dim / max_dim
    source = image

    if aspect_ratio > max_aspect:
        target_width, target_height = max_dim, min_dim
        crop_width = int(round(height * max_aspect))
        left = (width - crop_width) // 2
        source = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < min_aspect:
        target_width, target_height = min_dim, max_dim
        crop_height = int(round(width / min_aspect))
        top = (height - crop_height) // 2
        source = image.crop((0, top, width, top + crop_height))
    elif width > height:
        target_width = max_dim
        target_height = int(round(target_width / aspect_ratio))
    else:
        target_height = max_dim
        target_width = int(round(target_height * aspect_ratio))

    model_multiple = int(
        pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    )
    multiple = max(config.SPATIAL_MULTIPLE, model_multiple)
    final_width = round(target_width / multiple) * multiple
    final_height = round(target_height / multiple) * multiple
    final_width = max(min_dim, min(max_dim, final_width))
    final_height = max(min_dim, min(max_dim, final_height))
    return source.resize((final_width, final_height), Image.Resampling.LANCZOS)


def _resize_and_crop_to_match(
    image: Image.Image,
    reference: Image.Image,
) -> Image.Image:
    image = image.convert("RGB")
    scale = max(reference.width / image.width, reference.height / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    left = (resized.width - reference.width) // 2
    top = (resized.height - reference.height) // 2
    return resized.crop(
        (left, top, left + reference.width, top + reference.height)
    )


def _validate_video_frames(frames) -> None:
    """在编码/超分/上传之前拒绝 NaN/Inf 和整段纯黑，不将坏结果标为成功。"""
    import numpy as np

    if len(frames) == 0:
        raise ValueError("invalid generated video: no frames")
    all_black = True
    for index, frame in enumerate(frames):
        array = np.asarray(frame)
        if array.size == 0 or not np.isfinite(array).all():
            raise ValueError(f"invalid generated video: empty/non-finite values at frame {index}; export blocked")
        if np.any(array != 0):
            all_black = False
    if all_black:
        raise ValueError("invalid generated video: all frames are black; export blocked")


def generate_video(
    prompt: str,
    output_path: str,
    first_frame_path: str | None,
    last_frame_path: str | None,
    duration: float,
    seed: int | None,
    steps: int | None,
    negative_prompt: str | None = None,
    quality: int | None = None,
    resolution: str | None = None,
    metrics: dict | None = None,
) -> int:
    """生成单个 MP4，返回实际使用的 seed。"""
    used_seed = int(seed) if seed is not None else secrets.randbelow(2**31)
    used_steps = steps or config.NUM_STEPS
    used_quality = quality or config.VIDEO_QUALITY
    used_negative = (negative_prompt or "").strip() or config.NEGATIVE_PROMPT

    if not first_frame_path:
        if last_frame_path:
            raise ValueError("last_frame must be provided with first_frame")
        raise ValueError("WAMU I2V must provide first_frame")
    if config.DRY_RUN:
        min_dim, max_dim, _square = generate_dims(resolution)
        logger.info(
            "dry-run placeholder seed=%s duration=%s resolution=%s canvas=%sx%s",
            used_seed,
            duration,
            resolution or "480p",
            min_dim,
            max_dim,
        )
        _write_placeholder(output_path)
        return used_seed

    import torch
    from diffusers.utils import export_to_video
    from wan22.infer import attention
    from wan22.media.image import open_rgb

    pipe = load_pipe()
    quant_started = time.perf_counter()
    transformer_quant = _select_transformers(pipe, resolution)
    quant_select_s = time.perf_counter() - quant_started
    min_dim, max_dim, square_dim = generate_dims(resolution)
    image = _resize_for_wan(
        open_rgb(first_frame_path),
        pipe,
        min_dim=min_dim,
        max_dim=max_dim,
        square_dim=square_dim,
    )

    last_image = None
    if last_frame_path:
        if not _supports_last_image():
            raise ValueError("current diffusers does not support last_image")
        last_image = _resize_and_crop_to_match(open_rgb(last_frame_path), image)

    num_frames = _snap_frames(duration)
    generator = torch.Generator(device="cuda").manual_seed(used_seed)
    mode = "flf2v" if last_image is not None else "i2v"
    logger.info(
        "generating %s %sx%s %sf steps=%s cfg=%s/%s seed=%s",
        mode,
        image.width,
        image.height,
        num_frames,
        used_steps,
        config.GUIDANCE_SCALE,
        config.GUIDANCE_SCALE_2,
        used_seed,
    )

    kwargs = {
        "image": image,
        "prompt": prompt,
        "negative_prompt": used_negative,
        "height": image.height,
        "width": image.width,
        "num_frames": num_frames,
        "num_inference_steps": used_steps,
        "guidance_scale": config.GUIDANCE_SCALE,
        "guidance_scale_2": config.GUIDANCE_SCALE_2,
        "generator": generator,
        "output_type": "np",
    }
    if last_image is not None:
        kwargs["last_image"] = last_image

    is_1080p = (resolution or "").lower() == "1080p"
    stage_started = time.perf_counter()
    resident = _resident_experts(image.width, image.height, num_frames)
    restore_needed = config.OFFLOAD == "none" and not resident
    attention_before = attention.snapshot()
    stats = {}
    profile = config.PROFILE == "on" or (config.PROFILE == "auto" and is_1080p)
    try:
        on_step_end = _stage_experts(pipe, num_frames, resident=resident)
        stage_s = time.perf_counter() - stage_started
        if on_step_end is not None:
            kwargs["callback_on_step_end"] = on_step_end
        if profile:
            with _profile_generation(pipe) as stats:
                frames = pipe(**kwargs).frames[0]
        else:
            frames = pipe(**kwargs).frames[0]
    finally:
        attention_stats = attention.delta(attention_before)
        restore_started = time.perf_counter()
        if restore_needed:
            _restore_staged_pipe(pipe)
        restore_s = time.perf_counter() - restore_started
    export_started = time.perf_counter()
    _validate_video_frames(frames)
    export_to_video(
        frames,
        output_path,
        fps=config.FPS,
        quality=used_quality,
    )
    export_s = time.perf_counter() - export_started
    if metrics is not None:
        metrics.update(
            **stats, attention=attention_stats, width=image.width, height=image.height,
            num_frames=num_frames, resident=resident, quant=transformer_quant,
            quant_select_s=quant_select_s, stage_s=stage_s, restore_s=restore_s, export_s=export_s,
        )
    logger.info(
        "generation resolution=%s quant=%s quant_select_s=%.3f resident=%s stage_s=%.3f restore_s=%.3f export_s=%.3f seed=%s",
        resolution or "480p", transformer_quant, quant_select_s, int(resident), stage_s, restore_s, export_s, used_seed,
    )
    return used_seed


def _write_placeholder(output_path: str) -> None:
    Path(output_path).write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
