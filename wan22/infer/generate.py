from __future__ import annotations

import inspect
import json
import secrets
import threading
from pathlib import Path

from PIL import Image

from wan22 import config
from wan22.log import get_logger

logger = get_logger(__name__)

_pipe = None
_load_error: str | None = None
_pipe_lock = threading.Lock()

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
        raise ValueError(f"量化配置 {name!r} 不受当前 torchao 支持，可用预设: {supported}")
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
        raise ValueError(f"无法读取 {index_path}: {exc}") from exc

    if metadata.get("_class_name") != "WanImageToVideoPipeline":
        raise ValueError(
            f"{index_path} 不是 WanImageToVideoPipeline: "
            f"{metadata.get('_class_name')!r}"
        )
    if metadata.get("boundary_ratio") is None:
        raise ValueError("模型缺少 boundary_ratio，无法正确切换高/低噪声 Transformer")

    scheduler_path = Path(config.MODEL_DIR, "scheduler", "scheduler_config.json")
    try:
        scheduler = json.loads(scheduler_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取 {scheduler_path}: {exc}") from exc
    scheduler_name = scheduler.get("_class_name")
    flow_shift = scheduler.get("flow_shift", scheduler.get("shift"))
    if scheduler_name != "UniPCMultistepScheduler" or float(flow_shift) != 3.0:
        raise ValueError(
            "WAMU 参考配置要求 UniPCMultistepScheduler + flow_shift=3.0，"
            f"实际为 {scheduler_name} + {flow_shift}"
        )
    logger.info("scheduler %s flow_shift=%s", scheduler_name, flow_shift)


def _validate_config() -> None:
    if config.OFFLOAD not in {"model", "sequential", "none"}:
        raise ValueError("WAN22_OFFLOAD 只能是 model、sequential 或 none")
    if config.FOLEY_SIZE not in {"xl", "xxl"}:
        raise ValueError("WAN22_FOLEY_SIZE 只能是 xl 或 xxl")
    if config.FOLEY_STEPS < 1:
        raise ValueError("WAN22_FOLEY_STEPS 必须大于 0")
    if config.FOLEY_TIMEOUT < 1:
        raise ValueError("WAN22_FOLEY_TIMEOUT 必须大于 0")
    if config.NUM_STEPS < 1:
        raise ValueError("WAN22_STEPS 必须大于 0")
    if config.FPS < 1:
        raise ValueError("WAN22_FPS 必须大于 0")
    if config.MAX_FRAMES < 1:
        raise ValueError("WAN22_MAX_FRAMES 必须大于 0")
    if config.SPATIAL_MULTIPLE < 1:
        raise ValueError("WAN22_SPATIAL_MULTIPLE 必须大于 0")
    if config.MIN_DIM > config.MAX_DIM:
        raise ValueError("WAN22_MIN_DIM 不能大于 WAN22_MAX_DIM")
    for name, value in (
        ("WAN22_MIN_DIM", config.MIN_DIM),
        ("WAN22_MAX_DIM", config.MAX_DIM),
        ("WAN22_SQUARE_DIM", config.SQUARE_DIM),
    ):
        if value < 1 or value % config.SPATIAL_MULTIPLE:
            raise ValueError(
                f"{name} 必须是正数且能被 WAN22_SPATIAL_MULTIPLE 整除"
            )
    if not 1 <= config.VIDEO_QUALITY <= 10:
        raise ValueError("WAN22_VIDEO_QUALITY 必须在 1..10")
    for name, value in (
        ("WAN22_NSFW_HIGH_SCALE", config.NSFW_HIGH_SCALE),
        ("WAN22_NSFW_LOW_SCALE", config.NSFW_LOW_SCALE),
    ):
        if value < 0:
            raise ValueError(f"{name} 不能为负数")
    if config.FOLEY_ENABLE and config.OFFLOAD != "none":
        raise ValueError("Foley 需要 WAN22_OFFLOAD=none，成片后才能把 Wan 整模挪到 CPU")


def preflight() -> None:
    """在加载百 GB 权重之前验证目录、依赖和关键推理配置。"""
    from wan22.log import setup_logging

    setup_logging()
    _validate_config()

    model_dir = Path(config.MODEL_DIR)
    if not (model_dir / "model_index.json").is_file():
        raise FileNotFoundError(
            f"WAN22_MODEL_DIR={config.MODEL_DIR!r} 下没有 model_index.json"
        )
    for subfolder in ("transformer", "transformer_2", "vae", "text_encoder", "scheduler"):
        if not (model_dir / subfolder).is_dir():
            raise FileNotFoundError(f"{config.MODEL_DIR} 缺少子目录 {subfolder}")

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
            raise ValueError(f"LoRA {name} 不可用: {status} ({path})")

    if config.FOLEY_ENABLE:
        from wan22.infer import foley as foley_mod

        foley_mod.preflight()

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

    _fuse_nsfw_loras(pipe)
    _quantize_pipe(pipe)
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


def pause_gpu() -> None:
    """把 Wan pipeline 挪到 CPU，把显存让给 Foley。"""
    import torch

    with _pipe_lock:
        if _pipe is None:
            return
        _log_memory("wan pause gpu")
        _pipe.to("cpu")
        torch.cuda.empty_cache()
        _log_memory("wan on cpu")


def resume_gpu() -> None:
    """Foley 结束后把 Wan 放回 GPU。"""
    with _pipe_lock:
        if _pipe is None:
            return
        _log_memory("wan resume gpu")
        _pipe.to("cuda")
        _log_memory("wan on cuda")


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
    getattr(pipe, active).to("cuda")
    other = getattr(pipe, idle, None)
    if other is not None:
        other.to("cpu")
    torch.cuda.empty_cache()
    return active


def _stage_experts(pipe, num_frames: int):
    """长视频激活显存大，高/低噪声 transformer 分时上卡。"""
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    if num_frames <= 81 or getattr(pipe, "transformer_2", None) is None:
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
        timesteps = pipeline.scheduler.timesteps
        nxt = step_index + 1
        if nxt < len(timesteps):
            name = _place_experts(pipeline, timesteps[nxt], boundary)
            logger.info("expert swap step=%s next=%s", step_index + 1, name)
        return callback_kwargs

    return _on_step_end


def _snap_frames(seconds: float) -> int:
    """复刻参考 Space 的向下吸附规则；5 秒请求得到 77 帧。"""
    raw = max(9, round(float(seconds) * config.FPS))
    limit = max(1, ((config.MAX_FRAMES - 1) // 4) * 4 + 1)
    return min(((raw - 1) // 4) * 4 + 1, limit)


def _resize_for_wan(image: Image.Image, pipe) -> Image.Image:
    """按参考 Space 的 480×832 / 640×640 规则裁切并缩放首帧。"""
    image = image.convert("RGB")
    width, height = image.size
    if width == height:
        return image.resize((config.SQUARE_DIM, config.SQUARE_DIM), Image.Resampling.LANCZOS)

    aspect_ratio = width / height
    max_aspect = config.MAX_DIM / config.MIN_DIM
    min_aspect = config.MIN_DIM / config.MAX_DIM
    source = image

    if aspect_ratio > max_aspect:
        target_width, target_height = config.MAX_DIM, config.MIN_DIM
        crop_width = int(round(height * max_aspect))
        left = (width - crop_width) // 2
        source = image.crop((left, 0, left + crop_width, height))
    elif aspect_ratio < min_aspect:
        target_width, target_height = config.MIN_DIM, config.MAX_DIM
        crop_height = int(round(width / min_aspect))
        top = (height - crop_height) // 2
        source = image.crop((0, top, width, top + crop_height))
    elif width > height:
        target_width = config.MAX_DIM
        target_height = int(round(target_width / aspect_ratio))
    else:
        target_height = config.MAX_DIM
        target_width = int(round(target_height * aspect_ratio))

    model_multiple = int(
        pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    )
    multiple = max(config.SPATIAL_MULTIPLE, model_multiple)
    final_width = round(target_width / multiple) * multiple
    final_height = round(target_height / multiple) * multiple
    final_width = max(config.MIN_DIM, min(config.MAX_DIM, final_width))
    final_height = max(config.MIN_DIM, min(config.MAX_DIM, final_height))
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
) -> int:
    """生成单个 MP4，返回实际使用的 seed。"""
    used_seed = int(seed) if seed is not None else secrets.randbelow(2**31)
    used_steps = steps or config.NUM_STEPS
    used_quality = quality or config.VIDEO_QUALITY
    used_negative = (negative_prompt or "").strip() or config.NEGATIVE_PROMPT

    if not first_frame_path:
        if last_frame_path:
            raise ValueError("last_frame 必须与 first_frame 一起提供")
        raise ValueError("WAMU I2V 必须提供 first_frame")
    if config.DRY_RUN:
        logger.info("dry-run placeholder seed=%s duration=%s", used_seed, duration)
        _write_placeholder(output_path)
        return used_seed

    import torch
    from diffusers.utils import export_to_video
    from wan22.media.image import open_rgb

    pipe = load_pipe()
    image = _resize_for_wan(open_rgb(first_frame_path), pipe)

    last_image = None
    if last_frame_path:
        if not _supports_last_image():
            raise ValueError("当前 diffusers 不支持 last_image")
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

    on_step_end = _stage_experts(pipe, num_frames)
    if on_step_end is not None:
        kwargs["callback_on_step_end"] = on_step_end
    try:
        frames = pipe(**kwargs).frames[0]
    finally:
        if on_step_end is not None:
            try:
                pipe.to("cuda")
            except Exception:
                logger.exception("restore pipeline to cuda failed")
    export_to_video(
        frames,
        output_path,
        fps=config.FPS,
        quality=used_quality,
    )
    return used_seed


def _write_placeholder(output_path: str) -> None:
    Path(output_path).write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
