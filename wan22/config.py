from __future__ import annotations

import os
from pathlib import Path


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


_SERVER_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("WAN22_DATA_DIR", _SERVER_ROOT / "data"))
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR = os.environ.get(
    "WAN22_MODEL_DIR",
    "/data/models/wan22/base/WAMU_v3_WAN2.2_I2V_LIGHTNING",
)
LORA_DIR = Path(os.environ.get("WAN22_LORA_DIR", "/data/models/wan22/loras"))
NSFW_HIGH = os.environ.get(
    "WAN22_NSFW_HIGH",
    str(LORA_DIR / "nsfw" / "NSFW-22-H-e8.safetensors"),
)
NSFW_LOW = os.environ.get(
    "WAN22_NSFW_LOW",
    str(LORA_DIR / "nsfw" / "NSFW-22-L-e8.safetensors"),
)
NSFW_HIGH_SCALE = float(os.environ.get("WAN22_NSFW_HIGH_SCALE", "1.0"))
NSFW_LOW_SCALE = float(os.environ.get("WAN22_NSFW_LOW_SCALE", "1.0"))

DRY_RUN = os.environ.get("WAN22_DRY_RUN", "0") == "1"
PRELOAD = os.environ.get("WAN22_PRELOAD", "1") == "1"
OFFLOAD = os.environ.get("WAN22_OFFLOAD", "none").lower()
QUANT = os.environ.get("WAN22_QUANT", "int8wo").lower()
TEXT_ENCODER_QUANT = os.environ.get("WAN22_TEXT_ENCODER_QUANT", "int8wo").lower()
VAE_TILING = os.environ.get("WAN22_VAE_TILING", "1") == "1"
ATTENTION_BACKEND = os.environ.get("WAN22_ATTENTION_BACKEND", "native").lower()

NUM_STEPS = int(os.environ.get("WAN22_STEPS", "6"))
GUIDANCE_SCALE = float(os.environ.get("WAN22_GUIDANCE_SCALE", "1.0"))
GUIDANCE_SCALE_2 = float(os.environ.get("WAN22_GUIDANCE_SCALE_2", "1.0"))
FPS = int(os.environ.get("WAN22_FPS", "16"))
MAX_DIM = int(os.environ.get("WAN22_MAX_DIM", "832"))
MIN_DIM = int(os.environ.get("WAN22_MIN_DIM", "480"))
SQUARE_DIM = int(os.environ.get("WAN22_SQUARE_DIM", "640"))
SPATIAL_MULTIPLE = int(os.environ.get("WAN22_SPATIAL_MULTIPLE", "16"))
MAX_FRAMES = int(os.environ.get("WAN22_MAX_FRAMES", "321"))
DEFAULT_DURATION = float(os.environ.get("WAN22_DURATION", "5"))
VIDEO_QUALITY = int(os.environ.get("WAN22_VIDEO_QUALITY", "6"))

S3_BUCKET = os.environ.get("WAN22_S3_BUCKET", "")
S3_REGION = os.environ.get("WAN22_S3_REGION", os.environ.get("AWS_DEFAULT_REGION", ""))
S3_PREFIX = os.environ.get("WAN22_S3_PREFIX", "wan22/")
S3_PUBLIC_BASE_URL = os.environ.get("WAN22_S3_PUBLIC_BASE_URL", "").rstrip("/")
S3_ENDPOINT = os.environ.get("WAN22_S3_ENDPOINT", "")

REDIS_URL = os.environ.get("WAN22_REDIS_URL", "redis://127.0.0.1:6379/0")
QUEUE_MAX = int(os.environ.get("WAN22_QUEUE_MAX", "50"))
UPLOAD_MAX_BYTES = int(os.environ.get("WAN22_UPLOAD_MAX_BYTES", str(10 * 1024 * 1024)))
ENABLE_DOCS = os.environ.get("WAN22_DOCS", "0") == "1"
IMAGE_HOSTS = _csv("WAN22_IMAGE_HOSTS")
WEBHOOK_HOSTS = _csv("WAN22_WEBHOOK_HOSTS")
WEBHOOK_SECRET = os.environ.get("WAN22_WEBHOOK_SECRET", "")
DOWNLOAD_TIMEOUT = int(os.environ.get("WAN22_DOWNLOAD_TIMEOUT", "30"))
WEBHOOK_TIMEOUT = float(os.environ.get("WAN22_WEBHOOK_TIMEOUT", "5"))
WEBHOOK_RETRIES = int(os.environ.get("WAN22_WEBHOOK_RETRIES", "3"))

LOG_DIR = os.environ.get("WAN22_LOG_DIR", str(_SERVER_ROOT / "logs"))
LOG_LEVEL = os.environ.get("WAN22_LOG_LEVEL", "INFO").upper()
LOG_BACKUP_DAYS = int(os.environ.get("WAN22_LOG_BACKUP_DAYS", "30"))
LOG_CONSOLE = os.environ.get("WAN22_LOG_CONSOLE", "1") == "1"

FOLEY_ENABLE = os.environ.get("WAN22_FOLEY_ENABLE", "0") == "1"
FOLEY_REQUIRED = os.environ.get("WAN22_FOLEY_REQUIRED", "0") == "1"
FOLEY_PYTHON = os.environ.get("WAN22_FOLEY_PYTHON", "").strip()
FOLEY_REPO = Path(os.environ.get("WAN22_FOLEY_REPO", "")).expanduser() if os.environ.get("WAN22_FOLEY_REPO") else None
FOLEY_MODEL_DIR = Path(
    os.environ.get("WAN22_FOLEY_MODEL_DIR", "/data/models/hunyuanvideo-foley")
).expanduser()
FOLEY_SIZE = os.environ.get("WAN22_FOLEY_SIZE", "xl").strip().lower()
FOLEY_PROMPT = os.environ.get(
    "WAN22_FOLEY_PROMPT",
    "sound effects matching the video, ambient Foley, no music, no speech",
)
FOLEY_NEG_PROMPT = os.environ.get("WAN22_FOLEY_NEG_PROMPT", "noisy, harsh, music, speech")
FOLEY_STEPS = int(os.environ.get("WAN22_FOLEY_STEPS", "50"))
FOLEY_GUIDANCE = float(os.environ.get("WAN22_FOLEY_GUIDANCE", "4.5"))
FOLEY_TIMEOUT = int(os.environ.get("WAN22_FOLEY_TIMEOUT", "180"))

DEFAULT_PROMPT = os.environ.get(
    "WAN22_DEFAULT_PROMPT",
    "make this image come alive, cinematic motion, smooth animation",
)
NEGATIVE_PROMPT = os.environ.get(
    "WAN22_NEGATIVE",
    "色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 整体发灰, 最差质量, 低质量, "
    "JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, "
    "畸形的, 毁容的, 形态畸形的肢体, 手指融合, 杂乱的背景, 三条腿",
)

ROOT.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
