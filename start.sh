#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export HF_HOME="${HF_HOME:-/data/hf-cache}"
export MODEL_ROOT="${MODEL_ROOT:-/data/models/wan22}"
export WAN22_MODEL_DIR="${WAN22_MODEL_DIR:-$MODEL_ROOT/base/WAMU_v3_WAN2.2_I2V_LIGHTNING}"
export WAN22_LORA_DIR="${WAN22_LORA_DIR:-$MODEL_ROOT/loras}"
export WAN22_NSFW_HIGH="${WAN22_NSFW_HIGH:-$WAN22_LORA_DIR/nsfw/NSFW-22-H-e8.safetensors}"
export WAN22_NSFW_LOW="${WAN22_NSFW_LOW:-$WAN22_LORA_DIR/nsfw/NSFW-22-L-e8.safetensors}"

VENV_DIR="${WAN22_VENV_DIR:-/opt/wan22-venv}"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "[wan22] 虚拟环境不存在，请先运行 ./deploy.sh" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

for command_name in python3 uvicorn; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "[wan22] missing command: $command_name" >&2
    exit 1
  fi
done

export WAN22_QUANT="${WAN22_QUANT:-int8wo}"
export WAN22_TEXT_ENCODER_QUANT="${WAN22_TEXT_ENCODER_QUANT:-int8wo}"
export WAN22_OFFLOAD="${WAN22_OFFLOAD:-none}"
export WAN22_VAE_TILING="${WAN22_VAE_TILING:-1}"
export WAN22_STEPS="${WAN22_STEPS:-6}"
export WAN22_GUIDANCE_SCALE="${WAN22_GUIDANCE_SCALE:-1.0}"
export WAN22_GUIDANCE_SCALE_2="${WAN22_GUIDANCE_SCALE_2:-1.0}"
export WAN22_NSFW_HIGH_SCALE="${WAN22_NSFW_HIGH_SCALE:-1.0}"
export WAN22_NSFW_LOW_SCALE="${WAN22_NSFW_LOW_SCALE:-1.0}"
export WAN22_MAX_DIM="${WAN22_MAX_DIM:-832}"
export WAN22_MIN_DIM="${WAN22_MIN_DIM:-480}"
export WAN22_SQUARE_DIM="${WAN22_SQUARE_DIM:-640}"
export WAN22_FPS="${WAN22_FPS:-16}"
export WAN22_MAX_FRAMES="${WAN22_MAX_FRAMES:-321}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${WAN22_DRY_RUN:-0}" != "1" ]]; then
  echo "[wan22] running preflight"
  python3 -c "from wan22.infer.generate import preflight; preflight(); print('[wan22] preflight OK')"
fi

echo "[wan22] starting FastAPI service"
exec uvicorn wan22.api.app:app \
  --host "${WAN22_HOST:-0.0.0.0}" \
  --port "${WAN22_PORT:-8000}"
