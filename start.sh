#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/worker-env.sh"

if [[ -z "${WAN22_FOLEY_ENABLE:-}" ]]; then
  if [[ -x "$WAN22_FOLEY_PYTHON" ]]; then
    WAN22_FOLEY_ENABLE=1
  else
    WAN22_FOLEY_ENABLE=0
  fi
fi
export WAN22_FOLEY_ENABLE

if [[ -z "${WAN22_UPSCALE_ENABLE:-}" ]]; then
  if [[ -x "$WAN22_UPSCALE_PYTHON" ]]; then
    WAN22_UPSCALE_ENABLE=1
  else
    WAN22_UPSCALE_ENABLE=0
  fi
fi
export WAN22_UPSCALE_ENABLE

if [[ ! -f "$WAN22_VENV_DIR/bin/activate" ]]; then
  echo "[wan22] 虚拟环境不存在，请先运行 ./deploy.sh" >&2
  exit 1
fi
if [[ "$WAN22_FOLEY_ENABLE" == "1" && ! -x "$WAN22_FOLEY_PYTHON" ]]; then
  echo "[wan22] Foley 已打开但 $WAN22_FOLEY_PYTHON 不存在，请先运行 ./deploy.sh" >&2
  exit 1
fi
if [[ "$WAN22_UPSCALE_ENABLE" == "1" && ! -x "$WAN22_UPSCALE_PYTHON" ]]; then
  echo "[wan22] 超分已打开但 $WAN22_UPSCALE_PYTHON 不存在，请先运行 ./deploy.sh" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$WAN22_VENV_DIR/bin/activate"

for command_name in python uvicorn; do
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

echo "[wan22] layout=$WAN22_LAYOUT host=$WAN22_HOST:$WAN22_PORT"
echo "[wan22] torch=$(python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-cuda")')"
echo "[wan22] model=$WAN22_MODEL_DIR"
echo "[wan22] foley=$WAN22_FOLEY_ENABLE"
echo "[wan22] upscale=$WAN22_UPSCALE_ENABLE"

if [[ "${WAN22_DRY_RUN:-0}" != "1" ]]; then
  echo "[wan22] running preflight"
  python -c "from wan22.infer.generate import preflight; preflight(); print('[wan22] preflight OK')"
fi

echo "[wan22] starting GPU API"
exec uvicorn wan22.api.app:app \
  --host "${WAN22_HOST}" \
  --port "${WAN22_PORT}"
