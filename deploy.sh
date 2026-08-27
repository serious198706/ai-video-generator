#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${WAN22_VENV_DIR:-/opt/wan22-venv}"

export HF_HOME="${HF_HOME:-/data/hf-cache}"
export MODEL_ROOT="${MODEL_ROOT:-/data/models/wan22}"
export WAN22_MODEL_DIR="${WAN22_MODEL_DIR:-$MODEL_ROOT/base/WAMU_v3_WAN2.2_I2V_LIGHTNING}"
export WAN22_LORA_DIR="${WAN22_LORA_DIR:-$MODEL_ROOT/loras}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[wan22] missing command: python3" >&2
  exit 1
fi

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "[wan22] creating virtual environment: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[wan22] installing Python dependencies"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --upgrade -r "$SCRIPT_DIR/requirements.txt"

if ! command -v hf >/dev/null 2>&1; then
  echo "[wan22] requirements 安装后仍找不到 hf 命令" >&2
  exit 1
fi

mkdir -p "$WAN22_MODEL_DIR" "$WAN22_LORA_DIR/nsfw" "$HF_HOME"

echo "[wan22] downloading WAMU v3 Lightning base model"
hf download thornmaze/WAMU_v3_WAN2.2_I2V_LIGHTNING \
  --local-dir "$WAN22_MODEL_DIR"

echo "[wan22] downloading General NSFW Booster"
hf download lopi999/Wan2.2-I2V_General-NSFW-LoRA \
  NSFW-22-H-e8.safetensors \
  NSFW-22-L-e8.safetensors \
  --revision aeef17d7fa51 \
  --local-dir "$WAN22_LORA_DIR/nsfw"

echo "[wan22] deployment complete; run: $SCRIPT_DIR/start.sh"
