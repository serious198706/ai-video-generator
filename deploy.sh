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

VENV_DIR="${WAN22_VENV_DIR:-/opt/wan22-venv}"
FOLEY_VENV_DIR="${WAN22_FOLEY_VENV_DIR:-/opt/foley-venv}"
FOLEY_REPO="${WAN22_FOLEY_REPO:-/opt/HunyuanVideo-Foley}"
FOLEY_SKIP="${WAN22_FOLEY_SKIP:-0}"

export HF_HOME="${HF_HOME:-/data/hf-cache}"
export MODEL_ROOT="${MODEL_ROOT:-/data/models/wan22}"
export WAN22_MODEL_DIR="${WAN22_MODEL_DIR:-$MODEL_ROOT/base/WAMU_v3_WAN2.2_I2V_LIGHTNING}"
export WAN22_LORA_DIR="${WAN22_LORA_DIR:-$MODEL_ROOT/loras}"
export WAN22_FOLEY_MODEL_DIR="${WAN22_FOLEY_MODEL_DIR:-/data/models/hunyuanvideo-foley}"

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

if [[ "$FOLEY_SKIP" == "1" ]]; then
  echo "[wan22] WAN22_FOLEY_SKIP=1, skip Foley"
else
  if ! command -v git >/dev/null 2>&1; then
    echo "[wan22] Foley 需要 git，请先安装" >&2
    exit 1
  fi

  if [[ -d "$FOLEY_REPO/.git" ]]; then
    echo "[wan22] updating HunyuanVideo-Foley: $FOLEY_REPO"
    git -C "$FOLEY_REPO" pull --ff-only || \
      echo "[wan22] Foley repo not fast-forward, keep local clone" >&2
  elif [[ -d "$FOLEY_REPO/hunyuanvideo_foley" ]]; then
    echo "[wan22] Foley repo already present: $FOLEY_REPO"
  else
    echo "[wan22] cloning HunyuanVideo-Foley: $FOLEY_REPO"
    git clone --depth 1 \
      https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley \
      "$FOLEY_REPO"
  fi

  if [[ ! -f "$FOLEY_VENV_DIR/bin/python" ]]; then
    echo "[wan22] creating Foley virtual environment: $FOLEY_VENV_DIR"
    python3 -m venv "$FOLEY_VENV_DIR"
  fi

  FOLEY_PY="$FOLEY_VENV_DIR/bin/python"
  FOLEY_CONSTRAINTS="$SCRIPT_DIR/foley-constraints.txt"
  echo "[wan22] Foley pip using $FOLEY_PY (do not rebuild venv)"
  echo "[wan22] Foley: binary-only pillow/numpy via $FOLEY_CONSTRAINTS"
  "$FOLEY_PY" -m pip install --upgrade pip setuptools wheel
  "$FOLEY_PY" -m pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
  "$FOLEY_PY" -m pip install --upgrade \
    --only-binary=pillow,numpy \
    -c "$FOLEY_CONSTRAINTS" \
    'pillow>=11.3' 'numpy>=2.2'
  FOLEY_REQ="$(mktemp)"
  WAN22_FOLEY_REQ_IN="$FOLEY_REPO/requirements.txt" WAN22_FOLEY_REQ_OUT="$FOLEY_REQ" \
    "$FOLEY_PY" - <<'PY'
from pathlib import Path
import os
src = Path(os.environ["WAN22_FOLEY_REQ_IN"]).read_text().splitlines()
skip = (
    "numpy",
    "pillow",
    "gradio",  # 3.50 锁 numpy~=1.0，sidecar 不用
    "black",
    "isort",
    "flake8",
    "mypy",
    "pre-commit",
    "pandas",
    "pyarrow",
)
out = []
for line in src:
    raw = line.strip().lower().split("==")[0].split(">=")[0].split("~=")[0].strip()
    if any(raw == name or raw.startswith(name + "[") for name in skip):
        continue
    out.append(line)
Path(os.environ["WAN22_FOLEY_REQ_OUT"]).write_text("\n".join(out) + "\n")
PY
  echo "[wan22] Foley filtered requirements (no pillow/numpy/gradio/dev):"
  cat "$FOLEY_REQ"
  # descript-audiotools 会再拉 pillow；不加 only-binary 就会源码编 jpeg 失败。
  "$FOLEY_PY" -m pip install \
    --upgrade-strategy only-if-needed \
    --only-binary=pillow,numpy \
    -c "$FOLEY_CONSTRAINTS" \
    -r "$FOLEY_REQ"
  rm -f "$FOLEY_REQ"
  "$FOLEY_PY" -m pip install --upgrade \
    --only-binary=pillow,numpy \
    -c "$FOLEY_CONSTRAINTS" \
    'pillow>=11.3' 'numpy>=2.2'
  "$FOLEY_PY" -m pip install --upgrade torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
  "$FOLEY_PY" -m pip install --no-deps -e "$FOLEY_REPO"

  mkdir -p "$WAN22_FOLEY_MODEL_DIR"
  echo "[wan22] downloading HunyuanVideo-Foley weights"
  hf download tencent/HunyuanVideo-Foley \
    --local-dir "$WAN22_FOLEY_MODEL_DIR"
fi

echo "[wan22] deployment complete; run: $SCRIPT_DIR/start.sh"
