#!/usr/bin/env bash
# GPU 部署：在本目录建 venv、下权重。不写死 AutoDL / EC2 / vast.ai 路径。
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

FOLEY_SKIP="${WAN22_FOLEY_SKIP:-1}"
UPSCALE_SKIP="${WAN22_UPSCALE_SKIP:-0}"

_python() {
  if [[ -n "${WAN22_PYTHON:-}" ]]; then
    echo "$WAN22_PYTHON"
    return
  fi
  echo "[wan22] 找不到 python3 / python。" >&2
  echo "[wan22] 先装 Python，或在 .env 里设 WAN22_PYTHON=/绝对路径/python" >&2
  echo "[wan22] PATH=$PATH" >&2
  exit 1
}

_fetch() {
  local dest="$1"
  shift
  if [[ -f "$dest" ]]; then
    echo "[wan22] have $(basename "$dest")"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  local url
  for url in "$@"; do
    echo "[wan22] download $url"
    if curl -L --fail --retry 3 --retry-delay 2 -o "$dest.tmp" "$url"; then
      mv "$dest.tmp" "$dest"
      return 0
    fi
    rm -f "$dest.tmp"
  done
  echo "[wan22] failed: $(basename "$dest")" >&2
  return 1
}

_upscale_venv() {
  if [[ ! -f "$WAN22_UPSCALE_VENV_DIR/bin/python" ]]; then
    echo "[wan22] creating upscale virtual environment: $WAN22_UPSCALE_VENV_DIR"
    upscale_venv_args=()
    if [[ "${WAN22_VENV_SYSTEM_SITE}" == "1" ]]; then
      upscale_venv_args+=(--system-site-packages)
    fi
    "$BASE_PY" -m venv "${upscale_venv_args[@]}" "$WAN22_UPSCALE_VENV_DIR"
  fi
}

BASE_PY="$(_python)"

if [[ -z "${WAN22_INSTALL_TORCH:-}" ]]; then
  if "$BASE_PY" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
    WAN22_INSTALL_TORCH=0
  else
    WAN22_INSTALL_TORCH=1
  fi
fi
if [[ -z "${WAN22_VENV_SYSTEM_SITE:-}" ]]; then
  if [[ "$WAN22_INSTALL_TORCH" == "0" ]]; then
    WAN22_VENV_SYSTEM_SITE=1
  else
    WAN22_VENV_SYSTEM_SITE=0
  fi
fi
export WAN22_INSTALL_TORCH WAN22_VENV_SYSTEM_SITE

echo "[wan22] root=$SCRIPT_DIR python=$BASE_PY"
echo "[wan22] venv=$WAN22_VENV_DIR system_site=$WAN22_VENV_SYSTEM_SITE install_torch=$WAN22_INSTALL_TORCH"
echo "[wan22] models=$WAN22_MODEL_DIR"
echo "[wan22] hf_home=$HF_HOME endpoint=${HF_ENDPOINT:-https://huggingface.co}"
echo "[wan22] github_mirror=${WAN22_GITHUB_MIRROR:-https://github.com/} pip_index=${PIP_INDEX_URL:-pypi.org}"

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME" "$WAN22_DATA_DIR" "$WAN22_LOG_DIR"

for command_name in git curl; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "[wan22] missing command: $command_name" >&2
    exit 1
  fi
done

if [[ ! -f "$WAN22_VENV_DIR/bin/activate" ]]; then
  echo "[wan22] creating virtual environment: $WAN22_VENV_DIR"
  mkdir -p "$(dirname "$WAN22_VENV_DIR")"
  venv_args=()
  if [[ "${WAN22_VENV_SYSTEM_SITE}" == "1" ]]; then
    venv_args+=(--system-site-packages)
  fi
  "$BASE_PY" -m venv "${venv_args[@]}" "$WAN22_VENV_DIR"
fi

# shellcheck disable=SC1091
source "$WAN22_VENV_DIR/bin/activate"

echo "[wan22] installing Python dependencies"
python -m pip install --upgrade pip setuptools wheel

if [[ "${WAN22_INSTALL_TORCH}" == "1" ]]; then
  echo "[wan22] installing torch from cu128"
  python -m pip install --upgrade torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128
else
  echo "[wan22] WAN22_INSTALL_TORCH=0, keep image torch"
fi

python -c "import torch; assert torch.cuda.is_available(), 'torch.cuda.is_available() is false'; print('[wan22] torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

python -m pip install --upgrade -r "$SCRIPT_DIR/requirements.txt"

if ! command -v hf >/dev/null 2>&1; then
  echo "[wan22] requirements 安装后仍找不到 hf 命令" >&2
  exit 1
fi

mkdir -p "$WAN22_MODEL_DIR" "$WAN22_LORA_DIR/nsfw" "$HF_HOME" "$WAN22_DATA_DIR" "$WAN22_LOG_DIR"

HF_TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_TOKEN_ARGS=(--token "$HF_TOKEN")
fi

echo "[wan22] downloading WAMU v3 Lightning base model"
hf download "${HF_TOKEN_ARGS[@]}" thornmaze/WAMU_v3_WAN2.2_I2V_LIGHTNING \
  --local-dir "$WAN22_MODEL_DIR"

echo "[wan22] downloading General NSFW Booster"
hf download "${HF_TOKEN_ARGS[@]}" lopi999/Wan2.2-I2V_General-NSFW-LoRA \
  NSFW-22-H-e8.safetensors \
  NSFW-22-L-e8.safetensors \
  --revision aeef17d7fa51 \
  --local-dir "$WAN22_LORA_DIR/nsfw"

if [[ "$FOLEY_SKIP" == "1" ]]; then
  echo "[wan22] WAN22_FOLEY_SKIP=1, skip Foley"
else
  if [[ ! -d "$WAN22_FOLEY_REPO/.git" && ! -d "$WAN22_FOLEY_REPO/hunyuanvideo_foley" ]]; then
    echo "[wan22] cloning HunyuanVideo-Foley: $WAN22_FOLEY_REPO"
    git clone --depth 1 \
      https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley.git \
      "$WAN22_FOLEY_REPO"
  elif [[ -d "$WAN22_FOLEY_REPO/.git" ]]; then
    echo "[wan22] updating HunyuanVideo-Foley: $WAN22_FOLEY_REPO"
    git -C "$WAN22_FOLEY_REPO" pull --ff-only || \
      echo "[wan22] Foley repo not fast-forward, keep local clone" >&2
  else
    echo "[wan22] Foley repo already present: $WAN22_FOLEY_REPO"
  fi

  if [[ ! -f "$WAN22_FOLEY_VENV_DIR/bin/python" ]]; then
    echo "[wan22] creating Foley virtual environment: $WAN22_FOLEY_VENV_DIR"
    foley_venv_args=()
    if [[ "${WAN22_VENV_SYSTEM_SITE}" == "1" ]]; then
      foley_venv_args+=(--system-site-packages)
    fi
    "$BASE_PY" -m venv "${foley_venv_args[@]}" "$WAN22_FOLEY_VENV_DIR"
  fi

  FOLEY_PY="$WAN22_FOLEY_PYTHON"
  FOLEY_CONSTRAINTS="$SCRIPT_DIR/foley-constraints.txt"
  echo "[wan22] Foley pip using $FOLEY_PY"
  "$FOLEY_PY" -m pip install --upgrade pip setuptools wheel
  if [[ "${WAN22_INSTALL_TORCH}" == "1" ]]; then
    "$FOLEY_PY" -m pip install --upgrade torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu128
  fi
  "$FOLEY_PY" -m pip install --upgrade \
    --only-binary=pillow,numpy \
    -c "$FOLEY_CONSTRAINTS" \
    'pillow>=11.3' 'numpy>=2.2'
  FOLEY_REQ="$(mktemp)"
  WAN22_FOLEY_REQ_IN="$WAN22_FOLEY_REPO/requirements.txt" WAN22_FOLEY_REQ_OUT="$FOLEY_REQ" \
    "$FOLEY_PY" - <<'PY'
from pathlib import Path
import os
src = Path(os.environ["WAN22_FOLEY_REQ_IN"]).read_text().splitlines()
skip = (
    "numpy",
    "pillow",
    "gradio",
    "tensorboard",
    "tb-nightly",
    "tensorboard-data-server",
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
  "$FOLEY_PY" -m pip uninstall -y tensorboard tensorboard-data-server tb-nightly || true
  "$FOLEY_PY" -m pip install --upgrade 'protobuf>=5.29'
  if [[ "${WAN22_INSTALL_TORCH}" == "1" ]]; then
    "$FOLEY_PY" -m pip install --upgrade torch torchvision torchaudio \
      --index-url https://download.pytorch.org/whl/cu128
  fi
  "$FOLEY_PY" -m pip install --no-deps -e "$WAN22_FOLEY_REPO"

  mkdir -p "$WAN22_FOLEY_MODEL_DIR"
  echo "[wan22] downloading HunyuanVideo-Foley weights"
  hf download "${HF_TOKEN_ARGS[@]}" tencent/HunyuanVideo-Foley \
    --local-dir "$WAN22_FOLEY_MODEL_DIR"
fi

if [[ "$UPSCALE_SKIP" == "1" ]]; then
  echo "[wan22] WAN22_UPSCALE_SKIP=1, skip upscale"
else
  _upscale_venv
  UPSCALE_PY="$WAN22_UPSCALE_PYTHON"
  echo "[wan22] compact pip using $UPSCALE_PY"
  "$UPSCALE_PY" -m pip install --upgrade pip setuptools wheel
  if [[ "${WAN22_INSTALL_TORCH}" == "1" ]]; then
    "$UPSCALE_PY" -m pip install --upgrade torch torchvision \
      --index-url https://download.pytorch.org/whl/cu128
  fi
  "$UPSCALE_PY" -m pip install --upgrade \
    'numpy>=1.26.0' \
    'opencv-python-headless>=4.10.0' \
    'pillow>=10.0.0' \
    'imageio-ffmpeg>=0.6.0' \
    'spandrel>=0.4.0' \
    'spandrel_extra_arches>=0.2.0'
  mkdir -p "$WAN22_UPSCALE_MODEL_DIR"
  UPSCALE_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
  echo "[wan22] downloading compact realesr-general-x4v3"
  if [[ -n "${WAN22_GHFAST:-}" ]]; then
    _fetch "$WAN22_UPSCALE_MODEL_DIR/$WAN22_UPSCALE_MODEL" \
      "${WAN22_GHFAST}${UPSCALE_URL}" \
      "$UPSCALE_URL"
  else
    _fetch "$WAN22_UPSCALE_MODEL_DIR/$WAN22_UPSCALE_MODEL" "$UPSCALE_URL"
  fi
fi

echo "[wan22] deployment complete; run: $SCRIPT_DIR/start.sh"
