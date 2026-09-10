#!/usr/bin/env bash
# GPU worker 部署：venv + 权重。AutoDL 不覆盖镜像 Torch；EC2 仍装 cu128。
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

if [[ -z "${WAN22_FOLEY_SKIP:-}" ]]; then
  if [[ "$WAN22_LAYOUT" == "autodl" ]]; then
    FOLEY_SKIP=1
  else
    FOLEY_SKIP=0
  fi
else
  FOLEY_SKIP="$WAN22_FOLEY_SKIP"
fi
UPSCALE_SKIP="${WAN22_UPSCALE_SKIP:-0}"

_python() {
  if [[ -x "$WAN22_PYTHON" ]]; then
    echo "$WAN22_PYTHON"
    return
  fi
  if command -v "$WAN22_PYTHON" >/dev/null 2>&1; then
    command -v "$WAN22_PYTHON"
    return
  fi
  echo "[wan22] 找不到 Python: $WAN22_PYTHON" >&2
  exit 1
}

BASE_PY="$(_python)"
echo "[wan22] layout=$WAN22_LAYOUT python=$BASE_PY"
echo "[wan22] venv=$WAN22_VENV_DIR"
echo "[wan22] models=$WAN22_MODEL_DIR"
echo "[wan22] hf_home=$HF_HOME endpoint=${HF_ENDPOINT:-https://huggingface.co}"
echo "[wan22] github_mirror=${WAN22_GITHUB_MIRROR:-https://github.com/} pip_index=${PIP_INDEX_URL:-pypi.org}"

if [[ "$WAN22_LAYOUT" == "autodl" ]]; then
  if [[ ! -d /root/autodl-tmp ]]; then
    echo "[wan22] WAN22_LAYOUT=autodl 但没有 /root/autodl-tmp" >&2
    exit 1
  fi
  autodl_src="$(df -P /root/autodl-tmp | awk 'NR==2 { print $1 }')"
  root_src="$(df -P / | awk 'NR==2 { print $1 }')"
  if [[ -z "$autodl_src" || "$autodl_src" == "$root_src" ]]; then
    echo "[wan22] /root/autodl-tmp 还在系统盘 overlay 上，下载会把 30G 写满。" >&2
    echo "[wan22] 打开 AutoDL 控制台给实例加数据盘，确认: df -h /root/autodl-tmp 不是 30G overlay。" >&2
    df -h / /root/autodl-tmp >&2 || true
    exit 1
  fi
  case "$WAN22_MODEL_DIR" in
    /root/autodl-tmp/*) ;;
    *)
      echo "[wan22] AutoDL 系统盘只有 30G，WAN22_MODEL_DIR 必须在 /root/autodl-tmp 下: $WAN22_MODEL_DIR" >&2
      exit 1
      ;;
  esac
  mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$XDG_CACHE_HOME"
fi

for command_name in git; do
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

echo "[wan22] downloading WAMU v3 Lightning base model"
hf download --token $HF_TOKEN thornmaze/WAMU_v3_WAN2.2_I2V_LIGHTNING \
  --local-dir "$WAN22_MODEL_DIR"

echo "[wan22] downloading General NSFW Booster"
hf download --token $HF_TOKEN lopi999/Wan2.2-I2V_General-NSFW-LoRA \
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
      https://gitee.com/cy198706/HunyuanVideo-Foley.git \
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
  hf download --token $HF_TOKEN tencent/HunyuanVideo-Foley \
    --local-dir "$WAN22_FOLEY_MODEL_DIR"
fi

if [[ "$UPSCALE_SKIP" == "1" ]]; then
  echo "[wan22] WAN22_UPSCALE_SKIP=1, skip SeedVR2"
else
  if [[ -z "${WAN22_UPSCALE_REPO:-}" ]]; then
    echo "[wan22] WAN22_UPSCALE_REPO is empty" >&2
    exit 1
  fi
  if [[ ! -f "$WAN22_UPSCALE_REPO/inference_cli.py" ]]; then
    echo "[wan22] cloning SeedVR2: $WAN22_UPSCALE_REPO"
    git clone --depth 1 \
      https://gitee.com/cy198706/ComfyUI-SeedVR2_VideoUpscaler.git \
      "$WAN22_UPSCALE_REPO"
  elif [[ -d "$WAN22_UPSCALE_REPO/.git" ]]; then
    echo "[wan22] updating SeedVR2: $WAN22_UPSCALE_REPO"
    git -C "$WAN22_UPSCALE_REPO" pull --ff-only || \
      echo "[wan22] SeedVR2 repo not fast-forward, keep local clone" >&2
  else
    echo "[wan22] SeedVR2 repo already present: $WAN22_UPSCALE_REPO"
  fi

  if [[ ! -f "$WAN22_UPSCALE_VENV_DIR/bin/python" ]]; then
    echo "[wan22] creating SeedVR2 virtual environment: $WAN22_UPSCALE_VENV_DIR"
    upscale_venv_args=()
    if [[ "${WAN22_VENV_SYSTEM_SITE}" == "1" ]]; then
      upscale_venv_args+=(--system-site-packages)
    fi
    "$BASE_PY" -m venv "${upscale_venv_args[@]}" "$WAN22_UPSCALE_VENV_DIR"
  fi

  UPSCALE_PY="$WAN22_UPSCALE_PYTHON"
  echo "[wan22] SeedVR2 pip using $UPSCALE_PY"
  "$UPSCALE_PY" -m pip install --upgrade pip setuptools wheel
  if [[ "${WAN22_INSTALL_TORCH}" == "1" ]]; then
    "$UPSCALE_PY" -m pip install --upgrade torch torchvision \
      --index-url https://download.pytorch.org/whl/cu128
  fi
  "$UPSCALE_PY" -m pip install --upgrade -r "$WAN22_UPSCALE_REPO/requirements.txt"
  "$UPSCALE_PY" -m pip install --upgrade 'imageio-ffmpeg>=0.6.0'
  mkdir -p "$WAN22_UPSCALE_MODEL_DIR"
  echo "[wan22] downloading SeedVR2 3B FP8 + VAE"
  hf download --token $HF_TOKEN numz/SeedVR2_comfyUI \
    seedvr2_ema_3b_fp8_e4m3fn.safetensors \
    ema_vae_fp16.safetensors \
    --local-dir "$WAN22_UPSCALE_MODEL_DIR"
fi

echo "[wan22] deployment complete; run: $SCRIPT_DIR/start.sh"
