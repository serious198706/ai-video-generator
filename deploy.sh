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

# SageAttention：Wan 官方推荐的 SDPA 替换。2++/3 都不发 PyPI（PyPI 只有无效的 1.0.x），
# 一律源码编译。sa3 需要 python>=3.13；3.12 只装 sa2++（sm80+ 全兼容）。
# 编译失败不影响启动，运行时自动回退 SDPA。
if [[ "${WAN22_SAGE_SKIP:-0}" != "1" ]]; then
  SAGE_SRC="$SCRIPT_DIR/.cache/SageAttention"
  SAGE_GIT_URL="https://github.com/thu-ml/SageAttention.git"
  if [[ -n "${WAN22_GITHUB_MIRROR:-}" ]]; then
    SAGE_GIT_URL="${WAN22_GITHUB_MIRROR}thu-ml/SageAttention.git"
  fi
  if [[ ! -d "$SAGE_SRC" ]]; then
    git clone --depth 1 "$SAGE_GIT_URL" "$SAGE_SRC" || true
  fi
  if [[ -d "$SAGE_SRC" ]]; then
    # torch 是 cu128 编译的，系统 nvcc 可能是别的版本（如 13.2）；cpp_extension 会做
    # 版本一致性检查并拒绝编译。必须用和 torch 匹配的 toolkit。
    # 注意：vast/cuda 镜像常全局 export CUDA_HOME=/usr/local/cuda（跟着系统走），
    # 环境里的 CUDA_HOME 不能盲信，先验版本，不匹配就当没设。
    TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda)')"
    _match_toolkit() {
      local detected
      [[ -x "$1/bin/nvcc" ]] || return 1
      detected="$("$1/bin/nvcc" --version 2>/dev/null)" || return 1
      detected="$(printf '%s\n' "$detected" | sed -nE 's/.*release ([0-9]+\.[0-9]+),.*/\1/p')"
      [[ "$detected" == "$TORCH_CUDA" ]]
    }
    SAGE_CUDA_HOME=""
    if [[ -n "${CUDA_HOME:-}" ]] && _match_toolkit "$CUDA_HOME"; then
      SAGE_CUDA_HOME="$CUDA_HOME"
    else
      echo "[wan22] CUDA_HOME=${CUDA_HOME:-<unset>} does not match torch cu$TORCH_CUDA, ignored" >&2
    fi
    if [[ -z "$SAGE_CUDA_HOME" ]]; then
      for cand in "${CUDA_PATH:-}" /usr/local/cuda-"$TORCH_CUDA" /usr/local/cuda; do
        [[ -n "$cand" ]] || continue
        if _match_toolkit "$cand"; then
          SAGE_CUDA_HOME="$cand"
          break
        fi
      done
    fi
    if [[ -z "$SAGE_CUDA_HOME" && "$(id -u)" == "0" && "$TORCH_CUDA" == "12.8" \
          && ! -x "/usr/local/cuda-12.8/bin/nvcc" ]]; then
      # runfile --toolkit 只写 /usr/local/cuda-12.8，不碰驱动；但它会把
      # /usr/local/cuda 软链改指到 12.8，装完恢复原样，别影响系统里的 13.2。
      echo "[wan22] no matching toolkit; installing CUDA 12.8 toolkit (root, runfile, no driver)"
      OLD_LINK=""
      [[ -L /usr/local/cuda ]] && OLD_LINK="$(readlink /usr/local/cuda)"
      CUDA_RUN="$(mktemp /tmp/cuda-toolkit.XXXXXX.run)"
      if curl -L --fail --retry 3 -o "$CUDA_RUN" \
        "https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run"; then
        sh "$CUDA_RUN" --silent --toolkit --toolkitpath=/usr/local/cuda-12.8 --override --no-man-page \
          || echo "[wan22] cuda toolkit install failed" >&2
        if [[ -n "$OLD_LINK" ]]; then
          ln -sfn "$OLD_LINK" /usr/local/cuda
        fi
      else
        echo "[wan22] cuda runfile download failed" >&2
      fi
      rm -f "$CUDA_RUN"
    fi
    if [[ -z "$SAGE_CUDA_HOME" ]] && _match_toolkit "/usr/local/cuda-${TORCH_CUDA}"; then
      SAGE_CUDA_HOME="/usr/local/cuda-${TORCH_CUDA}"
    fi
    if [[ -n "$SAGE_CUDA_HOME" ]]; then
      echo "[wan22] building sageattention2++ with CUDA_HOME=$SAGE_CUDA_HOME (torch cu$TORCH_CUDA)"
      # 两套扩展共用同一编译环境，仅在子 shell 中覆盖，避免影响后面的部署步骤。
      (
        export CUDA_HOME="$SAGE_CUDA_HOME" CUDA_PATH="$SAGE_CUDA_HOME"
        export CUDACXX="$SAGE_CUDA_HOME/bin/nvcc" PYTORCH_NVCC="$SAGE_CUDA_HOME/bin/nvcc"
        export PATH="$SAGE_CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$SAGE_CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
        (cd "$SAGE_SRC" && python setup.py install) \
          || echo "[wan22] sageattn2 build failed; will fall back to sdpa" >&2
        if python -c "import sys; assert sys.version_info >= (3, 13)" 2>/dev/null; then
          echo "[wan22] python>=3.13: also building sageattn3 (blackwell fp4)"
          if [[ -d "$SAGE_SRC/sageattention3_blackwell" ]]; then
            (cd "$SAGE_SRC/sageattention3_blackwell" && python setup.py install) \
              || echo "[wan22] sageattn3 build failed; keep sa2++" >&2
          fi
        fi
      )
    else
      echo "[wan22] no toolkit matching torch cu$TORCH_CUDA; skip SageAttention build, use sdpa" >&2
      echo "[wan22] install CUDA Toolkit $TORCH_CUDA and rerun with CUDA_HOME=/path/to/cuda-$TORCH_CUDA" >&2
    fi
  else
    echo "[wan22] SageAttention clone failed; will fall back to sdpa" >&2
  fi
fi

# AWS 凭证：vast.ai 没有 IAM Role，SQS 接单和 S3 上传都靠专用 IAM 用户的 key。
# 交互式终端才会问；非交互（CI / 云-init）跳过，由环境变量直接注入。
# 已配过就不再问，轮换用 ./setup-aws.sh --force。WAN22_AWS_SETUP=0 可完全跳过。
if [[ "${WAN22_AWS_SETUP:-1}" == "1" ]]; then
  "$SCRIPT_DIR/setup-aws.sh" || {
    echo "[wan22] setup-aws.sh failed; fix it and rerun, or WAN22_AWS_SETUP=0 ./deploy.sh" >&2
    exit 1
  }
fi

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
