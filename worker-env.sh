#!/usr/bin/env bash
# 路径全部落在本仓库（server/）。deploy.sh / start.sh 在 source .env 之后再 source。
# 已在环境里的变量不会被覆盖。不假设 AutoDL / EC2 / vast.ai 的盘符或 conda 路径。

if [[ -z "${SCRIPT_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

_resolve_python() {
  local candidate
  if [[ -n "${WAN22_PYTHON:-}" ]]; then
    if [[ -x "$WAN22_PYTHON" ]]; then
      echo "$WAN22_PYTHON"
      return 0
    fi
    if command -v "$WAN22_PYTHON" >/dev/null 2>&1; then
      command -v "$WAN22_PYTHON"
      return 0
    fi
    echo "[wan22] WAN22_PYTHON=$WAN22_PYTHON 不存在，改从 PATH 找 python3/python" >&2
  fi
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}

if RESOLVED_PY="$(_resolve_python)"; then
  WAN22_PYTHON="$RESOLVED_PY"
else
  WAN22_PYTHON=""
fi
unset RESOLVED_PY

: "${WAN22_DATA_ROOT:=$SCRIPT_DIR}"
: "${WAN22_VENV_DIR:=$WAN22_DATA_ROOT/.venv}"
: "${WAN22_FOLEY_VENV_DIR:=$WAN22_DATA_ROOT/.venv-foley}"
: "${WAN22_FOLEY_REPO:=$WAN22_DATA_ROOT/HunyuanVideo-Foley}"
: "${WAN22_UPSCALE_VENV_DIR:=$WAN22_DATA_ROOT/.venv-upscale}"
: "${WAN22_CACHE_DIR:=$WAN22_DATA_ROOT/.cache}"
: "${HF_HOME:=$WAN22_CACHE_DIR/hf}"
: "${HF_HUB_CACHE:=$HF_HOME}"
: "${HUGGINGFACE_HUB_CACHE:=$HF_HOME}"
: "${XDG_CACHE_HOME:=$WAN22_CACHE_DIR/xdg}"
: "${PIP_CACHE_DIR:=$WAN22_CACHE_DIR/pip}"
: "${TMPDIR:=$WAN22_CACHE_DIR/tmp}"
: "${TMP:=$TMPDIR}"
: "${TEMP:=$TMPDIR}"
: "${MODEL_ROOT:=$WAN22_DATA_ROOT/models}"
: "${WAN22_DATA_DIR:=$WAN22_DATA_ROOT/data}"
: "${WAN22_LOG_DIR:=$WAN22_DATA_ROOT/logs}"
: "${WAN22_HOST:=0.0.0.0}"
: "${WAN22_PORT:=8000}"

: "${WAN22_MODEL_DIR:=$MODEL_ROOT/base/WAMU_v3_WAN2.2_I2V_LIGHTNING}"
: "${WAN22_LORA_DIR:=$MODEL_ROOT/loras}"
: "${WAN22_NSFW_HIGH:=$WAN22_LORA_DIR/nsfw/NSFW-22-H-e8.safetensors}"
: "${WAN22_NSFW_LOW:=$WAN22_LORA_DIR/nsfw/NSFW-22-L-e8.safetensors}"
: "${WAN22_FOLEY_PYTHON:=$WAN22_FOLEY_VENV_DIR/bin/python}"
: "${WAN22_FOLEY_MODEL_DIR:=$MODEL_ROOT/hunyuanvideo-foley}"
: "${WAN22_FOLEY_SIZE:=xl}"
: "${WAN22_FOLEY_PROMPT:=intimate erotic Foley matching the video, soft sensual ambient music, sultry atmosphere, breathy room tone, no speech, no lyrics}"
: "${WAN22_FOLEY_NEG_PROMPT:=noisy, harsh, speech, lyrics, shouting}"
: "${WAN22_FOLEY_STEPS:=50}"
: "${WAN22_FOLEY_GUIDANCE:=4.5}"
: "${WAN22_FOLEY_TIMEOUT:=180}"
: "${WAN22_FOLEY_REQUIRED:=0}"
: "${WAN22_UPSCALE_PYTHON:=$WAN22_UPSCALE_VENV_DIR/bin/python}"
: "${WAN22_UPSCALE_MODEL_DIR:=$MODEL_ROOT/realesrgan}"
: "${WAN22_UPSCALE_MODEL:=realesr-general-x4v3.pth}"
: "${WAN22_UPSCALE_SCALE:=1.5}"
: "${WAN22_UPSCALE_TIMEOUT:=300}"

export WAN22_DATA_ROOT WAN22_PYTHON WAN22_VENV_DIR
export WAN22_FOLEY_VENV_DIR WAN22_FOLEY_REPO WAN22_FOLEY_PYTHON WAN22_FOLEY_MODEL_DIR
export WAN22_FOLEY_SIZE WAN22_FOLEY_PROMPT WAN22_FOLEY_NEG_PROMPT
export WAN22_FOLEY_STEPS WAN22_FOLEY_GUIDANCE WAN22_FOLEY_TIMEOUT WAN22_FOLEY_REQUIRED
export WAN22_UPSCALE_VENV_DIR WAN22_UPSCALE_PYTHON WAN22_UPSCALE_MODEL_DIR
export WAN22_UPSCALE_MODEL WAN22_UPSCALE_SCALE WAN22_UPSCALE_TIMEOUT
export WAN22_CACHE_DIR WAN22_GHFAST
export HF_HOME HF_HUB_CACHE HUGGINGFACE_HUB_CACHE HF_ENDPOINT
export XDG_CACHE_HOME PIP_CACHE_DIR TMPDIR TMP TEMP
export MODEL_ROOT WAN22_MODEL_DIR WAN22_LORA_DIR WAN22_NSFW_HIGH WAN22_NSFW_LOW
export WAN22_DATA_DIR WAN22_LOG_DIR WAN22_INSTALL_TORCH WAN22_VENV_SYSTEM_SITE
export WAN22_HOST WAN22_PORT

if [[ -n "${WAN22_GITHUB_MIRROR:-}" ]]; then
  export WAN22_GITHUB_MIRROR
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="url.${WAN22_GITHUB_MIRROR}.insteadof"
  export GIT_CONFIG_VALUE_0="https://github.com/"
fi
if [[ -n "${PIP_INDEX_URL:-}" ]]; then
  export PIP_INDEX_URL
  export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-}"
fi
