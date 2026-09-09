#!/usr/bin/env bash
# GPU worker 路径。由 deploy.sh / start.sh 在 source .env 之后再 source。
# 已在环境里的变量不会被覆盖。
#
# AutoDL：系统盘 30G，权重必须在 /root/autodl-tmp；Python/Torch 用镜像 conda。
# 现网 EC2：/opt venv + /data 权重，pip 装 cu128。

if [[ -z "${WAN22_LAYOUT:-}" ]]; then
  if [[ -d /root/autodl-tmp ]]; then
    WAN22_LAYOUT=autodl
  else
    WAN22_LAYOUT=ec2
  fi
fi
export WAN22_LAYOUT

if [[ "$WAN22_LAYOUT" == "autodl" ]]; then
  : "${WAN22_DATA_ROOT:=/root/autodl-tmp/wan22}"
  : "${WAN22_PYTHON:=/root/miniconda3/bin/python}"
  : "${WAN22_VENV_DIR:=$WAN22_DATA_ROOT/venv}"
  : "${WAN22_FOLEY_VENV_DIR:=$WAN22_DATA_ROOT/foley-venv}"
  : "${WAN22_FOLEY_REPO:=$WAN22_DATA_ROOT/HunyuanVideo-Foley}"
  : "${HF_HOME:=$WAN22_DATA_ROOT/hf-cache}"
  : "${HF_ENDPOINT:=https://hf-mirror.com}"
  # git clone / pip git+https。挂了就换：https://ghfast.top/https://github.com/ 或 https://kkgithub.com/
  : "${WAN22_GITHUB_MIRROR:=https://gitclone.com/github.com/}"
  : "${PIP_INDEX_URL:=https://pypi.tuna.tsinghua.edu.cn/simple}"
  : "${PIP_TRUSTED_HOST:=pypi.tuna.tsinghua.edu.cn}"
  : "${MODEL_ROOT:=$WAN22_DATA_ROOT/models}"
  : "${WAN22_DATA_DIR:=$WAN22_DATA_ROOT/runtime}"
  : "${WAN22_LOG_DIR:=$WAN22_DATA_ROOT/logs}"
  : "${WAN22_INSTALL_TORCH:=0}"
  : "${WAN22_VENV_SYSTEM_SITE:=1}"
  : "${WAN22_FOLEY_MODEL_DIR:=$WAN22_DATA_ROOT/models/hunyuanvideo-foley}"
  : "${WAN22_HOST:=0.0.0.0}"
  : "${WAN22_PORT:=8000}"
else
  : "${WAN22_DATA_ROOT:=/data}"
  : "${WAN22_PYTHON:=python3}"
  : "${WAN22_VENV_DIR:=/opt/wan22-venv}"
  : "${WAN22_FOLEY_VENV_DIR:=/opt/foley-venv}"
  : "${WAN22_FOLEY_REPO:=/opt/HunyuanVideo-Foley}"
  : "${HF_HOME:=/data/hf-cache}"
  : "${MODEL_ROOT:=/data/models/wan22}"
  : "${WAN22_FOLEY_MODEL_DIR:=/data/models/hunyuanvideo-foley}"
  : "${WAN22_DATA_DIR:=$SCRIPT_DIR/data}"
  : "${WAN22_LOG_DIR:=$SCRIPT_DIR/logs}"
  : "${WAN22_INSTALL_TORCH:=1}"
  : "${WAN22_VENV_SYSTEM_SITE:=0}"
  : "${WAN22_HOST:=127.0.0.1}"
  : "${WAN22_PORT:=8000}"
fi

: "${WAN22_MODEL_DIR:=$MODEL_ROOT/base/WAMU_v3_WAN2.2_I2V_LIGHTNING}"
: "${WAN22_LORA_DIR:=$MODEL_ROOT/loras}"
: "${WAN22_NSFW_HIGH:=$WAN22_LORA_DIR/nsfw/NSFW-22-H-e8.safetensors}"
: "${WAN22_NSFW_LOW:=$WAN22_LORA_DIR/nsfw/NSFW-22-L-e8.safetensors}"
: "${WAN22_FOLEY_PYTHON:=$WAN22_FOLEY_VENV_DIR/bin/python}"
: "${WAN22_FOLEY_SIZE:=xl}"
: "${WAN22_FOLEY_PROMPT:=intimate erotic Foley matching the video, soft sensual ambient music, sultry atmosphere, breathy room tone, no speech, no lyrics}"
: "${WAN22_FOLEY_NEG_PROMPT:=noisy, harsh, speech, lyrics, shouting}"
: "${WAN22_FOLEY_STEPS:=50}"
: "${WAN22_FOLEY_GUIDANCE:=4.5}"
: "${WAN22_FOLEY_TIMEOUT:=180}"
: "${WAN22_FOLEY_REQUIRED:=0}"

export WAN22_DATA_ROOT WAN22_PYTHON WAN22_VENV_DIR
export WAN22_FOLEY_VENV_DIR WAN22_FOLEY_REPO WAN22_FOLEY_PYTHON WAN22_FOLEY_MODEL_DIR
export WAN22_FOLEY_SIZE WAN22_FOLEY_PROMPT WAN22_FOLEY_NEG_PROMPT
export WAN22_FOLEY_STEPS WAN22_FOLEY_GUIDANCE WAN22_FOLEY_TIMEOUT WAN22_FOLEY_REQUIRED
export HF_HOME HF_ENDPOINT MODEL_ROOT WAN22_MODEL_DIR WAN22_LORA_DIR WAN22_NSFW_HIGH WAN22_NSFW_LOW
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
