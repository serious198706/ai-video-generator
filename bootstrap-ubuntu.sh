#!/usr/bin/env bash
# 干净 Ubuntu 26.04（裸金属或未用 DLAMI 的机器）的系统层部署。
# PyTorch / 权重仍走 ./deploy.sh，不重复装。
#
#   sudo ./bootstrap-ubuntu.sh                 # 装 apt 依赖，GPU 已可用则接着 deploy.sh
#   sudo ./bootstrap-ubuntu.sh --install-driver  # 额外用 ubuntu-drivers 装 NVIDIA 驱动（装完通常要 reboot）
#   sudo ./bootstrap-ubuntu.sh --skip-deploy   # 只做系统包，不跑 deploy.sh
#
# 不必先上 EC2 翻配置。若要对齐现网版本，在 EC2 上先跑 ./inspect-host.sh。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

INSTALL_DRIVER=0
SKIP_DEPLOY=0
for arg in "$@"; do
  case "$arg" in
    --install-driver) INSTALL_DRIVER=1 ;;
    --skip-deploy) SKIP_DEPLOY=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "[wan22] unknown arg: $arg" >&2
      exit 1
      ;;
  esac
done

if [[ "$(id -u)" -ne 0 ]]; then
  echo "[wan22] 请用 root 跑：sudo $0 $*" >&2
  exit 1
fi

if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  echo "[wan22] os=${PRETTY_NAME:-unknown}"
  if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "[wan22] 只在 Ubuntu 上自动装包；当前是 ${ID:-unknown}" >&2
    exit 1
  fi
fi

export DEBIAN_FRONTEND=noninteractive
echo "[wan22] apt update + 基础包"
apt-get update -y
apt-get install -y --no-install-recommends \
  python3 \
  python3-venv \
  python3-dev \
  python3-pip \
  git \
  build-essential \
  pkg-config \
  ffmpeg \
  ca-certificates \
  curl

if [[ "$INSTALL_DRIVER" -eq 1 ]]; then
  echo "[wan22] installing NVIDIA driver via ubuntu-drivers"
  apt-get install -y ubuntu-drivers-common
  ubuntu-drivers autoinstall
  echo "[wan22] 驱动已提交安装。若 nvidia-smi 还不可用，reboot 后再跑一次本脚本（不要再加 --install-driver）。"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[wan22] 找不到 nvidia-smi。先装驱动再继续：" >&2
  echo "  sudo $0 --install-driver --skip-deploy && reboot" >&2
  echo "  sudo $0" >&2
  exit 1
fi
if ! nvidia-smi >/dev/null 2>&1; then
  echo "[wan22] nvidia-smi 失败。驱动可能还没加载，reboot 后再跑 sudo $0" >&2
  nvidia-smi >&2 || true
  exit 1
fi
echo "[wan22] gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n1)"

mkdir -p /opt /data/models /data/hf-cache
if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    echo "[wan22] 没有 .env，已从 .env.example 复制一份，请改 hosts / S3 后再 start.sh"
    cp .env.example .env
  else
    echo "[wan22] 没有 .env，deploy.sh 仍可装依赖，start.sh 会 source .env" >&2
  fi
fi

if [[ "$SKIP_DEPLOY" -eq 1 ]]; then
  echo "[wan22] --skip-deploy：系统层完成。接下来：./deploy.sh && ./start.sh"
  exit 0
fi

echo "[wan22] running deploy.sh（venv + torch cu128 + 权重）"
# deploy.sh 里的 python3 -m venv /opt/... 需要 root，当前已是
bash "$SCRIPT_DIR/deploy.sh"
echo "[wan22] bootstrap 完成。改好 .env 后执行：./start.sh"
