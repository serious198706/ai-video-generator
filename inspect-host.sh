#!/usr/bin/env bash
# 在现网 GPU（DLAMI / 已部署机）上跑，把版本打到 stdout。裸金属对照这份即可。
# 不读 .env，不含密钥。缺什么就标 missing，不中断。
set -u

VENV_DIR="${WAN22_VENV_DIR:-/opt/wan22-venv}"
FOLEY_PY="${WAN22_FOLEY_PYTHON:-/opt/foley-venv/bin/python}"
AMI_PY="/opt/pytorch/bin/python"

kv() {
  printf '%s=%s\n' "$1" "$2"
}

section() {
  printf '\n# %s\n' "$1"
}

section os
if [[ -f /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  kv os "${PRETTY_NAME:-unknown}"
else
  kv os missing
fi
kv uname "$(uname -srm)"
kv hostname "$(hostname)"

section nvidia
if command -v nvidia-smi >/dev/null 2>&1; then
  kv nvidia_smi present
  kv driver "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]' || echo missing)"
  kv gpu "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n1 | sed 's/,/; /g' || echo missing)"
  kv cuda_reported "$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]' || echo missing)"
else
  kv nvidia_smi missing
fi

section python
kv system_python "$(command -v python3 2>/dev/null || echo missing)"
if command -v python3 >/dev/null 2>&1; then
  kv system_python_version "$(python3 -c 'import sys; print("%s.%s.%s" % sys.version_info[:3])' 2>/dev/null || echo missing)"
fi

dump_torch() {
  local label="$1"
  local bin="$2"
  if [[ ! -x "$bin" ]]; then
    kv "${label}_python" missing
    return
  fi
  kv "${label}_python" "$bin"
  kv "${label}_python_version" "$("$bin" -c 'import sys; print("%s.%s.%s" % sys.version_info[:3])' 2>/dev/null || echo missing)"
  "$bin" - <<'PY' 2>/dev/null | while IFS='=' read -r k v; do kv "${label}_${k}" "$v"; done
import sys
try:
    import torch
    print("torch=" + torch.__version__)
    print("torch_cuda=" + str(torch.version.cuda or "none"))
    print("cuda_available=" + str(torch.cuda.is_available()).lower())
    if torch.cuda.is_available():
        print("gpu_name=" + torch.cuda.get_device_name(0).replace(" ", "_"))
except Exception as exc:
    print("torch=missing")
    print("torch_error=" + type(exc).__name__)
PY
}

section ami_pytorch
dump_torch ami "$AMI_PY"
if [[ -d /opt/pytorch/cuda ]]; then
  kv ami_cuda_home /opt/pytorch/cuda
else
  kv ami_cuda_home missing
fi

section wan22_venv
dump_torch wan22 "${VENV_DIR}/bin/python"

section foley_venv
dump_torch foley "$FOLEY_PY"

section apt
for pkg in python3 python3-venv python3-dev git build-essential ffmpeg; do
  if command -v dpkg >/dev/null 2>&1; then
    ver="$(dpkg-query -W -f='${Version}' "$pkg" 2>/dev/null || true)"
    kv "apt_${pkg}" "${ver:-missing}"
  fi
done

section commands
for cmd in git ffmpeg python3 nvidia-smi hf; do
  if command -v "$cmd" >/dev/null 2>&1; then
    kv "cmd_${cmd}" "$(command -v "$cmd")"
  else
    kv "cmd_${cmd}" missing
  fi
done

printf '\n# 把以上输出保存下来，裸金属 ./bootstrap-ubuntu.sh 不依赖这份也能跑。\n'
printf '# 若要对齐现网：看 wan22_torch / wan22_torch_cuda / system_python_version / driver。\n'
