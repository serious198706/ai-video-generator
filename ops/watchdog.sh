#!/usr/bin/env bash
# GPU 本机：内存 / 磁盘 / systemd / nvidia-smi。正常不发，异常才 webhook。
set -euo pipefail

OPS="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$OPS/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

UNIT="${WAN22_SYSTEMD_UNIT:-wan22-gpu}"
MEM_PCT="${WAN22_WATCHDOG_MEM_PCT:-85}"
DISK_PCT="${WAN22_WATCHDOG_DISK_PCT:-85}"
GPU_TEMP="${WAN22_WATCHDOG_GPU_TEMP:-85}"
COOLDOWN="${WAN22_WATCHDOG_COOLDOWN:-1800}"
STATE_DIR="${WAN22_WATCHDOG_STATE:-/var/tmp/wan22-watchdog}"
mkdir -p "$STATE_DIR"
NOW="$(date +%s)"

problems=()
keys=()

add() {
  keys+=("$1")
  problems+=("$2")
}

mem_line="$(awk '
  /^MemTotal:/ { t=$2 }
  /^MemAvailable:/ { a=$2 }
  END {
    if (t <= 0) { print "0 0 0"; exit }
    used = t - a
    pct = used * 100 / t
    printf "%.0f %.0f %.0f\n", pct, used/1024, t/1024
  }
' /proc/meminfo)"
read -r mem_used_pct mem_used_mb mem_total_mb <<<"$mem_line"
mem_used_pct="${mem_used_pct:-0}"
if [[ "$mem_used_pct" -ge "$MEM_PCT" ]]; then
  add mem "内存 ${mem_used_pct}%（已用 ${mem_used_mb}MB / ${mem_total_mb}MB，阈值 ${MEM_PCT}%）"
fi

check_disk() {
  local path="$1"
  local label="$2"
  [[ -d "$path" ]] || return 0
  local pct
  pct="$(df -P "$path" | awk 'NR==2 { gsub("%","",$5); print $5 }')"
  [[ -n "$pct" ]] || return 0
  if [[ "$pct" -ge "$DISK_PCT" ]]; then
    add "disk:${path}" "磁盘 ${label} ${pct}%（阈值 ${DISK_PCT}%）"
  fi
}
check_disk / 根盘
check_disk /data /data
check_disk /opt /opt

if command -v systemctl >/dev/null 2>&1; then
  if ! systemctl is-active --quiet "$UNIT"; then
    state="$(systemctl is-active "$UNIT" 2>/dev/null || true)"
    add service "systemd ${UNIT} 不是 active（当前 ${state:-unknown}）"
  fi
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  add gpu "找不到 nvidia-smi"
else
  if ! smi_out="$(nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    add gpu "nvidia-smi 失败"
  elif [[ -z "${smi_out// }" ]]; then
    add gpu "nvidia-smi 没有 GPU"
  else
    while IFS=',' read -r idx name temp used total util; do
      idx="$(echo "$idx" | xargs)"
      name="$(echo "$name" | xargs)"
      temp="$(echo "$temp" | xargs)"
      used="$(echo "$used" | xargs)"
      total="$(echo "$total" | xargs)"
      util="$(echo "$util" | xargs)"
      if [[ "$temp" =~ ^[0-9]+$ ]] && [[ "$temp" -ge "$GPU_TEMP" ]]; then
        add "gpu-temp:${idx}" "GPU${idx} ${name} 温度 ${temp}°C（阈值 ${GPU_TEMP}°C），显存 ${used}/${total} MiB，利用率 ${util}%"
      fi
    done <<<"$smi_out"
  fi
fi

if [[ ${#problems[@]} -eq 0 ]]; then
  # 恢复后清冷却，下次异常立刻能发
  find "$STATE_DIR" -type f -name 'alert-*' -delete 2>/dev/null || true
  exit 0
fi

fresh=()
fresh_keys=()
for i in "${!keys[@]}"; do
  key="${keys[$i]}"
  stamp_file="$STATE_DIR/alert-$(echo "$key" | tr '/:' '__')"
  last=0
  if [[ -f "$stamp_file" ]]; then
    last="$(cat "$stamp_file" 2>/dev/null || echo 0)"
  fi
  if (( NOW - last >= COOLDOWN )); then
    fresh+=("${problems[$i]}")
    fresh_keys+=("$key")
  fi
done

if [[ ${#fresh[@]} -eq 0 ]]; then
  exit 0
fi

host="$(hostname -s 2>/dev/null || hostname)"
{
  echo "【Wan22 watchdog】${host} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo
  for line in "${fresh[@]}"; do
    echo "- $line"
  done
} >"$STATE_DIR/last-message.txt"

python3 "$OPS/feishu.py" --file "$STATE_DIR/last-message.txt"

for key in "${fresh_keys[@]}"; do
  echo "$NOW" >"$STATE_DIR/alert-$(echo "$key" | tr '/:' '__')"
done
