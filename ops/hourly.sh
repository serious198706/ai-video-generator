#!/usr/bin/env bash
# Mac：SSH 拉上一小时 GPU journal，有 ERROR/失败才推飞书。
set -euo pipefail

OPS="$(cd "$(dirname "$0")" && pwd)"
NOTIFY=0
for arg in "$@"; do
  if [[ "$arg" == "--notify" ]]; then
    NOTIFY=1
  fi
done

if [[ -f "$OPS/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$OPS/.env"
  set +a
fi

HOST="${WAN22_GPU_SSH:?请在 ops/.env 里设置 WAN22_GPU_SSH=user@gpu}"
UNIT="${WAN22_SYSTEMD_UNIT:-wan22-gpu}"
# shellcheck disable=SC2206
SSH_OPTS=(${WAN22_GPU_SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=15})

window="$(python3 "$OPS/window.py" hourly)"
SINCE="$(printf '%s\n' "$window" | sed -n '1p')"
UNTIL="$(printf '%s\n' "$window" | sed -n '2p')"

raw="$(ssh "${SSH_OPTS[@]}" "$HOST" \
  "journalctl -u $(printf %q "$UNIT") --since $(printf %q "$SINCE") --until $(printf %q "$UNTIL") --no-pager")"
filtered="$(printf '%s\n' "$raw" | python3 "$OPS/filter_hourly.py")"
if [[ -z "${filtered// }" ]]; then
  exit 0
fi

host_label="${WAN22_GPU_LABEL:-$HOST}"
report="【Wan22 小时告警】${host_label}
窗口 ${SINCE} → ${UNTIL}

${filtered}"
printf '%s\n' "$report"
if [[ "$NOTIFY" == "1" && -n "${WAN22_FEISHU_WEBHOOK:-}" ]]; then
  printf '%s\n' "$report" | python3 "$OPS/feishu.py"
fi
