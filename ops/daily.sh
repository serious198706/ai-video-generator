#!/usr/bin/env bash
# Mac：SSH 拉上海昨天 08:00 到今天 08:00 的 timing，写日报。每天都发。
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

window="$(python3 "$OPS/window.py" daily)"
SINCE="$(printf '%s\n' "$window" | sed -n '1p')"
UNTIL="$(printf '%s\n' "$window" | sed -n '2p')"
host_label="${WAN22_GPU_LABEL:-$HOST}"

# grep 无匹配时不要让 ssh 失败
raw="$(ssh "${SSH_OPTS[@]}" "$HOST" \
  "journalctl -u $(printf %q "$UNIT") --since $(printf %q "$SINCE") --until $(printf %q "$UNTIL") --no-pager | grep timing || true")"

spec="$(printf '%s\n' "$raw" | python3 "$OPS/digest.py" --json --start "$SINCE" --end "$UNTIL" --host "$host_label")"
python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d["title"]); print(); print("\n\n".join(d["sections"])); print(); print(d["note"])' <<<"$spec"
if [[ "$NOTIFY" == "1" && -n "${WAN22_FEISHU_WEBHOOK:-}" ]]; then
  printf '%s\n' "$spec" | python3 "$OPS/feishu.py" --json
fi
