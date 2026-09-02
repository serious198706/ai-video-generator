#!/usr/bin/env bash
# 只测飞书 webhook。逐步打日志，避免 source 整个 .env 被 FOLEY_PROMPT 空格弄死。
set -uo pipefail

echo "[test] start"

OPS="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$OPS/.." && pwd)"
echo "[test] ops=$OPS"

load_feishu_env() {
  local file="$1"
  [[ -f "$file" ]] || return 0
  echo "[test] reading $file"
  local line key val
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    case "$line" in
      ''|\#*) continue ;;
    esac
    key="${line%%=*}"
    val="${line#*=}"
    [[ "$key" == WAN22_FEISHU_WEBHOOK || "$key" == WAN22_FEISHU_SECRET ]] || continue
    val="${val#\'}"
    val="${val%\'}"
    val="${val#\"}"
    val="${val%\"}"
    export "$key=$val"
  done <"$file"
}

load_feishu_env "$ROOT/.env"
load_feishu_env "$OPS/.env"

webhook="${WAN22_FEISHU_WEBHOOK:-}"
if [[ -z "$webhook" ]]; then
  echo "[test] FAIL: 没有 WAN22_FEISHU_WEBHOOK" >&2
  echo "[test] 请写在 $ROOT/.env 或 $OPS/.env" >&2
  exit 1
fi
echo "[test] webhook=${webhook:0:48}…"
echo "[test] secret=$([ -n "${WAN22_FEISHU_SECRET:-}" ] && echo set || echo empty)"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[test] FAIL: 找不到 python3" >&2
  exit 1
fi

host="$(hostname -s 2>/dev/null || hostname || echo unknown)"
set +e
printf '%s\n' "**主机** ${host}" | python3 -u "$OPS/feishu.py" \
  --title "✅ Wan22 测试 | webhook 连通" \
  --template green \
  --note "$(date '+%Y-%m-%d %H:%M:%S %Z')"
status=$?
set -e
echo "[test] feishu.py exit=$status"
if [[ "$status" -ne 0 ]]; then
  echo "[test] FAIL: 发送失败。" >&2
  exit "$status"
fi
echo "[test] OK：去飞书群看绿色卡片。"
