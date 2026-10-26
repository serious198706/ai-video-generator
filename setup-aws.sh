#!/usr/bin/env bash
# 把专用 IAM 用户的 key / secret / region 写进 server/.env（start.sh / deploy.sh 会 source）。
# vast.ai 上没有 IAM Role，SQS 消费和 S3 上传都靠这份身份。
#
# 用法：
#   ./setup-aws.sh                 # 缺什么问什么，已有的跳过
#   ./setup-aws.sh --force         # 重新输入整套（轮换 key 时用）
# 非交互：直接带环境变量跑，不落盘
#   AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=ap-northeast-1 ./deploy.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# .env 是 shell 语法，值里有空格/特殊字符要加引号（单引号内再转义单引号）。
put_env() {
  local key="$1" value="$2" tmp escaped
  touch "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  escaped="$(printf '%s' "$value" | sed "s/'/'\\\\''/g")"
  tmp="$(mktemp "$ENV_FILE.XXXXXX")"
  grep -v "^${key}=" "$ENV_FILE" >"$tmp" || true
  printf "%s='%s'\n" "$key" "$escaped" >>"$tmp"
  mv "$tmp" "$ENV_FILE"
}

env_value() {
  [[ -f "$ENV_FILE" ]] || return 0
  sed -n "s/^$1=//p" "$ENV_FILE" | tail -n 1 | sed "s/^'//; s/'$//"
}

# 非交互（CI / 云-init）：环境里给了就直接用，不写文件。
if [[ ! -t 0 ]]; then
  if [[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    export AWS_SECRET_ACCESS_KEY AWS_ACCESS_KEY_ID
    export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-}}"
    echo "[aws] non-interactive, using environment credentials" >&2
    exit 0
  fi
  echo "[aws] no tty and no AWS_ACCESS_KEY_ID; skip credential setup" >&2
  exit 0
fi

current_key="$(env_value AWS_ACCESS_KEY_ID)"
current_region="$(env_value AWS_DEFAULT_REGION)"
if [[ "$FORCE" != "1" && -n "${AWS_ACCESS_KEY_ID:-}${current_key}" && -n "$current_region" ]]; then
  echo "[aws] credentials already configured (${AWS_ACCESS_KEY_ID:-$current_key}), skip; use --force to rotate"
  exit 0
fi

echo "[aws] 专用 IAM 用户需要：sqs:ReceiveMessage / DeleteMessage / ChangeMessageVisibility；s3:PutObject（见 README）"
read -r -p "AWS Access Key ID（回车跳过）: " access_key || access_key=""
if [[ -z "$access_key" ]]; then
  echo "[aws] skipped; 之后可单独运行 ./setup-aws.sh" >&2
  exit 0
fi

read -r -s -p "AWS Secret Access Key（不回显）: " secret_key || secret_key=""
echo
if [[ -z "$secret_key" ]]; then
  echo "[aws] secret 为空，放弃" >&2
  exit 1
fi

default_region="${current_region:-${AWS_DEFAULT_REGION:-${WAN22_S3_REGION:-ap-northeast-1}}}"
read -r -p "AWS Region [$default_region]: " region || region=""
region="${region:-$default_region}"

# 先真打一次 STS：key 错、被禁用、region 写错都能当场发现，别等生成跑完上传才失败。
_PY="$SCRIPT_DIR/.venv/bin/python"
[[ -x "$_PY" ]] || _PY="$(command -v python3 || command -v python)"
verify_py="$(mktemp /tmp/wan22-aws.XXXXXX.py)"
trap 'rm -f "$verify_py"' EXIT
cat >"$verify_py" <<'PY'
import boto3

identity = boto3.client("sts").get_caller_identity()
print(identity.get("Account", "?"), identity.get("Arn", "?"))
PY

echo "[aws] verifying credentials ..."
if ! identity="$(
  AWS_ACCESS_KEY_ID="$access_key" \
  AWS_SECRET_ACCESS_KEY="$secret_key" \
  AWS_DEFAULT_REGION="$region" \
    "$_PY" "$verify_py" 2>&1
)"; then
  # 只回最后一行，别把整段 traceback 甩给用户。
  echo "[aws] STS 验证失败：$(printf '%s\n' "$identity" | tail -n 1)" >&2
  echo "[aws] 检查 key/secret/region 或网络；未写入 .env" >&2
  exit 1
fi
echo "[aws] ok $identity"

put_env AWS_ACCESS_KEY_ID "$access_key"
put_env AWS_SECRET_ACCESS_KEY "$secret_key"
put_env AWS_DEFAULT_REGION "$region"
# 成片上传和 SQS 各自读 S3_REGION / SQS_REGION；没显式设过就沿用同一个 region。
if [[ -z "$(env_value WAN22_S3_REGION)" ]]; then
  put_env WAN22_S3_REGION "$region"
fi
if [[ -z "$(env_value SQS_REGION)" ]]; then
  put_env SQS_REGION "$region"
fi

echo "[aws] written to $ENV_FILE (chmod 600)"
