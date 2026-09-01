#!/usr/bin/env bash
# 在 Mac 上交叉编译 Linux x86_64 / Python 3.12 wheel，避免 pydantic_core 对本机扩展。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-/tmp/wan22-lambda.zip}"
PKG="$(mktemp -d /tmp/wan22-lambda-pkg.XXXXXX)"
trap 'rm -rf "$PKG"' EXIT

export PIP_DISABLE_PIP_VERSION_CHECK=1
# pyenv 的 pip 成功后常因 rehash 返回非 0，只要 linux wheel 落盘就继续。
python3 -m pip install \
  -r "$ROOT/lambda_api/requirements-lambda.txt" \
  -t "$PKG" \
  --python-version 3.12 \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --only-binary=:all: \
  --upgrade \
  || true
if ! find "$PKG/pydantic_core" -name '*linux-gnu.so' | grep -q .; then
  echo "[lambda] 未装上 Linux x86_64 的 pydantic_core，请检查 pip / 网络" >&2
  exit 1
fi

rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ROOT/wan22/" "$PKG/wan22/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude 'package.sh' \
  "$ROOT/lambda_api/" "$PKG/lambda_api/"
rm -rf "$PKG/wan22/infer" "$PKG/wan22/media" \
  "$PKG/wan22/queue/worker.py" "$PKG/wan22/api/app.py"

# 确认打进去的是 Linux 扩展，不是 macOS .so
find "$PKG" -name '*.so' \( -name '*darwin*' -o -name '*arm64*' \) -delete
if find "$PKG" -name '*pydantic_core*' -name '*.so' | grep -q 'darwin\|arm64\|aarch64'; then
  echo "[lambda] pydantic_core 仍是本机二进制，打包失败" >&2
  exit 1
fi
if ! find "$PKG/pydantic_core" -name '*linux-gnu.so' | grep -q .; then
  echo "[lambda] 未装上 Linux x86_64 的 pydantic_core，请检查 pip / 网络" >&2
  exit 1
fi

rm -f "$OUT"
(
  cd "$PKG"
  zip -qr "$OUT" .
)
echo "[lambda] wrote $OUT ($(du -h "$OUT" | awk '{print $1}'))"
echo "[lambda] handler: lambda_api.handler.handler"
echo "[lambda] runtime: Python 3.12 / x86_64"
