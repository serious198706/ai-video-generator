#!/usr/bin/env bash
# 把系统盘 overlay 上已经下到一半的 HF / pip 缓存挪到 AutoDL 数据盘。
# 用法：在 GPU 容器里
#   cd /path/to/server && ./migrate-autodl-cache.sh
set -euo pipefail

DEST_ROOT="${WAN22_DATA_ROOT:-/root/autodl-tmp/wan22}"
DEST_HF="${HF_HOME:-$DEST_ROOT/hf-cache}"
DEST_PIP="${PIP_CACHE_DIR:-$DEST_ROOT/pip-cache}"
DEST_TMP="${TMPDIR:-$DEST_ROOT/tmp}"

if [[ ! -d /root/autodl-tmp ]]; then
  echo "[wan22] 没有 /root/autodl-tmp" >&2
  exit 1
fi

dest_src="$(df -P /root/autodl-tmp | awk 'NR==2 { print $1 }')"
root_src="$(df -P / | awk 'NR==2 { print $1 }')"
if [[ -z "$dest_src" || "$dest_src" == "$root_src" ]]; then
  echo "[wan22] /root/autodl-tmp 还在 30G overlay 上，挪过去等于没挪。" >&2
  echo "[wan22] 先在控制台挂上数据盘，确认: df -h /root/autodl-tmp 不是 overlay。" >&2
  df -h / /root/autodl-tmp >&2 || true
  exit 1
fi

mkdir -p "$DEST_HF" "$DEST_PIP" "$DEST_TMP"

echo "[wan22] 系统盘占用："
df -h /
echo "[wan22] 数据盘："
df -h /root/autodl-tmp
echo

move_dir() {
  local src="$1"
  local dest="$2"
  if [[ ! -e "$src" ]]; then
    return 0
  fi
  src_fs="$(df -P "$src" | awk 'NR==2 { print $1 }')"
  dest_fs="$(df -P "$dest" | awk 'NR==2 { print $1 }')"
  if [[ "$src_fs" == "$dest_fs" ]]; then
    echo "[wan22] skip $src（已经在数据盘）"
    return 0
  fi
  echo "[wan22] $src  ->  $dest"
  rsync -a --info=stats1 "$src"/ "$dest"/
  rm -rf "$src"
}

# huggingface_hub 默认写这里；hf download --local-dir 也会先占 cache
move_dir /root/.cache/huggingface "$DEST_HF"
move_dir /root/.cache/huggingface-hub "$DEST_HF"
move_dir /root/.cache/pip "$DEST_PIP"

# 若之前把 wan22 写在 overlay 的 autodl-tmp，数据盘挂上后这些文件会被挡住。
# 先试常见可见路径；挡住的见脚本末尾说明。
if [[ -d /root/autodl-tmp-overlay ]]; then
  move_dir /root/autodl-tmp-overlay/wan22/hf-cache "$DEST_HF"
  move_dir /root/autodl-tmp-overlay/wan22/models "$DEST_ROOT/models"
fi

echo
echo "[wan22] 挪完后系统盘："
df -h /
echo "[wan22] 数据盘："
df -h /root/autodl-tmp
echo
echo "[wan22] 接着："
echo "  export HF_HOME=$DEST_HF TMPDIR=$DEST_TMP PIP_CACHE_DIR=$DEST_PIP"
echo "  ./deploy.sh"
echo
echo "[wan22] 若系统盘仍接近 30G：多半是挂载数据盘之前写进 /root/autodl-tmp 的文件被挡住了。"
echo "  不要 umount。把数据盘临时挂到旁边再拷："
echo "    mkdir -p /mnt/data-disk"
echo "    mount --bind /root/autodl-tmp /mnt/data-disk"
echo "    # 然后按 AutoDL 文档把原 overlay 目录露出来，或关机前把缓存只从 ~/.cache 挪（本脚本已处理）"
