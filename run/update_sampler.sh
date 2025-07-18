#!/usr/bin/env bash
# update_sampler.sh
# 下载指定分支代码并覆盖本地 vllm sampler，同时注入温度超参数

set -euo pipefail

###################### 可调超参数（或用环境变量覆盖） ######################
WINDOW_SIZE=${WINDOW_SIZE:-100000}   # 最近多少个 token 的熵
PERCENTILE=${PERCENTILE:-0.20}       # 熵位于前 p% 时 → 高温
T_BASE=${T_BASE:-1.2}                # 正常温度
T_MAX=${T_MAX:-1.6}                  # 高温
###########################################################################

REPO_URL="https://github.com/zhmzm/vllm.git"
BRANCH="0.8.5"

echo "[1/4] 克隆分支 $BRANCH ..."
TMP_DIR="$(mktemp -d)"
git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$TMP_DIR"

echo "[2/4] 拷贝 sample.py → 系统安装目录 ..."
SRC_FILE="$TMP_DIR/vllm/v1/sample/sampler.py"
DST_DIR="/usr/local/lib/python3.11/dist-packages/vllm/v1/sample"
sudo mkdir -p "$DST_DIR"
sudo cp "$SRC_FILE" "$DST_DIR/sampler.py"

echo "[3/4] 注入自定义超参数 ..."
sudo sed -i -E \
  -e "s/(\"window_size\"[[:space:]]*:[[:space:]]*)[0-9_]+/\\1$WINDOW_SIZE/" \
  -e "s/(\"percentile\"[[:space:]]*:[[:space:]]*)[0-9.]+/\\1$PERCENTILE/" \
  -e "s/(\"T_base\"[[:space:]]*:[[:space:]]*)[0-9.]+/\\1$T_BASE/" \
  -e "s/(\"T_max\"[[:space:]]*:[[:space:]]*)[0-9.]+/\\1$T_MAX/" \
  "$DST_DIR/sampler.py"

echo "[4/4] 完成！当前参数："
grep -n -A3 -B1 "_GLOBAL_TEMP_CFG" "$DST_DIR/sampler.py"
