#!/usr/bin/env bash
# update_sampler.sh <mode> <max_resp_len> <high_T>
#   mode = uniform  → window_size=0，保持默认 T
#   mode = hl       → 高低温：window_size=2*max_resp_len, T_base=0.6, T_max=high_T
set -euo pipefail

mode="$1"; max_r="$2"; highT="${3:-1.0}"

site_dir=$(python - <<'PY'
import importlib.util, pathlib, sys
spec = importlib.util.find_spec("vllm")
print(pathlib.Path(spec.origin).parent.parent)
PY
)
sampler="$site_dir/vllm/v1/sample/sampler.py"

if [[ "$mode" == "uniform" ]]; then
  sed -i -E \
      -e 's/("window_size"[[:space:]]*:[[:space:]]*)[0-9]+/\10/' \
      "$sampler"
else  # hl
  ws=$((max_r * 2))
  sed -i -E \
      -e "s/(\"window_size\"[[:space:]]*:[[:space:]]*)[0-9]+/\\1$ws/" \
      -e 's/("T_base"[[:space:]]*:[[:space:]]*)[0-9.]+/\10.6/' \
      -e "s/(\"T_max\"[[:space:]]*:[[:space:]]*)[0-9.]+/\\1$highT/" \
      "$sampler"
fi
echo "[✓] sampler.py 已更新 → $mode"
