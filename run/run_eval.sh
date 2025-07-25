#!/usr/bin/env bash
# run_eval.sh <launch.sh> <uniform|hl>
set -euo pipefail

[[ $# -eq 2 ]] || { echo "用法: $0 /path/launch.sh uniform|hl"; exit 1; }
orig_launch=$(realpath "$1")
mode="$2"                              # uniform 或 hl

SELF_DIR=$(cd "$(dirname "$0")" && pwd)
UPDATE="$SELF_DIR/update_sampler.sh"   # 改 sampler.py 的脚本

# -------- 解析原脚本默认 project / exp / max_rlen --------
grab() { awk -F'=' -v k="$1" '$1==k {gsub(/.*=/,""); gsub(/["'\'' ]/,""); print; exit}' "$orig_launch"; }
base_proj=$(grab project_name)
base_exp=$(grab exp_name)
max_rlen=$(grab max_response_length)
[[ -z $base_proj || -z $base_exp || -z $max_rlen ]] && { echo "无法解析启动脚本变量"; exit 2; }

# -------- Sweep 列表与模式设置 --------
if [[ $mode == uniform ]]; then
  sweep=(0.2 0.4 0.6 0.8 1.0 1.2 1.4); suffix="-EVAL"; tag=T
else
  sweep=(0.8 1.0 1.2 1.4 1.6);         suffix="-HL";   tag=H
fi
full_proj="${base_proj}${suffix}"
new_test='eval-box-aime256-amc32.parquet'   # 新测试集

# -------- 等待 W&B 首条指标 ----------
wait_wandb() { python - "$@" <<'PY'
import sys, time, wandb
proj, rid = sys.argv[1:3]
api = wandb.TrackingApi()                   # TrackingApi 也行
run = api.run(project=proj, run_id=rid)
print('run.history()', run.history())
while not run.history():   # ← 这里把 sample 改成 samples
    time.sleep(60)
    print('run.history()', run.history())
    
PY
}

# ================= 主循环 =================
for val in "${sweep[@]}"; do
  # 1) 修改 sampler.py
  if [[ $mode == uniform ]]; then
    "$UPDATE" uniform "$max_rlen"
    T="$val"
  else
    "$UPDATE" hl "$max_rlen" "$val"    # T_max = val
    T="0.6"                            # T_base 固定
  fi

  exp_tag="${base_exp}-${tag}${val}"
  tmp_sh=$(mktemp --suffix=.sh)

  # 2) 复制并 patch 启动脚本
  sed -e "s|^\s*project_name=.*|project_name='${full_proj}' \\\\|" \
      -e "s|^\s*exp_name=.*|exp_name='${exp_tag}' \\\\|"           \
      -e "s|^\s*TEST_FILE=.*|TEST_FILE=\${RAY_DATA_HOME}/eval-box-aime256-amc32.parquet \\\\|" \
      -e "s|^\s*trainer\.save_freq=.*|trainer.save_freq=99999 \\\\|" \
      -e "s|^\s*trainer\.total_training_steps=.*|trainer.total_training_steps=99999 \\\\|" \
      -e "s|^\s*trainer\.total_epochs=.*|trainer.total_epochs=99999 \\\\|" \
      -e "s|^\s*trainer\.log_val_generations=.*|trainer.log_val_generations=1000 \\\\|" \
      -e "s|^\s*trainer\.val_before_train=.*|trainer.val_before_train=True \\\\|" \
      "$orig_launch" > "$tmp_sh"
  chmod +x "$tmp_sh"

  # 3) 提交 Ray Job，直接从回显抓 job_id
  echo "▶ 提交 $exp_tag  (temperature=$T)"
  submit_out=$(TEMPERATURE="$T" bash "$tmp_sh")
  jid=$(echo "$submit_out" | grep -oE "Job '[A-Za-z0-9_]+'" | cut -d"'" -f2)
  [[ -z $jid ]] && { echo "无法捕获 job_id"; echo "$submit_out"; rm -f "$tmp_sh"; exit 3; }
  echo "  ↳ Ray Job = $jid"

  # 4) 解析 run_id
  rid=""
  until [[ -n $rid ]]; do
    rid=$(ray job logs "$jid" 2>&1 | grep -oE 'run_[0-9]{8}_[0-9a-f]{8}' | head -n1 || true)
    sleep 30
  done
  echo "  ↳ 绑定 run_id = $rid"

  # 5) 等首条指标 → 停止 Job
  wait_wandb "$full_proj" "$rid"
  ray job stop "$jid" >/dev/null
  echo "  ↳ $exp_tag 完成 ✅"

  rm -f "$tmp_sh"
done

echo -e "\n[✓] 所有温度 sweep 完毕，结果已写入 W&B ➜ $full_proj"
