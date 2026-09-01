#!/usr/bin/env bash
# 恢复 v3 训练（--resume：恢复模型/优化器/调度器/EMA/epoch/早停状态）
set -euo pipefail
cd /home/li/face-to-ck3-datasetmake

RUN_DIR="runs/convnext_tiny_multiview_male_small_105865_v3"
CONFIG="configs/finetune_male_small_105865_v3.json"
CKPT="$RUN_DIR/last.pt"
LOG_FILE="train_105865_v3.log"
PID_FILE="$RUN_DIR/train.pid"

if [[ -f "$PID_FILE" ]]; then
  OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo)
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "already running pid=$OLD_PID"; exit 1
  fi
  rm -f "$PID_FILE"
fi

N_GPU=$(nvidia-smi --list-gpus | wc -l)
{
  echo "===== resume at $(date -Iseconds) from $CKPT ====="
  echo "torchrun --nproc_per_node=$N_GPU train.py --config $CONFIG --resume $CKPT --device auto"
} >> "$LOG_FILE"

nohup .venv/bin/torchrun --nproc_per_node="$N_GPU" train.py \
  --config "$CONFIG" \
  --resume "$CKPT" \
  --device auto \
  </dev/null \
  >>"$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"
disown $PID 2>/dev/null || true
sleep 1
echo "Resumed pid=$PID"
