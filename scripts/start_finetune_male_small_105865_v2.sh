#!/usr/bin/env bash
# 后台启动 105865 第二轮：从 v1 last.pt (epoch 29 EMA) 权重初始化，
# 优化器/调度器/epoch 全新，学习率按最初配置从头走完整调度。3 卡 torchrun。
# 用法: bash scripts/start_finetune_male_small_105865_v2.sh
set -euo pipefail
cd /home/li/face-to-ck3-datasetmake

RUN_DIR="runs/convnext_tiny_multiview_male_small_105865_v2"
SOURCE_CKPT="runs/convnext_tiny_multiview_male_small_105865_v1/last.pt"
CONFIG="configs/finetune_male_small_105865_v2.json"
LOG_FILE="train_105865_v2.log"
PID_FILE="$RUN_DIR/train.pid"

mkdir -p "$RUN_DIR"

# 避免残留旧进程
if [[ -f "$PID_FILE" ]]; then
  OLD_PID=$(cat "$PID_FILE" 2>/dev/null || echo)
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Killing previous training pid=$OLD_PID"
    kill -TERM "$OLD_PID" 2>/dev/null || true
    sleep 3
    kill -KILL "$OLD_PID" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi

N_GPU=$(nvidia-smi --list-gpus | wc -l)
echo "Starting round-2 finetune:"
echo "  source: $SOURCE_CKPT (epoch 29, EMA)"
echo "  config: $CONFIG"
echo "  output: $RUN_DIR"
echo "  log:    $LOG_FILE"
echo "  gpus:   $N_GPU"

{
  echo "===== launch at $(date -Iseconds) ====="
  echo "torchrun --nproc_per_node=$N_GPU train.py --config $CONFIG --finetune-from $SOURCE_CKPT --device auto"
} > "$LOG_FILE"

nohup .venv/bin/torchrun --nproc_per_node="$N_GPU" train.py \
  --config "$CONFIG" \
  --finetune-from "$SOURCE_CKPT" \
  --device auto \
  </dev/null \
  >>"$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$PID_FILE"
disown $PID 2>/dev/null || true
sleep 1

echo "Launched pid=$PID"
echo "Tail log:    tail -f $LOG_FILE"
echo "Check alive: kill -0 \$(cat $PID_FILE) && echo running"
