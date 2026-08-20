#!/usr/bin/env bash
# 后台启动 finetune v2 (从 v1 epoch 29 EMA 续训 20 epoch)
# 用法: bash scripts/start_finetune_male_small_34780_v2.sh
set -euo pipefail
cd /mnt/d/workspace/face-to-ck3-datasetmake
source .venv-linux/bin/activate

RUN_DIR="runs/convnext_tiny_multiview_male_small_34780_v2"
SOURCE_CKPT="runs/convnext_tiny_multiview_male_small_34780_v1/best.pt"
CONFIG="configs/finetune_male_small_34780_v2.json"
LOG_FILE="$RUN_DIR/train.log"

mkdir -p "$RUN_DIR"

# 避免残留旧进程
if [[ -f "$RUN_DIR/train.pid" ]]; then
  OLD_PID=$(cat "$RUN_DIR/train.pid" 2>/dev/null || echo)
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Killing previous training pid=$OLD_PID"
    kill -TERM "$OLD_PID" 2>/dev/null || true
    sleep 3
    kill -KILL "$OLD_PID" 2>/dev/null || true
  fi
  rm -f "$RUN_DIR/train.pid"
fi

echo "Starting finetune:"
echo "  source: $SOURCE_CKPT"
echo "  config: $CONFIG"
echo "  output: $RUN_DIR"
echo "  log:    $LOG_FILE"

# 不在 WSL 父 shell 退出时被 SIGHUP 杀
# nohup + & + disown + 重定向 stdin/stdout/stderr 全套
{
  echo "===== launch at $(date -Iseconds) ====="
  echo "python train.py --config $CONFIG --finetune-from $SOURCE_CKPT --device auto"
} > "$LOG_FILE"

nohup python train.py \
  --config "$CONFIG" \
  --finetune-from "$SOURCE_CKPT" \
  --device auto \
  </dev/null \
  >>"$LOG_FILE" 2>&1 &

PID=$!
echo $PID > "$RUN_DIR/train.pid"
disown $PID 2>/dev/null || true

# 让 WSL 这层也 detach（关键：避免 PowerShell 退出时把进程带走）
sleep 1

echo "Launched pid=$PID"
echo "Tail log:    tail -f $LOG_FILE"
echo "Check alive: kill -0 \$(cat $RUN_DIR/train.pid) && echo running"
