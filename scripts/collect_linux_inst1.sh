#!/usr/bin/env bash
# Linux 原生采集实例 1（Xorg :90 / GPU 0 / Wine CK3 / x11vnc 5902）
# 前置：Xorg :90 与 CK3@Wine 已启动（见 scripts/ck3_linux_stack.md）
# 用法: bash scripts/collect_linux_inst1.sh [新数据集目录名]
set -euo pipefail
cd /home/li/face-to-ck3-datasetmake
BASE_DIR="${1:-face_to_ck3_dataset_linux_pilot}"
export DISPLAY=:90 XDG_SESSION_TYPE=x11
export WINEPREFIX=$HOME/ck3inst1/prefix
exec .venv/bin/python face_to_ck3_tool.py --base-dir "$BASE_DIR"
