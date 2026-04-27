#!/usr/bin/env bash
# start_training.sh — Linux / macOS 训练启动脚本
set -e

cd "$(dirname "$0")/../Anti-Interference-2D-Vision"

echo "========================================"
echo "  AGEANet FLARE Training"
echo "  OS: $(uname -s)"
echo "========================================"
echo ""

# 优先使用 launch.py
if [ -f "launch.py" ]; then
    echo "使用统一启动器 (launch.py)..."
    python3 launch.py --mode training "$@"
    exit 0
fi

# 回退：手动启动
echo "启动训练 (50 epochs)..."
python3 training/train.py --synth_dir ./dataset --epochs 50 --batch_size 8 --lr 1e-4 --save_dir ./checkpoints --base_ch 32
