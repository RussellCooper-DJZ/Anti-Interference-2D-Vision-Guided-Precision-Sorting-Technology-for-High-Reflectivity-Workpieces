#!/usr/bin/env bash
# start_demo.sh — Linux / macOS 启动脚本
set -e

cd "$(dirname "$0")/../Anti-Interference-2D-Vision"

echo "========================================"
echo "  AGEANet FLARE Demo Server"
echo "  OS: $(uname -s)"
echo "========================================"
echo ""

# 优先使用 launch.py 自动管理环境
if [ -f "launch.py" ]; then
    echo "使用统一启动器 (launch.py)..."
    python3 launch.py --mode demo
    exit 0
fi

# 回退：手动启动
echo "启动 Streamlit 演示界面..."
echo "请访问: http://localhost:8501"
echo "按 Ctrl+C 停止"
echo ""

python3 -m streamlit run demo_streamlit.py --server.port 8501
