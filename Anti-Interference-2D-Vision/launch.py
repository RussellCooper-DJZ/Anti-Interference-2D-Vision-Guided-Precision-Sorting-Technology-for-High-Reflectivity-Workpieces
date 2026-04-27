#!/usr/bin/env python3
"""
launch.py — AGEANet 统一跨平台启动器

功能：
  1. 自动检测操作系统（Windows / Linux / macOS）
  2. 检查并创建虚拟环境（如不存在）
  3. 自动安装依赖（requirements-dev.txt）
  4. 启动 Streamlit / Gradio / Training / Demo 等模式

用法::
    python launch.py --mode demo          # 启动 Streamlit 演示
    python launch.py --mode gradio        # 启动 Gradio 演示
    python launch.py --mode training      # 启动训练
    python launch.py --mode pipeline      # 启动主流程（命令行）
    python launch.py --mode check         # 仅检查环境，不启动服务
"""

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / ".venv_auto"
PYTHON_EXE = (
    VENV_DIR / "Scripts" / "python.exe"
    if platform.system() == "Windows"
    else VENV_DIR / "bin" / "python"
)
PIP_EXE = (
    VENV_DIR / "Scripts" / "pip.exe"
    if platform.system() == "Windows"
    else VENV_DIR / "bin" / "pip"
)


def log(msg: str):
    print(f"[launch] {msg}", flush=True)


def ensure_venv():
    """确保虚拟环境存在，如不存在则创建。"""
    if PYTHON_EXE.exists():
        log(f"虚拟环境已存在: {VENV_DIR}")
        return

    log("正在创建虚拟环境...")
    system_py = sys.executable
    subprocess.check_call([system_py, "-m", "venv", str(VENV_DIR)])
    log(f"虚拟环境创建完成: {VENV_DIR}")


def ensure_deps():
    """确保依赖已安装。"""
    req_file = PROJECT_ROOT / "requirements-dev.txt"
    if not req_file.exists():
        req_file = PROJECT_ROOT / "requirements.txt"

    log(f"正在安装依赖 ({req_file.name})...")
    subprocess.check_call([str(PYTHON_EXE), "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([str(PIP_EXE), "install", "-r", str(req_file)])
    log("依赖安装完成")


def check_torch_backend() -> str:
    """检测 PyTorch 可用后端。"""
    try:
        result = subprocess.run(
            [str(PYTHON_EXE), "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True,
            text=True,
        )
        cuda = result.stdout.strip() == "True"
    except Exception:
        cuda = False

    try:
        result = subprocess.run(
            [str(PYTHON_EXE), "-c", "import torch; print(torch.backends.mps.is_available())"],
            capture_output=True,
            text=True,
        )
        mps = result.stdout.strip() == "True"
    except Exception:
        mps = False

    if cuda:
        return "CUDA"
    if mps:
        return "MPS (Apple Silicon)"
    return "CPU"


def run_demo():
    """启动 Streamlit 演示。"""
    log("启动 Streamlit 演示界面...")
    log(f"请访问: http://localhost:8501")
    subprocess.check_call(
        [str(PYTHON_EXE), "-m", "streamlit", "run", "demo_streamlit.py", "--server.port", "8501"],
        cwd=PROJECT_ROOT,
    )


def run_gradio():
    """启动 Gradio 演示。"""
    log("启动 Gradio 演示界面...")
    log(f"请访问: http://localhost:7860")
    subprocess.check_call(
        [str(PYTHON_EXE), "demo_gradio.py", "--port", "7860"],
        cwd=PROJECT_ROOT,
    )


def run_gui(args):
    """启动原生桌面 GUI。"""
    log("启动 PyQt6 桌面 GUI...")
    cmd = [str(PYTHON_EXE), "gui/main_window.py"]
    if args.model_path:
        cmd += ["--model_path", args.model_path]
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)


def run_training(args):
    """启动训练。"""
    log("启动训练流程...")
    cmd = [
        str(PYTHON_EXE), "training/train.py",
        "--synth_dir", args.synth_dir or "./dataset",
        "--epochs", str(args.epochs or 50),
        "--batch_size", str(args.batch_size or 8),
        "--lr", str(args.lr or 1e-4),
        "--save_dir", args.save_dir or "./checkpoints",
    ]
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)


def run_pipeline(args):
    """启动主流程。"""
    log("启动主流程...")
    cmd = [str(PYTHON_EXE), "main_pipeline.py", "--mode", args.pipeline_mode or "demo"]
    if args.model_path:
        cmd += ["--model_path", args.model_path]
    subprocess.check_call(cmd, cwd=PROJECT_ROOT)


def run_check():
    """环境检查模式。"""
    log("=" * 50)
    log("AGEANet 环境检查报告")
    log("=" * 50)
    log(f"操作系统: {platform.system()} {platform.release()}")
    log(f"Python: {sys.version.split()[0]}")
    log(f"虚拟环境: {VENV_DIR}")
    log(f"虚拟环境 Python: {'存在' if PYTHON_EXE.exists() else '不存在'}")

    if PYTHON_EXE.exists():
        backend = check_torch_backend()
        log(f"PyTorch 加速后端: {backend}")

        # 检查关键包
        for pkg in ["torch", "cv2", "numpy", "streamlit", "gradio", "yaml", "pytest"]:
            try:
                subprocess.run(
                    [str(PYTHON_EXE), "-c", f"import {pkg}"],
                    capture_output=True,
                    check=True,
                )
                log(f"  ✅ {pkg}")
            except subprocess.CalledProcessError:
                log(f"  ❌ {pkg} (未安装)")
    else:
        log("提示: 运行 'python launch.py --mode check --init' 可自动创建环境")

    log("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="AGEANet 统一跨平台启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python launch.py --mode demo
  python launch.py --mode training --epochs 100
  python launch.py --mode pipeline --pipeline-mode image --model_path ./checkpoints/best.pth
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "gradio", "gui", "training", "pipeline", "check"],
        default="demo",
        help="启动模式 (默认: demo)",
    )
    parser.add_argument("--no-init", action="store_true", help="跳过虚拟环境和依赖初始化")
    parser.add_argument("--synth-dir", default="./dataset", help="训练: 合成数据集目录")
    parser.add_argument("--epochs", type=int, default=50, help="训练: 轮数")
    parser.add_argument("--batch-size", type=int, default=8, help="训练: 批次大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="训练: 学习率")
    parser.add_argument("--save-dir", default="./checkpoints", help="训练: 保存目录")
    parser.add_argument("--pipeline-mode", default="demo", help="主流程: demo/camera/image")
    parser.add_argument("--model-path", default=None, help="模型权重路径")

    args = parser.parse_args()

    log(f"AGEANet 启动器 | 模式: {args.mode} | 系统: {platform.system()}")

    # 初始化环境
    if not args.no_init and args.mode != "check":
        ensure_venv()
        ensure_deps()

    # 执行模式
    if args.mode == "demo":
        run_demo()
    elif args.mode == "gradio":
        run_gradio()
    elif args.mode == "gui":
        run_gui(args)
    elif args.mode == "training":
        run_training(args)
    elif args.mode == "pipeline":
        run_pipeline(args)
    elif args.mode == "check":
        run_check()


if __name__ == "__main__":
    main()
