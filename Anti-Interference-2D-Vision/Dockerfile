# AGEANet Docker 镜像
# 支持 CPU / CUDA 两种运行模式
#
# 构建 (CPU 版本):
#   docker build -t ageanet:latest .
#
# 构建 (CUDA 版本):
#   docker build --build-arg BASE_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04 -t ageanet:cuda .
#
# 运行 Streamlit 演示:
#   docker run -p 8501:8501 -v $(pwd)/checkpoints:/app/checkpoints ageanet:latest --mode demo
#
# 运行 Gradio 演示:
#   docker run -p 7860:7860 -v $(pwd)/checkpoints:/app/checkpoints ageanet:latest --mode gradio
#
# 运行训练 (需要 GPU):
#   docker run --gpus all -v $(pwd)/dataset:/app/dataset -v $(pwd)/checkpoints:/app/checkpoints ageanet:cuda --mode training

ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

LABEL maintainer="RussellCooper"
LABEL description="AGEANet — Anti-Interference 2D Vision-Guided Precision Sorting"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8501 7860

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; import cv2; print('OK')" || exit 1

# 默认入口：统一启动器
ENTRYPOINT ["python", "launch.py"]
CMD ["--mode", "demo", "--no-init"]
