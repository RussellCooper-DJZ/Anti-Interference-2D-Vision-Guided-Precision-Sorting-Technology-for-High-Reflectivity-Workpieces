# 使用带有 CUDA 支持的 PyTorch 基础镜像
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装系统依赖 (OpenCV 所需)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app

# 默认运行主管线演示模式
CMD ["python", "src/main_pipeline.py", "--mode", "demo"]
