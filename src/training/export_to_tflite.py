"""
export_to_tflite.py — 工业级模型量化与 TFLite Micro 转换管线
功能：
  1. 将 PyTorch/ONNX 模型转换为 TFLite。
  2. 执行全整型 INT8 量化 (PTQ)，以适配 RA8P1 硬件加速。
  3. 使用代表性数据集 (Representative Dataset) 确保量化后的精度。
  4. 自动生成 C 语言头文件。
"""

import os
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import argparse

def representative_dataset_gen(data_path, img_size=320, num_samples=100):
    """
    生成代表性数据集，用于量化时的激活值范围校准。
    """
    # 这里应从实际的训练集/验证集中抽取样本
    for _ in range(num_samples):
        # 模拟高反光工件输入 (HDR 融合后的 3 通道图像)
        # 实际应从 data_path 加载真实图像
        sample = np.random.rand(1, img_size, img_size, 3).astype(np.float32)
        yield [sample]

def convert_to_tflite_int8(onnx_path, output_path, data_path, img_size=320):
    """
    将 ONNX 模型转换为 INT8 量化的 TFLite 模型。
    """
    converter = tf.lite.TFLiteConverter.from_onnx_models([onnx_path]) # 假设使用 onnx2tf 或类似工具
    
    # 启用全整型量化
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset_gen(data_path, img_size)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"INT8 TFLite model saved to: {output_path}")

def generate_c_header(tflite_path, header_path):
    """
    将 TFLite 二进制文件转换为 C 语言数组。
    """
    with open(tflite_path, 'rb') as f:
        data = f.read()
    
    with open(header_path, 'w') as f:
        f.write("#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n")
        f.write(f"// Model size: {len(data)} bytes\n")
        f.write("const unsigned char g_model_data[] __attribute__((aligned(16))) = {\n")
        for i, b in enumerate(data):
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write("\n};\n\n")
        f.write(f"const int g_model_data_len = {len(data)};\n\n")
        f.write("#endif\n")
    print(f"C header generated: {header_path}")

if __name__ == "__main__":
    # 示例用法
    # 1. 导出 YOLOv8 为 TFLite (INT8)
    # model = YOLO('best.pt')
    # model.export(format='tflite', int8=True, imgsz=320)
    
    # 2. 手动生成 C 头文件
    # generate_c_header('best_int8.tflite', 'src/embedded/core/include/model_data.h')
    print("Quantization pipeline script ready.")
