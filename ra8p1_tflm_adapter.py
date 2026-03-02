"""
ra8p1_tflm_adapter.py — RA8P1 嵌入式部署适配器

功能：
  1. PyTorch → ONNX → TFLite 模型转换
  2. INT8 量化 (Post-Training Quantization)
  3. TFLite 模型验证
  4. C 头文件生成 (用于嵌入式部署)
  5. 代表性数据集生成 (用于量化校准)
"""

import os
import sys
import struct
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. PyTorch → ONNX 导出
# ===========================================================================

def export_to_onnx(model, output_path, input_size=(1, 3, 256, 256),
                    opset_version=11):
    """
    将 PyTorch 模型导出为 ONNX 格式。

    参数:
        model:         PyTorch 模型 (需在 CPU 上)
        output_path:   ONNX 文件保存路径
        input_size:    输入张量尺寸
        opset_version: ONNX opset 版本
    """
    import torch

    model.eval()
    model = model.cpu()
    dummy_input = torch.randn(*input_size)

    # 获取输出键名
    with torch.no_grad():
        sample_output = model(dummy_input)

    output_names = list(sample_output.keys()) if isinstance(sample_output, dict) else ['output']

    # 对于 dict 输出的模型，需要包装
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            out = self.model(x)
            if isinstance(out, dict):
                return out['seg']  # 嵌入式只需要分割输出
            return out

    wrapped = ModelWrapper(model)

    torch.onnx.export(
        wrapped, dummy_input, output_path,
        input_names=['input'],
        output_names=['seg_output'],
        opset_version=opset_version,
        dynamic_axes={'input': {0: 'batch'}, 'seg_output': {0: 'batch'}},
    )

    logger.info(f"ONNX 模型导出成功: {output_path}")
    logger.info(f"  输入尺寸: {input_size}")
    logger.info(f"  文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")


# ===========================================================================
# 2. ONNX → TFLite 转换 (含 INT8 量化)
# ===========================================================================

def convert_onnx_to_tflite(onnx_path, output_path, quantize=True,
                            representative_data=None, input_size=(256, 256)):
    """
    将 ONNX 模型转换为 TFLite 格式，可选 INT8 量化。

    参数:
        onnx_path:           ONNX 模型路径
        output_path:         TFLite 输出路径
        quantize:            是否进行 INT8 量化
        representative_data: 代表性数据集 (numpy 数组列表)
        input_size:          输入图像尺寸 (H, W)
    """
    try:
        import tensorflow as tf
    except ImportError:
        logger.error("需要安装 tensorflow: pip install tensorflow")
        return False

    try:
        import onnx
        from onnx_tf.backend import prepare
    except ImportError:
        logger.error("需要安装 onnx 和 onnx-tf: pip install onnx onnx-tf")
        logger.info("替代方案: 使用 onnx2tf 工具")
        return _convert_via_onnx2tf(onnx_path, output_path, quantize, representative_data)

    # ONNX → TF SavedModel
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    saved_model_dir = str(Path(output_path).parent / 'saved_model_temp')
    tf_rep.export_graph(saved_model_dir)

    # TF SavedModel → TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if representative_data is not None:
            def representative_dataset():
                for data in representative_data:
                    yield [data.astype(np.float32)]

            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
            logger.info("使用 INT8 全量化")
        else:
            logger.info("使用动态范围量化 (无代表性数据)")

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    logger.info(f"TFLite 模型保存: {output_path}")
    logger.info(f"  文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")

    # 清理临时文件
    import shutil
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)

    return True


def _convert_via_onnx2tf(onnx_path, output_path, quantize, representative_data):
    """使用 onnx2tf 工具进行转换的备选方案。"""
    logger.info("尝试使用 onnx2tf 进行转换...")
    try:
        import subprocess
        cmd = f"onnx2tf -i {onnx_path} -o {Path(output_path).parent / 'tf_model'}"
        subprocess.run(cmd.split(), check=True)
        logger.info("onnx2tf 转换成功")
        return True
    except Exception as e:
        logger.error(f"onnx2tf 转换失败: {e}")
        logger.info("请手动使用以下工具进行转换:")
        logger.info("  pip install onnx2tf")
        logger.info(f"  onnx2tf -i {onnx_path} -o output_dir")
        return False


# ===========================================================================
# 3. 代表性数据集生成
# ===========================================================================

def generate_representative_dataset(count=100, input_size=(256, 256)):
    """
    生成代表性数据集用于 INT8 量化校准。

    使用合成金属工件数据，确保量化后模型在高反光场景下的精度。

    参数:
        count:      样本数量
        input_size: 输入尺寸 (H, W)

    返回:
        data_list: numpy 数组列表，每个形状为 (1, 3, H, W)
    """
    from data_augmentation import SyntheticMetalWorkpieceGenerator

    generator = SyntheticMetalWorkpieceGenerator(
        img_size=input_size[0], min_objects=1, max_objects=3
    )

    data_list = []
    for _ in range(count):
        sample = generator.generate()
        img = sample['image'].astype(np.float32) / 255.0
        img_tensor = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, 3, H, W)
        data_list.append(img_tensor)

    logger.info(f"生成 {count} 个代表性样本 (尺寸: {input_size})")
    return data_list


# ===========================================================================
# 4. TFLite 模型验证
# ===========================================================================

def validate_tflite_model(tflite_path, test_images=None):
    """
    验证 TFLite 模型的推理结果。

    参数:
        tflite_path:  TFLite 模型路径
        test_images:  测试图像列表 (numpy 数组)

    返回:
        validation_result: dict
    """
    try:
        import tensorflow as tf
    except ImportError:
        logger.error("需要安装 tensorflow")
        return None

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    logger.info(f"TFLite 模型信息:")
    logger.info(f"  输入: {input_details[0]['shape']} ({input_details[0]['dtype']})")
    logger.info(f"  输出: {output_details[0]['shape']} ({output_details[0]['dtype']})")

    if test_images is None:
        test_images = generate_representative_dataset(5)

    results = []
    for img in test_images[:5]:
        input_data = img.astype(input_details[0]['dtype'])

        # INT8 量化模型需要量化输入
        if input_details[0]['dtype'] == np.int8:
            scale = input_details[0]['quantization'][0]
            zero_point = input_details[0]['quantization'][1]
            input_data = (img / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        results.append(output)

    logger.info(f"验证完成: {len(results)} 个样本")
    return {
        'input_shape': input_details[0]['shape'].tolist(),
        'output_shape': output_details[0]['shape'].tolist(),
        'input_dtype': str(input_details[0]['dtype']),
        'output_dtype': str(output_details[0]['dtype']),
        'num_tested': len(results),
    }


# ===========================================================================
# 5. C 头文件生成
# ===========================================================================

def generate_c_array(tflite_path, c_header_path, array_name='g_model_data'):
    """
    将 TFLite 模型转换为 C 头文件。

    参数:
        tflite_path:   TFLite 模型路径
        c_header_path: C 头文件输出路径
        array_name:    C 数组名称
    """
    with open(tflite_path, 'rb') as f:
        model_data = f.read()

    model_size = len(model_data)
    guard_name = Path(c_header_path).stem.upper() + '_H'

    with open(c_header_path, 'w') as f:
        f.write(f'/**\n')
        f.write(f' * Auto-generated TFLite model data for RA8P1 deployment.\n')
        f.write(f' * Model size: {model_size} bytes ({model_size / 1024:.1f} KB)\n')
        f.write(f' * Source: {Path(tflite_path).name}\n')
        f.write(f' */\n\n')
        f.write(f'#ifndef {guard_name}\n')
        f.write(f'#define {guard_name}\n\n')
        f.write(f'#include <stdint.h>\n\n')

        # 对齐到 16 字节 (ARM Cortex-M85 优化)
        f.write(f'/* Aligned to 16 bytes for Cortex-M85 Helium SIMD */\n')
        f.write(f'__attribute__((aligned(16)))\n')
        f.write(f'const uint8_t {array_name}[] = {{\n')

        # 每行 16 字节
        for i in range(0, model_size, 16):
            chunk = model_data[i:i + 16]
            hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
            f.write(f'    {hex_str},\n')

        f.write(f'}};\n\n')
        f.write(f'const uint32_t {array_name}_len = {model_size};\n\n')
        f.write(f'#endif /* {guard_name} */\n')

    logger.info(f"C 头文件生成: {c_header_path}")
    logger.info(f"  模型大小: {model_size} bytes ({model_size / 1024:.1f} KB)")
    logger.info(f"  数组名: {array_name}")


# ===========================================================================
# 6. 完整导出流程
# ===========================================================================

def full_export_pipeline(model, output_dir, input_size=(256, 256),
                          quantize=True, num_calibration=100):
    """
    完整的嵌入式部署导出流程。

    步骤:
    1. PyTorch → ONNX
    2. 生成代表性数据集
    3. ONNX → TFLite (INT8 量化)
    4. 验证 TFLite 模型
    5. 生成 C 头文件

    参数:
        model:             PyTorch 模型
        output_dir:        输出目录
        input_size:        输入尺寸 (H, W)
        quantize:          是否量化
        num_calibration:   量化校准样本数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("嵌入式部署导出流程")
    logger.info("=" * 60)

    # 1. ONNX 导出
    onnx_path = output_dir / 'ageanet_lite.onnx'
    export_to_onnx(model, str(onnx_path),
                    input_size=(1, 3, *input_size))

    # 2. 生成代表性数据集
    if quantize:
        logger.info("\n生成量化校准数据...")
        rep_data = generate_representative_dataset(num_calibration, input_size)
    else:
        rep_data = None

    # 3. TFLite 转换
    tflite_path = output_dir / 'ageanet_lite.tflite'
    success = convert_onnx_to_tflite(
        str(onnx_path), str(tflite_path),
        quantize=quantize, representative_data=rep_data,
    )

    if success and tflite_path.exists():
        # 4. 验证
        logger.info("\n验证 TFLite 模型...")
        validate_tflite_model(str(tflite_path))

        # 5. C 头文件
        c_header_path = output_dir / 'model_data.h'
        generate_c_array(str(tflite_path), str(c_header_path))
    else:
        logger.warning("TFLite 转换未完成，跳过验证和 C 头文件生成")
        logger.info("请手动完成 ONNX → TFLite 转换后运行:")
        logger.info(f"  python ra8p1_tflm_adapter.py --generate-header {tflite_path}")

    logger.info("\n导出完成！输出文件:")
    for f in output_dir.iterdir():
        logger.info(f"  {f.name}: {f.stat().st_size / 1024:.1f} KB")


# ===========================================================================
# 入口点
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='RA8P1 嵌入式部署工具')
    parser.add_argument('--export', action='store_true', help='执行完整导出流程')
    parser.add_argument('--model-path', type=str, default=None, help='模型权重路径')
    parser.add_argument('--output-dir', type=str, default='./export', help='输出目录')
    parser.add_argument('--input-size', type=int, default=256, help='输入尺寸')
    parser.add_argument('--no-quantize', action='store_true', help='不进行量化')
    parser.add_argument('--generate-header', type=str, default=None,
                        help='从 TFLite 文件生成 C 头文件')
    args = parser.parse_args()

    if args.generate_header:
        c_path = Path(args.generate_header).with_suffix('.h')
        generate_c_array(args.generate_header, str(c_path))
    elif args.export:
        import torch
        from feature_extraction import AGEANetLite

        model = AGEANetLite(in_channels=3, out_channels=1)
        if args.model_path:
            checkpoint = torch.load(args.model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

        full_export_pipeline(
            model, args.output_dir,
            input_size=(args.input_size, args.input_size),
            quantize=not args.no_quantize,
        )
    else:
        # 快速测试
        print("RA8P1 TFLM 适配器已加载。")
        print("用法:")
        print("  --export          执行完整导出流程")
        print("  --generate-header 从 TFLite 生成 C 头文件")
        print("\n测试 ONNX 导出...")

        import torch
        from feature_extraction import AGEANetLite

        model = AGEANetLite(in_channels=3, out_channels=1)
        onnx_path = '/tmp/test_export.onnx'
        export_to_onnx(model, onnx_path, input_size=(1, 3, 256, 256))
        print(f"ONNX 导出成功: {os.path.getsize(onnx_path) / 1024:.1f} KB")
