import tensorflow as tf
import numpy as np

def convert_to_tflite_micro(keras_model_path, output_path, representative_dataset):
    """
    将 Keras 模型转换为 TFLite Micro 格式，并进行 8-bit 量化。
    """
    # 1. 加载模型
    model = tf.keras.models.load_data(keras_model_path)
    
    # 2. 创建转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # 3. 设置量化参数
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # 4. 转换模型
    tflite_model = converter.convert()
    
    # 5. 保存模型
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model converted and saved to {output_path}")

def generate_c_array(tflite_path, c_header_path):
    """
    将 TFLite 模型转换为 C 语言数组，以便在 RA8P1 上使用。
    """
    with open(tflite_path, 'rb') as f:
        tflite_content = f.read()
    
    hex_lines = [', '.join([f'0x{b:02x}' for b in tflite_content[i:i+12]]) for i in range(0, len(tflite_content), 12)]
    
    with open(c_header_path, 'w') as f:
        f.write('#ifndef MODEL_DATA_H\n#define MODEL_DATA_H\n\n')
        f.write(f'const unsigned char g_model_data[] = {{\n  ')
        f.write(',\n  '.join(hex_lines))
        f.write('\n};\n')
        f.write(f'const int g_model_data_len = {len(tflite_content)};\n\n')
        f.write('#endif // MODEL_DATA_H\n')
    
    print(f"C header file generated at {c_header_path}")

if __name__ == "__main__":
    # 示例用法说明
    print("TFLM Adapter Module Loaded.")
    # 假设已有训练好的轻量化模型 'lightweight_unet.h5'
    # representative_data = ... # 提供代表性数据集用于量化
    # convert_to_tflite_micro('lightweight_unet.h5', 'model.tflite', representative_data)
    # generate_c_array('model.tflite', 'model_data.h')
