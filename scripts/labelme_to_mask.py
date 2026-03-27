import json
import os
import cv2
import numpy as np
from tqdm import tqdm

def convert_labelme_to_mask(json_dir, image_dir, output_mask_dir, label_name='workpiece'):
    """
    将LabelMe标注的JSON文件转换为二值分割Mask图像。

    Args:
        json_dir (str): 包含LabelMe JSON文件的目录路径。
        image_dir (str): 包含原始图像的目录路径，用于获取图像尺寸。
        output_mask_dir (str): 输出Mask图像的目录路径。
        label_name (str): 在LabelMe中用于标注工件的标签名称。
    """
    os.makedirs(output_mask_dir, exist_ok=True)

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        print(f"Warning: No JSON files found in {json_dir}")
        return

    print(f"Converting {len(json_files)} LabelMe JSON files to masks...")

    for json_file in tqdm(json_files):
        json_path = os.path.join(json_dir, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading JSON file {json_path}: {e}")
            continue

        image_filename = data.get('imagePath')
        if not image_filename:
            print(f"Warning: 'imagePath' not found in {json_file}. Skipping.")
            continue

        original_image_path = os.path.join(image_dir, image_filename)
        if not os.path.exists(original_image_path):
            print(f"Warning: Original image {original_image_path} not found. Skipping {json_file}.")
            continue

        # 读取原始图像以获取尺寸
        original_image = cv2.imread(original_image_path)
        if original_image is None:
            print(f"Warning: Could not read image {original_image_path}. Skipping {json_file}.")
            continue

        height, width, _ = original_image.shape

        # 创建空白Mask图像
        mask = np.zeros((height, width), dtype=np.uint8)

        # 遍历所有形状
        for shape in data['shapes']:
            if shape['label'] == label_name and shape['shape_type'] == 'polygon':
                points = np.array(shape['points'], dtype=np.int32)
                # 填充多边形
                cv2.fillPoly(mask, [points], 255) # 255 for workpiece, 0 for background

        # 保存Mask图像
        mask_filename = os.path.splitext(image_filename)[0] + '.png'
        output_mask_path = os.path.join(output_mask_dir, mask_filename)
        cv2.imwrite(output_mask_path, mask)

    print("Conversion complete.")

if __name__ == '__main__':
    # 示例用法
    # 假设你的目录结构如下：
    # your_dataset/
    # ├── images/
    # │   ├── img1.jpg
    # │   └── img2.jpg
    # └── annotations/
    #     ├── img1.json
    #     └── img2.json

    # 请根据你的实际路径修改以下变量
    base_dataset_dir = '/home/ubuntu/your_dataset' # 你的数据集根目录
    json_annotations_dir = os.path.join(base_dataset_dir, 'annotations')
    original_images_dir = os.path.join(base_dataset_dir, 'images')
    output_masks_dir = os.path.join(base_dataset_dir, 'masks')

    # 确保安装了必要的库：pip install opencv-python numpy tqdm

    # 运行转换
    convert_labelme_to_mask(
        json_dir=json_annotations_dir,
        image_dir=original_images_dir,
        output_mask_dir=output_masks_dir,
        label_name='workpiece' # 你的工件标签名称
    )
    print(f"Masks saved to: {output_masks_dir}")

