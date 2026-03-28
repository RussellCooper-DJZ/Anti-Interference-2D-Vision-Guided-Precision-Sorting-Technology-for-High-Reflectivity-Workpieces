import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def inspect_data_pair(image_path, mask_path, output_path=None):
    """
    可视化单对图像和掩膜，检查对齐情况。
    """
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Error reading {image_path} or {mask_path}")
        return

    # 调整掩膜尺寸以匹配图像（如果不同）
    if image.shape[:2] != mask.shape[:2]:
        print(f"Warning: Size mismatch! Image: {image.shape[:2]}, Mask: {mask.shape[:2]}")
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 创建叠加图
    overlay = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2) # 绿色轮廓

    # 可视化
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Overlay (Verification)")
    plt.imshow(overlay)
    plt.axis('off')

    if output_path:
        plt.savefig(output_path)
        print(f"Inspection saved to {output_path}")
    else:
        plt.show()

def validate_dataset_consistency(image_dir, mask_dir):
    """
    校验整个数据集的一致性。
    """
    img_path = Path(image_dir)
    mask_path = Path(mask_dir)

    images = sorted([f.name for f in img_path.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg')])
    masks = sorted([f.name for f in mask_path.iterdir() if f.suffix.lower() in ('.jpg', '.png', '.jpeg')])

    print(f"Found {len(images)} images and {len(masks)} masks.")

    missing_masks = []
    for img in images:
        # 尝试匹配同名或不同后缀的掩膜
        base_name = os.path.splitext(img)[0]
        match_found = False
        for ext in ['.png', '.jpg', '.jpeg']:
            if (mask_path / (base_name + ext)).exists():
                match_found = True
                break
        if not match_found:
            missing_masks.append(img)

    if missing_masks:
        print(f"CRITICAL: {len(missing_masks)} images are missing corresponding masks!")
        print(f"First 5 missing: {missing_masks[:5]}")
    else:
        print("SUCCESS: All images have corresponding masks.")

if __name__ == '__main__':
    # 使用示例
    # validate_dataset_consistency('data/images', 'data/masks')
    # inspect_data_pair('data/images/test.jpg', 'data/masks/test.png', 'inspect.png')
    print("Dataset inspection tool ready.")
