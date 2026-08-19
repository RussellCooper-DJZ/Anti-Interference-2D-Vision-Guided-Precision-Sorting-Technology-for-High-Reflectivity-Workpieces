"""
glare_simulator.py — 物理级光斑模拟与数据增强
用于模拟高反光工件在不同光照下的镜面反射（Specular Glare）。
"""

import cv2
import numpy as np
import random
from typing import List, Optional

class GlareSimulator:
    """
    合成光斑生成器，支持多种形状和物理衰减模型。
    """
    def __init__(self, 
                 max_glare_blobs: int = 3,
                 intensity_range: tuple = (180, 255),
                 size_range: tuple = (0.05, 0.2)): # 占图像宽度的比例
        self.max_glare_blobs = max_glare_blobs
        self.intensity_range = intensity_range
        self.size_range = size_range

    def apply(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        在图像上叠加随机光斑。
        Args:
            image: 输入图像 (H, W, 3)
            mask:  工件掩膜 (H, W)，若提供，则光斑主要生成在工件区域。
        """
        h, w = image.shape[:2]
        output = image.copy().astype(np.float32)
        
        num_blobs = random.randint(1, self.max_glare_blobs)
        
        for _ in range(num_blobs):
            # 随机位置
            if mask is not None and np.any(mask > 0):
                y_coords, x_coords = np.where(mask > 0)
                idx = random.randint(0, len(x_coords) - 1)
                cx, cy = x_coords[idx], y_coords[idx]
            else:
                cx, cy = random.randint(0, w), random.randint(0, h)
            
            # 随机大小与形状
            base_size = random.uniform(self.size_range[0], self.size_range[1]) * w
            axes = (int(base_size), int(base_size * random.uniform(0.3, 1.0)))
            angle = random.randint(0, 180)
            
            # 创建光斑核 (Gaussian Kernel)
            blob_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(blob_mask, (cx, cy), axes, angle, 0, 360, 1.0, -1)
            blob_mask = cv2.GaussianBlur(blob_mask, (0, 0), sigmaX=base_size/4)
            
            # 随机强度与颜色 (通常反光是白色的，但也可能带光源色)
            intensity = random.uniform(self.intensity_range[0], self.intensity_range[1])
            color = np.array([intensity, intensity, intensity], dtype=np.float32)
            
            # 叠加光斑 (Additive Blending with saturation)
            glare = blob_mask[:, :, np.newaxis] * color
            output += glare
            
        return np.clip(output, 0, 255).astype(np.uint8)

    def generate_multi_exposure_sim(self, image: np.ndarray) -> List[np.ndarray]:
        """
        模拟多重曝光序列 (Under, Normal, Over)
        """
        # 正常曝光 (已带模拟光斑)
        normal = self.apply(image)
        
        # 欠曝光 (模拟低增益/短曝光，光斑变小且变暗)
        under = cv2.convertScaleAbs(normal, alpha=0.4, beta=0)
        
        # 过曝光 (模拟高增益/长曝光，光斑扩大且背景过曝)
        over = cv2.convertScaleAbs(normal, alpha=1.8, beta=30)
        
        return [under, normal, over]

if __name__ == "__main__":
    # 测试代码
    test_img = np.zeros((512, 512, 3), dtype=np.uint8) + 100
    sim = GlareSimulator()
    result = sim.apply(test_img)
    cv2.imwrite("glare_test.png", result)
    print("Glare simulation test image saved.")
