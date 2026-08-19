"""
yolo_detector.py — 抗高光 YOLO 检测器
集成 Ultralytics YOLOv8，支持多曝光输入与亚像素位姿提取。
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import List, Dict, Optional
from .hdr_processing import exposure_fusion
from .localization_and_calibration import SubpixelLocalizer

class GlareAwareYOLODetector:
    """
    针对高反光工件优化的 YOLO 检测器。
    """
    def __init__(self, 
                 model_path: str, 
                 conf_threshold: float = 0.5,
                 use_hdr: bool = True):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(model_path).to(self.device)
        self.conf_threshold = conf_threshold
        self.use_hdr = use_hdr
        self.localizer = SubpixelLocalizer()

    def detect(self, images: List[np.ndarray]) -> List[Dict]:
        """
        输入图像序列（如多曝光图），输出亚像素位姿。
        Args:
            images: 图像列表 [Under, Normal, Over]
        """
        # 1. 图像预处理与融合
        if len(images) > 1 and self.use_hdr:
            input_img = exposure_fusion(images)
        else:
            input_img = images[0] if isinstance(images, list) else images

        # 2. YOLO 推理
        results = self.model.predict(input_img, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for res in results:
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            masks = res.masks.data.cpu().numpy() if res.masks is not None else None
            
            for i, box in enumerate(boxes):
                # 3. 亚像素精化 (使用局部区域进行边缘检测)
                x1, y1, x2, y2 = map(int, box)
                roi = input_img[y1:y2, x1:x2]
                
                # 如果有分割掩膜，使用掩膜精化
                if masks is not None:
                    roi_mask = masks[i][y1:y2, x1:x2]
                    subpixel_res = self.localizer.localize(roi_mask, intensity_image=cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                else:
                    # 降级方案：使用矩形框内的灰度中心
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    subpixel_res = self.localizer.localize(binary, intensity_image=gray_roi)

                if subpixel_res:
                    # 转换回全局坐标
                    item = subpixel_res[0]
                    cx, cy = item['centroid_px']
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'score': float(scores[i]),
                        'centroid_px': (x1 + cx, y1 + cy),
                        'orientation_deg': item['orientation_deg'],
                        'feature_type': item['feature_type']
                    })
        
        return detections

if __name__ == "__main__":
    # 模拟运行
    print("GlareAwareYOLODetector module loaded.")
