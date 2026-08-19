"""
inference_engine.py — 统一视觉推理引擎
支持 AGEANet (U-Net) 和 YOLOv8，具备抗高光处理能力。
"""

import torch
import numpy as np
import cv2
from typing import Tuple, Dict, Optional, List
from .feature_extraction import AGEANet, AGEANetLite
from ultralytics import YOLO

class UnifiedInferenceEngine:
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = config.get('type', 'ageanet').lower()
        self.img_size = config.get('img_size', 512)
        
        if self.model_type == 'ageanet':
            self.model = AGEANet(in_channels=3).to(self.device)
            if config.get('path'):
                self.model.load_state_dict(torch.load(config['path'], map_location=self.device), strict=False)
            self.model.eval()
        elif self.model_type == 'yolo':
            self.model = YOLO(config['path']).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    @torch.no_grad()
    def infer(self, image_bgr: np.ndarray) -> Dict:
        """执行推理并返回统一格式的结果"""
        h_orig, w_orig = image_bgr.shape[:2]
        
        if self.model_type == 'ageanet':
            # AGEANet 推理逻辑
            img_resized = cv2.resize(image_bgr, (self.img_size, self.img_size))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(self.device)
            
            outputs = self.model(tensor)
            seg_prob = outputs['seg'][0, 0].cpu().numpy()
            edge_prob = outputs['edge'][0, 0].cpu().numpy()
            
            seg_mask = (seg_prob > 0.5).astype(np.uint8) * 255
            edge_mask = (edge_prob > 0.3).astype(np.uint8) * 255
            
            return {
                'seg_mask': cv2.resize(seg_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST),
                'edge_mask': cv2.resize(edge_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST),
                'detections': [] # AGEANet 通常后续接定位器
            }
            
        elif self.model_type == 'yolo':
            # YOLO 推理逻辑
            results = self.model.predict(image_bgr, imgsz=self.img_size, conf=self.config.get('conf_threshold', 0.5), verbose=False)
            res = results[0]
            
            detections = []
            if res.boxes:
                for i, box in enumerate(res.boxes.xyxy.cpu().numpy()):
                    detections.append({
                        'bbox': box.tolist(),
                        'score': float(res.boxes.conf[i]),
                        'class': int(res.boxes.cls[i])
                    })
            
            seg_mask = None
            if res.masks:
                seg_mask = res.masks.data[0].cpu().numpy()
                seg_mask = (cv2.resize(seg_mask, (w_orig, h_orig)) > 0.5).astype(np.uint8) * 255
                
            return {
                'seg_mask': seg_mask,
                'edge_mask': None,
                'detections': detections
            }
