"""
data/gc10_dataset.py — GC10-DET 数据集适配器
借鉴论文《高反光钢板视觉检测代码实现与开源项目指南》

GC10-DET 数据集：
  - 包含 10 种类型的表面缺陷
  - 图像分辨率：2048×1000
  - 适合训练和测试各种缺陷检测算法

本模块将其适配到 ageanet 的统一数据接口，支持目标检测格式（YOLO格式标注）。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["GC10Dataset", "GC10DefectType"]


class GC10DefectType:
    """GC10 数据集的 10 种缺陷类型"""

    PUNCHING_HOLE = "punching_hole"
    WELD_LINE = "weld_line"
    CRESCENT_GAP = "crescent_gap"
    WATER_SPOT = "water_spot"
    OIL_SPOT = "oil_spot"
    SILK_SPOT = "silk_spot"
    INCLUSION = "inclusion"
    ROLLED_PIT = "rolled_pit"
    CREASE = "crease"
    WAIST_FOLD = "waist_fold"

    ALL = [
        PUNCHING_HOLE, WELD_LINE, CRESCENT_GAP, WATER_SPOT, OIL_SPOT,
        SILK_SPOT, INCLUSION, ROLLED_PIT, CREASE, WAIST_FOLD,
    ]


class GC10Dataset(Dataset):
    """
    GC10-DET 数据集适配器。

    假设数据集目录结构::

        gc10/
            images/
                00001.jpg
                ...
            labels/
                00001.txt   # YOLO格式: class x_center y_center width height
                ...

    用法::

        from data.gc10_dataset import GC10Dataset
        dataset = GC10Dataset(root_dir="./data/gc10", img_size=512)
    """

    def __init__(
        self,
        root_dir: str,
        img_size: int = 512,
        defect_types: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        augmentation: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.transform = transform
        self.augmentation = augmentation
        self.defect_types = defect_types or GC10DefectType.ALL

        self.img_dir = self.root_dir / "images"
        self.lbl_dir = self.root_dir / "labels"

        self.samples: List[Tuple[str, Optional[str]]] = []
        self._load_samples()

    def _load_samples(self):
        if not self.img_dir.exists():
            raise RuntimeError(f"图像目录不存在: {self.img_dir}")

        for img_path in self.img_dir.glob("*.jpg"):
            lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
            self.samples.append((str(img_path), str(lbl_path) if lbl_path.exists() else None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_path, lbl_path = self.samples[idx]

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")

        h_orig, w_orig = img.shape[:2]

        # 调整尺寸（保持长宽比，填充）
        scale = min(self.img_size / w_orig, self.img_size / h_orig)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        # 填充到正方形
        pad_top = (self.img_size - new_h) // 2
        pad_bottom = self.img_size - new_h - pad_top
        pad_left = (self.img_size - new_w) // 2
        pad_right = self.img_size - new_w - pad_left
        img_padded = cv2.copyMakeBorder(
            img_resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(128, 128, 128),
        )

        # 读取标注
        bboxes = []
        labels = []
        if lbl_path:
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:])
                        # 转换到填充后的坐标系
                        xc = xc * scale + pad_left
                        yc = yc * scale + pad_top
                        w = w * scale
                        h = h * scale
                        bboxes.append([xc, yc, w, h])
                        labels.append(cls_id)

        # 数据增强
        if self.augmentation:
            img_padded = self._augment(img_padded)

        img_norm = img_padded.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1)

        target = {
            "bboxes": torch.tensor(bboxes, dtype=torch.float32) if bboxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros(0, dtype=torch.long),
            "img_path": img_path,
            "scale": scale,
            "pad": (pad_left, pad_top),
        }

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, target

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        return img
