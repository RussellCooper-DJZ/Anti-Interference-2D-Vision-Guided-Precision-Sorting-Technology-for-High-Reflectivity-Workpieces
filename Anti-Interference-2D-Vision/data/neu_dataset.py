"""
data/neu_dataset.py — NEU 表面缺陷数据库适配器
借鉴论文《高反光钢板视觉检测代码实现与开源项目指南》

NEU 表面缺陷数据库：
  - 东北大学发布的热轧钢带表面缺陷数据集
  - 包含 6 种常见缺陷： rolled-in scale, patches, crazing, pitted surface,
    inclusion, scratches
  - 图像尺寸：200x200 像素
  - 适合学术研究和算法验证

本模块将其适配到 ageanet 的统一数据接口。
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["NEUDataset", "NEUDefectType"]


class NEUDefectType:
    """NEU 数据集的 6 种缺陷类型"""

    ROLLED_IN_SCALE = "rolled-in_scale"
    PATCHES = "patches"
    CRAZING = "crazing"
    PITTED_SURFACE = "pitted_surface"
    INCLUSION = "inclusion"
    SCRATCHES = "scratches"

    ALL = [
        ROLLED_IN_SCALE,
        PATCHES,
        CRAZING,
        PITTED_SURFACE,
        INCLUSION,
        SCRATCHES,
    ]


class NEUDataset(Dataset):
    """
    NEU 表面缺陷数据集适配器。

    假设数据集目录结构::

        neu_dataset/
            rolled-in_scale/
                001.bmp
                002.bmp
                ...
            patches/
                001.bmp
                ...
            ...

    用法::

        from data.neu_dataset import NEUDataset
        from torch.utils.data import DataLoader

        dataset = NEUDataset(root_dir="./data/neu_dataset", img_size=512)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
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

        self.defect_types = defect_types or NEUDefectType.ALL
        self.samples: List[Tuple[str, int]] = []  # (path, label)

        self._load_samples()

    def _load_samples(self):
        """扫描数据集目录，收集样本路径"""
        for label_idx, defect_type in enumerate(self.defect_types):
            type_dir = self.root_dir / defect_type
            if not type_dir.exists():
                continue
            for img_path in type_dir.glob("*.bmp"):
                self.samples.append((str(img_path), label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"未在 {self.root_dir} 中找到 NEU 数据集图像")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]

        # 读取图像（NEU 为 BMP 灰度图）
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")

        # 转为 3 通道
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 调整尺寸
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 数据增强
        if self.augmentation:
            img = self._augment(img)

        # 归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0

        # 转为 Tensor (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label_tensor

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """简单的在线数据增强"""
        # 随机水平翻转
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
        # 随机垂直翻转
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
        # 随机旋转 90°
        if random.random() > 0.7:
            k = random.choice([1, 2, 3])
            img = np.rot90(img, k).copy()
        return img

    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重（用于处理类别不平衡）"""
        labels = [label for _, label in self.samples]
        counts = np.bincount(labels, minlength=len(self.defect_types))
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(weights)
        return torch.from_numpy(weights).float()
