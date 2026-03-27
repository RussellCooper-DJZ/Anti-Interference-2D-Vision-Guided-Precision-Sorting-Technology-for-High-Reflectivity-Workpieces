"""
real_world_dataloader.py — 船舶场景数据加载器
:Author: RussellCooper

支持两种数据源：
  1. SynthShipDataset  — 从 synth_dataset_generator 生成的合成数据集目录加载
  2. RealShipDataset   — 真实标注数据集（images/ + masks/ + edges/ 目录结构）
  3. MixedShipDataset  — 合成 + 真实数据混合（可调比例）

目录结构约定::

    dataset_root/
      images/   *.png / *.jpg
      masks/    *.png（与 images/ 同名，二值掩膜 0/255）
      edges/    *.png（与 images/ 同名，边缘掩膜 0/255，可选）

用法::

    from real_world_dataloader import build_dataloaders
    train_loader, val_loader = build_dataloaders(
        synth_dir='./datasets/synth_ship',
        real_dir=None,           # 若有真实数据则填写路径
        img_size=512,
        batch_size=8,
        num_workers=4,
    )
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_augmentation import ShipHullAugPipeline, generate_edge_from_mask


# ============================================================
# 1. 合成数据集
# ============================================================

class SynthShipDataset(Dataset):
    """
    从 synth_dataset_generator 生成的合成数据集加载器。

    Args:
        root:      数据集根目录（含 images/ masks/ edges/ 子目录）
        img_size:  输入分辨率（正方形）
        augment:   是否启用数据增强
        aug_p:     增强概率
        split:     'train' 或 'val'
        val_ratio: 验证集比例
        seed:      划分随机种子
    """

    def __init__(self, root: str, img_size: int = 512,
                 augment: bool = True, aug_p: float = 0.85,
                 split: str = 'train', val_ratio: float = 0.1,
                 seed: int = 42):
        self.root     = Path(root)
        self.img_size = img_size
        self.augment  = augment and (split == 'train')
        self.aug      = ShipHullAugPipeline(p=aug_p)

        img_dir = self.root / 'images'
        if not img_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {img_dir}")

        all_names = sorted([
            p.stem for p in img_dir.iterdir()
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ])
        if not all_names:
            raise ValueError(f"数据集为空: {img_dir}")

        # 按 seed 划分训练/验证集
        rng = random.Random(seed)
        shuffled = all_names.copy()
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        if split == 'val':
            self.names = shuffled[:n_val]
        else:
            self.names = shuffled[n_val:]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        name = self.names[idx]

        # 加载图像
        img_path = self._find_file(self.root / 'images', name)
        image = cv2.imread(str(img_path))
        if image is None:
            raise IOError(f"无法读取图像: {img_path}")

        # 加载掩膜
        mask_path = self._find_file(self.root / 'masks', name, required=True)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # 加载边缘（可选，若不存在则从掩膜生成）
        edge_path = self._find_file(self.root / 'edges', name, required=False)
        if edge_path and edge_path.exists():
            edge = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
        else:
            edge = generate_edge_from_mask(mask)

        # 缩放到目标尺寸
        image = cv2.resize(image, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST)
        edge  = cv2.resize(edge,  (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST)

        # 数据增强
        if self.augment:
            image, mask, edge = self.aug(image, mask, edge)

        return {
            'image': self._to_tensor(image),
            'mask':  self._mask_to_tensor(mask),
            'edge':  self._mask_to_tensor(edge),
            'name':  name,
        }

    @staticmethod
    def _find_file(directory: Path, stem: str,
                   required: bool = False) -> Optional[Path]:
        for ext in ('.png', '.jpg', '.jpeg'):
            p = directory / f"{stem}{ext}"
            if p.exists():
                return p
        if required:
            # 返回 .png 路径（即使不存在，调用方处理）
            return directory / f"{stem}.png"
        return None

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        """BGR uint8 (H,W,3) → float32 (3,H,W) [0,1]"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    @staticmethod
    def _mask_to_tensor(mask: np.ndarray) -> torch.Tensor:
        """uint8 (H,W) → float32 (1,H,W) [0,1]"""
        return torch.from_numpy(mask).unsqueeze(0).float() / 255.0


# ============================================================
# 2. 真实数据集（相同目录结构）
# ============================================================

class RealShipDataset(SynthShipDataset):
    """
    真实标注数据集加载器（与 SynthShipDataset 相同接口）。
    真实数据通常增强强度较低，val_ratio 可设为 0.2。
    """

    def __init__(self, root: str, img_size: int = 512,
                 augment: bool = True, aug_p: float = 0.7,
                 split: str = 'train', val_ratio: float = 0.2,
                 seed: int = 42):
        super().__init__(root, img_size, augment, aug_p,
                         split, val_ratio, seed)


# ============================================================
# 3. 混合数据集
# ============================================================

class MixedShipDataset(Dataset):
    """
    合成 + 真实数据混合数据集。

    每个 epoch 从合成数据集和真实数据集中按比例采样，
    真实数据因为稀缺会被重复采样（oversampling）。

    Args:
        synth_dataset: 合成数据集实例
        real_dataset:  真实数据集实例（可为 None）
        real_ratio:    真实数据占比（0~1），0 = 纯合成
    """

    def __init__(self, synth_dataset: Dataset,
                 real_dataset: Optional[Dataset] = None,
                 real_ratio: float = 0.3):
        self.synth = synth_dataset
        self.real  = real_dataset
        self.real_ratio = real_ratio if real_dataset is not None else 0.0

        n_synth = len(synth_dataset)
        if real_dataset is not None and len(real_dataset) > 0:
            n_real_target = int(n_synth * real_ratio / (1 - real_ratio + 1e-8))
            n_real_target = min(n_real_target, n_synth * 3)  # 最多 3 倍过采样
            # 构建索引：(source, idx)
            synth_indices = [('synth', i) for i in range(n_synth)]
            # 循环重复真实数据直到达到目标数量
            real_indices = [
                ('real', i % len(real_dataset))
                for i in range(n_real_target)
            ]
            self.indices = synth_indices + real_indices
            random.shuffle(self.indices)
        else:
            self.indices = [('synth', i) for i in range(n_synth)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        source, data_idx = self.indices[idx]
        if source == 'synth':
            return self.synth[data_idx]
        else:
            return self.real[data_idx]


# ============================================================
# 4. 在线合成数据集（无需预先生成，每次动态合成）
# ============================================================

class OnlineSynthDataset(Dataset):
    """
    在线合成数据集：每次 __getitem__ 动态调用 synthesize_one_sample。

    优点：无需预先生成大量文件，每个 epoch 样本都不同（无限多样性）。
    缺点：CPU 占用较高，建议配合 num_workers >= 4。

    Args:
        epoch_size: 每个 epoch 的虚拟样本数
        img_size:   图像尺寸
        augment:    是否叠加额外数据增强
    """

    def __init__(self, epoch_size: int = 1000, img_size: int = 512,
                 augment: bool = True):
        self.epoch_size = epoch_size
        self.img_size   = img_size
        self.augment    = augment
        self.aug        = ShipHullAugPipeline(p=0.9)

        # 延迟导入（避免循环依赖）
        from synth_dataset_generator import synthesize_one_sample
        self._synthesize = synthesize_one_sample

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._synthesize(h=self.img_size, w=self.img_size)
        image, mask, edge = sample['image'], sample['mask'], sample['edge']

        if self.augment:
            image, mask, edge = self.aug(image, mask, edge)

        return {
            'image': SynthShipDataset._to_tensor(image),
            'mask':  SynthShipDataset._mask_to_tensor(mask),
            'edge':  SynthShipDataset._mask_to_tensor(edge),
            'name':  f'online_{idx:06d}',
        }


# ============================================================
# 5. 工厂函数
# ============================================================

def build_dataloaders(
    synth_dir: Optional[str] = None,
    real_dir:  Optional[str] = None,
    img_size:  int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    real_ratio: float = 0.3,
    online_epoch_size: int = 2000,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    构建训练/验证 DataLoader。

    优先级：
      1. 若 synth_dir 存在 → 使用磁盘合成数据集
      2. 若 synth_dir 为 None → 使用在线合成（动态生成）
      3. 若 real_dir 存在 → 与合成数据混合

    Args:
        synth_dir:          合成数据集根目录（None=在线合成）
        real_dir:           真实数据集根目录（None=不使用）
        img_size:           图像尺寸
        batch_size:         批次大小
        num_workers:        DataLoader 工作进程数
        real_ratio:         真实数据占比（MixedShipDataset 用）
        online_epoch_size:  在线合成模式下每 epoch 样本数
        val_ratio:          验证集比例
        seed:               随机种子

    Returns:
        (train_loader, val_loader)
    """
    # --- 构建训练集 ---
    if synth_dir and Path(synth_dir).exists():
        train_synth = SynthShipDataset(
            synth_dir, img_size=img_size, augment=True,
            split='train', val_ratio=val_ratio, seed=seed,
        )
        val_synth = SynthShipDataset(
            synth_dir, img_size=img_size, augment=False,
            split='val', val_ratio=val_ratio, seed=seed,
        )
    else:
        print("[dataloader] synth_dir 不存在，使用在线合成模式")
        train_synth = OnlineSynthDataset(
            epoch_size=online_epoch_size, img_size=img_size, augment=True
        )
        val_synth = OnlineSynthDataset(
            epoch_size=max(100, online_epoch_size // 10),
            img_size=img_size, augment=False
        )

    # --- 混合真实数据 ---
    if real_dir and Path(real_dir).exists():
        train_real = RealShipDataset(
            real_dir, img_size=img_size, augment=True,
            split='train', val_ratio=0.2, seed=seed,
        )
        val_real = RealShipDataset(
            real_dir, img_size=img_size, augment=False,
            split='val', val_ratio=0.2, seed=seed,
        )
        train_dataset = MixedShipDataset(train_synth, train_real, real_ratio)
        val_dataset   = MixedShipDataset(val_synth,   val_real,   real_ratio * 0.5)
        print(f"[dataloader] 混合数据集: 合成 {len(train_synth)} + 真实 {len(train_real)}")
    else:
        train_dataset = train_synth
        val_dataset   = val_synth

    print(f"[dataloader] 训练集: {len(train_dataset)} 样本")
    print(f"[dataloader] 验证集: {len(val_dataset)} 样本")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


# ============================================================
# 命令行验证
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--synth_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--online_size", type=int, default=20)
    args = parser.parse_args()

    train_loader, val_loader = build_dataloaders(
        synth_dir=args.synth_dir,
        batch_size=args.batch_size,
        num_workers=0,
        online_epoch_size=args.online_size,
    )

    print("\n--- 验证训练批次 ---")
    for batch in train_loader:
        print(f"  image: {tuple(batch['image'].shape)}, dtype={batch['image'].dtype}")
        print(f"  mask:  {tuple(batch['mask'].shape)},  min={batch['mask'].min():.2f}, max={batch['mask'].max():.2f}")
        print(f"  edge:  {tuple(batch['edge'].shape)},  min={batch['edge'].min():.2f}, max={batch['edge'].max():.2f}")
        break

    print("\n--- 验证验证批次 ---")
    for batch in val_loader:
        print(f"  image: {tuple(batch['image'].shape)}")
        break

    print("\n数据加载器验证通过！")
