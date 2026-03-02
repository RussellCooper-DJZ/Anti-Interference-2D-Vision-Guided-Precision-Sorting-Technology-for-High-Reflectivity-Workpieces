"""
train.py — AGEANet 完整训练流程

功能：
  1. 组合损失函数：Dice Loss + Focal Loss + Boundary Loss + 高光一致性损失
  2. 数据集加载器：支持真实数据和合成数据混合训练
  3. 完整训练循环：学习率调度、早停、模型保存、TensorBoard 日志
  4. 验证与评估：IoU、Dice、边缘精度等指标

用法：
  python train.py --data_dir ./datasets/metal_workpieces --epochs 100 --batch_size 8
  python train.py --synthetic_only --syn_count 2000 --epochs 50   # 纯合成数据冷启动
"""

import os
import sys
import argparse
import time
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

from feature_extraction import AGEANet, AGEANetLite, get_model_info
from data_augmentation import (
    HighReflectivityAugPipeline,
    SyntheticMetalWorkpieceGenerator,
    generate_edge_from_mask,
    generate_sobel_edge,
)

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. 损失函数
# ===========================================================================

class DiceLoss(nn.Module):
    """Dice Loss — 处理类别不平衡，适合分割任务。"""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / \
               (pred_flat.sum() + target_flat.sum() + self.smooth)


class FocalLoss(nn.Module):
    """Focal Loss — 聚焦于难分类样本，适合边缘像素。"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class BoundaryLoss(nn.Module):
    """
    边界损失 — 专门优化边缘精度。

    原理：计算预测边缘与真实边缘之间的距离加权损失。
    对于高反光工件，边缘精度是最关键的指标。
    """

    def __init__(self, theta=3):
        super().__init__()
        self.theta = theta

    def forward(self, pred, target):
        # 使用 Sobel 算子提取边缘
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=pred.device).view(1, 1, 3, 3)

        # 提取预测和目标的边缘
        pred_edge_x = F.conv2d(pred, sobel_x, padding=1)
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-8)

        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-8)

        # 边界区域加权
        boundary_weight = torch.exp(-target_edge / self.theta)
        loss = F.mse_loss(pred_edge * (1 - boundary_weight),
                          target_edge * (1 - boundary_weight))
        return loss


class SpecularConsistencyLoss(nn.Module):
    """
    高光一致性损失 — 确保高光区域的分割不受干扰。

    原理：在检测到的高光区域内，分割结果应与非高光区域保持一致。
    """

    def __init__(self, weight=0.5):
        super().__init__()
        self.weight = weight

    def forward(self, seg_pred, glare_map, target):
        # 高光区域的分割损失 (加权)
        glare_weight = 1.0 + glare_map * self.weight
        loss = F.binary_cross_entropy(seg_pred, target, weight=glare_weight)
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失函数 — 整合所有损失项。

    L_total = λ_dice * L_dice + λ_focal * L_focal + λ_boundary * L_boundary
            + λ_edge * L_edge + λ_specular * L_specular
    """

    def __init__(self, dice_w=1.0, focal_w=1.0, boundary_w=0.5,
                 edge_w=0.5, specular_w=0.3):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.boundary_loss = BoundaryLoss()
        self.specular_loss = SpecularConsistencyLoss()
        self.edge_bce = nn.BCELoss()

        self.dice_w = dice_w
        self.focal_w = focal_w
        self.boundary_w = boundary_w
        self.edge_w = edge_w
        self.specular_w = specular_w

    def forward(self, outputs, targets):
        """
        参数:
            outputs: dict {'seg', 'edge', 'glare'}
            targets: dict {'mask', 'edge_mask'}
        """
        seg_pred = outputs['seg']
        edge_pred = outputs['edge']
        seg_target = targets['mask']
        edge_target = targets['edge_mask']

        # 主分割损失
        l_dice = self.dice_loss(seg_pred, seg_target)
        l_focal = self.focal_loss(seg_pred, seg_target)
        l_boundary = self.boundary_loss(seg_pred, seg_target)

        # 边缘损失
        l_edge = self.edge_bce(edge_pred, edge_target)

        # 高光一致性损失 (仅标准模型有 glare 输出)
        l_specular = torch.tensor(0.0, device=seg_pred.device)
        if 'glare' in outputs:
            l_specular = self.specular_loss(seg_pred, outputs['glare'], seg_target)

        total = (self.dice_w * l_dice +
                 self.focal_w * l_focal +
                 self.boundary_w * l_boundary +
                 self.edge_w * l_edge +
                 self.specular_w * l_specular)

        loss_dict = {
            'total': total.item(),
            'dice': l_dice.item(),
            'focal': l_focal.item(),
            'boundary': l_boundary.item(),
            'edge': l_edge.item(),
            'specular': l_specular.item(),
        }

        return total, loss_dict


# ===========================================================================
# 2. 数据集
# ===========================================================================

class MetalWorkpieceDataset(Dataset):
    """
    金属工件分割数据集。

    目录结构:
        data_dir/
        ├── images/     # 原始图像 (.png, .jpg, .bmp)
        ├── masks/      # 分割掩膜 (同名，单通道，0/255)
        └── edges/      # 边缘掩膜 (可选，自动生成)
    """

    def __init__(self, data_dir, mode='train', img_size=256):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.img_size = img_size
        self.augment = HighReflectivityAugPipeline(mode=mode, img_size=img_size)

        # 收集图像文件
        img_dir = self.data_dir / 'images'
        self.image_files = sorted([
            f for f in img_dir.iterdir()
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        ]) if img_dir.exists() else []

        if len(self.image_files) == 0:
            logger.warning(f"数据目录 {img_dir} 中未找到图像文件！")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.data_dir / 'masks' / img_path.name

        # 读取图像
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")

        # 读取掩膜
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # 如果没有掩膜，尝试不同扩展名
            mask = None
            for ext in ['.png', '.jpg', '.bmp']:
                alt_path = mask_path.with_suffix(ext)
                if alt_path.exists():
                    mask = cv2.imread(str(alt_path), cv2.IMREAD_GRAYSCALE)
                    break
            if mask is None:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                logger.warning(f"未找到掩膜: {mask_path}，使用空掩膜")

        # 数据增强
        augmented = self.augment(image=image, mask=mask)

        # 转换为 tensor
        img_tensor = torch.from_numpy(augmented['image']).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(augmented['mask']).float() / 255.0
        edge_tensor = torch.from_numpy(augmented['edge_mask']).float() / 255.0

        # 确保维度正确
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if edge_tensor.dim() == 2:
            edge_tensor = edge_tensor.unsqueeze(0)

        return img_tensor, {'mask': mask_tensor, 'edge_mask': edge_tensor}


class SyntheticDataset(Dataset):
    """合成数据集 — 用于冷启动训练或数据增强。"""

    def __init__(self, count=1000, img_size=256, mode='train'):
        self.count = count
        self.img_size = img_size
        self.mode = mode
        self.generator = SyntheticMetalWorkpieceGenerator(
            img_size=img_size, min_objects=1, max_objects=4
        )
        self.augment = HighReflectivityAugPipeline(mode=mode, img_size=img_size)

        # 预生成数据以加速训练
        logger.info(f"预生成 {count} 张合成训练数据...")
        self.data = [self.generator.generate() for _ in range(count)]
        logger.info("合成数据生成完成。")

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.mode == 'train':
            augmented = self.augment(image=sample['image'], mask=sample['mask'])
        else:
            augmented = {
                'image': cv2.resize(sample['image'], (self.img_size, self.img_size)),
                'mask': cv2.resize(sample['mask'], (self.img_size, self.img_size),
                                   interpolation=cv2.INTER_NEAREST),
                'edge_mask': generate_edge_from_mask(
                    cv2.resize(sample['mask'], (self.img_size, self.img_size),
                               interpolation=cv2.INTER_NEAREST)
                ),
            }

        img_tensor = torch.from_numpy(augmented['image']).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(augmented['mask']).float() / 255.0
        edge_tensor = torch.from_numpy(augmented['edge_mask']).float() / 255.0

        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if edge_tensor.dim() == 2:
            edge_tensor = edge_tensor.unsqueeze(0)

        return img_tensor, {'mask': mask_tensor, 'edge_mask': edge_tensor}


# ===========================================================================
# 3. 评估指标
# ===========================================================================

class MetricsCalculator:
    """计算分割和边缘检测的评估指标。"""

    @staticmethod
    def iou(pred, target, threshold=0.5):
        pred_bin = (pred > threshold).float()
        intersection = (pred_bin * target).sum()
        union = pred_bin.sum() + target.sum() - intersection
        return (intersection / (union + 1e-8)).item()

    @staticmethod
    def dice(pred, target, threshold=0.5):
        pred_bin = (pred > threshold).float()
        intersection = (pred_bin * target).sum()
        return (2.0 * intersection / (pred_bin.sum() + target.sum() + 1e-8)).item()

    @staticmethod
    def boundary_accuracy(pred, target, threshold=0.5, tolerance=2):
        """边缘精度：预测边缘在真实边缘 tolerance 像素内的比例。"""
        pred_bin = (pred > threshold).float()

        # 膨胀目标边缘
        kernel = torch.ones(1, 1, 2 * tolerance + 1, 2 * tolerance + 1,
                            device=target.device)
        target_dilated = F.conv2d(target, kernel, padding=tolerance)
        target_dilated = (target_dilated > 0).float()

        # 计算预测边缘在容差范围内的比例
        correct = (pred_bin * target_dilated).sum()
        total = pred_bin.sum()
        return (correct / (total + 1e-8)).item()


# ===========================================================================
# 4. 训练器
# ===========================================================================

class Trainer:
    """AGEANet 训练器。"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.save_dir = Path(args.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 初始化模型
        if args.model == 'standard':
            self.model = AGEANet(in_channels=3, out_channels=1, base_ch=args.base_ch)
        else:
            self.model = AGEANetLite(in_channels=3, out_channels=1, base_ch=args.base_ch)
        self.model = self.model.to(self.device)

        info = get_model_info(self.model)
        logger.info(f"模型: {args.model}, 参数量: {info['total_params']:,}, "
                     f"大小: {info['total_params_mb']:.1f} MB")

        # 损失函数
        self.criterion = CombinedLoss(
            dice_w=args.dice_w, focal_w=args.focal_w,
            boundary_w=args.boundary_w, edge_w=args.edge_w,
            specular_w=args.specular_w,
        )

        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=args.lr, weight_decay=args.weight_decay,
        )

        # 学习率调度
        self.scheduler = None  # 在 train() 中初始化

        # 评估指标
        self.metrics = MetricsCalculator()

        # 训练状态
        self.best_iou = 0.0
        self.patience_counter = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'val_dice': []}

    def build_dataloaders(self):
        """构建训练和验证数据加载器。"""
        args = self.args
        datasets_train = []
        datasets_val = []

        # 真实数据
        if args.data_dir and Path(args.data_dir).exists():
            real_train = MetalWorkpieceDataset(
                args.data_dir, mode='train', img_size=args.img_size
            )
            if len(real_train) > 0:
                # 按 8:2 分割训练/验证
                n_val = max(1, int(len(real_train) * 0.2))
                n_train = len(real_train) - n_val
                train_set, val_set = torch.utils.data.random_split(
                    real_train, [n_train, n_val]
                )
                datasets_train.append(train_set)

                real_val = MetalWorkpieceDataset(
                    args.data_dir, mode='val', img_size=args.img_size
                )
                val_subset_indices = val_set.indices
                val_dataset = torch.utils.data.Subset(real_val, val_subset_indices)
                datasets_val.append(val_dataset)
                logger.info(f"真实数据: {n_train} 训练 / {n_val} 验证")

        # 合成数据
        if args.synthetic_only or args.syn_count > 0:
            syn_count = args.syn_count if args.syn_count > 0 else 1000
            syn_train = SyntheticDataset(
                count=int(syn_count * 0.8), img_size=args.img_size, mode='train'
            )
            syn_val = SyntheticDataset(
                count=int(syn_count * 0.2), img_size=args.img_size, mode='val'
            )
            datasets_train.append(syn_train)
            datasets_val.append(syn_val)
            logger.info(f"合成数据: {len(syn_train)} 训练 / {len(syn_val)} 验证")

        if not datasets_train:
            logger.error("没有可用的训练数据！请指定 --data_dir 或 --synthetic_only")
            sys.exit(1)

        train_dataset = ConcatDataset(datasets_train) if len(datasets_train) > 1 else datasets_train[0]
        val_dataset = ConcatDataset(datasets_val) if len(datasets_val) > 1 else datasets_val[0]

        self.train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )

        logger.info(f"训练批次: {len(self.train_loader)}, 验证批次: {len(self.val_loader)}")

    def train_one_epoch(self, epoch):
        """训练一个 epoch。"""
        self.model.train()
        total_loss = 0.0
        loss_components = {}

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss, loss_dict = self.criterion(outputs, targets)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v

            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                logger.info(f"  Epoch {epoch} [{batch_idx+1}/{len(self.train_loader)}] "
                             f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        avg_components = {k: v / len(self.train_loader) for k, v in loss_components.items()}
        return avg_loss, avg_components

    @torch.no_grad()
    def validate(self):
        """验证。"""
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0
        total_edge_acc = 0.0
        n_batches = 0

        for images, targets in self.val_loader:
            images = images.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            outputs = self.model(images)
            loss, _ = self.criterion(outputs, targets)

            total_loss += loss.item()
            total_iou += self.metrics.iou(outputs['seg'], targets['mask'])
            total_dice += self.metrics.dice(outputs['seg'], targets['mask'])
            total_edge_acc += self.metrics.boundary_accuracy(
                outputs['edge'], targets['edge_mask']
            )
            n_batches += 1

        return {
            'loss': total_loss / max(n_batches, 1),
            'iou': total_iou / max(n_batches, 1),
            'dice': total_dice / max(n_batches, 1),
            'edge_acc': total_edge_acc / max(n_batches, 1),
        }

    def train(self):
        """完整训练流程。"""
        args = self.args

        logger.info("=" * 60)
        logger.info("开始训练 AGEANet")
        logger.info("=" * 60)

        self.build_dataloaders()

        # 初始化学习率调度器
        total_steps = len(self.train_loader) * args.epochs
        self.scheduler = OneCycleLR(
            self.optimizer, max_lr=args.lr,
            total_steps=total_steps, pct_start=0.1,
            anneal_strategy='cos',
        )

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            # 训练
            train_loss, train_components = self.train_one_epoch(epoch)

            # 验证
            val_metrics = self.validate()

            epoch_time = time.time() - epoch_start

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['val_dice'].append(val_metrics['dice'])

            logger.info(
                f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | "
                f"Val Dice: {val_metrics['dice']:.4f} | "
                f"Edge Acc: {val_metrics['edge_acc']:.4f}"
            )

            # 保存最佳模型
            if val_metrics['iou'] > self.best_iou:
                self.best_iou = val_metrics['iou']
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"  ★ 新最佳模型！IoU: {self.best_iou:.4f}")
            else:
                self.patience_counter += 1

            # 定期保存
            if epoch % args.save_every == 0:
                self._save_checkpoint(epoch, val_metrics, is_best=False)

            # 早停
            if self.patience_counter >= args.patience:
                logger.info(f"早停触发 (patience={args.patience})")
                break

        # 保存训练历史
        self._save_history()

        logger.info("=" * 60)
        logger.info(f"训练完成！最佳 IoU: {self.best_iou:.4f}")
        logger.info(f"模型保存在: {self.save_dir}")
        logger.info("=" * 60)

    def _save_checkpoint(self, epoch, metrics, is_best=False):
        """保存模型检查点。"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'args': vars(self.args),
        }

        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'

        torch.save(checkpoint, path)

    def _save_history(self):
        """保存训练历史。"""
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)


# ===========================================================================
# 5. 命令行参数
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='AGEANet 训练脚本')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default=None,
                        help='真实数据集目录 (包含 images/ 和 masks/ 子目录)')
    parser.add_argument('--synthetic_only', action='store_true',
                        help='仅使用合成数据训练')
    parser.add_argument('--syn_count', type=int, default=0,
                        help='合成数据数量 (0=不使用合成数据)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='训练图像尺寸')

    # 模型参数
    parser.add_argument('--model', type=str, default='standard',
                        choices=['standard', 'lite'],
                        help='模型类型: standard (AGEANet) 或 lite (AGEANet-Lite)')
    parser.add_argument('--base_ch', type=int, default=64,
                        help='基础通道数 (standard: 64, lite: 32)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=20,
                        help='早停耐心值')
    parser.add_argument('--save_every', type=int, default=10,
                        help='每 N 个 epoch 保存一次检查点')

    # 损失权重
    parser.add_argument('--dice_w', type=float, default=1.0)
    parser.add_argument('--focal_w', type=float, default=1.0)
    parser.add_argument('--boundary_w', type=float, default=0.5)
    parser.add_argument('--edge_w', type=float, default=0.5)
    parser.add_argument('--specular_w', type=float, default=0.3)

    # 系统参数
    parser.add_argument('--device', type=str, default='auto',
                        help='设备: auto, cpu, cuda')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据加载线程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')

    args = parser.parse_args()

    # 自动设备选择
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Lite 模型默认通道数
    if args.model == 'lite' and args.base_ch == 64:
        args.base_ch = 32

    return args


# ===========================================================================
# 6. 入口点
# ===========================================================================

if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)

    # 恢复训练
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"从检查点恢复: {args.resume}")

    trainer.train()
