"""
train.py — FLARE 训练主程序
:Author: RussellCooper

功能：
  - 支持合成数据 / 真实数据 / 混合训练
  - 双头损失：分割（BCE + Dice）+ 边缘（BCE + Focal）
  - 学习率调度：Cosine Annealing with Warmup
  - 混合精度训练（AMP）
  - 模型检查点保存（best_val_iou）
  - TensorBoard 日志
  - 支持从检查点恢复训练

用法::

    # 纯合成数据训练（在线生成）
    python3 train.py --epochs 50 --batch_size 4 --img_size 512

    # 使用预生成合成数据集
    python3 train.py --synth_dir ./datasets/synth_ship --epochs 100 --batch_size 8

    # 混合真实数据
    python3 train.py --synth_dir ./datasets/synth_ship --real_dir ./datasets/real_ship \\
                     --real_ratio 0.3 --epochs 150

    # 使用轻量版模型
    python3 train.py --model lite --base_ch 32 --epochs 80
"""

import argparse
import math
import numpy as np
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from vision.feature_extraction import FLARE, FLARELite, get_model_info, CIoULoss
from data.real_world_dataloader import build_dataloaders
from data.data_augmentation import apply_cutmix_batch, apply_mixup_batch
from core.metrics import MetricTracker, tta_inference as core_tta_inference


# ============================================================
# 1. 损失函数
# ============================================================

class DiceLoss(nn.Module):
    """Dice Loss（对类别不平衡鲁棒）。"""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred   = pred.view(-1)
        target = target.view(-1)
        inter  = (pred * target).sum()
        return 1.0 - (2.0 * inter + self.smooth) / \
               (pred.sum() + target.sum() + self.smooth)


class FocalLoss(nn.Module):
    """
    Focal Loss（专注于难样本，对边缘细节有效）。

    alpha: 正样本权重（边缘像素稀少，alpha > 0.5 可补偿）
    gamma: 聚焦参数（gamma=2 是标准设置）

    输入：logits（未经 sigmoid）
    """

    def __init__(self, alpha: float = 0.9, gamma: float = 3.0, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Label smoothing
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        # 使用 binary_cross_entropy_with_logits（AMP 兼容）
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pred = torch.sigmoid(logits)
        pt  = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# ============================================================
# 边缘专用损失函数
# ============================================================

class EdgeAwareFocalLoss(nn.Module):
    """
    边缘感知 Focal Loss - 专注于边缘像素的检测。

    核心改进：
    1. 边缘区域加权 - 对边缘像素给予更高权重
    2. 难例挖掘 - 聚焦于预测困难的像素
    3. 类别平衡 - 处理边缘像素稀少的问题

    边缘权重由分割掩膜的梯度计算得到。
    """

    def __init__(self, alpha: float = 0.9, gamma: float = 3.0, edge_weight: float = 3.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.edge_weight = edge_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                seg_logits: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) 边缘预测 logits
            target: (B, 1, H, W) 边缘 ground truth
            seg_logits: (B, 1, H, W) 可选的分割 logits 用于计算边缘权重
        """
        # 计算边缘权重
        if seg_logits is not None:
            # 从分割 logits 计算边缘权重
            seg_prob = torch.sigmoid(seg_logits)
            grad_x = torch.abs(seg_prob[:, :, :, :-1] - seg_prob[:, :, :, 1:])
            grad_y = torch.abs(seg_prob[:, :, :-1, :] - seg_prob[:, :, 1:, :])
            grad_x = F.pad(grad_x, (0, 1, 0, 0))
            grad_y = F.pad(grad_y, (0, 0, 0, 1))
            edge_grad = grad_x + grad_y
            # 边缘梯度归一化
            edge_grad = edge_grad / (edge_grad.max() + 1e-8)
            # 边缘像素权重更高
            edge_weights = 1.0 + self.edge_weight * edge_grad
        else:
            edge_weights = torch.ones_like(logits)

        # 标准 BCE
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pred = torch.sigmoid(logits)

        # Focal weight
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma

        # 加权损失
        weighted_bce = (focal_weight * bce * edge_weights)

        return weighted_bce.mean()


class EdgeGradientConsistencyLoss(nn.Module):
    """
    边缘梯度一致性损失 - 确保边缘预测与分割掩膜边界一致。

    原理：
    1. 边缘应该是分割掩膜的边界
    2. 分割掩膜的梯度应该与边缘预测一致
    3. 边缘一致性可提升边缘定位精度
    """

    def __init__(self, weight: float = 0.5):
        super().__init__()
        self.weight = weight

    def forward(self, edge_logits: torch.Tensor, seg_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_logits: (B, 1, H, W) 边缘预测
            seg_logits: (B, 1, H, W) 分割预测
        """
        # 分割概率
        seg_prob = torch.sigmoid(seg_logits)

        # 计算分割掩膜的梯度（边缘）
        grad_x = torch.abs(seg_prob[:, :, :, :-1] - seg_prob[:, :, :, 1:])
        grad_y = torch.abs(seg_prob[:, :, :-1, :] - seg_prob[:, :, 1:, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        seg_edge = grad_x + grad_y

        # 边缘预测的梯度
        edge_prob = torch.sigmoid(edge_logits)
        edge_grad_x = torch.abs(edge_prob[:, :, :, :-1] - edge_prob[:, :, :, 1:])
        edge_grad_y = torch.abs(edge_prob[:, :, :-1, :] - edge_prob[:, :, 1:, :])
        edge_grad_x = F.pad(edge_grad_x, (0, 1, 0, 0))
        edge_grad_y = F.pad(edge_grad_y, (0, 0, 0, 1))
        edge_grad = edge_grad_x + edge_grad_y

        # MSE 损失：边缘预测的梯度应该与分割边缘相似
        consistency_loss = F.mse_loss(edge_grad, seg_edge.detach())

        return self.weight * consistency_loss


class BalancedBCELoss(nn.Module):
    """
    平衡 BCE 损失 - 边缘像素更少，需要更高权重。

    使用自适应权重平衡前景和背景。
    """

    def __init__(self, pos_weight: float = 15.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) 预测 logits
            target: (B, 1, H, W) ground truth
        """
        # 计算正负样本比例
        pos_count = target.sum()
        neg_count = (1 - target).sum()
        total = pos_count + neg_count + 1e-8

        # 自适应权重：正样本少所以权重高
        pos_weight = (neg_count / (pos_count + 1e-8)).clamp(max=10.0)
        pos_weight = pos_weight * self.pos_weight

        # 加权 BCE
        loss = F.binary_cross_entropy_with_logits(
            logits, target,
            pos_weight=torch.tensor([pos_weight], device=logits.device)
        )
        return loss


class MultiScaleEdgeLoss(nn.Module):
    """
    多尺度边缘损失 - 从 RCF 侧输出监督每个尺度。

    对每个解码器层的侧输出计算边缘损失，
    确保每个尺度都能学习到边缘特征。
    """

    def __init__(self, alpha: float = 0.9, gamma: float = 3.0):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)

    def forward(self, side_edges: list, target_edge: torch.Tensor) -> torch.Tensor:
        """
        Args:
            side_edges: 各层侧输出边缘预测列表
            target_edge: (B, 1, H, W) 边缘 ground truth
        """
        total_loss = 0.0
        n_levels = len(side_edges)

        for i, side_edge in enumerate(side_edges):
            # 对齐到 target 尺寸
            if side_edge.shape[-2:] != target_edge.shape[-2:]:
                side_edge = F.interpolate(
                    side_edge, size=target_edge.shape[-2:],
                    mode='bilinear', align_corners=False
                )
            # 动态权重：浅层权重较低（容易学习），深层权重较高（难学习）
            weight = (i + 1) / n_levels
            loss = self.focal(side_edge, target_edge)
            total_loss += weight * loss

        return total_loss


# ============================================================
# EMA (指数移动平均) - 训练稳定性优化
# ============================================================

class EMA:
    """
    Exponential Moving Average for model parameters.

    优势：
    - 训练更稳定，减少抖动
    - 通常能获得更好的泛化性能
    - 对预测时使用 averaging 而不是最后一个 checkpoint

    用法：
        ema = EMA(model, decay=0.999)
        # 训练步骤后
        ema.update()
        # 验证/推理时
        ema.apply_shadow()
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        self.model = model
        self.decay = decay
        self.device = device or next(model.parameters()).device
        self.shadow = {}
        self.backup = {}

        # 初始化 shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """更新 EMA 参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"Parameter {name} not in shadow"
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """应用 shadow parameters 到模型（用于验证/推理）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """恢复原始 parameters（训练模式）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# ============================================================
# 新增损失函数：Lovász Loss + Boundary Loss
# ============================================================

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """计算 Lovász-Softmax损失的梯度（IoU 直通估计）。"""
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if len(gt_sorted) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


class LovaszLoss(nn.Module):
    """
    Lovász-Softmax Loss — 直接优化 IoU/Dice 指标，比 BCE/Dice 收敛更直接。

    适用于二值分割，比 Dice Loss 对小物体更友好。
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        signs = 2.0 * target - 1.0
        errors = 1.0 - pred * signs
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = target[perm]
        grad = _lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss


class BoundaryLoss(nn.Module):
    """
    Boundary Loss — 专门优化边缘/轮廓质量，对细线/小物体分割更有效。

    通过边缘区域加权 BCE，边缘像素获得更高权重。
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:  (B, 1, H, W) sigmoid 输出 [0,1]
        target: (B, 1, H, W) 二值掩膜 [0,1]
        """
        target = target.float()

        # 计算边缘掩膜：膨胀后减去原图
        kernel = torch.ones(1, 1, 3, 3, device=pred.device)
        target_dilated = F.conv2d(target, kernel, padding=1) > 0
        edge_mask = target_dilated & ~target.bool()
        edge_mask = edge_mask.float()

        # 距离加权（pred 是 logits，用 binary_cross_entropy_with_logits）
        pred_edge = pred * edge_mask
        loss = F.binary_cross_entropy_with_logits(pred_edge, edge_mask, reduction='mean')
        return loss


class MixedLoss(nn.Module):
    """
    混合损失：Lovász + Dice + BCE + Boundary + CIoU（加权组合）。

    L_total = w_bce * BCE + w_dice * Dice + w_lovasz * Lovász + w_boundary * Boundary + w_ciou * CIoU
    """

    def __init__(
        self,
        w_bce: float = 0.5,
        w_dice: float = 0.5,
        w_lovasz: float = 0.5,
        w_boundary: float = 0.3,
        w_ciou: float = 0.5,
        edge_w: float = 0.5,
    ):
        super().__init__()
        self.edge_w = edge_w
        # BCEWithLogitsLoss：接收 logits，AMP 兼容
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.lovasz = LovaszLoss()
        self.boundary = BoundaryLoss()
        self.ciou = CIoULoss()
        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_lovasz = w_lovasz
        self.w_boundary = w_boundary
        self.w_ciou = w_ciou

    def forward(
        self,
        pred_seg: torch.Tensor,
        pred_edge: torch.Tensor,
        target_seg: torch.Tensor,
        target_edge: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # 分割损失：Lovász + Dice + BCE (logits 输入)
        l_bce = self.bce(pred_seg, target_seg)
        l_dice = self.dice(pred_seg, target_seg)
        l_lovasz = self.lovasz(pred_seg, target_seg)
        l_boundary = self.boundary(pred_seg, target_seg)
        l_seg = (self.w_bce * l_bce + self.w_dice * l_dice +
                 self.w_lovasz * l_lovasz + self.w_boundary * l_boundary)

        # CIoU 损失：从分割掩膜计算 bbox 回归
        l_ciou = torch.tensor(0.0, device=pred_seg.device)
        if self.w_ciou > 0 and pred_seg.shape == target_seg.shape:
            l_ciou = self.ciou(pred_seg, target_seg)

        # 边缘损失：Focal + BCE (logits 输入) - 增强参数
        l_edge_bce = self.bce(pred_edge, target_edge)
        l_edge_focal = FocalLoss(alpha=0.9, gamma=3.0)(pred_edge, target_edge)
        l_edge = self.edge_w * (l_edge_bce + l_edge_focal)

        total = l_seg + l_edge + self.w_ciou * l_ciou

        return {
            'total': total,
            'seg': l_seg,
            'seg_bce': l_bce,
            'seg_dice': l_dice,
            'seg_lovasz': l_lovasz,
            'seg_boundary': l_boundary,
            'seg_ciou': l_ciou,
            'edge': l_edge,
            'edge_bce': l_edge_bce,
            'edge_focal': l_edge_focal,
        }


class FLARELoss(nn.Module):
    """
    FLARE 双头联合损失（增强版）。

    使用混合损失：
      L_seg  = BCE + Dice + Lovász + Boundary（加权）
      L_edge = BCE + Focal + 边缘感知Focal + 梯度一致性

    相比原版：
    - 增加了边缘感知 Focal Loss（专注边缘像素）
    - 增加了边缘梯度一致性损失（确保边缘与分割边界一致）
    - 增加了多尺度边缘损失（从 RCF 侧输出监督）
    - 支持多尺度侧输出监督
    """

    def __init__(self, w_seg: float = 1.0, w_edge: float = 4.0,
                 use_mixed_loss: bool = True,
                 use_edge_aware: bool = True):
        super().__init__()
        self.w_seg = w_seg
        self.w_edge = w_edge
        self.use_mixed_loss = use_mixed_loss
        self.use_edge_aware = use_edge_aware

        if use_mixed_loss:
            # 分割损失：Lovász + Dice + BCE + Boundary
            self.seg_bce = nn.BCEWithLogitsLoss()
            self.seg_dice = DiceLoss()
            self.seg_lovasz = LovaszLoss()
            self.seg_boundary = BoundaryLoss()
        else:
            self.bce = nn.BCEWithLogitsLoss()
            self.dice = DiceLoss()
            self.focal = FocalLoss(alpha=0.75, gamma=2.0)

        # 边缘专用损失
        self.edge_focal = FocalLoss(alpha=0.9, gamma=3.0)
        self.edge_aware_focal = EdgeAwareFocalLoss(alpha=0.9, gamma=3.0, edge_weight=3.0)
        self.edge_gradient_consistency = EdgeGradientConsistencyLoss(weight=0.3)
        self.edge_balanced_bce = BalancedBCELoss(pos_weight=15.0)
        self.multi_scale_edge = MultiScaleEdgeLoss(alpha=0.9, gamma=3.0)

    def forward(self, pred: Dict[str, torch.Tensor],
                target_seg: torch.Tensor,
                target_edge: torch.Tensor) -> Dict[str, torch.Tensor]:

        seg_pred = pred['seg']
        edge_pred = pred['edge']

        if self.use_mixed_loss:
            # 分割损失：Lovász + Dice + BCE + Boundary
            l_seg_bce = self.seg_bce(seg_pred, target_seg)
            l_seg_dice = self.seg_dice(seg_pred, target_seg)
            l_seg_lovasz = self.seg_lovasz(seg_pred, target_seg)
            l_seg_boundary = self.seg_boundary(seg_pred, target_seg)
            l_seg = l_seg_bce + l_seg_dice + l_seg_lovasz + 0.3 * l_seg_boundary
        else:
            # 原版 BCE + Dice
            l_seg_bce = self.bce(seg_pred, target_seg)
            l_seg_dice = self.dice(seg_pred, target_seg)
            l_seg = l_seg_bce + l_seg_dice

        # 边缘损失：多种损失组合
        l_edge_bce = self.edge_balanced_bce(edge_pred, target_edge)
        l_edge_focal = self.edge_focal(edge_pred, target_edge)

        # 边缘感知 Focal（使用分割 logits 计算边缘权重）
        if self.use_edge_aware:
            l_edge_aware = self.edge_aware_focal(edge_pred, target_edge, seg_pred)
        else:
            l_edge_aware = torch.tensor(0.0, device=edge_pred.device)

        # 边缘梯度一致性（增加到30%，确保边缘与分割掩膜边界一致）
        l_edge_gradient = self.edge_gradient_consistency(edge_pred, seg_pred)

        # 多尺度边缘损失（如果有侧输出）
        l_edge_multi = torch.tensor(0.0, device=edge_pred.device)
        if 'side_edges' in pred and pred['side_edges'] is not None:
            l_edge_multi = self.multi_scale_edge(pred['side_edges'], target_edge)

        # 组合边缘损失（简化：去掉互相干扰的多头监督）
        # 边缘任务需要集中优化，5个头各占15-25%反而无法突出主信号
        l_edge = (
            l_edge_bce * 0.4 +           # 40% 平衡 BCE - 主损失
            l_edge_focal * 0.3 +          # 30% Focal - 难例挖掘
            l_edge_gradient * 0.3         # 30% 梯度一致性 - 与分割边界对齐
        )

        # 总损失
        total = self.w_seg * l_seg + self.w_edge * l_edge

        return {
            'total': total,
            'seg': l_seg,
            'seg_bce': l_seg_bce,
            'seg_dice': l_seg_dice if self.use_mixed_loss else l_seg_dice,
            'seg_lovasz': l_seg_lovasz if self.use_mixed_loss else torch.tensor(0.0),
            'edge': l_edge,
            'edge_bce': l_edge_bce,
            'edge_focal': l_edge_focal,
            'edge_aware': l_edge_aware,
            'edge_gradient': l_edge_gradient,
            'edge_multi': l_edge_multi,
        }


# ============================================================
# 2. 评估指标
# ============================================================
# 使用 core.metrics 中的统一 MetricTracker（已导入）
# 注意：边缘阈值已统一为 0.40


# ============================================================
# 3. 学习率调度（Cosine + Warmup）
# ============================================================

def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """
    Cosine Annealing with Linear Warmup。

    前 warmup_steps 步线性增大 LR，之后 Cosine 衰减到 min_lr_ratio * base_lr。
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return min_lr_ratio + (1.0 - min_lr_ratio) * \
               0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


# ============================================================
# 4. 训练 / 验证 epoch
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: FLARELoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    cutmix_p: float = 0.0,
    mixup_p: float = 0.0,
    cutmix_alpha: float = 1.0,
    mixup_alpha: float = 0.4,
    ema: Optional[EMA] = None,
) -> Dict[str, float]:

    model.train()
    tracker = MetricTracker()

    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        masks  = batch['mask'].to(device, non_blocking=True)
        edges  = batch['edge'].to(device, non_blocking=True)

        # CutMix / MixUp 数据增强
        if np.random.random() < cutmix_p:
            images, masks, edges = apply_cutmix_batch(
                images, masks, edges, alpha=cutmix_alpha, p=1.0
            )
        elif np.random.random() < mixup_p:
            images, masks, edges = apply_mixup_batch(
                images, masks, edges, alpha=mixup_alpha, p=1.0
            )

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type='cuda', enabled=use_amp):
            pred  = model(images)
            losses = criterion(pred, masks, edges)
            loss   = losses['total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # EMA 更新
        if ema is not None:
            ema.update()

        tracker.update(
            pred['seg'].detach(), masks,
            pred['edge'].detach(), edges,
            loss.item(),
        )

    return tracker.compute()


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: FLARELoss,
    device: torch.device,
    use_amp: bool = True,
    tta: bool = False,
) -> Dict[str, float]:

    model.eval()
    tracker = MetricTracker()

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks  = batch['mask'].to(device, non_blocking=True)
        edges  = batch['edge'].to(device, non_blocking=True)

        with autocast(device_type='cuda', enabled=use_amp):
            if tta:
                pred = core_tta_inference(model, images, device)
            else:
                pred = model(images)
            losses = criterion(pred, masks, edges)

        tracker.update(
            pred['seg'], masks,
            pred['edge'], edges,
            losses['total'].item(),
        )

    return tracker.compute()

# 使用 core.metrics 中的统一 tta_inference（已导入为 core_tta_inference）


# ============================================================
# 5. 检查点保存/加载
# ============================================================

def save_checkpoint(state: dict, path: str):
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler=None) -> int:
    """加载检查点，返回起始 epoch。"""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    if optimizer and 'optimizer' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except (ValueError, RuntimeError) as e:
            print(f"[train] 警告: 无法加载优化器状态 ({e})，从头开始训练")
    if scheduler and 'scheduler' in ckpt:
        try:
            scheduler.load_state_dict(ckpt['scheduler'])
        except (ValueError, RuntimeError) as e:
            print(f"[train] 警告: 无法加载调度器状态 ({e})，从头开始训练")
    start_epoch = ckpt.get('epoch', 0) + 1
    print(f"[train] 从检查点恢复: {path}（epoch {ckpt.get('epoch', 0)}）")
    return start_epoch


# ============================================================
# 5.1 自蒸馏 (Self-Distillation) - 自有算法优化
# ============================================================

class SelfDistillationLoss(nn.Module):
    """
    自蒸馏损失 - 使用更深层的特征作为软标签监督浅层网络

    FLARE 专利技术：
    - 深解码器特征 → 监督浅解码器特征
    - 多尺度分割头相互监督
    - 边缘感知一致性损失

    优势：不依赖外部预训练模型，纯自有算法迭代优化
    """

    def __init__(self, temperature: float = 3.0, edge_weight: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.edge_weight = edge_weight

    def forward(self, student_outputs, teacher_outputs, target_seg, target_edge):
        """
        自蒸馏：深层教师特征监督浅层学生特征

        Args:
            student_outputs: 学生模型输出 {'seg': logits, 'edge': logits, 'feat': ...}
            teacher_outputs: 教师模型输出（同结构，detach 不更新梯度）
            target_seg: 硬标签分割掩膜
            target_edge: 硬标签边缘掩膜
        """
        T = self.temperature

        # 硬标签损失
        hard_seg = F.binary_cross_entropy_with_logits(
            student_outputs['seg'], target_seg
        )
        hard_edge = F.binary_cross_entropy_with_logits(
            student_outputs['edge'], target_edge
        )
        hard_loss = hard_seg + self.edge_weight * hard_edge

        # 软标签损失：教师→学生特征匹配
        soft_loss = torch.tensor(0.0, device=student_outputs['seg'].device)

        if 'feat' in student_outputs and 'feat' in teacher_outputs:
            # 多尺度特征蒸馏
            student_feat = student_outputs['feat']
            teacher_feat = teacher_outputs['feat'].detach()

            # 在通道维度做注意力匹配
            student_soft = torch.sigmoid(student_feat / T)
            teacher_soft = torch.sigmoid(teacher_feat.detach() / T)
            soft_loss = F.mse_loss(student_soft, teacher_soft) * (T * T)

        # 边缘一致性损失：分割边缘与边缘检测一致性
        edge_consistency = torch.tensor(0.0, device=student_outputs['seg'].device)
        if 'edge' in student_outputs and 'seg' in student_outputs:
            seg_prob = torch.sigmoid(student_outputs['seg'])
            edge_prob = torch.sigmoid(student_outputs['edge'])

            # 边缘应该是分割区域内部的梯度极大值处
            grad_x = torch.abs(seg_prob[:, :, :, :-1] - seg_prob[:, :, :, 1:])
            grad_y = torch.abs(seg_prob[:, :, :-1, :] - seg_prob[:, :, 1:, :])
            edge_gradient = grad_x.mean() + grad_y.mean()

            # 边缘预测与梯度应该一致
            edge_consistency = F.mse_loss(edge_prob, edge_gradient.detach())

        # 总损失
        total = 0.6 * hard_loss + 0.3 * soft_loss + 0.1 * edge_consistency

        return {
            'total': total,
            'hard_loss': hard_loss,
            'soft_loss': soft_loss,
            'edge_consistency': edge_consistency,
        }


# ============================================================
# 5.2 TensorRT 优化推理
# ============================================================

def optimize_for_tensorrt(model: nn.Module, input_shape: tuple = (1, 3, 512, 512),
                          output_path: str = "checkpoints/model.engine") -> str:
    """
    使用 torch2trt 将 PyTorch 模型转换为 TensorRT

    Args:
        model: PyTorch 模型
        input_shape: 输入形状 (B, C, H, W)
        output_path: 输出 TensorRT engine 路径

    Returns:
        TensorRT engine 路径
    """
    try:
        import torch_tensorrt as torch_trt

        print("[tensorrt] 开始 TensorRT 优化...")

        # 编译为 TensorRT
        trt_model = torch_trt.compile(
            model,
            inputs=[torch.randn(input_shape)],
            enabled_precisions={torch.float16},
            workspace_size=1 << 30,
        )

        # 保存
        torch.jit.save(trt_model, output_path)
        print(f"[tensorrt] TensorRT 模型已保存: {output_path}")

        return output_path
    except ImportError:
        print("[tensorrt] torch_tensorrt 未安装，跳过优化")
        print("[tensorrt] 安装: pip install torch-tensorrt")
        return None


def apply_quantization(model: nn.Module, method: str = "dynamic"):
    """
    模型量化 - 减少模型大小并加速推理

    Args:
        model: PyTorch 模型
        method: "dynamic" | "static" | "qat"
    """
    if method == "dynamic":
        model.qconfig = torch.quantization.default_dynamic_qconfig
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)
        print("[quantize] 动态量化已应用")
    elif method == "static":
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        # 需要校准数据
        print("[quantize] 静态量化已准备，需要校准")
    return model


# ============================================================
# 6. 主训练入口
# ============================================================

def main(args):
    # --- 设备 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = torch.cuda.is_available()
    print(f"[train] 设备: {device}, AMP: {use_amp}")

    # --- 数据加载器 ---
    train_loader, val_loader = build_dataloaders(
        synth_dir=args.synth_dir,
        real_dir=args.real_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        real_ratio=args.real_ratio,
        online_epoch_size=args.online_epoch_size,
        val_ratio=0.1,
    )

    # --- 模型 ---
    if args.model == 'lite':
        model = FLARELite(in_channels=3, base_ch=args.base_ch).to(device)
    else:
        model = FLARE(in_channels=3, base_ch=args.base_ch).to(device)

    info = get_model_info(model)
    print(f"[train] 模型: {args.model}, 参数量: {info['total_params']:,}, "
          f"FP32: {info['size_mb_fp32']:.1f} MB")

    # --- EMA (指数移动平均) ---
    ema = EMA(model, decay=0.999, device=device)
    print(f"[train] EMA: 启用 (decay=0.999)")

    # --- 损失函数 ---
    criterion = FLARELoss(w_seg=1.0, w_edge=args.edge_weight,
                            use_mixed_loss=args.use_mixed_loss)
    print(f"[train] 损失函数: {'Lovász+Boundary 混合损失' if args.use_mixed_loss else 'BCE+Dice+Focal'}")

    # --- 优化器 ---
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)

    # --- 学习率调度 ---
    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = int(total_steps * 0.10)  # 增加到 10% warmup 更稳定
    scheduler     = build_cosine_warmup_scheduler(
        optimizer, total_steps, warmup_steps, min_lr_ratio=0.01
    )
    scaler = GradScaler('cuda', enabled=use_amp)

    # --- 检查点恢复 ---
    start_epoch = 0
    os.makedirs(args.save_dir, exist_ok=True)
    resume_path = os.path.join(args.save_dir, 'last.pth')
    if args.resume and os.path.exists(resume_path):
        start_epoch = load_checkpoint(resume_path, model, optimizer, scheduler)

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))

    # --- 训练循环 ---
    best_iou = 0.0
    print(f"\n[train] 开始训练: {args.epochs} epochs\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            scaler, device, epoch, use_amp,
            cutmix_p=args.cutmix_p,
            mixup_p=args.mixup_p,
            cutmix_alpha=args.cutmix_alpha,
            mixup_alpha=args.mixup_alpha,
            ema=ema,
        )

        # 验证时使用 EMA 模型
        ema.apply_shadow()
        val_metrics = validate(model, val_loader, criterion, device, use_amp,
                              tta=args.tta)
        ema.restore()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        # 打印
        print(
            f"Epoch [{epoch + 1:3d}/{args.epochs}] "
            f"lr={lr_now:.2e}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"train_iou={train_metrics['iou']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_iou={val_metrics['iou']:.4f}  "
            f"val_edge_f1={val_metrics['edge_f1']:.4f}  "
            f"[{elapsed:.1f}s]"
        )

        # TensorBoard
        for k, v in train_metrics.items():
            writer.add_scalar(f'train/{k}', v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        writer.add_scalar('lr', lr_now, epoch)

        # 保存最新检查点
        save_checkpoint({
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_iou':   val_metrics['iou'],
        }, resume_path)

        # 保存最优检查点
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            best_path = os.path.join(args.save_dir, 'best.pth')
            save_checkpoint({
                'epoch':   epoch,
                'model':   model.state_dict(),
                'val_iou': best_iou,
                'args':    vars(args),
            }, best_path)
            print(f"  ★ 新最优 IoU: {best_iou:.4f} → 已保存 {best_path}")

    writer.close()
    print(f"\n[train] 训练完成！最优 val IoU: {best_iou:.4f}")
    print(f"[train] 最优模型: {os.path.join(args.save_dir, 'best.pth')}")


# ============================================================
# 7. 命令行参数
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLARE 训练程序")

    # 数据
    parser.add_argument("--synth_dir",         type=str,   default=None)
    parser.add_argument("--real_dir",           type=str,   default=None)
    parser.add_argument("--real_ratio",         type=float, default=0.3)
    parser.add_argument("--online_epoch_size",  type=int,   default=2000)
    parser.add_argument("--img_size",           type=int,   default=512)

    # 模型
    parser.add_argument("--model",   type=str, default='standard',
                        choices=['standard', 'lite'])
    parser.add_argument("--base_ch", type=int, default=128,
                        help="基础通道数（默认128，头结构用128）")

    # 训练
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=3e-5,
                        help="学习率（默认3e-5，更低的LR配合更大的base_ch）")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--edge_weight",  type=float, default=2.0,
                        help="边缘损失权重（相对于分割损失，默认2.0增强边缘检测）")
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--use_mixed_loss", action="store_true", default=True,
                        help="使用 Lovász + Boundary 混合损失（默认开启）")
    parser.add_argument("--no_mixed_loss", dest="use_mixed_loss", action="store_false",
                        help="禁用混合损失，使用原版 BCE+Dice+Focal")
    parser.add_argument("--tta", action="store_true",
                        help="启用 Test Time Augmentation（水平翻转 + 对角翻转）")
    # CutMix / MixUp
    parser.add_argument("--cutmix_p", type=float, default=0.3,
                        help="CutMix 应用概率（0=禁用）")
    parser.add_argument("--mixup_p", type=float, default=0.3,
                        help="MixUp 应用概率（0=禁用）")
    parser.add_argument("--cutmix_alpha", type=float, default=1.0,
                        help="CutMix Beta 分布参数")
    parser.add_argument("--mixup_alpha", type=float, default=0.4,
                        help="MixUp Beta 分布参数")

    # 检查点
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume",   action="store_true",
                        help="从 save_dir/last.pth 恢复训练")

    args = parser.parse_args()
    main(args)
