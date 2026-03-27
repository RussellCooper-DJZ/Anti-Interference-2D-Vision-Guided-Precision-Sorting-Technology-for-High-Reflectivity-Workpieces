"""
train.py — AGEANet 训练主程序

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
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from feature_extraction import AGEANet, AGEANetLite, get_model_info
from real_world_dataloader import build_dataloaders


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
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt  = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class AGEANetLoss(nn.Module):
    """
    AGEANet 双头联合损失。

    L_total = w_seg * L_seg + w_edge * L_edge

    L_seg  = BCE + Dice
    L_edge = BCE + Focal（边缘像素稀少，Focal 更有效）
    """

    def __init__(self, w_seg: float = 1.0, w_edge: float = 0.5):
        super().__init__()
        self.w_seg  = w_seg
        self.w_edge = w_edge
        self.bce    = nn.BCELoss()
        self.dice   = DiceLoss()
        self.focal  = FocalLoss(alpha=0.75, gamma=2.0)

    def forward(self, pred: Dict[str, torch.Tensor],
                target_seg: torch.Tensor,
                target_edge: torch.Tensor) -> Dict[str, torch.Tensor]:

        seg_pred  = pred['seg']
        edge_pred = pred['edge']

        # 分割损失
        l_seg_bce  = self.bce(seg_pred, target_seg)
        l_seg_dice = self.dice(seg_pred, target_seg)
        l_seg      = l_seg_bce + l_seg_dice

        # 边缘损失
        l_edge_bce   = self.bce(edge_pred, target_edge)
        l_edge_focal = self.focal(edge_pred, target_edge)
        l_edge       = l_edge_bce + l_edge_focal

        total = self.w_seg * l_seg + self.w_edge * l_edge

        return {
            'total':      total,
            'seg':        l_seg,
            'seg_bce':    l_seg_bce,
            'seg_dice':   l_seg_dice,
            'edge':       l_edge,
            'edge_bce':   l_edge_bce,
            'edge_focal': l_edge_focal,
        }


# ============================================================
# 2. 评估指标
# ============================================================

class MetricTracker:
    """在线累积计算 IoU、Dice、边缘 F1。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = 0
        self.edge_tp = self.edge_fp = self.edge_fn = 0
        self.loss_sum = 0.0
        self.n = 0

    def update(self, pred_seg: torch.Tensor, target_seg: torch.Tensor,
               pred_edge: torch.Tensor, target_edge: torch.Tensor,
               loss: float):
        # 二值化
        ps = (pred_seg  > 0.5).float()
        ts = (target_seg > 0.5).float()
        pe = (pred_edge  > 0.3).float()
        te = (target_edge > 0.5).float()

        self.tp += (ps * ts).sum().item()
        self.fp += (ps * (1 - ts)).sum().item()
        self.fn += ((1 - ps) * ts).sum().item()

        self.edge_tp += (pe * te).sum().item()
        self.edge_fp += (pe * (1 - te)).sum().item()
        self.edge_fn += ((1 - pe) * te).sum().item()

        self.loss_sum += loss
        self.n += 1

    def compute(self) -> Dict[str, float]:
        eps = 1e-8
        iou  = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        prec = self.edge_tp / (self.edge_tp + self.edge_fp + eps)
        rec  = self.edge_tp / (self.edge_tp + self.edge_fn + eps)
        ef1  = 2 * prec * rec / (prec + rec + eps)
        return {
            'loss':      self.loss_sum / max(self.n, 1),
            'iou':       iou,
            'dice':      dice,
            'edge_f1':   ef1,
            'edge_prec': prec,
            'edge_rec':  rec,
        }


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
    criterion: AGEANetLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
) -> Dict[str, float]:

    model.train()
    tracker = MetricTracker()

    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        masks  = batch['mask'].to(device, non_blocking=True)
        edges  = batch['edge'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            pred  = model(images)
            losses = criterion(pred, masks, edges)
            loss   = losses['total']

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

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
    criterion: AGEANetLoss,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:

    model.eval()
    tracker = MetricTracker()

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks  = batch['mask'].to(device, non_blocking=True)
        edges  = batch['edge'].to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            pred   = model(images)
            losses = criterion(pred, masks, edges)

        tracker.update(
            pred['seg'], masks,
            pred['edge'], edges,
            losses['total'].item(),
        )

    return tracker.compute()


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
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt.get('epoch', 0) + 1
    print(f"[train] 从检查点恢复: {path}（epoch {ckpt.get('epoch', 0)}）")
    return start_epoch


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
        model = AGEANetLite(in_channels=3, base_ch=args.base_ch).to(device)
    else:
        model = AGEANet(in_channels=3, base_ch=args.base_ch).to(device)

    info = get_model_info(model)
    print(f"[train] 模型: {args.model}, 参数量: {info['total_params']:,}, "
          f"FP32: {info['size_mb_fp32']:.1f} MB")

    # --- 损失函数 ---
    criterion = AGEANetLoss(w_seg=1.0, w_edge=args.edge_weight)

    # --- 优化器 ---
    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)

    # --- 学习率调度 ---
    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = int(total_steps * 0.05)
    scheduler     = build_cosine_warmup_scheduler(
        optimizer, total_steps, warmup_steps, min_lr_ratio=0.01
    )
    scaler = GradScaler(enabled=use_amp)

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
        )
        val_metrics = validate(model, val_loader, criterion, device, use_amp)

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
    parser = argparse.ArgumentParser(description="AGEANet 训练程序")

    # 数据
    parser.add_argument("--synth_dir",         type=str,   default=None)
    parser.add_argument("--real_dir",           type=str,   default=None)
    parser.add_argument("--real_ratio",         type=float, default=0.3)
    parser.add_argument("--online_epoch_size",  type=int,   default=2000)
    parser.add_argument("--img_size",           type=int,   default=512)

    # 模型
    parser.add_argument("--model",   type=str, default='standard',
                        choices=['standard', 'lite'])
    parser.add_argument("--base_ch", type=int, default=64)

    # 训练
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--edge_weight",  type=float, default=0.5,
                        help="边缘损失权重（相对于分割损失）")
    parser.add_argument("--num_workers",  type=int,   default=4)

    # 检查点
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume",   action="store_true",
                        help="从 save_dir/last.pth 恢复训练")

    args = parser.parse_args()
    main(args)
