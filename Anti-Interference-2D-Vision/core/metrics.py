"""
core/metrics.py — 统一的指标追踪模块
统一 MetricTracker、TTA 推理等功能
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class MetricTracker:
    """
    在线累积计算 IoU、Dice、边缘 F1。

    用法::

        tracker = MetricTracker()
        for batch in loader:
            tracker.update(pred_seg, target_seg, pred_edge, target_edge, loss)
        metrics = tracker.compute()
    """

    def __init__(self, edge_threshold: float = 0.40):
        """
        Args:
            edge_threshold: 边缘检测阈值（默认 0.40）
        """
        self.edge_threshold = edge_threshold
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = 0
        self.edge_tp = self.edge_fp = self.edge_fn = 0
        self.loss_sum = 0.0
        self.n = 0

    def update(
        self,
        pred_seg: torch.Tensor,
        target_seg: torch.Tensor,
        pred_edge: torch.Tensor,
        target_edge: torch.Tensor,
        loss: float = 0.0,
    ):
        # 二值化 - 必须对 logits 应用 sigmoid
        ps = (torch.sigmoid(pred_seg) > 0.5).float()
        ts = (target_seg > 0.5).float()
        pe = (torch.sigmoid(pred_edge) > self.edge_threshold).float()
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
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        prec = self.edge_tp / (self.edge_tp + self.edge_fp + eps)
        rec = self.edge_tp / (self.edge_tp + self.edge_fn + eps)
        ef1 = 2 * prec * rec / (prec + rec + eps)
        return {
            'loss': self.loss_sum / max(self.n, 1),
            'iou': iou,
            'dice': dice,
            'edge_f1': ef1,
            'edge_prec': prec,
            'edge_rec': rec,
        }


def tta_inference(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Test Time Augmentation（TTA）：4x 翻转平均

    Args:
        model: 分割模型
        images: (B, C, H, W) 输入图像
        device: 计算设备

    Returns:
        {'seg': (B, 1, H, W), 'edge': (B, 1, H, W)} 融合后的预测
    """
    preds: List[Dict[str, torch.Tensor]] = []

    with torch.no_grad():
        # 原图
        preds.append(model(images))

        # 水平翻转
        images_hflip = torch.flip(images, dims=[3])
        p = model(images_hflip)
        p['seg'] = torch.flip(p['seg'], dims=[3])
        p['edge'] = torch.flip(p['edge'], dims=[3])
        preds.append(p)

        # 垂直翻转
        images_vflip = torch.flip(images, dims=[2])
        p = model(images_vflip)
        p['seg'] = torch.flip(p['seg'], dims=[2])
        p['edge'] = torch.flip(p['edge'], dims=[2])
        preds.append(p)

        # 对角翻转
        images_dflip = torch.flip(images, dims=[2, 3])
        p = model(images_dflip)
        p['seg'] = torch.flip(p['seg'], dims=[2, 3])
        p['edge'] = torch.flip(p['edge'], dims=[2, 3])
        preds.append(p)

    # 平均融合
    seg_avg = torch.stack([p['seg'] for p in preds]).mean(dim=0)
    edge_avg = torch.stack([p['edge'] for p in preds]).mean(dim=0)

    return {'seg': seg_avg, 'edge': edge_avg}
