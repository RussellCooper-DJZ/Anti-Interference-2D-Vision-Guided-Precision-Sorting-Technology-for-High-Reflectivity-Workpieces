"""
evaluate.py — FLARE 模型评估脚本
:Author: RussellCooper

功能：
  - 加载检查点并在指定数据集上评估模型性能
  - 计算 IoU、Dice、边缘 F1、Precision、Recall
  - 生成预测可视化对比图
  - 支持 TTA（Test Time Augmentation）推理

用法::

    # 评估最佳检查点
    python training/evaluate.py --checkpoint checkpoints/best.pth --data dataset_merged

    # 使用 TTA 评估
    python training/evaluate.py --checkpoint checkpoints/best.pth --data dataset_merged --tta

    # 指定输出目录
    python training/evaluate.py --checkpoint checkpoints/best.pth --data dataset_merged --output eval_results
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.feature_extraction import FLARE, FLARELite, get_model_info
from data.real_world_dataloader import SynthShipDataset
from core.metrics import MetricTracker, tta_inference


# ============================================================
# 1. 指标计算
# ============================================================
# 使用 core.metrics 中的统一 MetricTracker


# ============================================================
# 3. 可视化
# ============================================================

def visualize_prediction(image: np.ndarray, pred_seg: np.ndarray,
                        pred_edge: np.ndarray, target_seg: np.ndarray,
                        target_edge: np.ndarray, name: str,
                        output_dir):
    """生成预测结果可视化对比图"""
    output_dir = Path(output_dir)

    # 确保图像格式正确
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (512, 512))

    # 归一化预测
    pred_seg_vis = (pred_seg * 255).astype(np.uint8)
    pred_edge_vis = (pred_edge * 255).astype(np.uint8)
    target_seg_vis = (target_seg * 255).astype(np.uint8)
    target_edge_vis = (target_edge * 255).astype(np.uint8)

    # 创建彩色分割叠加
    def blend_overlay(img, mask, color=(0, 255, 0), alpha=0.5):
        mask_colored = np.zeros_like(img)
        mask_colored[mask > 0.5] = color
        return cv2.addWeighted(img, 1, mask_colored, alpha, 0)

    # 原图 + 预测分割
    pred_overlay = blend_overlay(image.copy(), pred_seg > 0.5, (0, 255, 0))
    # 原图 + 真值分割
    gt_overlay = blend_overlay(image.copy(), target_seg > 0.5, (0, 255, 255))

    # 边缘对比
    pred_edge_color = cv2.cvtColor(pred_edge_vis, cv2.COLOR_GRAY2BGR)
    pred_edge_color[:, :, 2] = np.where(pred_edge > 0.40, 255, pred_edge_color[:, :, 2])
    target_edge_color = cv2.cvtColor(target_edge_vis, cv2.COLOR_GRAY2BGR)
    target_edge_color[:, :, 0] = np.where(target_edge > 0.5, 255, target_edge_color[:, :, 0])

    # 拼接为一行
    top_row = np.hstack([
        image,
        pred_overlay,
        gt_overlay,
    ])

    bottom_row = np.hstack([
        cv2.cvtColor(pred_seg_vis, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(target_seg_vis, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(pred_edge_vis, cv2.COLOR_GRAY2BGR),
    ])

    result = np.vstack([top_row, bottom_row])

    # 添加标签
    cv2.putText(result, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, "Pred Seg", (522, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, "GT Seg", (1034, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 保存
    output_path = output_dir / f"{name}_eval.png"
    cv2.imwrite(str(output_path), result)


# ============================================================
# 4. 评估函数
# ============================================================

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_tta: bool = False,
    visualize_samples: int = 10,
    output_dir: Path = None,
) -> Dict[str, float]:
    """在数据集上评估模型"""

    model.eval()
    tracker = MetricTracker()
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    vis_counter = 0

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks  = batch['mask'].to(device, non_blocking=True)
        edges  = batch['edge'].to(device, non_blocking=True)
        names  = batch.get('name', [f"sample_{i}" for i in range(len(images))])

        # 推理
        with autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            if use_tta:
                pred = tta_inference(model, images, device)
            else:
                pred = model(images)

        # 计算指标
        tracker.update(pred['seg'], masks, pred['edge'], edges)

        # 可视化前 N 个样本
        if output_dir and vis_counter < visualize_samples:
            for i in range(min(len(images), visualize_samples - vis_counter)):
                idx = vis_counter + i
                if idx >= visualize_samples:
                    break

                # 转为 CPU numpy
                img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                if img_np.shape[-1] == 1:
                    img_np = img_np.squeeze(-1)

                pred_seg_np = pred['seg'][i, 0].cpu().numpy()
                pred_edge_np = pred['edge'][i, 0].cpu().numpy()
                mask_np = masks[i, 0].cpu().numpy()
                edge_np = edges[i, 0].cpu().numpy()

                name = names[i] if isinstance(names, list) else f"sample_{idx}"
                visualize_prediction(
                    img_np, pred_seg_np, pred_edge_np, mask_np, edge_np,
                    str(name), output_dir
                )
                vis_counter += 1

    return tracker.compute()


# ============================================================
# 5. 主程序
# ============================================================

def main(args):
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[evaluate] 设备: {device}")

    # 加载模型
    if args.model == 'lite':
        model = FLARELite(in_channels=3, base_ch=args.base_ch)
    else:
        model = FLARE(in_channels=3, base_ch=args.base_ch)

    # 加载检查点
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"[evaluate] 加载检查点: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt)
        model.load_state_dict(state, strict=False)

        # 显示检查点信息
        if 'epoch' in ckpt:
            print(f"  - Epoch: {ckpt['epoch']}")
        if 'val_iou' in ckpt:
            print(f"  - Val IoU: {ckpt['val_iou']:.4f}")
    else:
        print(f"[evaluate] 警告: 检查点不存在 {args.checkpoint}，使用随机权重")

    model = model.to(device)

    # 打印模型信息
    info = get_model_info(model)
    print(f"[evaluate] 模型: {args.model}, 参数量: {info['total_params']:,}")

    # 数据加载器
    print(f"[evaluate] 加载数据集: {args.data}")
    dataset = SynthShipDataset(
        root=args.data,
        img_size=512,
        augment=False,  # 评估时不增强
        split='val',
        val_ratio=1.0,  # 评估时使用全部数据
        seed=42,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"[evaluate] 样本数: {len(dataset)}, Batch size: {args.batch_size}")
    print(f"[evaluate] TTA: {'启用' if args.tta else '禁用'}")

    # 评估
    t0 = time.time()
    metrics = evaluate(
        model, loader, device,
        use_tta=args.tta,
        visualize_samples=args.visualize,
        output_dir=args.output,
    )
    elapsed = time.time() - t0

    # 输出结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)
    print(f"  IoU:        {metrics['iou']:.4f}")
    print(f"  Dice:       {metrics['dice']:.4f}")
    print(f"  Edge F1:    {metrics['edge_f1']:.4f}")
    print(f"  Edge Prec:  {metrics['edge_prec']:.4f}")
    print(f"  Edge Rec:   {metrics['edge_rec']:.4f}")
    print("=" * 50)
    print(f"用时: {elapsed:.1f}s")

    # 保存结果到文件
    result_path = Path(args.output) / "metrics.txt"
    with open(result_path, 'w') as f:
        f.write(f"IoU: {metrics['iou']:.4f}\n")
        f.write(f"Dice: {metrics['dice']:.4f}\n")
        f.write(f"Edge F1: {metrics['edge_f1']:.4f}\n")
        f.write(f"Edge Precision: {metrics['edge_prec']:.4f}\n")
        f.write(f"Edge Recall: {metrics['edge_rec']:.4f}\n")
    print(f"\n结果已保存: {result_path}")

    # 可视化样本
    if args.visualize > 0:
        print(f"可视化样本已保存: {args.output}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FLARE 模型评估")

    # 模型
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/best.pth",
                        help="模型检查点路径")
    parser.add_argument("--model", type=str, default='standard',
                        choices=['standard', 'lite'])
    parser.add_argument("--base_ch", type=int, default=64)

    # 数据
    parser.add_argument("--data", type=str, default="dataset_merged",
                        help="数据集路径")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    # 推理
    parser.add_argument("--tta", action="store_true",
                        help="启用 Test Time Augmentation")
    parser.add_argument("--visualize", type=int, default=10,
                        help="可视化样本数量")

    # 输出
    parser.add_argument("--output", type=str, default="eval_results",
                        help="输出目录")

    args = parser.parse_args()
    main(args)
