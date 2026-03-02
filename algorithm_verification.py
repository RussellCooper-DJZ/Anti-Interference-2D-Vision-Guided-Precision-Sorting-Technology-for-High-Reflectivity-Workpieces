"""
algorithm_verification.py — 算法验证与评估模块

功能：
  1. 合成高反光场景测试 — 自动生成测试用例并评估
  2. 模型精度评估 — IoU, Dice, 边缘精度, F1 等指标
  3. 抗干扰能力测试 — 不同反光强度/角度下的鲁棒性
  4. 处理速度基准测试 — 各模块耗时统计
  5. 可视化报告生成 — 对比图和指标图表
"""

import os
import time
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ===========================================================================
# 1. 评估指标
# ===========================================================================

def compute_iou(pred, gt, threshold=0.5):
    """计算 IoU (Intersection over Union)。"""
    pred_bin = (pred > threshold * 255 if pred.max() > 1 else pred > threshold).astype(np.uint8)
    gt_bin = (gt > threshold * 255 if gt.max() > 1 else gt > threshold).astype(np.uint8)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    return intersection / (union + 1e-8)


def compute_dice(pred, gt, threshold=0.5):
    """计算 Dice 系数。"""
    pred_bin = (pred > threshold * 255 if pred.max() > 1 else pred > threshold).astype(np.uint8)
    gt_bin = (gt > threshold * 255 if gt.max() > 1 else gt > threshold).astype(np.uint8)
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    return 2 * intersection / (pred_bin.sum() + gt_bin.sum() + 1e-8)


def compute_boundary_f1(pred, gt, tolerance=2):
    """
    计算边缘 F1 分数。

    在 tolerance 像素容差内评估边缘检测的精确率和召回率。
    """
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (2 * tolerance + 1, 2 * tolerance + 1))
    gt_dilated = cv2.dilate(gt_bin, kernel)
    pred_dilated = cv2.dilate(pred_bin, kernel)

    precision = np.logical_and(pred_bin, gt_dilated).sum() / max(pred_bin.sum(), 1)
    recall = np.logical_and(gt_bin, pred_dilated).sum() / max(gt_bin.sum(), 1)

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {'precision': float(precision), 'recall': float(recall), 'f1': float(f1)}


def compute_all_metrics(pred_mask, gt_mask, pred_edge=None, gt_edge=None):
    """计算所有评估指标。"""
    metrics = {
        'iou': compute_iou(pred_mask, gt_mask),
        'dice': compute_dice(pred_mask, gt_mask),
    }
    if pred_edge is not None and gt_edge is not None:
        edge_metrics = compute_boundary_f1(pred_edge, gt_edge)
        metrics.update({
            'edge_precision': edge_metrics['precision'],
            'edge_recall': edge_metrics['recall'],
            'edge_f1': edge_metrics['f1'],
        })
    return metrics


# ===========================================================================
# 2. 合成测试场景
# ===========================================================================

def generate_test_scenarios(count=10, img_size=256):
    """
    生成多种高反光测试场景。

    场景类型：baseline, light_glare, strong_glare
    """
    from data_augmentation import SyntheticMetalWorkpieceGenerator

    generator = SyntheticMetalWorkpieceGenerator(img_size=img_size)
    scenarios = []
    Y, X = np.ogrid[:img_size, :img_size]

    for i in range(count):
        sample = generator.generate()
        image = sample['image']
        mask = sample['mask']
        edge = sample['edge_mask']

        # 基线
        scenarios.append({
            'name': f'baseline_{i}', 'type': 'baseline',
            'image': image.copy(), 'mask': mask.copy(), 'edge': edge.copy(),
        })

        # 轻微反光
        light_glare = image.copy().astype(np.float32)
        cx, cy = np.random.randint(50, img_size - 50, 2)
        dist = np.sqrt((X - int(cx)) ** 2 + (Y - int(cy)) ** 2)
        glare = np.exp(-dist ** 2 / (2 * 30 ** 2)) * 80
        for c in range(3):
            light_glare[:, :, c] += glare
        scenarios.append({
            'name': f'light_glare_{i}', 'type': 'light_glare',
            'image': np.clip(light_glare, 0, 255).astype(np.uint8),
            'mask': mask.copy(), 'edge': edge.copy(),
        })

        # 强反光
        strong_glare = image.copy().astype(np.float32)
        cx2, cy2 = np.random.randint(50, img_size - 50, 2)
        dist2 = np.sqrt((X - int(cx2)) ** 2 + (Y - int(cy2)) ** 2)
        glare2 = np.exp(-dist2 ** 2 / (2 * 50 ** 2)) * 200
        for c in range(3):
            strong_glare[:, :, c] += glare2
        scenarios.append({
            'name': f'strong_glare_{i}', 'type': 'strong_glare',
            'image': np.clip(strong_glare, 0, 255).astype(np.uint8),
            'mask': mask.copy(), 'edge': edge.copy(),
        })

    return scenarios


# ===========================================================================
# 3. 模型评估
# ===========================================================================

def evaluate_model(model, scenarios, device, infer_size=256):
    """在测试场景上评估模型。"""
    from main_system import _predict_masks

    results = {}
    for scenario in scenarios:
        stype = scenario['type']
        if stype not in results:
            results[stype] = {'iou': [], 'dice': [], 'edge_f1': []}

        seg_mask, edge_map = _predict_masks(
            model, scenario['image'], device, infer_size=infer_size
        )
        metrics = compute_all_metrics(
            seg_mask, scenario['mask'],
            (edge_map > 0.3).astype(np.uint8) * 255, scenario['edge'],
        )
        results[stype]['iou'].append(metrics['iou'])
        results[stype]['dice'].append(metrics['dice'])
        if 'edge_f1' in metrics:
            results[stype]['edge_f1'].append(metrics['edge_f1'])

    summary = {}
    for stype, vals in results.items():
        summary[stype] = {
            k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
            for k, v in vals.items() if len(v) > 0
        }
    return summary


# ===========================================================================
# 4. 速度基准测试
# ===========================================================================

def benchmark_speed(model, device, infer_size=256, num_runs=20):
    """推理速度基准测试。"""
    import torch

    model.eval()
    dummy = torch.randn(1, 3, infer_size, infer_size).to(device)

    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.time()
            _ = model(dummy)
            times.append((time.time() - t0) * 1000)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
        'fps': float(1000 / np.mean(times)),
    }


# ===========================================================================
# 5. 可视化
# ===========================================================================

def visualize_prediction(image, seg_mask, edge_map=None, detections=None):
    """生成预测结果可视化图。"""
    vis = image.copy()

    mask_overlay = np.zeros_like(vis)
    mask_overlay[:, :, 1] = seg_mask
    vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

    if edge_map is not None:
        edge_bin = (edge_map > 0.3 if edge_map.max() <= 1 else edge_map > 128).astype(np.uint8)
        vis[edge_bin > 0] = [0, 0, 255]

    if detections:
        for i, det in enumerate(detections):
            cx, cy = int(det['center'][0]), int(det['center'][1])
            cv2.circle(vis, (cx, cy), 5, (255, 0, 0), -1)
            cv2.putText(vis, f"#{i+1}", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    return vis


# ===========================================================================
# 6. 完整验证流程
# ===========================================================================

def full_verification(model_path=None, output_dir='./verification_results'):
    """完整算法验证流程。"""
    import torch
    from feature_extraction import AGEANetLite

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cpu')

    model = AGEANetLite(in_channels=3, out_channels=1, base_ch=32)
    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print("=" * 60)
    print("AGEANet 算法验证")
    print("=" * 60)

    # 生成测试场景
    print("\n1. 生成测试场景...")
    scenarios = generate_test_scenarios(count=5, img_size=256)
    print(f"   共 {len(scenarios)} 个测试场景")

    # 评估精度
    print("\n2. 评估模型精度...")
    accuracy_results = evaluate_model(model, scenarios, device)
    for stype, metrics in accuracy_results.items():
        print(f"   {stype}:")
        for metric, vals in metrics.items():
            print(f"     {metric}: {vals['mean']:.4f} +/- {vals['std']:.4f}")

    # 速度测试
    print("\n3. 速度基准测试...")
    speed_results = benchmark_speed(model, device, num_runs=10)
    print(f"   推理时间: {speed_results['mean_ms']:.1f} +/- {speed_results['std_ms']:.1f} ms")
    print(f"   FPS: {speed_results['fps']:.1f}")

    # 保存报告
    report = {'accuracy': accuracy_results, 'speed': speed_results}
    report_path = output_dir / 'verification_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n报告已保存: {report_path}")

    # 可视化
    print("\n4. 生成可视化...")
    from main_system import _predict_masks
    for scenario in scenarios[:6]:
        seg_mask, edge_map = _predict_masks(model, scenario['image'], device)
        vis = visualize_prediction(scenario['image'], seg_mask, edge_map)
        vis_path = output_dir / f"vis_{scenario['name']}.png"
        cv2.imwrite(str(vis_path), vis)

    print(f"   可视化保存到: {output_dir}")
    print("\n验证完成！")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--output', type=str, default='./verification_results')
    args = parser.parse_args()
    full_verification(args.model, args.output)
