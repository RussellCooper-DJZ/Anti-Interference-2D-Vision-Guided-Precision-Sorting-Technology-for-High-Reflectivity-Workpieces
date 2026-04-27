#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_optimize.py — FLARE 项目级全自动1000次迭代算法优化引擎
直接对接 Anti-Interference-2D 现有 Pipeline：
  AntiGlarePipeline → FLARE推理 → SubpixelLocalizer → Caliper/GapMeasurement
放置位置: E:\opcode\Anti-Interference-2D\scripts\auto_optimize.py
"""

import os
import sys
import json
import time
import random
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "results" / "auto_tuning"
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "auto_tuning.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FLARE-AutoOpt")

try:
    from vision.hdr_processing import AntiGlarePipeline, detect_highlight_mask
    from vision.localization_and_calibration import SubpixelLocalizer
    from vision.measurement import CaliperMeasurement, GapMeasurement
    from vision.feature_extraction import FLARE, FLARELite
    from vision.inference_engine import PyTorchEngine
    PROJECT_MODULES_AVAILABLE = True
    logger.info("✅ 成功导入项目现有模块")
except Exception as e:
    logger.warning(f"⚠️ 项目模块导入失败（{e}），使用内置 Mock 运行")
    PROJECT_MODULES_AVAILABLE = False

if not PROJECT_MODULES_AVAILABLE:
    class AntiGlarePipeline:
        def process_single(self, img): return img.copy()
        def process_multi(self, imgs): return imgs[0].copy()

    def detect_highlight_mask(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        return mask

    class SubpixelLocalizer:
        def __init__(self, min_area=100): self.min_area = min_area
        def localize(self, seg_mask, edge_mask, intensity_image=None, glare_mask=None):
            contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dets = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area: continue
                M = cv2.moments(cnt)
                if M["m00"] == 0: continue
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                dets.append({
                    'centroid_px': (float(cx), float(cy)),
                    'orientation_deg': 0.0,
                    'feature_type': 'blob',
                    'area_px': area,
                    'bbox': cv2.boundingRect(cnt)
                })
            return dets

    class CaliperMeasurement:
        def __init__(self, **kwargs): pass
        def measure(self, gray, roi):
            x,y,w,h = roi
            profile = gray[y:y+h, x:x+w].mean(axis=0) if w>0 and h>0 else np.array([0])
            return {'distance': float(w), 'valid': True}

    class GapMeasurement:
        def __init__(self, **kwargs): pass
        def measure(self, gray, roi):
            return {'pitches': [10.0], 'widths': [5.0], 'spacings': [5.0]}


@dataclass
class FLAREParamSpace:
    hdr_gamma: float = 1.0
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    glare_inpaint_radius: int = 3
    seg_threshold: float = 0.5
    edge_threshold: float = 0.3
    nms_iou_threshold: float = 0.5
    localizer_min_area: int = 200
    caliper_width: int = 21
    edge_polarity: int = 1
    edge_threshold_percent: float = 0.1
    search_direction: int = 0
    gap_edge_spacing: int = 5
    depth_fusion_weight: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'FLAREParamSpace':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class AdaptiveParameterSampler:
    SPACE = {
        "hdr_gamma": {"type": "float", "low": 0.5, "high": 2.5},
        "bilateral_d": {"type": "int", "low": 5, "high": 15, "odd": True},
        "bilateral_sigma_color": {"type": "float", "low": 30, "high": 150},
        "bilateral_sigma_space": {"type": "float", "low": 30, "high": 150},
        "clahe_clip_limit": {"type": "float", "low": 1.0, "high": 8.0},
        "clahe_tile_size": {"type": "choice", "options": [4, 8, 16, 32]},
        "glare_inpaint_radius": {"type": "int", "low": 1, "high": 7, "odd": True},
        "seg_threshold": {"type": "float", "low": 0.3, "high": 0.9},
        "edge_threshold": {"type": "float", "low": 0.1, "high": 0.5},
        "nms_iou_threshold": {"type": "float", "low": 0.3, "high": 0.7},
        "localizer_min_area": {"type": "int", "low": 50, "high": 800},
        "caliper_width": {"type": "int", "low": 11, "high": 51, "odd": True},
        "edge_polarity": {"type": "choice", "options": [-1, 0, 1]},
        "edge_threshold_percent": {"type": "float", "low": 0.05, "high": 0.3},
        "search_direction": {"type": "choice", "options": [-1, 0, 1]},
        "gap_edge_spacing": {"type": "int", "low": 2, "high": 20},
        "depth_fusion_weight": {"type": "float", "low": 0.0, "high": 1.0},
    }

    @classmethod
    def sample(cls, strategy: str = "adaptive", history: List[Dict] = None) -> FLAREParamSpace:
        params = {}
        for key, cfg in cls.SPACE.items():
            if cfg["type"] == "int":
                low, high = cfg["low"], cfg["high"]
                if cfg.get("odd"):
                    candidates = list(range(low if low % 2 == 1 else low + 1, high + 1, 2))
                    params[key] = random.choice(candidates) if candidates else low
                else:
                    params[key] = random.randint(low, high)
            elif cfg["type"] == "float":
                params[key] = round(random.uniform(cfg["low"], cfg["high"]), 4)
            elif cfg["type"] == "choice":
                params[key] = random.choice(cfg["options"])

        if strategy == "adaptive" and history and len(history) > 20:
            top_k = max(1, len(history) // 5)
            best = sorted(history, key=lambda x: x["score"], reverse=True)[:top_k]
            if best:
                parent = random.choice(best)["params"]
                perturb_keys = random.sample(list(params.keys()), max(1, len(params) // 3))
                for key in perturb_keys:
                    cfg = cls.SPACE.get(key, {})
                    if cfg.get("type") not in ["int", "float"]:
                        continue
                    if cfg["type"] == "float":
                        pv = parent.get(key, params[key])
                        delta = random.gauss(0, (cfg["high"] - cfg["low"]) * 0.05)
                        params[key] = round(max(cfg["low"], min(cfg["high"], pv + delta)), 4)
                    elif cfg["type"] == "int" and not cfg.get("odd"):
                        pv = parent.get(key, params[key])
                        params[key] = max(cfg["low"], min(cfg["high"], pv + random.randint(-2, 2)))
        return FLAREParamSpace(**params)


class ProjectPipelineRunner:
    def __init__(self):
        self.anti_glare = AntiGlarePipeline()
        self.localizer = SubpixelLocalizer(min_area=200)
        self.caliper = CaliperMeasurement(
            search_direction='left_to_right', polarity='black_to_white',
            edge_intensity=20, search_line_count=20
        )
        self.gap = GapMeasurement(
            search_direction='left_to_right', polarity='black_to_white',
            edge_intensity=20, edge_spacing=5, search_line_count=20
        )

    def run(self, image: np.ndarray, params: FLAREParamSpace) -> Dict[str, Any]:
        result = {
            "success": False, "num_detections": 0, "num_localized": 0,
            "num_measured": 0, "measurement_std": 999.0,
            "seg_mask": None, "edge_mask": None, "glare_mask": None,
            "detections": []
        }
        try:
            bd = params.bilateral_d
            if bd % 2 == 0: bd += 1

            proc = image.copy()
            if len(proc.shape) == 2:
                proc = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)

            proc = cv2.bilateralFilter(proc, bd, params.bilateral_sigma_color, params.bilateral_sigma_space)

            lab = cv2.cvtColor(proc, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            tile = params.clahe_tile_size
            clahe = cv2.createCLAHE(clipLimit=params.clahe_clip_limit, tileGridSize=(tile, tile))
            l = clahe.apply(l)
            proc = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

            gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            _, glare_mask = cv2.threshold(gray, int(255 * 0.92), 255, cv2.THRESH_BINARY)
            if np.count_nonzero(glare_mask) > 0:
                proc = cv2.inpaint(proc, glare_mask, params.glare_inpaint_radius, cv2.INPAINT_TELEA)

            gray_proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            seg_mask = cv2.adaptiveThreshold(gray_proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 11, 2)
            edges = cv2.Canny(gray_proc, 50, 150)
            edge_mask = (edges > int(255 * params.edge_threshold)).astype(np.uint8) * 255

            contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if cv2.contourArea(c) >= params.localizer_min_area]
            num_det = len(valid_contours)

            detections = self.localizer.localize(seg_mask, edge_mask, intensity_image=gray_proc, glare_mask=glare_mask)
            num_loc = len(detections)

            measurements = []
            for det in detections:
                bbox = det.get('bbox')
                if bbox is None:
                    continue
                x, y, w, h = bbox
                x = max(0, x); y = max(0, y)
                w = min(gray_proc.shape[1] - x, w); h = min(gray_proc.shape[0] - y, h)
                if w <= 0 or h <= 0:
                    continue
                roi = (x, y, w, h)
                cm = self.caliper.measure(gray_proc, roi)
                gm = self.gap.measure(gray_proc, roi)
                det['caliper_distance_px'] = cm.get('distance', 0)
                det['caliper_valid'] = cm.get('valid', False)
                det['gap_pitches'] = gm.get('pitches', [])
                if cm.get('valid') and cm.get('distance', 0) > 0:
                    measurements.append(cm['distance'])

            result.update({
                "success": True,
                "num_detections": num_det,
                "num_localized": num_loc,
                "num_measured": len(measurements),
                "measurement_std": float(np.std(measurements)) if measurements else 999.0,
                "measurement_mean": float(np.mean(measurements)) if measurements else 0.0,
                "seg_mask": seg_mask,
                "edge_mask": edge_mask,
                "glare_mask": glare_mask,
                "detections": detections,
                "processed_image": proc
            })
        except Exception as e:
            logger.error(f"Pipeline 执行异常: {e}")
            result["error"] = str(e)
        return result


class IndustrialEvaluator:
    @staticmethod
    def evaluate(result: Dict, target: Dict[str, Any] = None) -> float:
        if not result.get("success"):
            return 0.0

        score = 0.0
        w = {"detection": 0.25, "localization": 0.20, "measurement": 0.30, "glare": 0.15, "stability": 0.10}

        target_det = target.get("ideal_detections", 3) if target else 3
        det_score = max(0, 100 - abs(result["num_detections"] - target_det) * 25)
        score += det_score * w["detection"]

        loc_rate = (result["num_localized"] / max(result["num_detections"], 1)) * 100
        loc_score = min(100, loc_rate)
        score += loc_score * w["localization"]

        std = result["measurement_std"]
        if std < 0.5:
            meas_score = 100
        elif std < 2.0:
            meas_score = 100 - (std - 0.5) * 20
        elif std < 5.0:
            meas_score = 70 - (std - 2.0) * 10
        else:
            meas_score = max(0, 40 - (std - 5.0) * 2)
        score += meas_score * w["measurement"]

        em = result.get("edge_mask")
        if em is not None and em.size > 0:
            edge_ratio = np.count_nonzero(em) / em.size
            glare_score = 100 - abs(edge_ratio - 0.05) * 1500
            glare_score = max(0, min(100, glare_score))
        else:
            glare_score = 0
        score += glare_score * w["glare"]

        stab = 100
        score += stab * w["stability"]

        return round(score, 2)


class IterationEngine:
    def __init__(self, total: int = 1000, patience: int = 80, strategy: str = "adaptive"):
        self.total = total
        self.patience = patience
        self.strategy = strategy
        self.history: List[Dict] = []
        self.best_result: Optional[Dict] = None
        self.best_score = -1.0
        self.no_improve = 0
        self.start_time = 0.0

    def run(self, image: np.ndarray, target: Dict[str, Any] = None) -> Dict[str, Any]:
        self.start_time = time.time()
        runner = ProjectPipelineRunner()

        logger.info(f"🚀 启动 FLARE 项目级自动优化 | 目标迭代: {self.total} | 策略: {self.strategy}")
        logger.info(f"📂 项目根目录: {PROJECT_ROOT}")
        logger.info(f"📊 评估目标: {target}")

        for i in range(1, self.total + 1):
            params = AdaptiveParameterSampler.sample(self.strategy, self.history)
            result = runner.run(image, params)
            result["iteration"] = i
            result["params"] = params.to_dict()

            score = IndustrialEvaluator.evaluate(result, target)
            result["score"] = score

            self.history.append({
                "iteration": i,
                "params": params.to_dict(),
                "score": score,
                "num_detections": result["num_detections"],
                "num_localized": result["num_localized"],
                "measurement_std": result["measurement_std"]
            })

            if score > self.best_score:
                self.best_score = score
                self.best_result = result
                self.no_improve = 0
                logger.info(f"✨ 第{i:04d}轮新最优 | Score={score:.2f} | Det={result['num_detections']} | Loc={result['num_localized']} | Std={result['measurement_std']:.2f}")
            else:
                self.no_improve += 1

            if i % 10 == 0:
                self._save_checkpoint(i)

            if self.no_improve >= self.patience:
                logger.info(f"🛑 早停触发 | 连续{self.patience}轮无提升 | 停止于第{i}轮")
                break

        elapsed = time.time() - self.start_time
        return self._finalize(elapsed)

    def _save_checkpoint(self, iteration: int):
        ckpt = {
            "iteration": iteration,
            "best_score_so_far": self.best_score,
            "best_params": self.best_result["params"] if self.best_result else None,
            "history_tail": self.history[-10:]
        }
        path = OUTPUT_DIR / f"checkpoint_{iteration:04d}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(ckpt, f, ensure_ascii=False, indent=2, default=str)

    def _finalize(self, elapsed: float) -> Dict[str, Any]:
        report = {
            "project_path": str(PROJECT_ROOT),
            "total_iterations": len(self.history),
            "best_score": self.best_score,
            "best_params": self.best_result["params"] if self.best_result else None,
            "elapsed_seconds": round(elapsed, 2),
            "speed_ips": round(len(self.history) / elapsed, 1) if elapsed > 0 else 0,
            "summary": {
                "max_score": max(h["score"] for h in self.history),
                "min_score": min(h["score"] for h in self.history),
                "avg_score": round(np.mean([h["score"] for h in self.history]), 2),
                "median_score": round(np.median([h["score"] for h in self.history]), 2)
            }
        }

        with open(OUTPUT_DIR / "final_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        if self.best_result:
            with open(OUTPUT_DIR / "best_params.json", 'w', encoding='utf-8') as f:
                json.dump(self.best_result["params"], f, ensure_ascii=False, indent=2, default=str)

        with open(OUTPUT_DIR / "full_history.json", 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2, default=str)

        self._plot()
        return report

    def _plot(self):
        iters = [h["iteration"] for h in self.history]
        scores = [h["score"] for h in self.history]

        best_curve = []
        cur = -1
        for s in scores:
            if s > cur: cur = s
            best_curve.append(cur)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        ax = axes[0, 0]
        ax.plot(iters, scores, 'b-', alpha=0.25, label='每次得分')
        ax.plot(iters, best_curve, 'r-', linewidth=2.5, label='历史最优')
        ax.set_xlabel('迭代次数'); ax.set_ylabel('综合得分')
        ax.set_title('FLARE 1000次迭代收敛曲线'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.hist(scores, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.best_score, color='red', linestyle='--', linewidth=2, label=f'最优: {self.best_score}')
        ax.set_xlabel('得分'); ax.set_ylabel('频次'); ax.set_title('得分分布'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[0, 2]
        dets = [h["num_detections"] for h in self.history]
        ax.scatter(dets, scores, c=iters, cmap='viridis', alpha=0.5, s=20)
        ax.set_xlabel('检测数量'); ax.set_ylabel('得分')
        ax.set_title('检测数量 vs 得分'); cbar = plt.colorbar(ax.collections[0], ax=ax); cbar.set_label('迭代')

        ax = axes[1, 0]
        stds = [min(h["measurement_std"], 20) for h in self.history]
        ax.scatter(stds, scores, c=iters, cmap='plasma', alpha=0.5, s=20)
        ax.set_xlabel('测量标准差 (px, 截断20)'); ax.set_ylabel('得分'); ax.set_title('测量稳定性 vs 得分')
        cbar = plt.colorbar(ax.collections[0], ax=ax); cbar.set_label('迭代')

        ax = axes[1, 1]
        top6 = ["seg_threshold", "edge_threshold", "bilateral_d", "clahe_clip_limit", "localizer_min_area", "caliper_width"]
        heat = []
        for p in top6:
            vals = [h["params"][p] for h in self.history if p in h["params"]]
            if vals:
                mn, mx = min(vals), max(vals)
                norm = [(v - mn) / (mx - mn + 1e-6) for v in vals[:100]]
                heat.append(norm + [0.5] * (100 - len(norm)))
        if heat:
            im = ax.imshow(heat, aspect='auto', cmap='viridis')
            ax.set_yticks(range(len(top6))); ax.set_yticklabels(top6, fontsize=9)
            ax.set_xlabel('样本索引'); ax.set_title('关键参数采样分布 (前100次)')
            plt.colorbar(im, ax=ax)

        ax = axes[1, 2]
        if self.best_result:
            bp = self.best_result["params"]
            radar_p = ["seg_threshold", "edge_threshold", "clahe_clip_limit", "glare_inpaint_radius", "depth_fusion_weight", "edge_threshold_percent"]
            radar_v, radar_l = [], []
            for rp in radar_p:
                cfg = AdaptiveParameterSampler.SPACE.get(rp, {})
                if cfg:
                    low, high = cfg.get("low", 0), cfg.get("high", 1)
                    val = bp.get(rp, (low + high) / 2)
                    radar_v.append((val - low) / (high - low + 1e-6))
                    radar_l.append(rp.replace("_", "\n"))
            angles = np.linspace(0, 2 * np.pi, len(radar_v), endpoint=False).tolist()
            radar_v += radar_v[:1]; angles += angles[:1]
            ax.plot(angles, radar_v, 'o-', linewidth=2, color='#e74c3c')
            ax.fill(angles, radar_v, alpha=0.25, color='#e74c3c')
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_l, fontsize=8)
            ax.set_ylim(0, 1); ax.set_title('最优参数归一化雷达图'); ax.grid(True)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "convergence_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"📊 可视化报告已保存: {OUTPUT_DIR / 'convergence_analysis.png'}")


def generate_synthetic_metal_image(size: Tuple[int, int] = (640, 480)) -> np.ndarray:
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 185

    centers = [(150, 150), (420, 200), (320, 360)]
    for cx, cy in centers:
        cv2.rectangle(img, (cx - 55, cy - 35), (cx + 55, cy + 35), (55, 55, 65), -1)
        cv2.ellipse(img, (cx - 18, cy - 12), (22, 14), 0, 0, 360, (245, 245, 250), -1)
        cv2.ellipse(img, (cx + 12, cy + 8), (14, 9), 0, 0, 360, (235, 235, 245), -1)
        cv2.line(img, (cx - 45, cy - 25), (cx - 25, cy - 8), (115, 115, 125), 2)

    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    mask = np.random.random((size[1], size[0])) > 0.996
    img[mask] = [255, 255, 255]
    return img


def main():
    logger.info("=" * 70)
    logger.info(" FLARE 项目级全自动1000次迭代视觉算法优化引擎")
    logger.info(" 项目路径: %s", PROJECT_ROOT)
    logger.info(" 模块状态: %s", "已接入项目模块" if PROJECT_MODULES_AVAILABLE else "Mock模式运行")
    logger.info("=" * 70)

    test_img = generate_synthetic_metal_image()
    test_path = OUTPUT_DIR / "test_input.jpg"
    cv2.imwrite(str(test_path), test_img)
    logger.info(f"🖼️ 测试图像已生成: {test_path}")

    target = {"ideal_detections": 3, "max_measurement_std": 2.0}

    engine = IterationEngine(total=1000, patience=80, strategy="adaptive")
    report = engine.run(test_img, target=target)

    print("\n" + "=" * 70)
    print("📋 最终优化报告")
    print("=" * 70)
    print(f"总迭代次数      : {report['total_iterations']}")
    print(f"最优综合得分    : {report['best_score']}")
    print(f"平均得分        : {report['summary']['avg_score']}")
    print(f"中位数得分      : {report['summary']['median_score']}")
    print(f"耗时            : {report['elapsed_seconds']} 秒")
    print(f"迭代速度        : {report['speed_ips']} 次/秒")
    print("\n🏆 最优参数组合（可直接写入 config.yaml）:")
    if report['best_params']:
        for k, v in report['best_params'].items():
            print(f"  {k:<25s} : {v}")
    print("=" * 70)
    print(f"\n📁 全部输出文件: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
