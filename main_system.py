"""
main_system.py — 高反光工件精准分拣系统主控

整合所有模块的完整推理管线：
  ISP 处理 → HDR 融合 → 反光抑制 → 深度学习分割 → 伪边缘剔除 →
  亚像素定位 → 坐标转换 → 抓取位姿输出

支持两种运行模式：
  1. 标准模式 — 使用 AGEANet 完整模型 (PC/GPU)
  2. 嵌入式模式 — 使用 AGEANet-Lite + TFLite (RA8P1)
"""

import time
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from simple_isp_simulator import SimpleISPSimulator
from hdr_processing import AntiGlarePipeline
from feature_extraction import AGEANet, AGEANetLite, get_model_info
from localization_and_calibration import (
    subpixel_edge_detection,
    filter_pseudo_edges,
    detect_multiple_workpieces,
    pixel_to_robot_coords,
    compute_grasp_pose,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ===========================================================================
# 推理辅助函数 (内存优化)
# ===========================================================================

def _predict_masks(model, image, device, infer_size=256,
                   seg_threshold=0.5, edge_threshold=0.3):
    """
    内存安全的推理：将大图缩放到 infer_size 后推理，再放大回原尺寸。

    参数:
        model:          AGEANet / AGEANetLite
        image:          BGR numpy (H, W, 3) uint8
        device:         torch.device
        infer_size:     推理尺寸 (正方形)
        seg_threshold:  分割阈值
        edge_threshold: 边缘阈值

    返回:
        seg_mask:  (H, W) uint8, 0/255
        edge_mask: (H, W) float32, [0,1]
    """
    model.eval()
    h, w = image.shape[:2]

    # 缩放到推理尺寸
    resized = cv2.resize(image, (infer_size, infer_size))

    with torch.no_grad():
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        tensor = tensor.to(device)
        output = model(tensor)

        seg = output['seg'].squeeze(0).squeeze(0).cpu().numpy()   # (infer_size, infer_size)
        edge = output['edge'].squeeze(0).squeeze(0).cpu().numpy()

    # 放大回原尺寸
    seg_full = cv2.resize(seg, (w, h), interpolation=cv2.INTER_LINEAR)
    edge_full = cv2.resize(edge, (w, h), interpolation=cv2.INTER_LINEAR)

    seg_mask = (seg_full > seg_threshold).astype(np.uint8) * 255
    return seg_mask, edge_full


# ===========================================================================
# 主系统类
# ===========================================================================

class HighReflectiveSortingSystem:
    """
    高反光工件精准分拣系统。

    完整处理流程：
    1. ISP 处理 — Bayer RAW → BGR (模拟 RZ/V2H ISP)
    2. HDR 融合 — 多重曝光融合 + 偏振模拟
    3. 反光抑制 — 高光检测/修复 + 自适应增强
    4. 深度学习分割 — AGEANet 边缘感知分割
    5. 后处理 — 伪边缘剔除 + 形态学优化
    6. 亚像素定位 — Zernike 矩精修
    7. 坐标转换 — 像素 → 机器人基座坐标
    8. 抓取位姿 — 计算最优抓取位置和角度
    """

    def __init__(self, model_path=None, model_type='lite',
                 isp_preset='high_reflectivity', device=None,
                 infer_size=256):
        """
        初始化系统。

        参数:
            model_path:  训练好的模型权重路径 (.pth)
            model_type:  'standard' (AGEANet) 或 'lite' (AGEANet-Lite)
            isp_preset:  ISP 预设
            device:      计算设备 (None=自动选择)
            infer_size:  推理图像尺寸 (缩放到此尺寸后推理)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"计算设备: {self.device}")

        self.infer_size = infer_size

        # 深度学习模型
        base_ch = 64 if model_type == 'standard' else 32
        if model_type == 'standard':
            self.model = AGEANet(in_channels=3, out_channels=1, base_ch=base_ch)
        else:
            self.model = AGEANetLite(in_channels=3, out_channels=1, base_ch=base_ch)
        self.model = self.model.to(self.device)

        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            logger.info(f"模型加载成功: {model_path}")
        else:
            logger.warning("未加载预训练权重，使用随机初始化模型")

        self.model.eval()
        info = get_model_info(self.model)
        logger.info(f"模型参数量: {info['total_params']:,} ({info['total_params_mb']:.1f} MB)")

        # ISP 模拟器
        self.isp = SimpleISPSimulator(preset=isp_preset)

        # 反光抑制管线
        self.anti_glare = AntiGlarePipeline(
            glare_threshold=235, repair_method='blend', use_polarization=True,
        )

        # 相机参数 (需实际标定替换)
        self.camera_matrix = np.array(
            [[1500, 0, 960], [0, 1500, 540], [0, 0, 1]], dtype=np.float64
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        self.R_cam2base = np.eye(3, dtype=np.float64)
        self.t_cam2base = np.array([[100], [200], [500]], dtype=np.float64)

        # 处理参数
        self.seg_threshold = 0.5
        self.edge_threshold = 0.3
        self.min_workpiece_area = 500
        self.z_plane = 0.0

    def set_camera_params(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix.astype(np.float64)
        self.dist_coeffs = dist_coeffs.astype(np.float64)

    def set_hand_eye_params(self, R_cam2base, t_cam2base):
        self.R_cam2base = R_cam2base.astype(np.float64)
        self.t_cam2base = t_cam2base.astype(np.float64)

    def process_frame(self, raw_images, return_debug=False):
        """
        完整处理流程。

        参数:
            raw_images:   BGR 图像列表 (多重曝光) 或单张
            return_debug: 是否返回中间结果

        返回:
            result: dict
        """
        total_start = time.time()
        timing = {}
        debug_info = {} if return_debug else None

        # --- 1. ISP 处理 ---
        t0 = time.time()
        processed_images = [self.isp.process_raw_image(img) for img in raw_images]
        timing['isp'] = (time.time() - t0) * 1000

        # --- 2. HDR 融合 + 反光抑制 ---
        t0 = time.time()
        pipeline_result = self.anti_glare.process(processed_images)
        enhanced = pipeline_result['enhanced']
        timing['hdr_antiglare'] = (time.time() - t0) * 1000

        if debug_info is not None:
            debug_info['fused'] = pipeline_result['fused']
            debug_info['enhanced'] = enhanced
            debug_info['glare_mask'] = pipeline_result['glare_mask']

        # --- 3. 深度学习分割 (内存安全) ---
        t0 = time.time()
        seg_mask, edge_map = _predict_masks(
            self.model, enhanced, self.device,
            infer_size=self.infer_size,
            seg_threshold=self.seg_threshold,
            edge_threshold=self.edge_threshold,
        )
        timing['inference'] = (time.time() - t0) * 1000

        if debug_info is not None:
            debug_info['seg_mask'] = seg_mask
            debug_info['edge_map'] = edge_map

        # --- 4. 后处理 ---
        t0 = time.time()
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced

        # 形态学优化
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        seg_cleaned = cv2.morphologyEx(seg_mask, cv2.MORPH_CLOSE, kernel)
        seg_cleaned = cv2.morphologyEx(seg_cleaned, cv2.MORPH_OPEN, kernel)
        timing['postprocess'] = (time.time() - t0) * 1000

        # --- 5. 多目标检测 + 亚像素定位 ---
        t0 = time.time()
        detections = detect_multiple_workpieces(
            seg_cleaned, gray, min_area=self.min_workpiece_area,
        )
        timing['localization'] = (time.time() - t0) * 1000

        # --- 6. 坐标转换 + 抓取位姿 ---
        t0 = time.time()
        grasp_poses = []
        for det in detections:
            grasp = compute_grasp_pose(
                det, self.R_cam2base, self.t_cam2base,
                self.camera_matrix, self.dist_coeffs,
                z_plane=self.z_plane,
            )
            grasp_poses.append(grasp)
        timing['coordinate'] = (time.time() - t0) * 1000

        total_time = (time.time() - total_start) * 1000
        timing['total'] = total_time

        result = {
            'detections': detections,
            'grasp_poses': grasp_poses,
            'processing_time': total_time,
            'timing': timing,
            'num_workpieces': len(detections),
        }
        if debug_info is not None:
            result['debug'] = debug_info
        return result

    def process_single_image(self, image, return_debug=False):
        """处理单张 BGR 图像。"""
        return self.process_frame([image], return_debug=return_debug)


# ===========================================================================
# 性能测试
# ===========================================================================

def performance_test():
    """性能测试。"""
    print("=" * 60)
    print("高反光工件分拣系统 — 性能测试")
    print("=" * 60)

    system = HighReflectiveSortingSystem(model_type='lite', infer_size=256)

    # 模拟图像 (640x480，节省内存)
    h, w = 480, 640
    base = np.zeros((h, w, 3), dtype=np.uint8) + 60
    cv2.circle(base, (320, 240), 80, (180, 180, 190), -1)
    cv2.rectangle(base, (100, 100), (250, 200), (170, 175, 180), -1)
    cv2.circle(base, (300, 220), 20, (255, 255, 255), -1)

    img_under = np.clip(base.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
    img_normal = base.copy()
    img_over = np.clip(base.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
    raw_images = [img_under, img_normal, img_over]

    # 预热
    print("\n预热中...")
    _ = system.process_frame(raw_images)

    # 测试 5 次
    print("开始性能测试 (5 次)...\n")
    times = []
    for i in range(5):
        result = system.process_frame(raw_images)
        t = result['processing_time']
        times.append(t)
        tm = result['timing']
        print(f"  迭代 {i+1}: {t:7.1f} ms | "
              f"ISP={tm['isp']:.0f} HDR={tm['hdr_antiglare']:.0f} "
              f"推理={tm['inference']:.0f} 后处理={tm['postprocess']:.0f} "
              f"定位={tm['localization']:.0f} 坐标={tm['coordinate']:.0f} | "
              f"工件={result['num_workpieces']}")

    avg = sum(times) / len(times)
    print(f"\n平均处理时间: {avg:.1f} ms")
    print(f"最小: {min(times):.1f} ms, 最大: {max(times):.1f} ms")

    # 单张图像
    print(f"\n{'='*60}")
    print("单张图像模式:")
    result = system.process_single_image(img_normal, return_debug=True)
    print(f"  处理时间: {result['processing_time']:.1f} ms")
    print(f"  检测工件: {result['num_workpieces']}")
    for i, pose in enumerate(result['grasp_poses']):
        print(f"  工件 {i+1}: pos={pose['position']}, angle={pose['angle']:.1f}°, "
              f"conf={pose['confidence']:.3f}")


if __name__ == "__main__":
    performance_test()
