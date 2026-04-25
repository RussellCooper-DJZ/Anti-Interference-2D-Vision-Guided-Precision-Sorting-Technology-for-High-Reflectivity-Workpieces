"""
main_pipeline.py — 船舶高光面视觉引导精密分拣主流水线
:Author: RussellCooper

完整流水线：
  摄像头采集 → HDR 融合 → 反光抑制 → AGEANet 推理
  → 亚像素定位 → 坐标变换 → ABB 机器人通信（模拟桩）

支持三种运行模式：
  --mode demo    : 使用合成图像演示完整流程（无需真实硬件）
  --mode camera  : 使用真实摄像头（OpenCV VideoCapture）
  --mode image   : 处理单张/目录图像

ABB 通信接口：
  本文件包含 AbbRobotStub（模拟桩），接口与真实 ABB EGM/RAPID 通信
  协议完全一致，替换时只需将 AbbRobotStub 换为 AbbRobotEGM 即可。

用法::

    # 演示模式（合成图像，无需硬件）
    python3 main_pipeline.py --mode demo --model_path ./checkpoints/best.pth

    # 单张图像处理
    python3 main_pipeline.py --mode image --input ./test.jpg --model_path ./checkpoints/best.pth

    # 摄像头实时处理
    python3 main_pipeline.py --mode camera --camera_id 0 --model_path ./checkpoints/best.pth
"""

import argparse
import logging
import json
import math
import os
import socket
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from vision.feature_extraction import AGEANet, AGEANetLite
from vision.hdr_processing import (
    exposure_fusion_mertens,
    repair_highlight_regions,
    detect_highlight_mask,
    AntiGlarePipeline,
)
from vision.localization_and_calibration import (
    SubpixelLocalizer,
    CoordinateTransformer,
    detect_glare_regions,
)


# ============================================================
# 1. ABB 机器人通信接口（模拟桩）
# ============================================================

class AbbRobotStub:
    """
    ABB 机器人通信模拟桩。

    接口与真实 ABB EGM（Externally Guided Motion）协议一致：
      - connect()        : 建立连接
      - disconnect()     : 断开连接
      - send_target()    : 发送目标位姿（x, y, z, rx, ry, rz）
      - get_status()     : 查询机器人状态
      - wait_done()      : 等待运动完成

    替换为真实 ABB EGM 时，只需将此类替换为 AbbRobotEGM，
    其他代码无需修改。

    真实 ABB EGM 通信参考：
      - ABB RobotWare EGM User Manual (3HAC073318-001)
      - UDP 端口：6510（默认）
      - 协议：Protobuf（egm.proto）
    """

    def __init__(self, host: str = "192.168.1.100", port: int = 6510,
                 verbose: bool = True):
        self.host    = host
        self.port    = port
        self.verbose = verbose
        self._connected = False
        self._move_count = 0
        self._current_pos = [0.0, 0.0, 500.0, 0.0, 0.0, 0.0]  # x,y,z,rx,ry,rz

    def connect(self) -> bool:
        """建立连接（模拟：始终成功）。"""
        self._connected = True
        if self.verbose:
            logger.info(f"[ABB-Stub] 已连接到 {self.host}:{self.port}（模拟模式）")
        return True

    def disconnect(self):
        """断开连接。"""
        self._connected = False
        if self.verbose:
            logger.info("[ABB-Stub] 已断开连接")

    def send_target(
        self,
        x_mm: float, y_mm: float, z_mm: float,
        rx_deg: float = 0.0, ry_deg: float = 0.0, rz_deg: float = 0.0,
        speed_mm_s: float = 100.0,
        zone: str = "z10",
    ) -> bool:
        """
        发送目标位姿到机器人。

        Args:
            x_mm, y_mm, z_mm:  目标位置（机器人基坐标系，毫米）
            rx_deg, ry_deg, rz_deg: 目标姿态（欧拉角，度）
            speed_mm_s:        运动速度（mm/s）
            zone:              到位区域（z0=精确到位，z10=10mm 区域）

        Returns:
            True = 指令已发送
        """
        if not self._connected:
            logger.info("[ABB-Stub] 错误：未连接")
            return False

        self._move_count += 1
        self._current_pos = [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]

        if self.verbose:
            print(f"[ABB-Stub] 发送目标 #{self._move_count:04d}: "
                  f"pos=({x_mm:.1f}, {y_mm:.1f}, {z_mm:.1f}) mm  "
                  f"rot=({rx_deg:.1f}, {ry_deg:.1f}, {rz_deg:.1f})°  "
                  f"speed={speed_mm_s:.0f}mm/s  zone={zone}")

        # 模拟运动延迟（真实 ABB 约 0.1~2s）
        dist = math.sqrt(sum((a - b) ** 2 for a, b in
                             zip(self._current_pos[:3], [x_mm, y_mm, z_mm])))
        delay = min(dist / speed_mm_s, 2.0)
        time.sleep(delay * 0.01)  # 模拟桩加速 100x

        return True

    def get_status(self) -> Dict:
        """查询机器人当前状态。"""
        return {
            'connected':    self._connected,
            'move_count':   self._move_count,
            'current_pos':  self._current_pos,
            'is_moving':    False,  # 模拟桩：始终不在运动中
            'error_code':   0,
        }

    def wait_done(self, timeout_s: float = 10.0) -> bool:
        """等待当前运动完成（模拟桩：立即返回）。"""
        return True

    def home(self) -> bool:
        """回零点。"""
        return self.send_target(0.0, 0.0, 500.0, 0.0, 0.0, 0.0, speed_mm_s=200.0)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.disconnect()


# ============================================================
# 2. 视觉推理引擎
# ============================================================

class VisionInferenceEngine:
    """
    封装 AGEANet 推理，支持 CPU/GPU，输入 BGR 图像，输出分割掩膜和边缘图。
    """

    def __init__(self, model_path: Optional[str] = None,
                 model_type: str = 'standard',
                 base_ch: int = 64,
                 img_size: int = 512,
                 device: Optional[str] = None):
        self.img_size = img_size
        self.device   = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )

        # 构建模型
        if model_type == 'lite':
            self.model = AGEANetLite(in_channels=3, base_ch=base_ch).to(self.device)
        else:
            self.model = AGEANet(in_channels=3, base_ch=base_ch).to(self.device)

        # 加载权重
        if model_path and Path(model_path).exists():
            ckpt = torch.load(model_path, map_location=self.device)
            state = ckpt.get('model', ckpt)
            self.model.load_state_dict(state, strict=False)
            logger.info(f"[VisionEngine] 已加载模型: {model_path}")
        else:
            logger.info("[VisionEngine] 警告：未加载预训练权重，使用随机初始化")

        self.model.eval()

    @torch.no_grad()
    def infer(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单张 BGR 图像执行推理。

        Args:
            image_bgr: (H, W, 3) uint8 BGR 图像

        Returns:
            seg_mask:  (H, W) uint8，分割掩膜（0/255）
            edge_mask: (H, W) uint8，边缘掩膜（0/255）
        """
        h_orig, w_orig = image_bgr.shape[:2]

        # 预处理
        img_resized = cv2.resize(image_bgr, (self.img_size, self.img_size))
        img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor      = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor      = tensor.unsqueeze(0).to(self.device)

        # 推理
        outputs = self.model(tensor)
        seg_prob  = outputs['seg'][0, 0].cpu().numpy()   # (H, W) float [0,1]
        edge_prob = outputs['edge'][0, 0].cpu().numpy()  # (H, W) float [0,1]

        # 后处理：二值化
        seg_mask  = (seg_prob  > 0.5).astype(np.uint8) * 255
        edge_mask = (edge_prob > 0.3).astype(np.uint8) * 255

        # 恢复原始分辨率
        if (h_orig, w_orig) != (self.img_size, self.img_size):
            seg_mask  = cv2.resize(seg_mask,  (w_orig, h_orig),
                                   interpolation=cv2.INTER_NEAREST)
            edge_mask = cv2.resize(edge_mask, (w_orig, h_orig),
                                   interpolation=cv2.INTER_NEAREST)

        return seg_mask, edge_mask


# ============================================================
# 3. 完整视觉流水线
# ============================================================

class ShipVisionPipeline:
    """
    船舶高光面视觉引导精密分拣完整流水线。

    流程：
      1. 多重曝光采集（或单帧输入）
      2. HDR 融合 + 反光抑制
      3. AGEANet 分割/边缘推理
      4. 高光区域检测与排除
      5. 亚像素定位（质心 + 方向）
      6. 坐标变换（像素 → 机器人基坐标系）
      7. ABB 机器人通信（发送目标位姿）
    """

    def __init__(
        self,
        model_path:    Optional[str] = None,
        model_type:    str = 'standard',
        img_size:      int = 512,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs:   Optional[np.ndarray] = None,
        T_cam2robot:   Optional[np.ndarray] = None,
        robot_host:    str = "192.168.1.100",
        robot_port:    int = 6510,
        verbose:       bool = True,
    ):
        self.verbose = verbose

        # 视觉推理引擎
        self.engine = VisionInferenceEngine(
            model_path=model_path,
            model_type=model_type,
            img_size=img_size,
        )

        # HDR 融合与反光抑制
        self.anti_glare = AntiGlarePipeline()

        # 亚像素定位器
        self.localizer = SubpixelLocalizer(min_area=200)

        # 坐标变换器（使用默认内参，实际使用时需标定）
        K = camera_matrix if camera_matrix is not None else np.array([
            [800.0,   0.0, 320.0],
            [  0.0, 800.0, 240.0],
            [  0.0,   0.0,   1.0],
        ], dtype=np.float64)
        dist = dist_coeffs if dist_coeffs is not None else np.zeros((1, 5))
        T    = T_cam2robot if T_cam2robot is not None else np.eye(4)
        self.transformer = CoordinateTransformer(K, dist, T)

        # ABB 机器人通信（模拟桩）
        self.robot = AbbRobotStub(host=robot_host, port=robot_port,
                                  verbose=verbose)

        # 性能统计
        self._frame_count = 0
        self._total_time  = 0.0

    def process_frame(
        self,
        image_bgr: np.ndarray,
        image_over: Optional[np.ndarray] = None,
        depth_map:  Optional[np.ndarray] = None,
        plane_d_mm: float = 800.0,
        send_to_robot: bool = True,
    ) -> Dict:
        """
        处理单帧图像，返回完整分析结果。

        Args:
            image_bgr:     主图像（正常曝光）BGR uint8
            image_over:    过曝图像（可选，用于 HDR 融合）
            depth_map:     深度图（可选，float32 mm）
            plane_d_mm:    无深度图时的平面距离假设（mm）
            send_to_robot: 是否将结果发送到机器人

        Returns:
            Dict 包含：
              image_hdr:     HDR 融合结果
              seg_mask:      分割掩膜
              edge_mask:     边缘掩膜
              glare_mask:    高光区域掩膜
              detections:    目标列表（含机器人坐标）
              vis_image:     可视化结果图
              timing:        各阶段耗时（ms）
        """
        t_start = time.perf_counter()
        timing  = {}

        # ---- Step 1: HDR 融合 ----
        t0 = time.perf_counter()
        if image_over is not None:
            image_hdr = exposure_fusion_mertens([image_bgr, image_over])
        else:
            image_hdr = image_bgr.copy()
        timing['hdr_ms'] = (time.perf_counter() - t0) * 1000

        # ---- Step 2: 反光抑制 ----
        t0 = time.perf_counter()
        # 使用 AntiGlarePipeline 进行高光修复
        glare_mask_raw = detect_highlight_mask(image_hdr)
        image_proc = repair_highlight_regions(image_hdr, glare_mask_raw)
        timing['glare_sup_ms'] = (time.perf_counter() - t0) * 1000

        # ---- Step 3: 深度学习推理 ----
        t0 = time.perf_counter()
        seg_mask, edge_mask = self.engine.infer(image_proc)
        timing['infer_ms'] = (time.perf_counter() - t0) * 1000

        # ---- Step 4: 高光区域检测 ----
        t0 = time.perf_counter()
        glare_mask = detect_glare_regions(image_hdr)
        timing['glare_detect_ms'] = (time.perf_counter() - t0) * 1000

        # ---- Step 5: 亚像素定位 ----
        t0 = time.perf_counter()
        gray = cv2.cvtColor(image_hdr, cv2.COLOR_BGR2GRAY)
        raw_results = self.localizer.localize(
            seg_mask, edge_mask,
            intensity_image=gray,
            glare_mask=glare_mask,
        )
        timing['localize_ms'] = (time.perf_counter() - t0) * 1000

        # ---- Step 6: 坐标变换 ----
        t0 = time.perf_counter()
        detections = self.transformer.transform_localization_results(
            raw_results, depth_map=depth_map, plane_d=plane_d_mm
        )
        timing['transform_ms'] = (time.perf_counter() - t0) * 1000

        # ---- Step 7: 机器人通信 ----
        t0 = time.perf_counter()
        if send_to_robot and self.robot._connected:
            for det in detections:
                if det.get('position_robot_mm') is not None:
                    pos = det['position_robot_mm']
                    # 根据目标方向计算末端姿态
                    rz = det['orientation_deg']
                    self.robot.send_target(
                        x_mm=float(pos[0]),
                        y_mm=float(pos[1]),
                        z_mm=float(pos[2]),
                        rz_deg=rz,
                        speed_mm_s=150.0,
                        zone="z5",
                    )
        timing['robot_ms'] = (time.perf_counter() - t0) * 1000

        # ---- 可视化 ----
        vis = self._visualize(image_hdr, seg_mask, edge_mask,
                               glare_mask, detections)

        total_ms = (time.perf_counter() - t_start) * 1000
        timing['total_ms'] = total_ms
        self._frame_count += 1
        self._total_time  += total_ms

        if self.verbose:
            print(f"[Pipeline] 帧 #{self._frame_count:04d}  "
                  f"目标数={len(detections)}  "
                  f"总耗时={total_ms:.1f}ms  "
                  f"(HDR={timing['hdr_ms']:.0f} "
                  f"推理={timing['infer_ms']:.0f} "
                  f"定位={timing['localize_ms']:.0f})")

        return {
            'image_hdr':   image_hdr,
            'seg_mask':    seg_mask,
            'edge_mask':   edge_mask,
            'glare_mask':  glare_mask,
            'detections':  detections,
            'vis_image':   vis,
            'timing':      timing,
        }

    def _visualize(
        self,
        image:      np.ndarray,
        seg_mask:   np.ndarray,
        edge_mask:  np.ndarray,
        glare_mask: np.ndarray,
        detections: List[Dict],
    ) -> np.ndarray:
        """生成可视化结果图（4 宫格：原图+分割+边缘+结果）。"""
        h, w = image.shape[:2]
        vis_orig = image.copy()

        # 分割叠加（绿色半透明）
        vis_seg = image.copy()
        seg_overlay = np.zeros_like(image)
        seg_overlay[seg_mask > 0] = [0, 200, 0]
        vis_seg = cv2.addWeighted(vis_seg, 0.7, seg_overlay, 0.3, 0)

        # 边缘叠加（青色）
        vis_edge = image.copy()
        vis_edge[edge_mask > 0] = [255, 200, 0]

        # 高光叠加（红色）
        vis_glare = image.copy()
        vis_glare[glare_mask > 0] = [0, 0, 220]

        # 结果图：标注目标
        vis_result = image.copy()
        for det in detections:
            cx, cy = int(det['centroid_px'][0]), int(det['centroid_px'][1])
            angle  = det['orientation_deg']
            ftype  = det['feature_type']

            # 颜色编码：blob=绿，line=蓝，region=橙
            color = {'blob': (0, 255, 0), 'line': (255, 100, 0),
                     'region': (0, 165, 255)}.get(ftype, (255, 255, 255))

            cv2.circle(vis_result, (cx, cy), 8, color, -1)
            cv2.circle(vis_result, (cx, cy), 8, (255, 255, 255), 1)

            # 方向箭头
            length = 40
            angle_rad = math.radians(angle)
            ex = int(cx + length * math.cos(angle_rad))
            ey = int(cy + length * math.sin(angle_rad))
            cv2.arrowedLine(vis_result, (cx, cy), (ex, ey), color, 2, tipLength=0.3)

            # 坐标标注
            pos = det.get('position_robot_mm')
            if pos is not None:
                label = f"({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})"
                cv2.putText(vis_result, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 拼接 4 宫格
        top    = np.hstack([vis_orig, vis_seg])
        bottom = np.hstack([vis_edge, vis_result])
        grid   = np.vstack([top, bottom])

        # 添加标签
        labels = ["原图 + 高光", "分割结果", "边缘检测", "定位结果"]
        positions = [(10, 25), (w + 10, 25), (10, h + 25), (w + 10, h + 25)]
        for label, pos in zip(labels, positions):
            cv2.putText(grid, label, pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

        return grid

    def get_stats(self) -> Dict:
        """返回性能统计信息。"""
        avg_ms = self._total_time / max(self._frame_count, 1)
        return {
            'frame_count': self._frame_count,
            'avg_ms':      avg_ms,
            'avg_fps':     1000.0 / max(avg_ms, 1),
        }


# ============================================================
# 4. 命令行入口
# ============================================================

def run_demo_mode(pipeline: ShipVisionPipeline, output_dir: str):
    """演示模式：使用合成图像运行流水线。"""
    from synth_dataset_generator import synthesize_one_sample, set_seed

    os.makedirs(output_dir, exist_ok=True)
    print("\n[Demo] 生成合成测试图像...")

    for i in range(3):
        set_seed(i * 7 + 42)
        sample = synthesize_one_sample(h=512, w=512)
        image  = sample['image']

        # 模拟过曝图像（亮度 +40）
        image_over = np.clip(image.astype(np.int32) + 40, 0, 255).astype(np.uint8)

        print(f"\n[Demo] 处理第 {i+1}/3 帧...")
        result = pipeline.process_frame(
            image_bgr=image,
            image_over=image_over,
            send_to_robot=True,
        )

        # 保存结果
        out_path = os.path.join(output_dir, f"demo_result_{i+1:02d}.png")
        cv2.imwrite(out_path, result['vis_image'])
        logger.info(f"[Demo] 可视化已保存: {out_path}")

        # 打印检测结果
        for j, det in enumerate(result['detections']):
            pos = det.get('position_robot_mm')
            pos_str = f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) mm" \
                      if pos is not None else "N/A"
            print(f"  目标 [{j}] 类型={det['feature_type']:8s}  "
                  f"质心=({det['centroid_px'][0]:.1f}, {det['centroid_px'][1]:.1f})  "
                  f"方向={det['orientation_deg']:.1f}°  "
                  f"机器人坐标={pos_str}")

    stats = pipeline.get_stats()
    print(f"\n[Demo] 性能统计: 平均耗时={stats['avg_ms']:.1f}ms  "
          f"平均帧率={stats['avg_fps']:.1f}fps")


def run_image_mode(pipeline: ShipVisionPipeline, input_path: str, output_dir: str):
    """图像模式：处理单张或目录中的图像。"""
    os.makedirs(output_dir, exist_ok=True)
    input_path = Path(input_path)

    if input_path.is_dir():
        images = sorted(input_path.glob("*.png")) + sorted(input_path.glob("*.jpg"))
    else:
        images = [input_path]

    print(f"\n[Image] 处理 {len(images)} 张图像...")
    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            logger.info(f"[Image] 跳过（无法读取）: {img_path}")
            continue

        result = pipeline.process_frame(image_bgr=image, send_to_robot=True)
        out_path = os.path.join(output_dir, f"result_{img_path.stem}.png")
        cv2.imwrite(out_path, result['vis_image'])
        logger.info(f"[Image] 已保存: {out_path}")


def run_camera_mode(pipeline: ShipVisionPipeline, camera_id: int, output_dir: str):
    """摄像头实时模式。"""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.info(f"[Camera] 错误：无法打开摄像头 {camera_id}")
        return

    logger.info(f"[Camera] 已打开摄像头 {camera_id}，按 'q' 退出，'s' 保存当前帧")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = pipeline.process_frame(frame, send_to_robot=True)
        cv2.imshow("Ship Vision Pipeline", result['vis_image'])

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            out_path = os.path.join(output_dir, f"capture_{frame_idx:04d}.png")
            cv2.imwrite(out_path, result['vis_image'])
            logger.info(f"[Camera] 已保存: {out_path}")
            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="船舶高光面视觉引导精密分拣系统")
    parser.add_argument("--mode", type=str, default="demo",
                        choices=["demo", "image", "camera"])
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="standard",
                        choices=["standard", "lite"])
    parser.add_argument("--img_size",   type=int, default=512)
    parser.add_argument("--input",      type=str, default=None,
                        help="图像路径或目录（image 模式）")
    parser.add_argument("--camera_id",  type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--robot_host", type=str, default="192.168.1.100")
    parser.add_argument("--robot_port", type=int, default=6510)
    parser.add_argument("--no_robot",   action="store_true",
                        help="不连接机器人（仅视觉处理）")
    args = parser.parse_args()

    # 构建流水线
    pipeline = ShipVisionPipeline(
        model_path=args.model_path,
        model_type=args.model_type,
        img_size=args.img_size,
        robot_host=args.robot_host,
        robot_port=args.robot_port,
        verbose=True,
    )

    # 连接机器人（模拟桩）
    if not args.no_robot:
        pipeline.robot.connect()
        pipeline.robot.home()

    # 运行对应模式
    if args.mode == "demo":
        run_demo_mode(pipeline, args.output_dir)
    elif args.mode == "image":
        if not args.input:
            print("错误：--input 参数必须指定图像路径")
            return
        run_image_mode(pipeline, args.input, args.output_dir)
    elif args.mode == "camera":
        run_camera_mode(pipeline, args.camera_id, args.output_dir)

    # 断开机器人
    if not args.no_robot:
        pipeline.robot.disconnect()

    # 打印最终统计
    stats = pipeline.get_stats()
    print(f"\n最终统计: {stats['frame_count']} 帧  "
          f"平均 {stats['avg_ms']:.1f}ms/帧  "
          f"{stats['avg_fps']:.1f}fps")


if __name__ == "__main__":
    main()
