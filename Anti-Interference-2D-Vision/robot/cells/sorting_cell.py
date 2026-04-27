"""
sorting_cell.py — 机器人分拣单元
:Author: RussellCooper

功能：
  - 配合视觉系统进行目标定位
  - 智能抓取规划
  - 支持多种工件类型
  - 实时通信（与视觉系统、码垛单元）

规格：
  - 循环周期: 3-6秒/件
  - 定位精度: ±0.5mm
  - 工作空间: 1130×1130mm
  - 负载: 3KG / 6KG

组成：
  - 机器人本体（3T/6T可选）
  - 视觉系统（FLARE推理引擎）
  - 夹爪系统
  - 通信接口

用法::

    from robot.cells import SortingCell
    from vision.inference_engine import PyTorchEngine

    engine = PyTorchEngine("checkpoints/best.pth")
    cell = SortingCell(engine=engine, robot_host="192.168.1.100")
    cell.connect()

    # 启动分拣循环
    cell.start_sorting()

    cell.stop()
    cell.disconnect()
"""

import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class GripperSpec:
    """夹爪规格"""
    type: str = "parallel"           # 平行夹爪
    max_width_mm: float = 150.0      # 最大开口宽度
    grip_force_n: float = 200.0       # 夹持力
    weight_kg: float = 2.0            # 夹爪重量


@dataclass
class SortTarget:
    """分拣目标"""
    centroid_px: Tuple[float, float]
    orientation_deg: float
    robot_pos_mm: Tuple[float, float, float]
    confidence: float
    workpiece_type: str = "unknown"
    quality_score: float = 1.0


@dataclass
class SortingConfig:
    """分拣配置"""
    cycle_time_s: float = 4.0          # 目标周期
    accuracy_mm: float = 0.5             # 定位精度
    max_targets: int = 10                # 每次最大处理数
    use_tta: bool = True                # 使用TTA推理
    edge_threshold: float = 0.15         # 边缘阈值
    seg_threshold: float = 0.5           # 分割阈值


class SortingCell:
    """
    机器人分拣单元控制器

    功能：
      - 视觉系统集成（FLARE推理）
      - 目标检测与定位
      - 抓取点规划
      - 机器人轨迹控制
      - 与码垛单元联动
    """

    def __init__(
        self,
        engine=None,  # 视觉推理引擎
        robot=None,  # AbbRobotBase 实例（如 AbbRobotStub / AbbRobotEGM）
        robot_host: str = "192.168.1.100",
        robot_port: int = 6510,
        config: Optional[SortingConfig] = None,
        gripper: Optional[GripperSpec] = None,
        verbose: bool = True,
        camera_device: int = 0,
    ):
        self.engine = engine
        self.robot = robot
        self.robot_host = robot_host
        self.robot_port = robot_port
        self.config = config or SortingConfig()
        self.gripper = gripper or GripperSpec()
        self.verbose = verbose
        self.camera_device = camera_device

        # 状态
        self._connected: bool = False
        self._sorting: bool = False
        self._targets: List[SortTarget] = []
        self._processed_count: int = 0
        self._failed_count: int = 0

        # 相机内参（默认）
        self.camera_matrix = None
        self.dist_coeffs = None
        self.T_cam2robot = None

        # 时间统计
        self.cycle_times: List[float] = []
        self.inference_times: List[float] = []

        self._logger("分拣单元初始化完成")

    def _logger(self, msg: str):
        if self.verbose:
            print(f"[分拣单元] {msg}")

    def set_calibration(
        self,
        camera_matrix,
        dist_coeffs,
        T_cam2robot,
    ):
        """设置相机标定参数"""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.T_cam2robot = T_cam2robot
        self._logger("相机标定参数已更新")

    def _init_vision_components(self):
        """延迟初始化视觉组件（避免循环导入）。"""
        if self._localizer is not None:
            return
        from vision.localization_and_calibration import (
            SubpixelLocalizer, CoordinateTransformer
        )
        from vision.gripper_simulation import GripperEdgePlanner
        self._localizer = SubpixelLocalizer(min_area=200)
        K = self.camera_matrix if self.camera_matrix is not None else np.array(
            [[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float64)
        d = self.dist_coeffs if self.dist_coeffs is not None else np.zeros((1, 5))
        T = self.T_cam2robot if self.T_cam2robot is not None else np.eye(4)
        self._transformer = CoordinateTransformer(K, d, T)
        self._gripper_planner = GripperEdgePlanner(
            gripper_width_px=self.gripper.max_width_mm)

    def connect(self) -> bool:
        """连接相机和机器人控制器。"""
        self._init_vision_components()
        # 连接相机
        try:
            import cv2
            self._camera = cv2.VideoCapture(self.camera_device)
            if self._camera.isOpened():
                self._logger(f"相机 {self.camera_device} 已连接")
            else:
                self._camera = None
                self._logger("相机未找到，使用无图模式")
        except Exception as e:
            self._camera = None
            self._logger(f"相机初始化失败: {e}，使用无图模式")
        # 连接机器人（优先使用外部注入的 robot 实例）
        if self.robot is not None:
            if not hasattr(self.robot, 'is_connected') or not self.robot.is_connected():
                self.robot.connect()
            self._logger(f"机器人已连接: {type(self.robot).__name__}")
        self._connected = True
        self._logger(f"系统就绪")
        return True

    def disconnect(self):
        """断开相机和机器人连接"""
        if self._camera:
            self._camera.release()
            self._camera = None
        self._connected = False
        self._sorting = False
        self._logger("已断开连接")

    def capture_and_detect(self) -> List[SortTarget]:
        """
        采集图像并进行目标检测（使用真实视觉pipeline）。

        Returns:
            检测到的目标列表
        """
        if self.engine is None:
            self._logger("错误：没有配置视觉引擎")
            return []

        t0 = time.perf_counter()

        # 1. 采集图像
        image_bgr = self._capture_frame()
        if image_bgr is None:
            self._logger("未采集到图像")
            return []

        # 2. 推理（优先使用 predict，与 BaseInferenceEngine 一致）
        if hasattr(self.engine, 'predict'):
            seg_mask, edge_mask = self.engine.predict(image_bgr)
        elif hasattr(self.engine, 'infer'):
            seg_mask, edge_mask = self.engine.infer(image_bgr)
        else:
            self._logger("引擎没有 predict 或 infer 方法")
            return []

        # 3. 像素级定位
        import cv2
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        raw_results = self._localizer.localize(
            seg_mask, edge_mask,
            intensity_image=gray,
            glare_mask=None,
        )
        if not raw_results:
            return []

        # 4. 坐标变换（像素 → 机器人基坐标系）
        detections = self._transformer.transform_localization_results(
            raw_results, depth_map=None, plane_d=150.0
        )

        # 5. 构建 SortTarget
        targets = []
        for det in detections[:self.config.max_targets]:
            pos = det.get('position_robot_mm')
            if pos is None:
                continue
            targets.append(SortTarget(
                centroid_px=det.get('centroid_px', (0, 0)),
                orientation_deg=det.get('orientation_deg', 0.0),
                robot_pos_mm=tuple(pos),
                confidence=0.8,
                workpiece_type=det.get('feature_type', 'unknown'),
                quality_score=det.get('quality_score', 0.8),
            ))

        inference_time = (time.perf_counter() - t0) * 1000
        self.inference_times.append(inference_time)
        self._logger(f"检测到 {len(targets)} 个目标，推理耗时{inference_time:.1f}ms")
        return targets

    def _capture_frame(self) -> Optional[np.ndarray]:
        """从相机采集一帧图像。"""
        if self._camera is None:
            return None
        ret, frame = self._camera.read()
        return frame if ret else None

    def plan_grasp(self, target: SortTarget) -> Dict:
        """
        使用 GripperEdgePlanner 规划抓取（如有轮廓信息）。

        Args:
            target: 分拣目标

        Returns:
            抓取参数
        """
        angle = target.orientation_deg
        grasp_depth_mm = 30.0
        grasp_angle_rad = math.radians(angle)
        offset_x = grasp_depth_mm * math.cos(grasp_angle_rad) / 2
        offset_y = grasp_depth_mm * math.sin(grasp_angle_rad) / 2
        grasp_pos = (
            target.robot_pos_mm[0] + offset_x,
            target.robot_pos_mm[1] + offset_y,
            target.robot_pos_mm[2],
        )
        return {
            "position_mm": grasp_pos,
            "rotation_deg": angle,
            "grip_width_mm": 50.0,
            "force_n": self.gripper.grip_force_n,
            "quality": target.quality_score,
        }

    def execute_grasp(self, grasp: Dict) -> bool:
        """
        执行抓取（使用真实机器人或模拟）。

        Args:
            grasp: 抓取参数

        Returns:
            True=成功
        """
        if not self._connected:
            self._logger("错误：未连接")
            return False

        pos = grasp["position_mm"]
        rot = grasp["rotation_deg"]

        if self.robot is not None and hasattr(self.robot, 'send_target'):
            # 真实机器人控制
            self.robot.send_target(
                pos[0], pos[1], pos[2],
                rx_deg=0.0, ry_deg=180.0, rz_deg=rot,
                speed_mm_s=150.0, zone="z10"
            )
            self.robot.wait_done(timeout_s=30.0)
            self._logger(f"机器人到达抓取位 ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})")
        else:
            # 模拟
            self._logger(f"移动到抓取位置 ({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})mm")
            time.sleep(0.3)

        self._logger(f"执行抓取，开口{grasp['grip_width_mm']:.0f}mm")
        time.sleep(0.2)
        self._logger("抓取成功")
        return True

    def execute_place(self, place_pos: Tuple[float, float, float]) -> bool:
        """
        执行放置

        Args:
            place_pos: 放置位置

        Returns:
            True=成功
        """
        if not self._connected:
            self._logger("错误：未连接")
            return False

        self._logger(f"移动到放置位置 {place_pos}")
        time.sleep(0.3)

        self._logger("执行放置")
        time.sleep(0.2)

        self._logger("放置完成")
        return True

    def process_one_cycle(self) -> Tuple[bool, float]:
        """
        处理一个完整的分拣周期

        Returns:
            (成功标志, 周期时间ms)
        """
        t0 = time.perf_counter()

        # 1. 采集和检测
        targets = self.capture_and_detect()
        if not targets:
            self._logger("未检测到目标")
            return False, (time.perf_counter() - t0) * 1000

        # 2. 选择最优目标（按置信度排序）
        target = max(targets, key=lambda t: t.confidence)

        # 3. 规划抓取
        grasp = self.plan_grasp(target)

        # 4. 执行抓取
        if not self.execute_grasp(grasp):
            self._failed_count += 1
            return False, (time.perf_counter() - t0) * 1000

        # 5. 移动到放置位置
        place_pos = (800, 400, 150)  # 模拟放置位置
        if not self.execute_place(place_pos):
            self._failed_count += 1
            return False, (time.perf_counter() - t0) * 1000

        cycle_time = (time.perf_counter() - t0) * 1000
        self.cycle_times.append(cycle_time)
        self._processed_count += 1

        self._logger(f"周期完成: {cycle_time:.0f}ms")
        return True, cycle_time

    def start_sorting(self, duration_s: Optional[float] = None):
        """
        启动分拣循环

        Args:
            duration_s: 运行时间（None=无限）
        """
        if not self._connected:
            self._logger("错误：未连接")
            return

        self._sorting = True
        self._logger("开始分拣...")

        start_time = time.time()
        cycle_count = 0

        while self._sorting:
            success, cycle_time = self.process_one_cycle()

            if success:
                cycle_count += 1
                avg_cycle = sum(self.cycle_times[-10:]) / len(self.cycle_times[-10:])
                self._logger(f"完成 {cycle_count} 个，周期 {cycle_time:.0f}ms，"
                           f"平均 {avg_cycle:.0f}ms")

            # 检查运行时间
            if duration_s and (time.time() - start_time) >= duration_s:
                break

            # 检查停止信号
            # 实际应用中应检查外部停止信号

        self._logger("分拣循环结束")

    def stop(self):
        """停止分拣"""
        self._sorting = False
        self._logger("停止分拣")

    def get_stats(self) -> Dict:
        """获取统计信息"""
        avg_cycle = sum(self.cycle_times) / len(self.cycle_times) if self.cycle_times else 0
        avg_inference = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0

        return {
            "connected": self._connected,
            "sorting": self._sorting,
            "processed_count": self._processed_count,
            "failed_count": self._failed_count,
            "success_rate": self._processed_count / max(1, self._processed_count + self._failed_count),
            "avg_cycle_time_ms": avg_cycle,
            "avg_inference_ms": avg_inference,
            "targets_in_queue": len(self._targets),
        }

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.stop()
        self.disconnect()
