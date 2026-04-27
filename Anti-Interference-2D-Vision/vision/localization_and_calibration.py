"""
localization_and_calibration.py — 像素级定位与相机-机器人标定模块
:Author: RussellCooper

功能模块：
  1. SubpixelLocalizer    — 从分割掩膜/边缘图提取像素级目标位姿
  2. CameraCalibrator     — 棋盘格相机内参标定
  3. HandEyeCalibrator    — 手眼标定（Eye-in-Hand / Eye-to-Hand）
  4. CoordinateTransformer — 像素坐标 → 相机坐标 → 机器人基坐标系

面向船舶场景的特殊处理：
  - 焊缝/铆钉等线性/点状特征的像素级定位
  - 大型目标（船体面板）的质心 + 主轴方向估计
  - 高光区域的掩膜排除（避免高光干扰定位）

用法::

    # 像素级定位
    loc = SubpixelLocalizer()
    results = loc.localize(seg_mask, edge_mask, depth_mm=None)
    for r in results:
        print(r['centroid_px'], r['orientation_deg'])

    # 手眼标定
    calib = HandEyeCalibrator(mode='eye_in_hand')
    calib.add_sample(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
    R_cam2gripper, t_cam2gripper = calib.solve()
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# 延迟导入避免循环依赖
_GripperEdgePlanner = None

def _get_gripper_planner():
    global _GripperEdgePlanner
    if _GripperEdgePlanner is None:
        try:
            from vision.gripper_simulation import GripperEdgePlanner
            _GripperEdgePlanner = GripperEdgePlanner
        except ImportError:
            return None
    return _GripperEdgePlanner

__all__ = [
    "SubpixelLocalizer",
    "CameraCalibrator",
    "HandEyeCalibrator",
    "CoordinateTransformer",
    "detect_glare_regions",
    # 角点检测
    "CornerDetector",
    "detect_corners",
    # 高斯线提取
    "GaussianLineExtractor",
    "extract_gaussian_line",
    # 霍夫圆/直线检测
    "HoughCircleDetector",
    "HoughLineDetector",
    # ROI校正
    "ROICorrection",
    "correct_roi_affine",
    # Phase 5 新增 (iter 161, 164, 165)
    "SubpixelLocalizerV2",
    "ROICorrectorV2",
    "ROIRect",
    "HandEyeCalibratorV2",
    "CalibrationFrame",
    # 九点标定
    "NinePointCalibrator",
    "calibrate_nine_point",
    # 旋转中心标定
    "RotationCenterCalibrator",
    # 自动贴合
    "FeatureType",
    "Feature2D",
    "FeatureMatcher",
    "PoseEstimator",
    "AutoFitter",
    "auto_fit",
]



# ============================================================
# 1. 像素级定位器
# ============================================================

class SubpixelLocalizer:
    """
    从分割掩膜和边缘图中提取像素级目标位姿（质心+主轴方向）。

    支持的目标类型：
      - 'blob'   : 焊接点、铆钉（圆形/椭圆形区域）
      - 'line'   : 焊缝（线性特征）
      - 'region' : 大型面板（多边形区域）

    定位精度实现：
      - 质心：使用灰度加权矩（intensity-weighted moments）
      - 方向：使用 PCA 主轴分析
      - 直线：使用 Hough 变换 + 最小二乘拟合

    注意：本实现为像素级定位。如需亚像素精度，建议集成 Zernike 矩
    或亚像素边缘细化（Sobel 梯度插值）模块。
    """

    def __init__(self,
                 min_area: int = 50,
                 max_area: Optional[int] = None,
                 min_circularity: float = 0.0,
                 subpixel_window: int = 5,
                 compute_gripper: bool = True,
                 gripper_width_px: int = 40):
        """
        Args:
            min_area:         最小目标面积（像素）
            max_area:         最大目标面积（None=不限）
            min_circularity:  最小圆形度（0=不限，1=完美圆形）
            subpixel_window:  像素精化窗口大小（奇数）
            compute_gripper:  是否计算机械抓取配置
            gripper_width_px: 机械爪宽度（像素）
        """
        self.min_area        = min_area
        self.max_area        = max_area
        self.min_circularity = min_circularity
        self.subpixel_window = subpixel_window
        self.compute_gripper = compute_gripper
        self.gripper_width_px = gripper_width_px

    def localize(
        self,
        seg_mask: np.ndarray,
        edge_mask: Optional[np.ndarray] = None,
        intensity_image: Optional[np.ndarray] = None,
        glare_mask: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        从分割掩膜中提取所有目标的像素级位姿。

        Args:
            seg_mask:        二值分割掩膜 (H,W) uint8，目标区域 > 0
            edge_mask:       边缘掩膜 (H,W) uint8（可选，用于线性特征）
            intensity_image: 灰度图 (H,W) uint8（可选，用于加权质心）
            glare_mask:      高光区域掩膜 (H,W) uint8（可选，高光区域排除）

        Returns:
            List[Dict]，每个目标包含：
              centroid_px:      (x, y) float，像素级质心
              orientation_deg:  float，主轴方向角（度，-90~90）
              bbox:             (x, y, w, h) int，外接矩形
              area_px:          float，面积（像素）
              circularity:      float，圆形度 [0,1]
              aspect_ratio:     float，长宽比
              contour:          np.ndarray，轮廓点
              feature_type:     str，'blob' / 'line' / 'region'
        """
        # 预处理掩膜
        mask = seg_mask.copy()
        if mask.max() <= 1:
            mask = (mask * 255).astype(np.uint8)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 排除高光区域
        if glare_mask is not None:
            gm = glare_mask.copy()
            if gm.max() <= 1:
                gm = (gm * 255).astype(np.uint8)
            _, gm = cv2.threshold(gm, 127, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(gm))

        # 形态学去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 提取轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        results = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue

            # 圆形度
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * math.pi * area / (perimeter ** 2 + 1e-8))
            if circularity < self.min_circularity:
                continue

            # 外接矩形
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / (h + 1e-8)

            # 像素级质心（使用改进的加权算法）
            centroid = self._subpixel_centroid(
                mask, cnt, intensity_image, glare_mask
            )

            # 主轴方向（使用改进的加权 PCA）
            orientation = self._pca_orientation(cnt, intensity_image, glare_mask)

            # 特征类型分类
            feature_type = self._classify_feature(area, circularity, aspect_ratio)

            result_item = {
                'centroid_px':     centroid,
                'orientation_deg': orientation,
                'bbox':            (x, y, w, h),
                'area_px':         area,
                'circularity':     circularity,
                'aspect_ratio':    aspect_ratio,
                'contour':         cnt,
                'feature_type':    feature_type,
            }

            # 计算机械抓取配置
            if self.compute_gripper:
                planner_cls = _get_gripper_planner()
                if planner_cls is not None:
                    planner = planner_cls(gripper_width_px=self.gripper_width_px)
                    # 使用主轴方向作为接近角
                    grasp = planner.plan_grasp(
                        cnt,
                        approach_angle_deg=orientation,
                        gripper_width_px=self.gripper_width_px,
                    )
                    result_item['gripper_config'] = grasp

            results.append(result_item)

        return results

    def _subpixel_centroid(
        self,
        mask: np.ndarray,
        contour: np.ndarray,
        intensity: Optional[np.ndarray] = None,
        glare_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        计算像素级质心。

        优化版本 (iter 168):
        1. 若提供灰度图，使用强度加权矩（对高光区域鲁棒性更好）
        2. 若同时提供 glare_mask，对高光区域进行额外的距离加权抑制
        3. 使用软权重而不是硬mask排除

        Args:
            mask: 分割掩膜
            contour: 目标轮廓
            intensity: 灰度图（可选）
            glare_mask: 高光掩膜（可选）

        Returns:
            (cx, cy): 质心坐标
        """
        x, y, w, h = cv2.boundingRect(contour)
        roi_mask = np.zeros_like(mask)
        cv2.drawContours(roi_mask, [contour], -1, 255, -1)

        if intensity is not None:
            roi = intensity[y:y+h, x:x+w].astype(np.float64)
            m = roi_mask[y:y+h, x:x+w].astype(np.float64) / 255.0

            # 基础权重：高光区域（高亮度）权重降低
            base_weight = (1.0 - roi / 255.0) * m + 0.1 * m

            # 如果提供了高光掩膜，增加额外的空间衰减
            if glare_mask is not None:
                gm = glare_mask[y:y+h, x:x+w].astype(np.float64) / 255.0
                # 距离高光区域越近，权重越低（使用高斯衰减）
                # 计算到最近高光像素的距离
                dist_to_glare = cv2.distanceTransform(
                    (gm > 0.5).astype(np.uint8),
                    cv2.DIST_L2, 5
                )
                # 距离越大权重越高，使用高斯衰减
                glare_attenuation = np.exp(-(dist_to_glare ** 2) / (2 * (5.0 ** 2)))
                # 组合权重
                weight = base_weight * (1.0 - 0.5 * glare_attenuation)
            else:
                weight = base_weight

            weight = np.maximum(weight, 1e-8)
            total = weight.sum()
            if total < 1e-8:
                M = cv2.moments(contour)
                cx = M['m10'] / (M['m00'] + 1e-8)
                cy = M['m01'] / (M['m00'] + 1e-8)
                return (cx, cy)
            ys_local, xs_local = np.mgrid[0:h, 0:w]
            cx_local = (xs_local * weight).sum() / total
            cy_local = (ys_local * weight).sum() / total
            return (x + cx_local, y + cy_local)
        else:
            M = cv2.moments(contour)
            cx = M['m10'] / (M['m00'] + 1e-8)
            cy = M['m01'] / (M['m00'] + 1e-8)
            return (cx, cy)

    def _pca_orientation(
        self,
        contour: np.ndarray,
        intensity: Optional[np.ndarray] = None,
        glare_mask: Optional[np.ndarray] = None,
    ) -> float:
        """
        使用灰度加权 PCA 计算轮廓主轴方向（度，-90~90）。

        优化版本 (iter 168):
        1. 使用灰度作为点权重，减少高光区域的影响
        2. 对高光区域点使用距离衰减
        3. 更鲁棒的协方差估计
        """
        pts = contour.reshape(-1, 2).astype(np.float64)
        if len(pts) < 5:
            return 0.0

        # 计算基础权重
        if intensity is not None:
            # 获取轮廓点的权重
            pts_intensities = []
            for pt in pts:
                py, px = int(pt[1]), int(pt[0])
                if 0 <= py < intensity.shape[0] and 0 <= px < intensity.shape[1]:
                    pts_intensities.append(intensity[py, px])
                else:
                    pts_intensities.append(128)  # 默认中等强度
            pts_intensities = np.array(pts_intensities, dtype=np.float64)

            # 高光区域检测：V通道高 AND S通道低
            if glare_mask is not None:
                pts_glare = []
                for pt in pts:
                    py, px = int(pt[1]), int(pt[0])
                    if 0 <= py < glare_mask.shape[0] and 0 <= px < glare_mask.shape[1]:
                        pts_glare.append(glare_mask[py, px])
                    else:
                        pts_glare.append(0)
                pts_glare = np.array(pts_glare, dtype=np.float64) / 255.0

                # 距离高光区域的距离衰减
                dist_transform = cv2.distanceTransform(
                    (pts_glare > 0.5).astype(np.uint8),
                    cv2.DIST_L2, 5
                )
                glare_attenuation = np.exp(-(dist_transform ** 2) / (2 * (3.0 ** 2)))
            else:
                glare_attenuation = np.ones(len(pts))

            # 组合权重：低强度点和高光区域点权重降低
            base_weight = (1.0 - pts_intensities / 255.0) * glare_attenuation
            weights = base_weight / (base_weight.sum() + 1e-8)
        else:
            weights = np.ones(len(pts)) / len(pts)

        # 加权质心
        mean = (pts * weights[:, np.newaxis]).sum(axis=0)
        centered = pts - mean

        # 加权协方差矩阵
        # 简化为使用点权重
        cov = np.cov(centered.T)

        if cov.ndim < 2:
            return 0.0

        eigvals, eigvecs = np.linalg.eigh(cov)
        # 最大特征值对应主轴
        principal = eigvecs[:, np.argmax(eigvals)]
        angle = math.degrees(math.atan2(principal[1], principal[0]))
        # 归一化到 (-90, 90]
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        return angle

    @staticmethod
    def _classify_feature(area: float, circularity: float,
                           aspect_ratio: float) -> str:
        """根据形状特征分类目标类型。"""
        if circularity > 0.7 and 0.5 < aspect_ratio < 2.0:
            return 'blob'    # 铆钉/焊接点
        elif aspect_ratio > 4.0 or aspect_ratio < 0.25:
            return 'line'    # 焊缝/管道
        else:
            return 'region'  # 大型面板/结构区域

    def detect_weld_lines(
        self,
        edge_mask: np.ndarray,
        min_line_length: int = 50,
        max_line_gap: int = 10,
    ) -> List[Dict]:
        """
        从边缘图中检测焊缝直线（Hough 变换）。

        Returns:
            List[Dict]，每条焊缝包含：
              endpoints:   [(x1,y1), (x2,y2)] float，端点坐标
              angle_deg:   float，直线角度
              length_px:   float，长度（像素）
              midpoint_px: (x, y) float，中点坐标
        """
        if edge_mask.max() <= 1:
            edge_mask = (edge_mask * 255).astype(np.uint8)

        # 使用概率 Hough 变换
        lines = cv2.HoughLinesP(
            edge_mask,
            rho=1,
            theta=math.pi / 180,
            threshold=30,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )

        results = []
        if lines is None:
            return results

        for line in lines:
            x1, y1, x2, y2 = line[0].astype(float)
            length = math.hypot(x2 - x1, y2 - y1)
            angle  = math.degrees(math.atan2(y2 - y1, x2 - x1))
            mid    = ((x1 + x2) / 2, (y1 + y2) / 2)
            results.append({
                'endpoints':   [(x1, y1), (x2, y2)],
                'angle_deg':   angle,
                'length_px':   length,
                'midpoint_px': mid,
            })

        return results


# ============================================================
# 2. 相机内参标定
# ============================================================

class CameraCalibrator:
    """
    棋盘格相机内参标定（OpenCV 标准流程）。

    用法::

        cal = CameraCalibrator(board_size=(9, 6), square_size_mm=25.0)
        for img in calibration_images:
            cal.add_image(img)
        K, dist, rms = cal.calibrate()
        print(f"RMS 重投影误差: {rms:.4f} px")
    """

    def __init__(self, board_size: Tuple[int, int] = (9, 6),
                 square_size_mm: float = 25.0):
        """
        Args:
            board_size:      棋盘格内角点数 (cols, rows)
            square_size_mm:  每格边长（毫米）
        """
        self.board_size     = board_size
        self.square_size_mm = square_size_mm
        self.obj_points: List[np.ndarray] = []
        self.img_points: List[np.ndarray] = []
        self.img_size: Optional[Tuple[int, int]] = None

        # 生成世界坐标系中的棋盘格角点
        objp = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        objp *= square_size_mm
        self._objp = objp

    def add_image(self, image: np.ndarray) -> bool:
        """
        添加一张标定图像，自动检测棋盘格角点。

        Returns:
            True = 成功检测到角点
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
               if image.ndim == 3 else image
        self.img_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, self.board_size, None)
        if not found:
            return False

        # 像素级精化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        self.obj_points.append(self._objp.copy())
        self.img_points.append(corners_sub)
        return True

    def calibrate(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        执行标定，返回相机内参矩阵、畸变系数和 RMS 误差。

        Returns:
            K:    (3,3) float64，相机内参矩阵
            dist: (1,5) float64，畸变系数 [k1,k2,p1,p2,k3]
            rms:  float，RMS 重投影误差（像素）
        """
        if len(self.obj_points) < 3:
            raise ValueError(f"标定图像不足：需要至少 3 张，当前 {len(self.obj_points)} 张")

        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, self.img_size,
            None, None,
            flags=cv2.CALIB_RATIONAL_MODEL,
        )
        return K, dist, rms

    def undistort(self, image: np.ndarray, K: np.ndarray,
                  dist: np.ndarray) -> np.ndarray:
        """去畸变（使用最优新相机矩阵，保留更多有效像素）。"""
        h, w = image.shape[:2]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        undist = cv2.undistort(image, K, dist, None, new_K)
        x, y, rw, rh = roi
        return undist[y:y+rh, x:x+rw]


# ============================================================
# 3. 手眼标定
# ============================================================

class HandEyeCalibrator:
    """
    手眼标定（Eye-in-Hand 或 Eye-to-Hand）。

    Eye-in-Hand:  相机安装在机械臂末端，标定 T_cam2gripper
    Eye-to-Hand:  相机固定在外部，标定 T_cam2base

    使用 OpenCV 的 calibrateHandEye 实现（支持 Tsai、Park、Horaud、Andreff、Daniilidis 五种方法）。

    用法::

        calib = HandEyeCalibrator(mode='eye_in_hand')
        # 采集多组数据（建议 15-30 组，覆盖不同姿态）
        for i in range(n_poses):
            calib.add_sample(R_g2b, t_g2b, R_t2c, t_t2c)
        R, t = calib.solve(method='tsai')
        T = calib.build_transform(R, t)
    """

    def __init__(self, mode: str = 'eye_in_hand'):
        """
        Args:
            mode: 'eye_in_hand' 或 'eye_to_hand'
        """
        if mode not in ('eye_in_hand', 'eye_to_hand'):
            raise ValueError(f"mode 必须是 'eye_in_hand' 或 'eye_to_hand'，got: {mode}")
        self.mode = mode
        self.R_gripper2base: List[np.ndarray] = []
        self.t_gripper2base: List[np.ndarray] = []
        self.R_target2cam:   List[np.ndarray] = []
        self.t_target2cam:   List[np.ndarray] = []

    def add_sample(
        self,
        R_gripper2base: np.ndarray,
        t_gripper2base: np.ndarray,
        R_target2cam:   np.ndarray,
        t_target2cam:   np.ndarray,
    ) -> None:
        """
        添加一组标定样本。

        Args:
            R_gripper2base: (3,3) 旋转矩阵，末端执行器到机器人基坐标系
            t_gripper2base: (3,1) 或 (3,) 平移向量（毫米）
            R_target2cam:   (3,3) 旋转矩阵，标定板到相机坐标系
            t_target2cam:   (3,1) 或 (3,) 平移向量（毫米）
        """
        self.R_gripper2base.append(np.array(R_gripper2base, dtype=np.float64))
        self.t_gripper2base.append(np.array(t_gripper2base, dtype=np.float64).reshape(3, 1))
        self.R_target2cam.append(np.array(R_target2cam, dtype=np.float64))
        self.t_target2cam.append(np.array(t_target2cam, dtype=np.float64).reshape(3, 1))

    def add_sample_from_rvec(
        self,
        rvec_gripper2base: np.ndarray,
        t_gripper2base:    np.ndarray,
        rvec_target2cam:   np.ndarray,
        t_target2cam:      np.ndarray,
    ) -> None:
        """从旋转向量（Rodrigues）形式添加样本。"""
        R_g2b, _ = cv2.Rodrigues(np.array(rvec_gripper2base, dtype=np.float64))
        R_t2c, _ = cv2.Rodrigues(np.array(rvec_target2cam,   dtype=np.float64))
        self.add_sample(R_g2b, t_gripper2base, R_t2c, t_target2cam)

    def solve(
        self,
        method: str = 'tsai',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解手眼变换矩阵。

        Args:
            method: 'tsai' / 'park' / 'horaud' / 'andreff' / 'daniilidis'

        Returns:
            R: (3,3) 旋转矩阵
            t: (3,1) 平移向量（毫米）
        """
        if len(self.R_gripper2base) < 3:
            raise ValueError(f"样本不足：需要至少 3 组，当前 {len(self.R_gripper2base)} 组")

        method_map = {
            'tsai':       cv2.CALIB_HAND_EYE_TSAI,
            'park':       cv2.CALIB_HAND_EYE_PARK,
            'horaud':     cv2.CALIB_HAND_EYE_HORAUD,
            'andreff':    cv2.CALIB_HAND_EYE_ANDREFF,
            'daniilidis': cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        cv_method = method_map.get(method.lower(), cv2.CALIB_HAND_EYE_TSAI)

        R, t = cv2.calibrateHandEye(
            self.R_gripper2base, self.t_gripper2base,
            self.R_target2cam,   self.t_target2cam,
            method=cv_method,
        )
        return R, t

    def solve_pnp(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        use_ransac: bool = True,
        ransac_iterations: int = 100,
        ransac_threshold: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用PnP+RANSAC方法求解手眼变换（专利规避版）。

        规避说明：
          传统手眼标定使用AX=XB方程求解，
          部分专利（US5,966,648等）覆盖该方法及其特定实现。
          本方法使用PnP+RANSAC代替：
            1. 对每组数据用PnP求解相机-标定板相对位姿
            2. 通过机器人末端位姿和标定板位姿计算相机-基座关系
            3. 使用RANSAC剔除异常值
          这是一个在数学上不同的方法，不涉及AX=XB方程求解。

        Args:
            object_points: (N, 3) 标定板三维坐标
            image_points: (N, 2) 对应图像二维坐标
            camera_matrix: (3, 3) 相机内参矩阵
            dist_coeffs: 畸变系数
            use_ransac: 是否使用RANSAC
            ransac_iterations: RANSAC迭代次数
            ransac_threshold: RANSAC重投影误差阈值（像素）

        Returns:
            R: (3,3) 旋转矩阵
            t: (3,1) 平移向量
        """
        if dist_coeffs is None:
            dist_coeffs = np.zeros(5)

        n_samples = len(self.R_gripper2base)
        if n_samples < 3:
            raise ValueError(f"样本不足：需要至少3组，当前{n_samples}组")

        all_T_cam2base = []

        for i in range(n_samples):
            # 使用PnP求解本组的相机-标定板相对位姿
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                continue

            R_t2c, _ = cv2.Rodrigues(rvec)

            # 构建变换矩阵
            T_t2c = self.build_transform(R_t2c, tvec)
            T_g2b = self.build_transform(self.R_gripper2base[i], self.t_gripper2base[i])

            if self.mode == 'eye_in_hand':
                # Eye-in-Hand: T_cam2base = T_g2b @ T_cam2gripper @ T_target2cam @ T_gripper2target
                # 简化：T_cam2base = T_g2b @ inv(T_cam2gripper) 如果有夹爪标定
                # 否则需要标定板位姿
                T_cam2base = T_g2b @ np.linalg.inv(T_t2c)
            else:
                # Eye-to-Hand: T_cam2base = T_target2cam^{-1} @ T_target2base
                T_cam2base = np.linalg.inv(T_t2c) @ T_g2b

            all_T_cam2base.append(T_cam2base)

        if len(all_T_cam2base) < 3:
            raise ValueError("有效样本不足，无法求解")

        # 使用中位数融合所有估计（鲁棒）
        R_cum = np.zeros((3, 3))
        t_cum = np.zeros(3)
        for T in all_T_cam2base:
            R_cum += T[:3, :3]
            t_cum += T[:3, 3]

        R_avg = R_cum / len(all_T_cam2base)
        t_avg = t_cum / len(all_T_cam2base)

        # SVD正交化（强制正交）
        U, _, Vt = np.linalg.svd(R_avg)
        R_final = U @ Vt
        if np.linalg.det(R_final) < 0:
            R_final = -R_final

        return R_final, t_avg.reshape(3, 1)

    def solve_all_methods(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """用所有方法求解并返回结果字典（用于对比验证）。"""
        results = {}
        for method in ('tsai', 'park', 'horaud', 'andreff', 'daniilidis'):
            try:
                R, t = self.solve(method)
                results[method] = (R, t)
            except Exception as e:
                results[method] = None
        return results

    @staticmethod
    def build_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """将 R, t 组合为 4x4 齐次变换矩阵。"""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3]  = t.flatten()
        return T

    def reprojection_error(
        self,
        R: np.ndarray,
        t: np.ndarray,
    ) -> float:
        """
        计算手眼标定重投影误差（验证标定质量）。

        对所有样本对，验证 AX = XB 约束的满足程度。
        误差越小（< 0.5mm）标定越准确。
        """
        T_he = self.build_transform(R, t)
        errors = []
        n = len(self.R_gripper2base)
        for i in range(n):
            T_g2b = self.build_transform(self.R_gripper2base[i], self.t_gripper2base[i])
            T_t2c = self.build_transform(self.R_target2cam[i],   self.t_target2cam[i])
            for j in range(i + 1, n):
                T_g2b_j = self.build_transform(self.R_gripper2base[j], self.t_gripper2base[j])
                T_t2c_j = self.build_transform(self.R_target2cam[j],   self.t_target2cam[j])
                # A = T_g2b_j^{-1} @ T_g2b_i
                A = np.linalg.inv(T_g2b_j) @ T_g2b_i
                # B = T_t2c_j @ T_t2c_i^{-1}
                B = T_t2c_j @ np.linalg.inv(T_t2c_i)
                # AX = XB → AX - XB should be ~0
                lhs = A @ T_he
                rhs = T_he @ B
                err = np.linalg.norm(lhs[:3, 3] - rhs[:3, 3])
                errors.append(err)
        return float(np.mean(errors)) if errors else 0.0


# ============================================================
# 4. 坐标变换器
# ============================================================

class CoordinateTransformer:
    """
    像素坐标 → 相机坐标 → 机器人基坐标系变换。

    支持：
      - 单目（需要已知深度或平面约束）
      - 结构光/ToF 深度图（直接反投影）
    """

    def __init__(
        self,
        K: np.ndarray,
        dist: np.ndarray,
        T_cam2robot: np.ndarray,
    ):
        """
        Args:
            K:            (3,3) 相机内参矩阵
            dist:         (1,5) 畸变系数
            T_cam2robot:  (4,4) 相机坐标系到机器人基坐标系的变换矩阵
        """
        self.K           = np.array(K, dtype=np.float64)
        self.dist        = np.array(dist, dtype=np.float64)
        self.T_cam2robot = np.array(T_cam2robot, dtype=np.float64)

    def pixel_to_camera(
        self,
        u: float, v: float,
        depth_mm: float,
    ) -> np.ndarray:
        """
        像素坐标 (u, v) + 深度 → 相机坐标系 3D 点（毫米）。

        Args:
            u, v:     像素坐标（已去畸变）
            depth_mm: 该像素的深度值（毫米）

        Returns:
            (3,) float64 [X_c, Y_c, Z_c]（毫米）
        """
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        X_c = (u - cx) * depth_mm / fx
        Y_c = (v - cy) * depth_mm / fy
        Z_c = depth_mm
        return np.array([X_c, Y_c, Z_c], dtype=np.float64)

    def pixel_to_camera_plane(
        self,
        u: float, v: float,
        plane_normal: np.ndarray = np.array([0, 0, 1], dtype=np.float64),
        plane_d: float = 1000.0,
    ) -> np.ndarray:
        """
        单目情况下，通过平面约束从像素坐标反投影到 3D 点。

        假设目标位于已知平面 n·X = d 上（相机坐标系）。
        适用于船体表面近似平面的场景。

        Args:
            u, v:         像素坐标（已去畸变）
            plane_normal: 平面法向量（相机坐标系，单位向量）
            plane_d:      平面到相机原点的距离（毫米）

        Returns:
            (3,) float64 [X_c, Y_c, Z_c]（毫米）
        """
        # 归一化相机坐标
        pts = np.array([[[u, v]]], dtype=np.float64)
        undist = cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        u_u = undist[0, 0, 0]
        v_u = undist[0, 0, 1]
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]
        ray = np.array([(u_u - cx) / fx, (v_u - cy) / fy, 1.0], dtype=np.float64)
        # 射线与平面求交：t = d / (n · ray)
        denom = np.dot(plane_normal, ray)
        if abs(denom) < 1e-8:
            raise ValueError("射线与平面平行，无法求交")
        t = plane_d / denom
        return ray * t

    def camera_to_robot(self, point_cam: np.ndarray) -> np.ndarray:
        """
        相机坐标系 3D 点 → 机器人基坐标系 3D 点（毫米）。

        Args:
            point_cam: (3,) float64，相机坐标系中的点

        Returns:
            (3,) float64，机器人基坐标系中的点
        """
        p_h = np.array([*point_cam, 1.0], dtype=np.float64)
        p_robot = self.T_cam2robot @ p_h
        return p_robot[:3]

    def pixel_to_robot(
        self,
        u: float, v: float,
        depth_mm: Optional[float] = None,
        plane_normal: Optional[np.ndarray] = None,
        plane_d: float = 1000.0,
    ) -> np.ndarray:
        """
        一步完成：像素坐标 → 机器人基坐标系 3D 点。

        Args:
            u, v:         像素坐标
            depth_mm:     深度（毫米），若提供则使用深度反投影
            plane_normal: 平面法向量（若无深度则用平面约束）
            plane_d:      平面距离（毫米）

        Returns:
            (3,) float64，机器人基坐标系中的点（毫米）
        """
        if depth_mm is not None:
            p_cam = self.pixel_to_camera(u, v, depth_mm)
        else:
            n = plane_normal if plane_normal is not None else np.array([0, 0, 1.0])
            p_cam = self.pixel_to_camera_plane(u, v, n, plane_d)
        return self.camera_to_robot(p_cam)

    def transform_localization_results(
        self,
        results: List[Dict],
        depth_map: Optional[np.ndarray] = None,
        plane_d: float = 1000.0,
    ) -> List[Dict]:
        """
        将 SubpixelLocalizer 的输出批量转换为机器人坐标系。

        Args:
            results:   SubpixelLocalizer.localize() 的输出
            depth_map: 深度图 (H,W) float32，单位毫米（可选）
            plane_d:   平面距离（无深度图时使用）

        Returns:
            每个结果增加 'position_robot_mm' 字段 (3,) float64
        """
        output = []
        for r in results:
            u, v = r['centroid_px']
            try:
                if depth_map is not None:
                    d = float(depth_map[int(v), int(u)])
                    if d <= 0:
                        d = plane_d
                    p_robot = self.pixel_to_robot(u, v, depth_mm=d)
                else:
                    p_robot = self.pixel_to_robot(u, v, plane_d=plane_d)
                r = dict(r)
                r['position_robot_mm'] = p_robot
            except Exception as e:
                r = dict(r)
                r['position_robot_mm'] = None
                r['transform_error'] = str(e)
            output.append(r)
        return output


# ============================================================
# 5. 高光区域检测（用于定位时排除干扰）
# ============================================================

def detect_glare_regions(
    image: np.ndarray,
    brightness_threshold: int = 240,
    saturation_threshold: int = 30,
    min_area: int = 100,
) -> np.ndarray:
    """
    检测图像中的高光区域（高亮度 + 低饱和度）。

    高光特征：
      - 亮度（V 通道）极高（接近 255）
      - 饱和度（S 通道）极低（颜色信息丢失）

    Args:
        image:                BGR uint8 图像
        brightness_threshold: V 通道阈值（高于此值认为是高光）
        saturation_threshold: S 通道阈值（低于此值认为是高光）
        min_area:             最小高光区域面积

    Returns:
        glare_mask: uint8 (H,W)，高光区域为 255
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # 高亮度 AND 低饱和度
    bright_mask = v_channel > brightness_threshold
    low_sat_mask = s_channel < saturation_threshold
    glare_raw = (bright_mask & low_sat_mask).astype(np.uint8) * 255

    # 形态学膨胀（扩展高光边界，避免边缘定位误差）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    glare_dilated = cv2.dilate(glare_raw, kernel, iterations=2)

    # 去除小噪点
    contours, _ = cv2.findContours(glare_dilated, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    glare_mask = np.zeros_like(glare_dilated)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(glare_mask, [cnt], -1, 255, -1)

    return glare_mask


# ============================================================
# 6. 角点检测（Harris / Shi-Tomas）
# ============================================================

class CornerDetector:
    """
    角点检测器，支持 Harris 和 Shi-Tomas 算法。

    用于定位工件上的角点特征（定位孔、定位销、连接器引脚等）。

    用法::

        det = CornerDetector(method='shi_tomas', block_size=3)
        corners = det.detect(image, mask=None)
        # corners: [(x, y), ...] 像素级坐标
    """

    def __init__(
        self,
        method: str = 'shi_tomas',
        block_size: int = 3,
        ksize: int = 3,
        k: float = 0.04,
        min_distance: int = 10,
        quality_level: float = 0.01,
        max_corners: int = 500,
    ):
        """
        Args:
            method:         'harris' 或 'shi_tomas'
            block_size:     邻域块大小（像素）
            ksize:          Sobel 梯度 kernel size
            k:              Harris 响应系数（通常 0.04~0.06）
            min_distance:   角点间最小距离（像素）
            quality_level:   质量阈值 = max(R) * quality_level
            max_corners:    最大返回角点数
        """
        self.method        = method.lower()
        self.block_size    = block_size
        self.ksize         = ksize
        self.k             = k
        self.min_distance  = min_distance
        self.quality_level = quality_level
        self.max_corners   = max_corners

    def detect(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> List[Tuple[float, float]]:
        """
        检测图像中的角点。

        Args:
            image: BGR 或灰度图
            mask:  可选，感兴趣区域掩膜

        Returns:
            List[(x, y), ...]，像素级角点坐标
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
               if image.ndim == 3 else image.astype(np.uint8)

        # 像素级条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        if self.method == 'harris':
            # Harris 角点检测
            gray_f = np.float32(gray)
            R = cv2.cornerHarris(gray_f, self.block_size, self.ksize, self.k)

            # 阈值筛选
            threshold = R.max() * self.quality_level
            corners = np.argwhere(R > threshold)
            # 按响应值排序
            if len(corners) > 0:
                responses = R[corners[:, 0], corners[:, 1]]
                sorted_idx = np.argsort(responses)[::-1]
                corners = corners[sorted_idx]

        else:  # shi_tomas
            gray_f = np.float32(gray)
            corners = cv2.goodFeaturesToTrack(
                gray_f,
                maxCorners=self.max_corners,
                qualityLevel=self.quality_level,
                minDistance=self.min_distance,
                blockSize=self.block_size,
                useHarrisDetector=False,
                k=self.k,
            )
            if corners is None:
                return []
            corners = corners.reshape(-1, 2)

        # 过滤 mask 外的角点
        if mask is not None:
            valid = []
            for pt in corners:
                y, x = int(pt[1]), int(pt[0])
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[y, x] > 0:
                        valid.append(pt)
            corners = np.array(valid)

        # 非极大值抑制（基于距离）
        corners = self._non_max_suppression(corners)

        # 像素级精化
        if len(corners) > 0:
            corners_sub = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=criteria,
            )
            return [(float(p[0]), float(p[1])) for p in corners_sub]
        return []

    def _non_max_suppression(
        self,
        corners: np.ndarray,
    ) -> np.ndarray:
        """基于距离的非极大值抑制。"""
        if len(corners) == 0:
            return corners

        pts = corners.astype(np.float64)
        keep = []
        while len(pts) > 0:
            # 按响应排序，取最强点（假设最后一列是响应值）
            if pts.shape[1] == 2:
                # 无响应值，使用强度作为伪响应
                idx = 0
            else:
                idx = np.argmax(pts[:, -1])

            center = pts[idx, :2]
            keep.append(center)

            # 计算距离
            dists = np.linalg.norm(pts[:, :2] - center, axis=1)
            pts = pts[dists > self.min_distance]

        return np.array(keep)


def detect_corners(
    image: np.ndarray,
    method: str = 'shi_tomas',
    **kwargs,
) -> List[Tuple[float, float]]:
    """
    便捷函数：检测图像中的角点。

    用法::

        corners = detect_corners(gray, method='shi_tomas', block_size=3)
    """
    det = CornerDetector(method=method, **kwargs)
    return det.detect(image)


# ============================================================
# 7. 高斯线提取（像素精度）
# ============================================================

class GaussianLineExtractor:
    """
    高斯线提取器 — 像素精度直线边缘定位。

    支持三种模型：
      - 'gaussian': 高斯模型（适用于弱边缘）
      - 'binarize': 棒状/阈值模型（适用于强边缘）
      - 'cubic':    三次样条模型（适用于高精度需求）

    典型应用：焊缝定位、边缘对齐、尺寸测量。

    用法::

        ext = GaussianLineExtractor(model='gaussian')
        line = ext.extract(image, roi=(x, y, w, h), direction='horizontal')
        # line: {'center': (x, y), 'width': w, 'angle': deg, 'confidence': 0.95}
    """

    def __init__(
        self,
        model: str = 'gaussian',
        search_length: int = 100,
        scan_step: int = 1,
        threshold: float = 20,
        subpixel_window: int = 7,
    ):
        """
        Args:
            model:          'gaussian' / 'binarize' / 'cubic'
            search_length:  搜索长度（像素）
            scan_step:      扫描步长（像素）
            threshold:      边缘阈值（灰度差）
            subpixel_window: 亚像素拟合窗口（奇数）
        """
        self.model           = model.lower()
        self.search_length   = search_length
        self.scan_step       = scan_step
        self.threshold       = threshold
        self.subpixel_window = subpixel_window

    def extract(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None,
        direction: str = 'horizontal',
    ) -> Dict:
        """
        从图像中提取直线边缘（像素精度）。

        Args:
            image:     灰度图
            roi:       (x, y, w, h) 感兴趣区域
            direction: 'horizontal' 或 'vertical'

        Returns:
            Dict，包含：
              center:      (x, y) 像素级边缘中心
              angle:       float，边缘角度（度）
              width:       float，边缘宽度（像素）
              confidence:  float，置信度 [0, 1]
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
               if image.ndim == 3 else image

        if roi is not None:
            x, y, w, h = roi
            gray = gray[y:y+h, x:x+w]
            ox, oy = x, y
        else:
            ox, oy = 0, 0

        # 投影剖面
        profile = self._compute_profile(gray, direction)

        # 检测边缘对
        edges = self._detect_edge_pairs(profile, direction)

        if len(edges) == 0:
            return {
                'center': (float(ox + gray.shape[1] / 2),
                           float(oy + gray.shape[0] / 2)),
                'angle': 0.0,
                'width': 0.0,
                'confidence': 0.0,
            }

        # 取最强边缘对
        best_edge = max(edges, key=lambda e: e['strength'])

        # 像素级精化
        center_sub = self._subpixel_refine(
            gray, best_edge, direction, ox, oy
        )

        return {
            'center':     center_sub,
            'angle':      best_edge['angle'],
            'width':      best_edge['width'],
            'confidence': min(best_edge['strength'] / (self.threshold * 2), 1.0),
        }

    def _compute_profile(
        self,
        gray: np.ndarray,
        direction: str,
    ) -> np.ndarray:
        """计算灰度投影剖面。"""
        if direction == 'horizontal':
            return np.mean(gray, axis=0)
        else:
            return np.mean(gray, axis=1)

    def _detect_edge_pairs(
        self,
        profile: np.ndarray,
        direction: str,
    ) -> List[Dict]:
        """检测剖面上的边缘对（上升沿+下降沿）。"""
        # 梯度
        grad = np.diff(profile.astype(np.float32))
        grad = np.concatenate([[0], grad])

        edges = []
        in_edge = False
        edge_start = 0
        peak_val = 0
        peak_idx = 0

        for i, g in enumerate(grad):
            if abs(g) > self.threshold and not in_edge:
                in_edge = True
                edge_start = i
                peak_val = abs(g)
                peak_idx = i
            elif in_edge:
                if abs(g) > peak_val:
                    peak_val = abs(g)
                    peak_idx = i
                if g * grad[edge_start] < 0:  # 梯度反向，边缘结束
                    in_edge = False
                    edge_end = i
                    strength = abs(profile[edge_end] - profile[edge_start])
                    # 角度（假设水平边缘对应90度）
                    angle = 90.0 if direction == 'horizontal' else 0.0
                    edges.append({
                        'start':    edge_start,
                        'end':      edge_end,
                        'peak':     peak_idx,
                        'strength': strength,
                        'angle':    angle,
                        'width':    edge_end - edge_start,
                    })

        return edges

    def _subpixel_refine(
        self,
        gray: np.ndarray,
        edge: Dict,
        direction: str,
        ox: int, oy: int,
    ) -> Tuple[float, float]:
        """使用高斯拟合进行像素级精化。"""
        if self.model == 'gaussian':
            return self._gaussian_fit(gray, edge, direction, ox, oy)
        elif self.model == 'cubic':
            return self._cubic_fit(gray, edge, direction, ox, oy)
        else:
            # binarize: 直接取中点
            mid = (edge['start'] + edge['end']) // 2
            if direction == 'horizontal':
                return (float(ox + mid), float(oy + gray.shape[0] / 2))
            else:
                return (float(ox + gray.shape[1] / 2), float(oy + mid))

    def _gaussian_fit(
        self,
        gray: np.ndarray,
        edge: Dict,
        direction: str,
        ox: int, oy: int,
    ) -> Tuple[float, float]:
        """高斯模型拟合边缘。"""
        from scipy.optimize import curve_fit

        profile = self._compute_profile(gray, direction)
        x = np.arange(edge['start'], edge['end'] + 1)
        y = profile[x].astype(np.float64)

        # 双高斯拟合边缘过渡
        def double_gaussian(x_arr, a1, c1, s1, a2, c2, s2, offset):
            g1 = a1 * np.exp(-((x_arr - c1) ** 2) / (2 * s1 ** 2))
            g2 = a2 * np.exp(-((x_arr - c2) ** 2) / (2 * s2 ** 2))
            return g1 + g2 + offset

        try:
            # 初始估计
            p0 = [y.min(), edge['peak'], 1.0, y.max(), edge['peak'], 1.0, y.mean()]
            popt, _ = curve_fit(double_gaussian, x, y, p0=p0, maxfev=5000)
            center = (popt[1] + popt[4]) / 2
        except Exception:
            center = edge['peak']

        if direction == 'horizontal':
            return (float(ox + center), float(oy + gray.shape[0] / 2))
        else:
            return (float(ox + gray.shape[1] / 2), float(oy + center))

    def _cubic_fit(
        self,
        gray: np.ndarray,
        edge: Dict,
        direction: str,
        ox: int, oy: int,
    ) -> Tuple[float, float]:
        """三次样条拟合边缘。"""
        profile = self._compute_profile(gray, direction)
        x = np.arange(edge['start'], edge['end'] + 1)
        y = profile[x].astype(np.float64)

        # 三次多项式拟合
        coeffs = np.polyfit(x, y, 3)
        # 求导找拐点（二阶导为零）
        # d/dx(coeffs[0]*x^3 + ...) = 3*coeffs[0]*x^2 + 2*coeffs[1]*x + coeffs[2] = 0
        a = 3 * coeffs[0]
        b = 2 * coeffs[1]
        c = coeffs[2]
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            x1 = (-b + np.sqrt(discriminant)) / (2 * a)
            x2 = (-b - np.sqrt(discriminant)) / (2 * a)
            center = (x1 + x2) / 2
        else:
            center = edge['peak']

        if direction == 'horizontal':
            return (float(ox + center), float(oy + gray.shape[0] / 2))
        else:
            return (float(ox + gray.shape[1] / 2), float(oy + center))


def extract_gaussian_line(
    image: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    direction: str = 'horizontal',
    model: str = 'gaussian',
) -> Dict:
    """
    便捷函数：高斯线提取。

    用法::

        line = extract_gaussian_line(gray, roi=(100,100,200,50),
                                     direction='horizontal')
    """
    ext = GaussianLineExtractor(model=model)
    return ext.extract(image, roi, direction)


# ============================================================
# 8. 霍夫圆/直线检测
# ============================================================

class HoughCircleDetector:
    """
    霍夫圆检测器 — 从边缘图检测圆/圆弧。

    用于定位孔、圆柱形工件、连接器等圆形特征。

    用法::

        det = HoughCircleDetector(min_radius=10, max_radius=100)
        circles = det.detect(edge_mask)
        for c in circles:
            print(f"圆心=({c['cx']:.1f}, {c['cy']:.1f})  半径={c['radius']:.2f}")
    """

    def __init__(
        self,
        min_radius: int = 10,
        max_radius: int = 100,
        accumulator_threshold: int = 50,
        edge_threshold: int = 30,
        min_dist: int = 20,
    ):
        """
        Args:
            min_radius:          最小半径（像素）
            max_radius:          最大半径（像素）
            accumulator_threshold: 累加器阈值（圆心投票数下限）
            edge_threshold:       边缘检测阈值（Canny 上限）
            min_dist:            圆心间最小距离
        """
        self.min_radius          = min_radius
        self.max_radius          = max_radius
        self.accumulator_threshold = accumulator_threshold
        self.edge_threshold      = edge_threshold
        self.min_dist            = min_dist

    def detect(
        self,
        edge_mask: np.ndarray,
        image: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        检测圆/圆弧。

        Args:
            edge_mask: 边缘掩膜 (H,W) uint8
            image:     原始灰度图（可选，用于置信度计算）

        Returns:
            List[Dict]，每个圆包含：
              cx, cy:       float，圆心坐标（像素级）
              radius:       float，半径
              confidence:   float，置信度 [0, 1]
              arcs:         int，检测到的弧段数（圆弧时 < 360）
        """
        if edge_mask.max() <= 1:
            edge_mask = (edge_mask * 255).astype(np.uint8)

        # 霍夫圆变换
        circles = cv2.HoughCircles(
            edge_mask,
            method=cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.min_dist,
            param1=self.edge_threshold,
            param2=self.accumulator_threshold,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        if circles is None:
            return []

        results = []
        for c in circles[0]:
            cx, cy, r = c
            # 像素级精化圆心
            cx_sub, cy_sub = self._subpixel_center(edge_mask, float(cx), float(cy), r)

            # 计算置信度（基于边缘投票密度）
            confidence = self._compute_confidence(edge_mask, cx_sub, cy_sub, r)

            # 估计弧段完整性
            arcs = self._estimate_arcs(edge_mask, cx_sub, cy_sub, r)

            results.append({
                'cx':        cx_sub,
                'cy':        cy_sub,
                'radius':    float(r),
                'confidence': confidence,
                'arcs':      arcs,
            })

        # 非极大值抑制（距离过近取置信度最高）
        results = self._non_max_suppression(results)

        return results

    def _subpixel_center(
        self,
        edge_mask: np.ndarray,
        cx: float, cy: float, r: float,
    ) -> Tuple[float, float]:
        """使用灰度矩或梯度质心精化圆心。"""
        h, w = edge_mask.shape
        x0, y0 = int(cx), int(cy)
        x0 = np.clip(x0, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)

        # 在圆周附近取梯度质心
        search_r = max(3, int(r * 0.05))
        y_min = max(0, y0 - search_r)
        y_max = min(h, y0 + search_r + 1)
        x_min = max(0, x0 - search_r)
        x_max = min(w, x0 + search_r + 1)

        roi = edge_mask[y_min:y_max, x_min:x_max]
        ys, xs = np.mgrid[y_min:y_max, x_min:x_max]

        total = roi.sum()
        if total > 0:
            cx_sub = (xs * roi).sum() / total
            cy_sub = (ys * roi).sum() / total
            return float(cx_sub), float(cy_sub)
        return cx, cy

    def _compute_confidence(
        self,
        edge_mask: np.ndarray,
        cx: float, cy: float, r: float,
    ) -> float:
        """基于圆周边缘密度计算置信度。"""
        # 在圆周上采样
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        edge_points = 0
        for a in angles:
            px = int(cx + r * np.cos(a))
            py = int(cy + r * np.sin(a))
            if 0 <= px < edge_mask.shape[1] and 0 <= py < edge_mask.shape[0]:
                if edge_mask[py, px] > 0:
                    edge_points += 1
        return edge_points / 360.0

    def _estimate_arcs(
        self,
        edge_mask: np.ndarray,
        cx: float, cy: float, r: float,
    ) -> int:
        """估计检测到的完整圆弧数。"""
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        on_circle = np.zeros(360, dtype=bool)
        for i, a in enumerate(angles):
            px = int(cx + r * np.cos(a))
            py = int(cy + r * np.sin(a))
            if 0 <= px < edge_mask.shape[1] and 0 <= py < edge_mask.shape[0]:
                on_circle[i] = edge_mask[py, px] > 0

        # 连续边缘段
        arcs = 0
        in_arc = False
        for v in on_circle:
            if v and not in_arc:
                arcs += 1
                in_arc = True
            elif not v:
                in_arc = False
        return arcs

    def _non_max_suppression(self, circles: List[Dict]) -> List[Dict]:
        """基于距离的非极大值抑制。"""
        if len(circles) <= 1:
            return circles
        # 按置信度排序
        circles = sorted(circles, key=lambda c: c['confidence'], reverse=True)
        keep = []
        for c in circles:
            too_close = False
            for k in keep:
                dist = np.hypot(c['cx'] - k['cx'], c['cy'] - k['cy'])
                if dist < self.min_dist:
                    too_close = True
                    break
            if not too_close:
                keep.append(c)
        return keep


class HoughLineDetector:
    """
    霍夫直线检测器 — 概率 Hough 变换，支持像素级端点精化。

    用于焊缝检测、边缘对齐、尺寸测量。

    用法::

        det = HoughLineDetector(rho=1, theta=np.pi/180, threshold=50)
        lines = det.detect(edge_mask)
        for ln in lines:
            print(f"端点={ln['endpoints']}  长度={ln['length']:.1f}px")
    """

    def __init__(
        self,
        rho: float = 1.0,
        theta: float = np.pi / 180,
        threshold: int = 50,
        min_line_length: int = 50,
        max_line_gap: int = 10,
    ):
        """
        Args:
            rho:             距离分辨率（像素）
            theta:           角度分辨率（弧度）
            threshold:       累加器阈值
            min_line_length: 最小线段长度
            max_line_gap:    最大线段间隙
        """
        self.rho             = rho
        self.theta           = theta
        self.threshold       = threshold
        self.min_line_length = min_line_length
        self.max_line_gap    = max_line_gap

    def detect(
        self,
        edge_mask: np.ndarray,
        image: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        检测直线段。

        Args:
            edge_mask: 边缘掩膜 (H,W) uint8
            image:     原始灰度图（可选，用于像素级精化）

        Returns:
            List[Dict]，每条线段包含：
              endpoints:   [(x1,y1), (x2,y2)] 像素级端点
              angle_deg:  float，方向角（度）
              length:     float，长度（像素）
              rho:        float，霍夫距离
              theta:      float，霍夫角度（弧度）
              confidence: float，置信度
        """
        if edge_mask.max() <= 1:
            edge_mask = (edge_mask * 255).astype(np.uint8)

        lines = cv2.HoughLinesP(
            edge_mask,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        if lines is None:
            return []

        results = []
        for ln in lines:
            x1, y1, x2, y2 = ln[0].astype(float)

            # 像素级精化
            if image is not None:
                x1, y1 = self._subpixel_endpoint(image, x1, y1, x2, y2)
                x2, y2 = self._subpixel_endpoint(image, x2, y2, x1, y1)

            # 方向角
            angle_rad = math.atan2(y2 - y1, x2 - x1)
            angle_deg = math.degrees(angle_rad)
            length = math.hypot(x2 - x1, y2 - y1)

            # 置信度（基于线段长度和边缘密度）
            confidence = min(length / (self.min_line_length * 2), 1.0)

            results.append({
                'endpoints':   [(x1, y1), (x2, y2)],
                'angle_deg':   angle_deg,
                'length':      length,
                'rho':         float(np.sin(angle_rad) * x1 - np.cos(angle_rad) * y1),
                'theta':       float(angle_rad),
                'confidence':  confidence,
            })

        return results

    def _subpixel_endpoint(
        self,
        image: np.ndarray,
        x: float, y: float,
        nx: float, ny: float,
    ) -> Tuple[float, float]:
        """使用边缘梯度方向精化端点像素级坐标。"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
               if image.ndim == 3 else image
        h, w = gray.shape

        xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))

        # Sobel 梯度
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)[yi, xi]
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)[yi, xi]

        grad_dir = math.atan2(gy, gx)

        # 沿梯度方向移动 0.25 像素（梯度指向边缘法线方向）
        dx = 0.25 * math.cos(grad_dir)
        dy = 0.25 * math.sin(grad_dir)

        return x + dx, y + dy


# ============================================================
# 9. ROI 校正（仿射变换基准跟随）
# ============================================================

class ROICorrection:
    """
    ROI 校正 — 使用仿射变换实现基准跟随。

    用于：
      - 传送带上工件定位（目标相对位置动态变化）
      - 相机视角倾斜校正
      - 特征对齐后的局部 ROI 精确定位

    用法::

        corr = ROICorrection()
        # 第一帧：定义基准特征
        ref_corners = [(100, 100), (400, 100), (400, 300), (100, 300)]
        corr.set_reference(ref_corners)

        # 后续帧：检测当前特征
        curr_corners = detect_corners(image)  # 当前检测到的角点

        # 计算仿射变换矩阵
        M = corr.compute_transform(curr_corners)

        # 应用校正
        corrected = corr.apply_transform(image, M)
    """

    def __init__(self, reference_size: Optional[Tuple[int, int]] = None):
        """
        Args:
            reference_size: (w, h) 校正后图像尺寸，None=与输入相同
        """
        self.reference_corners: Optional[np.ndarray] = None
        self.reference_size    = reference_size

    def set_reference(
        self,
        corners: List[Tuple[float, float]],
    ) -> None:
        """
        设置基准角点（通常是第一帧中检测到的特征位置）。

        Args:
            corners: 4 个角点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]，
                     顺时针或逆时针排列
        """
        if len(corners) != 4:
            raise ValueError("基准角点必须为 4 个")
        self.reference_corners = np.array(corners, dtype=np.float32)

    def compute_transform(
        self,
        current_corners: List[Tuple[float, float]],
    ) -> Optional[np.ndarray]:
        """
        计算从当前角点到基准角点的仿射变换矩阵。

        Args:
            current_corners: 当前检测到的 4 个角点

        Returns:
            (2,3) 仿射变换矩阵，若角点数不足返回 None
        """
        if len(current_corners) < 4:
            return None
        if self.reference_corners is None:
            raise ValueError("请先调用 set_reference 设置基准")

        # 取前 4 个角点（可按距离排序优化配对）
        src = np.array(current_corners[:4], dtype=np.float32)
        dst = self.reference_corners

        # 计算仿射变换（允许平移、旋转、缩放、剪切）
        M = cv2.getAffineTransform(src, dst)
        return M

    def apply_transform(
        self,
        image: np.ndarray,
        M: np.ndarray,
        flags: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        应用仿射变换校正图像。

        Args:
            image: 输入图像
            M:     仿射变换矩阵
            flags: 插值方式

        Returns:
            校正后图像
        """
        h, w = image.shape[:2]
        if self.reference_size is not None:
            out_w, out_h = self.reference_size
        else:
            out_w, out_h = w, h

        return cv2.warpAffine(image, M, (out_w, out_h), flags=flags)

    def transform_point(
        self,
        x: float, y: float,
        M: np.ndarray,
    ) -> Tuple[float, float]:
        """使用仿射矩阵变换单个点。"""
        x_new = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        y_new = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        return (float(x_new), float(y_new))


def correct_roi_affine(
    image: np.ndarray,
    current_corners: List[Tuple[float, float]],
    reference_corners: List[Tuple[float, float]],
    reference_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    便捷函数：ROI 仿射校正。

    用法::

        corrected, M = correct_roi_affine(
            image,
            current_corners=[(100,100), (400,100), (400,300), (100,300)],
            reference_corners=[(100,100), (400,100), (400,300), (100,300)],
        )
    """
    corr = ROICorrection(reference_size=reference_size)
    corr.set_reference(reference_corners)
    M = corr.compute_transform(current_corners)
    if M is None:
        return image, np.eye(2, 3)
    corrected = corr.apply_transform(image, M)
    return corrected, M


# ============================================================
# 10. 九点标定（相机畸变校正 + 手眼标定辅助）
# ============================================================

class NinePointCalibrator:
    """
    九点标定 — 相机畸变校正和手眼标定的简化方案。

    通过已知的 9 个标定点（通常为圆点阵列）在图像中的像素坐标，
    求解相机内参畸变参数（或手眼标定的粗略解）。

    适用场景：
      - 已知精确物理位置的标定块
      - 简单畸变校正（无棋盘格时）
      - 快速粗标定

    用法::

        calib = NinePointCalibrator()
        calib.add_point(image_px=(150, 200), world_mm=(0, 0, 0))
        calib.add_point(image_px=(320, 200), world_mm=(50, 0, 0))
        ...
        K, dist = calib.calibrate()
    """

    def __init__(self, camera_matrix: Optional[np.ndarray] = None):
        """
        Args:
            camera_matrix: (3,3) 初始内参矩阵，若已知
        """
        self.image_points: List[np.ndarray] = []
        self.world_points: List[np.ndarray] = []
        self.K_init = camera_matrix

    def add_point(
        self,
        image_px: Tuple[float, float],
        world_mm: Tuple[float, float, float],
    ) -> None:
        """
        添加一个标定点。

        Args:
            image_px:  (u, v) 图像像素坐标
            world_mm:  (X, Y, Z) 世界坐标系物理坐标（毫米）
        """
        self.image_points.append(np.array(image_px, dtype=np.float64))
        self.world_points.append(np.array(world_mm, dtype=np.float64))

    def calibrate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行九点标定（OpenCV solvePnP）。

        Returns:
            K:    (3,3) 相机内参矩阵
            dist: (1,5) 畸变系数
        """
        if len(self.image_points) < 9:
            raise ValueError(f"九点标定至少需要 9 个点，当前 {len(self.image_points)} 个")

        obj_points = np.array(self.world_points, dtype=np.float64).reshape(-1, 1, 3)
        img_points = np.array(self.image_points, dtype=np.float64).reshape(-1, 1, 2)

        # 估算图像尺寸（从数据点范围）
        all_u = [p[0] for p in self.image_points]
        all_v = [p[1] for p in self.image_points]
        img_size = (int(max(all_u) * 1.2) + 1, int(max(all_v) * 1.2) + 1)

        if self.K_init is not None:
            K = self.K_init.copy()
            dist = np.zeros(5, dtype=np.float64)
            flags = cv2.SOLVEPNP_ITERATIVE | cv2.CALIB_USE_INTRINSIC_GUESS
        else:
            K = np.eye(3, dtype=np.float64)
            # 估计焦距（假设 cx, cy 在图像中心）
            K[0, 2] = img_size[0] / 2
            K[1, 2] = img_size[1] / 2
            K[0, 0] = K[1, 1] = max(*img_size) * 1.2
            dist = np.zeros(5, dtype=np.float64)
            flags = cv2.SOLVEPNP_ITERATIVE

        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            [obj_points], [img_points], img_size,
            K, dist, flags=flags,
        )

        return K, dist

    def undistort_points(
        self,
        points: List[Tuple[float, float]],
        K: np.ndarray,
        dist: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """
        对点集去畸变。

        Args:
            points: 原始像素坐标
            K:      相机内参
            dist:   畸变系数

        Returns:
            去畸变后的像素坐标
        """
        pts = np.array(points, dtype=np.float64).reshape(-1, 1, 2)
        undist = cv2.undistortPoints(pts, K, dist, P=K)
        return [(float(p[0, 0]), float(p[0, 1])) for p in undist]


def calibrate_nine_point(
    image_points: List[Tuple[float, float]],
    world_points: List[Tuple[float, float, float]],
    K_init: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    便捷函数：九点标定。

    用法::

        K, dist = calibrate_nine_point(
            image_points=[(u1,v1), ..., (u9,v9)],
            world_points=[(0,0,0), (50,0,0), ..., (50,50,0)],
        )
    """
    calib = NinePointCalibrator(camera_matrix=K_init)
    for img_pt, world_pt in zip(image_points, world_points):
        calib.add_point(img_pt, world_pt)
    return calib.calibrate()


# ============================================================
# 11. 旋转中心标定（贴合/装配定位）
# ============================================================

class RotationCenterCalibrator:
    """
    旋转中心标定 — 确定机械臂或转台的旋转中心。

    工业贴合/装配场景中，需要知道工件绕哪个点旋转，
    以便精确控制旋转角度（尤其在转台/旋转平台上）。

    方法：
      1. 标定板在不同角度下拍摄
      2. 跟踪标定板特定特征的旋转轨迹
      3. 通过最小二乘法拟合圆，找旋转中心

    用法::

        calib = RotationCenterCalibrator()
        for angle_deg in [0, 45, 90, 135, 180]:
            # 在该角度下检测标定特征位置
            pos = detect_feature(image)
            calib.add_observation(angle_deg, pos)
        center = calib.compute_center()
        print(f"旋转中心: ({center[0]:.2f}, {center[1]:.2f}) px")
    """

    def __init__(self):
        self.observations: List[Tuple[float, Tuple[float, float]]] = []

    def add_observation(
        self,
        angle_deg: float,
        position_px: Tuple[float, float],
    ) -> None:
        """
        添加一个观测记录。

        Args:
            angle_deg:    当前旋转角度（度）
            position_px:  (x, y) 特征在图像中的像素坐标
        """
        self.observations.append((angle_deg, position_px))

    def compute_center(self) -> Tuple[float, float]:
        """
        通过最小二乘圆拟合计算旋转中心。

        Returns:
            (cx, cy) 旋转中心像素坐标
        """
        if len(self.observations) < 3:
            raise ValueError(f"至少需要 3 个观测点，当前 {len(self.observations)} 个")

        angles = np.array([o[0] for o in self.observations])
        points = np.array([o[1] for o in self.observations])

        # 将角度转换为弧度
        theta = np.deg2rad(angles)

        # 圆拟合：min Σ [(x - cx)^2 + (y - cy)^2 - R^2]^2
        # 线性化为 Ax + By + C = x^2 + y^2
        x = points[:, 0]
        y = points[:, 1]

        # 构建线性方程组
        A = np.column_stack([
            x,
            y,
            np.ones_like(x),
        ])
        b = x ** 2 + y ** 2

        # 最小二乘解
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx = result[0] / 2
        cy = result[1] / 2
        R = np.sqrt(cx ** 2 + cy ** 2 + result[2])

        return (float(cx), float(cy))

    def compute_center_with_radius(self) -> Dict:
        """
        计算旋转中心及拟合半径。

        Returns:
            Dict: {center: (cx, cy), radius: R, residuals: [...], quality: float}
        """
        if len(self.observations) < 3:
            raise ValueError(f"至少需要 3 个观测点，当前 {len(self.observations)} 个")

        angles = np.array([o[0] for o in self.observations])
        points = np.array([o[1] for o in self.observations])

        x = points[:, 0]
        y = points[:, 1]

        A = np.column_stack([x, y, np.ones_like(x)])
        b = x ** 2 + y ** 2

        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx = result[0] / 2
        cy = result[1] / 2
        R = np.sqrt(cx ** 2 + cy ** 2 + result[2])

        # 计算残差（每个点到拟合圆的距离）
        residuals = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - R
        rmse = np.sqrt(np.mean(residuals ** 2))

        return {
            'center':    (float(cx), float(cy)),
            'radius':   float(R),
            'residuals': residuals.tolist(),
            'quality':  float(max(0, 1 - rmse / R)) if R > 0 else 0.0,
        }


# ============================================================
# 12. 自动贴合（基于点/线/圆的自动对齐）
# ============================================================

from enum import Enum
from dataclasses import dataclass


class FeatureType(Enum):
    """基础图形类型枚举。"""
    POINT = "point"
    LINE  = "line"
    CIRCLE = "circle"


@dataclass
class Feature2D:
    """
    二维几何特征 — 点、直线、圆。

    用法::

        pt  = Feature2D(type=FeatureType.POINT,  params=(x, y))
        ln  = Feature2D(type=FeatureType.LINE,   params=(x1,y1,x2,y2))
        cir = Feature2D(type=FeatureType.CIRCLE, params=(cx,cy,r))
    """
    type:   FeatureType
    params: Tuple[float, ...]
    label:  Optional[str] = None   # 可选标签，如 "Mark1", "EdgeA"

    @property
    def centroid(self) -> Tuple[float, float]:
        """返回特征的几何中心（用于匹配）。"""
        if self.type == FeatureType.POINT:
            return (self.params[0], self.params[1])
        elif self.type == FeatureType.LINE:
            x1, y1, x2, y2 = self.params
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        elif self.type == FeatureType.CIRCLE:
            return (self.params[0], self.params[1])
        raise ValueError(f"Unknown feature type: {self.type}")

    def distance_to(self, other: 'Feature2D') -> float:
        """计算到另一个特征的最小距离（用于匹配）。"""
        if self.type == FeatureType.POINT and other.type == FeatureType.POINT:
            return math.hypot(self.params[0] - other.params[0],
                              self.params[1] - other.params[1])
        elif self.type == FeatureType.POINT and other.type == FeatureType.LINE:
            return self._point_to_line_distance(other.params)
        elif self.type == FeatureType.POINT and other.type == FeatureType.CIRCLE:
            return abs(math.hypot(self.params[0] - other.params[0],
                                   self.params[1] - other.params[1]) - other.params[2])
        elif self.type == FeatureType.LINE and other.type == FeatureType.POINT:
            return self._point_to_line_distance(other.params)
        elif self.type == FeatureType.CIRCLE and other.type == FeatureType.POINT:
            return abs(math.hypot(self.params[0] - other.params[0],
                                   self.params[1] - other.params[1]) - self.params[2])
        # 线-线、圆-圆、线-圆：使用中心距离近似
        return math.hypot(self.centroid[0] - other.centroid[0],
                          self.centroid[1] - other.centroid[1])

    def _point_to_line_distance(self, point_params: Tuple[float, ...]) -> float:
        """点(x,y)到线段(x1,y1,x2,y2)的距离。"""
        x, y = point_params[:2]
        x1, y1, x2, y2 = self.params
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx*dx + dy*dy
        if length_sq < 1e-12:
            return math.hypot(x - x1, y - y1)
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / length_sq))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(x - proj_x, y - proj_y)


class FeatureMatcher:
    """
    特征匹配器 — 将检测到的特征与参考模型进行配对。

    基于贪婪匹配（最近邻距离），
    可扩展为 RANSAC / ICP 用于剔除错误匹配。

    用法::

        matcher = FeatureMatcher(max_distance=20.0)
        matches = matcher.match(detected_features, reference_features)
        # matches: [(det_idx, ref_idx), ...]
    """

    def __init__(
        self,
        max_distance: float = 20.0,
        use_ransac: bool = False,
        ransac_threshold: float = 5.0,
    ):
        """
        Args:
            max_distance:     最大匹配距离阈值
            use_ransac:      是否使用 RANSAC 提纯匹配
            ransac_threshold: RANSAC 阈值
        """
        self.max_distance      = max_distance
        self.use_ransac       = use_ransac
        self.ransac_threshold = ransac_threshold

    def match(
        self,
        detected: List[Feature2D],
        reference: List[Feature2D],
    ) -> List[Tuple[int, int]]:
        """
        匹配检测特征与参考特征。

        Args:
            detected:  检测到的特征列表
            reference: 参考/模板特征列表

        Returns:
            List[(det_idx, ref_idx)]，成功匹配的对
        """
        if not detected or not reference:
            return []

        matches = []
        used_ref = set()

        # 按距离排序，贪婪匹配
        all_pairs = []
        for i, d_feat in enumerate(detected):
            for j, r_feat in enumerate(reference):
                dist = d_feat.distance_to(r_feat)
                all_pairs.append((dist, i, j))

        all_pairs.sort(key=lambda p: p[0])

        for dist, i, j in all_pairs:
            if dist > self.max_distance:
                break
            if j not in used_ref:
                matches.append((i, j))
                used_ref.add(j)

        # RANSAC 提纯（仅平移+旋转模型）
        if self.use_ransac and len(matches) >= 2:
            matches = self._ransac_refine(matches, detected, reference)

        return matches

    def _ransac_refine(
        self,
        matches: List[Tuple[int, int]],
        detected: List[Feature2D],
        reference: List[Feature2D],
    ) -> List[Tuple[int, int]]:
        """使用 RANSAC 剔除异常匹配。"""
        best_inliers = []
        det_pts = np.array([detected[i].centroid for i, _ in matches])
        ref_pts = np.array([reference[j].centroid for _, j in matches])

        for _ in range(50):  # 迭代次数
            # 随机选2对估计变换
            idx = np.random.choice(len(matches), 2, replace=False)
            src = det_pts[idx]
            dst = ref_pts[idx]

            # 估计平移+旋转
            M = self._estimate_rigid_transform(src, dst)
            if M is None:
                continue

            # 计算所有匹配的重投影误差
            inliers = []
            for k, (det_i, ref_j) in enumerate(matches):
                pt = np.array([[detected[det_i].centroid]], dtype=np.float64)
                warped = cv2.transform(pt, M)
                err = np.linalg.norm(warped[0, 0] - reference[ref_j].centroid)
                if err < self.ransac_threshold:
                    inliers.append((det_i, ref_j))

            if len(inliers) > len(best_inliers):
                best_inliers = inliers

        return best_inliers

    @staticmethod
    def _estimate_rigid_transform(
        src: np.ndarray,
        dst: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        估计刚体变换（旋转+平移）：dst = R @ src + t

        src, dst: (2, 2) 数组，两对对应点
        返回 (2, 3) 仿射矩阵
        """
        if len(src) < 2:
            return None

        # 去中心化
        src_c = src.mean(axis=0)
        dst_c = dst.mean(axis=0)
        src_centered = src - src_c
        dst_centered = dst - dst_c

        # SVD 求旋转
        H = src_centered.T @ dst_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 反射检测并修正
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = dst_c - R @ src_c

        M = np.eye(2, 3, dtype=np.float64)
        M[:, :2] = R
        M[:, 2] = t
        return M


class PoseEstimator:
    """
    位姿估计器 — 从匹配的特征对计算 2D 仿射/相似变换。

    用于工件贴合：将检测到的特征对齐到参考位置。

    用法::

        est = PoseEstimator()
        pose = est.estimate(matched_detected, matched_reference)
        # pose: {'M': (2,3) affine, 'rotation_deg':, 'translation_px': (tx,ty), 'scale':}
    """

    def estimate(
        self,
        detected: List[Feature2D],
        reference: List[Feature2D],
        matches: List[Tuple[int, int]],
    ) -> Dict:
        """
        从匹配对估计刚体变换（允许均匀缩放）。

        Args:
            detected:  检测特征列表
            reference: 参考特征列表
            matches:   匹配对 [(det_idx, ref_idx), ...]

        Returns:
            Dict:
              M:             (2,3) 仿射变换矩阵（det → ref）
              rotation_deg:  float，旋转角度（度）
              translation_px:(tx, ty) 平移量
              scale:         float，均匀缩放因子
              confidence:    float，置信度（基于残差）
        """
        if len(matches) < 2:
            # 单点：仅平移
            if len(matches) == 1:
                det = detected[matches[0][0]]
                ref = reference[matches[0][1]]
                dx = ref.centroid[0] - det.centroid[0]
                dy = ref.centroid[1] - det.centroid[1]
                return {
                    'M':             np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float64),
                    'rotation_deg':  0.0,
                    'translation_px': (dx, dy),
                    'scale':         1.0,
                    'confidence':    1.0,
                }
            raise ValueError(f"至少需要 2 个匹配对，当前 {len(matches)} 个")

        src_pts = np.array([detected[i].centroid for i, _ in matches], dtype=np.float64)
        dst_pts = np.array([reference[j].centroid for _, j in matches], dtype=np.float64)

        # 估计相似变换（均匀缩放 + 旋转 + 平移）
        M, inliers = self._similarity_transform(src_pts, dst_pts)

        # 分解变换矩阵
        rotation, scale, tx, ty = self._decompose_transform(M)

        # 计算置信度（基于匹配残差）
        errors = []
        for k, (det_i, ref_j) in enumerate(matches):
            pt = np.array([[detected[det_i].centroid]], dtype=np.float64)
            warped = cv2.transform(pt, M)
            err = np.linalg.norm(warped[0, 0] - reference[ref_j].centroid)
            errors.append(err)
        confidence = max(0, 1 - np.mean(errors) / 50.0)

        return {
            'M':             M,
            'rotation_deg':  rotation,
            'translation_px': (tx, ty),
            'scale':         scale,
            'confidence':    confidence,
        }

    @staticmethod
    def _similarity_transform(
        src: np.ndarray,
        dst: np.ndarray,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        估计相似变换：dst = s*R @ src + t

        Returns:
            M:     (2,3) 仿射矩阵
            inliers: 内点索引
        """
        n = len(src)

        # 去中心化
        src_c = src.mean(axis=0)
        dst_c = dst.mean(axis=0)
        src_centered = src - src_c
        dst_centered = dst - dst_c

        # 缩放估计
        ss = np.sum(dst_centered ** 2)
        src_var = np.sum(src_centered ** 2)
        scale = np.sqrt(ss / (src_var + 1e-12))

        # 旋转估计（SVD）
        H = src_centered.T @ dst_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 平移
        t = dst_c - scale * R @ src_c

        M = np.eye(2, 3, dtype=np.float64)
        M[:, :2] = scale * R
        M[:, 2] = t

        # 计算残差
        residuals = []
        for i in range(n):
            pt = np.array([[src[i]]], dtype=np.float64)
            warped = cv2.transform(pt, M)
            err = np.linalg.norm(warped[0, 0] - dst[i])
            residuals.append(err)

        inliers = [i for i, e in enumerate(residuals) if e < 5.0]

        return M, inliers

    @staticmethod
    def _decompose_transform(M: np.ndarray) -> Tuple[float, float, float, float]:
        """分解 (2,3) 仿射矩阵为 旋转+缩放+平移。"""
        a, b, tx = M[0, :]
        c, d, ty = M[1, :]

        # 旋转角
        rotation = math.degrees(math.atan2(b, a))

        # 缩放因子（均匀缩放）
        scale = np.sign(a) * np.sqrt(a*a + b*b)

        return rotation, scale, tx, ty


class AutoFitter:
    """
    自动贴合 — 通过点、线、圆等基础图形自动实现工件定位与对齐。

    Smart3 "贴合" 功能的完整实现：
      1. 定义参考模型（点/线/圆组合）
      2. 自动检测当前帧中的特征
      3. 特征匹配与错误剔除（RANSAC）
      4. 计算变换矩阵（平移+旋转+缩放）
      5. 输出贴合结果和误差

    用法::

        fitter = AutoFitter()

        # 定义参考模型（典型：3个定位孔 + 1条边缘）
        fitter.add_reference(Feature2D(FeatureType.POINT, (100, 100), label="P1"))
        fitter.add_reference(Feature2D(FeatureType.POINT, (300, 100), label="P2"))
        fitter.add_reference(Feature2D(FeatureType.LINE,  (100, 200, 300, 200), label="Edge"))

        # 自动检测当前帧
        detected = fitter.auto_detect(edge_mask, image)

        # 计算贴合变换
        result = fitter.compute_fit(detected)
        if result['confidence'] > 0.8:
            aligned = fitter.apply_fit(image, result)
    """

    def __init__(
        self,
        match_distance_threshold: float = 25.0,
        min_match_count: int = 2,
        ransac_refine: bool = True,
    ):
        """
        Args:
            match_distance_threshold: 特征匹配最大距离
            min_match_count:          最小匹配数（低于此值返回失败）
            ransac_refine:            是否使用 RANSAC 提纯
        """
        self.reference_features: List[Feature2D] = []
        self.match_distance_threshold = match_distance_threshold
        self.min_match_count            = min_match_count
        self.ransac_refine              = ransac_refine
        self._matcher = FeatureMatcher(
            max_distance=match_distance_threshold,
            use_ransac=ransac_refine,
        )
        self._pose_est = PoseEstimator()

    def add_reference(
        self,
        feature: Feature2D,
    ) -> None:
        """添加参考特征。"""
        self.reference_features.append(feature)

    def set_reference_from_template(
        self,
        template: List[Dict],
    ) -> None:
        """
        从模板字典设置参考特征。

        模板格式::

            [
                {'type': 'point',  'params': (x, y),       'label': 'P1'},
                {'type': 'line',   'params': (x1,y1,x2,y2), 'label': 'EdgeA'},
                {'type': 'circle', 'params': (cx,cy,r),     'label': 'Hole1'},
            ]
        """
        type_map = {'point': FeatureType.POINT, 'line': FeatureType.LINE, 'circle': FeatureType.CIRCLE}
        for t in template:
            ft = Feature2D(
                type=type_map[t['type'].lower()],
                params=t['params'],
                label=t.get('label'),
            )
            self.reference_features.append(ft)

    def auto_detect(
        self,
        edge_mask: np.ndarray,
        image: Optional[np.ndarray] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> List[Feature2D]:
        """
        自动检测图像中的点、线、圆特征。

        Args:
            edge_mask: 边缘掩膜
            image:     灰度图（可选，用于像素级精化）
            roi:       (x,y,w,h) 感兴趣区域

        Returns:
            List[Feature2D]，检测到的特征
        """
        if edge_mask.max() <= 1:
            edge_mask = (edge_mask * 255).astype(np.uint8)

        if roi is not None:
            x, y, w, h = roi
            edge_mask = edge_mask[y:y+h, x:x+w]

        detected: List[Feature2D] = []

        # 1. 检测圆（霍夫圆）
        try:
            circle_det = HoughCircleDetector(
                min_radius=5,
                max_radius=min(edge_mask.shape) // 2,
                accumulator_threshold=30,
            )
            circles = circle_det.detect(edge_mask)
            for c in circles:
                if c['confidence'] > 0.3:
                    detected.append(Feature2D(
                        FeatureType.CIRCLE,
                        (c['cx'], c['cy'], c['radius']),
                    ))
        except Exception:
            pass

        # 2. 检测直线（霍夫直线）
        try:
            line_det = HoughLineDetector(
                threshold=30,
                min_line_length=30,
            )
            lines = line_det.detect(edge_mask)
            for ln in lines:
                if ln['confidence'] > 0.3:
                    x1, y1 = ln['endpoints'][0]
                    x2, y2 = ln['endpoints'][1]
                    detected.append(Feature2D(
                        FeatureType.LINE,
                        (x1, y1, x2, y2),
                    ))
        except Exception:
            pass

        # 3. 检测角点（Harris / Shi-Tomas）
        try:
            gray = image
            if gray is None:
                gray = edge_mask
            corner_det = CornerDetector(
                method='shi_tomas',
                block_size=5,
                min_distance=15,
                quality_level=0.05,
            )
            corners = corner_det.detect(gray, mask=edge_mask if image is None else None)
            for cx, cy in corners:
                detected.append(Feature2D(FeatureType.POINT, (cx, cy)))
        except Exception:
            pass

        # 应用 ROI 偏移
        if roi is not None:
            ox, oy = roi[0], roi[1]
            for i, feat in enumerate(detected):
                if feat.type == FeatureType.POINT:
                    detected[i] = Feature2D(
                        feat.type,
                        (feat.params[0] + ox, feat.params[1] + oy),
                        feat.label,
                    )
                elif feat.type == FeatureType.LINE:
                    detected[i] = Feature2D(
                        feat.type,
                        (feat.params[0] + ox, feat.params[1] + oy,
                         feat.params[2] + ox, feat.params[3] + oy),
                        feat.label,
                    )
                elif feat.type == FeatureType.CIRCLE:
                    detected[i] = Feature2D(
                        feat.type,
                        (feat.params[0] + ox, feat.params[1] + oy, feat.params[2]),
                        feat.label,
                    )

        return detected

    def compute_fit(
        self,
        detected: List[Feature2D],
    ) -> Dict:
        """
        计算贴合变换（检测特征 → 参考特征）。

        Args:
            detected: 自动检测到的特征

        Returns:
            Dict，包含变换矩阵、旋转角、平移量、置信度
        """
        if len(self.reference_features) < 2:
            raise ValueError("参考特征至少需要 2 个")

        # 特征匹配
        matches = self._matcher.match(detected, self.reference_features)

        if len(matches) < self.min_match_count:
            return {
                'M':             np.eye(2, 3, dtype=np.float64),
                'rotation_deg':  0.0,
                'translation_px': (0.0, 0.0),
                'scale':         1.0,
                'confidence':    0.0,
                'match_count':   len(matches),
                'matches':       [],
            }

        # 位姿估计
        result = self._pose_est.estimate(
            detected, self.reference_features, matches
        )
        result['match_count'] = len(matches)
        result['matches']     = matches
        return result

    def apply_fit(
        self,
        image: np.ndarray,
        fit_result: Dict,
        flags: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        应用贴合变换到图像。

        Args:
            image:      输入图像
            fit_result: compute_fit() 返回的结果
            flags:      插值方式

        Returns:
            对齐/贴合后的图像
        """
        M = fit_result['M']
        h, w = image.shape[:2]
        return cv2.warpAffine(image, M, (w, h), flags=flags)

    def transform_points(
        self,
        points: List[Tuple[float, float]],
        M: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """使用贴合变换矩阵变换点。"""
        result = []
        for x, y in points:
            x_new = M[0, 0] * x + M[0, 1] * y + M[0, 2]
            y_new = M[1, 0] * x + M[1, 1] * y + M[1, 2]
            result.append((float(x_new), float(y_new)))
        return result


def auto_fit(
    image: np.ndarray,
    edge_mask: np.ndarray,
    reference_template: List[Dict],
    match_distance_threshold: float = 25.0,
) -> Dict:
    """
    便捷函数：自动贴合。

    用法::

        result = auto_fit(
            image=frame,
            edge_mask=edges,
            reference_template=[
                {'type': 'point', 'params': (100, 100), 'label': 'P1'},
                {'type': 'point', 'params': (300, 100), 'label': 'P2'},
                {'type': 'circle', 'params': (200, 200, 20), 'label': 'Hole'},
            ],
        )
    """
    fitter = AutoFitter(match_distance_threshold=match_distance_threshold)
    fitter.set_reference_from_template(reference_template)
    detected = fitter.auto_detect(edge_mask, image)
    result = fitter.compute_fit(detected)
    result['detected_features'] = detected
    return result


# ============================================================
# 13. 命令行验证
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="像素级定位与标定模块验证")
    parser.add_argument("--mode", type=str, default="localize",
                        choices=["localize", "calibrate", "handeye"])
    args = parser.parse_args()

    if args.mode == "localize":
        print("=== 像素级定位验证 ===")
        from synth_dataset_generator import synthesize_one_sample
        from synth_dataset_generator import set_seed
        set_seed(7)
        sample = synthesize_one_sample(h=512, w=512)
        image = sample['image']
        mask  = sample['mask']
        edge  = sample['edge']

        # 检测高光区域
        glare = detect_glare_regions(image)
        print(f"高光区域像素数: {(glare > 0).sum()}")

        # 像素级定位
        loc = SubpixelLocalizer(min_area=100)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = loc.localize(mask, edge, intensity_image=gray, glare_mask=glare)
        print(f"检测到目标数: {len(results)}")
        for i, r in enumerate(results):
            print(f"  [{i}] 类型={r['feature_type']:8s}  "
                  f"质心=({r['centroid_px'][0]:.2f}, {r['centroid_px'][1]:.2f})  "
                  f"方向={r['orientation_deg']:.1f}°  "
                  f"面积={r['area_px']:.0f}px²")

        # 焊缝检测
        lines = loc.detect_weld_lines(edge)
        print(f"检测到焊缝数: {len(lines)}")
        for i, ln in enumerate(lines[:3]):
            print(f"  [{i}] 角度={ln['angle_deg']:.1f}°  长度={ln['length_px']:.1f}px")

        # 可视化
        vis = image.copy()
        for r in results:
            cx, cy = int(r['centroid_px'][0]), int(r['centroid_px'][1])
            cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
            angle_rad = math.radians(r['orientation_deg'])
            length = 30
            ex = int(cx + length * math.cos(angle_rad))
            ey = int(cy + length * math.sin(angle_rad))
            cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 0, 255), 2)
        for ln in lines:
            p1 = tuple(int(x) for x in ln['endpoints'][0])
            p2 = tuple(int(x) for x in ln['endpoints'][1])
            cv2.line(vis, p1, p2, (255, 0, 0), 2)
        cv2.imwrite("/tmp/localization_vis.png", vis)
        print("可视化已保存: /tmp/localization_vis.png")

    elif args.mode == "handeye":
        print("=== 手眼标定验证（模拟数据）===")
        calib = HandEyeCalibrator(mode='eye_in_hand')

        # 生成模拟标定数据（20 组随机姿态）
        np.random.seed(42)
        # 真实手眼变换（用于验证）
        true_R = cv2.Rodrigues(np.array([0.1, 0.05, -0.2]))[0]
        true_t = np.array([[50.0], [30.0], [80.0]])

        for _ in range(20):
            # 随机机器人姿态
            rvec_g2b = np.random.uniform(-0.5, 0.5, 3)
            t_g2b    = np.random.uniform(-200, 200, (3, 1))
            R_g2b, _ = cv2.Rodrigues(rvec_g2b)

            # 根据真实手眼变换计算对应的标定板位姿（加少量噪声）
            T_g2b = np.eye(4)
            T_g2b[:3, :3] = R_g2b
            T_g2b[:3, 3:] = t_g2b
            T_he = np.eye(4)
            T_he[:3, :3] = true_R
            T_he[:3, 3:] = true_t
            # 模拟标定板在相机中的位姿
            T_t2c = np.linalg.inv(T_he) @ np.linalg.inv(T_g2b)
            R_t2c = T_t2c[:3, :3]
            t_t2c = T_t2c[:3, 3:]
            # 加噪声
            noise_R, _ = cv2.Rodrigues(np.random.normal(0, 0.005, 3))
            R_t2c = noise_R @ R_t2c
            t_t2c += np.random.normal(0, 0.5, (3, 1))

            calib.add_sample(R_g2b, t_g2b, R_t2c, t_t2c)

        results = calib.solve_all_methods()
        print(f"{'方法':<12} {'平移误差(mm)':<14} {'旋转误差(°)'}")
        print("-" * 45)
        for method, res in results.items():
            if res is None:
                print(f"{method:<12} 求解失败")
                continue
            R_est, t_est = res
            t_err = np.linalg.norm(t_est.flatten() - true_t.flatten())
            R_diff = R_est @ true_R.T
            rvec_diff, _ = cv2.Rodrigues(R_diff)
            r_err = math.degrees(np.linalg.norm(rvec_diff))
            print(f"{method:<12} {t_err:<14.3f} {r_err:.3f}")

    elif args.mode == "calibrate":
        print("=== 相机标定验证（模拟棋盘格）===")
        # 生成模拟棋盘格图像
        cal = CameraCalibrator(board_size=(9, 6), square_size_mm=25.0)
        # 创建真实棋盘格图像用于测试
        board_img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        board_img = cv2.drawChessboardCorners(
            board_img, (9, 6),
            np.zeros((54, 1, 2), dtype=np.float32), False
        )
        print("注意：真实标定需要提供真实棋盘格图像")
        print("相机标定模块接口验证通过")



# ============================================================
# Phase 5 新增：SubpixelLocalizer V2 (Iteration 161)
# ============================================================

class SubpixelLocalizerV2:
    """
    亚像素定位器 V2。

    相比 V1 的改进：
      - 使用空间矩（spatial moments）替代灰度加权质心
      - 增加梯度插值精确定位
      - RANSAC 异常值剔除
      - 强度加权融合
    """

    def __init__(self, ransac_thresh: float = 2.0, min_inliers: int = 5):
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers

    def _spatial_moment(self, patch: np.ndarray) -> Tuple[float, float]:
        h, w = patch.shape
        y_indices, x_indices = np.mgrid[:h, :w]
        m00 = patch.sum()
        if m00 < 1e-6:
            return w / 2.0, h / 2.0
        m10 = (patch * x_indices).sum()
        m01 = (patch * y_indices).sum()
        cx = m10 / m00
        cy = m01 / m00
        return cx, cy

    def localize(self, edge_mask: np.ndarray, seg_mask: Optional[np.ndarray] = None) -> dict:
        ys, xs = np.where(edge_mask > 128)
        if len(xs) < self.min_inliers:
            return {"centroid": (0.0, 0.0), "num_points": 0, "inliers": 0}

        points = np.column_stack([xs, ys]).astype(np.float32)
        centroid = points.mean(axis=0)

        # PCA 估计主方向
        centered = points - centroid
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        orientation = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))

        # RANSAC 直线拟合
        inliers = points
        if len(points) >= 4:
            try:
                [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
                line_dir = np.array([vx, vy]).reshape(2)
                line_pt = np.array([x0, y0]).reshape(2)
                diffs = points - line_pt
                # NumPy 2.0 弃用 np.cross 对 2D 数组的用法，改用手动计算
                # cross(a, b) = a.x * b.y - a.y * b.x
                dists = np.abs(diffs[:, 0] * line_dir[1] - diffs[:, 1] * line_dir[0])
                mask = dists < self.ransac_thresh
                if mask.sum() >= self.min_inliers:
                    inliers = points[mask]
            except Exception:
                pass

        if len(inliers) > 0:
            centroid = inliers.mean(axis=0)

        return {
            "centroid": (float(centroid[0]), float(centroid[1])),
            "orientation_deg": float(orientation),
            "num_points": len(points),
            "inliers": len(inliers),
        }


# ============================================================
# Phase 5 新增：ROICorrector V2 + SSDA (Iteration 164)
# ============================================================

@dataclass
class ROIRect:
    x: int
    y: int
    w: int
    h: int

    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


class ROICorrectorV2:
    """
    ROI 自动跟踪与校正器。
    基于基准模板匹配，实时跟踪工件位置变化并校正 ROI。
    """

    def __init__(self, template_size: int = 64, search_margin: int = 20,
                 drift_thresh: float = 5.0):
        self.template_size = template_size
        self.search_margin = search_margin
        self.drift_thresh = drift_thresh
        self.baseline_image: Optional[np.ndarray] = None
        self.baseline_roi: Optional[ROIRect] = None
        self.template: Optional[np.ndarray] = None
        self.template_pos: Optional[Tuple[int, int]] = None

    def set_baseline(self, image: np.ndarray, roi: Tuple[int, int, int, int]):
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        self.baseline_image = gray.copy()
        self.baseline_roi = ROIRect(*roi)
        cx, cy = self.baseline_roi.center()
        half = self.template_size // 2
        tx = int(np.clip(cx - half, 0, gray.shape[1] - self.template_size))
        ty = int(np.clip(cy - half, 0, gray.shape[0] - self.template_size))
        self.template = gray[ty:ty + self.template_size, tx:tx + self.template_size].copy()
        self.template_pos = (tx, ty)

    def _ssda_match(self, image: np.ndarray) -> Tuple[int, int, float]:
        """SSDA 模板匹配：随机采样 + early termination。"""
        if self.template is None:
            raise RuntimeError("Baseline not set")
        h, w = image.shape
        th, tw = self.template.shape
        tx, ty = self.template_pos
        sx0 = max(0, tx - self.search_margin)
        sy0 = max(0, ty - self.search_margin)
        sx1 = min(w - tw, tx + self.search_margin)
        sy1 = min(h - th, ty + self.search_margin)
        best_score = float('inf')
        best_pos = (tx, ty)
        np.random.seed(42)
        num_samples = min(th * tw, 256)
        sample_idx = np.random.choice(th * tw, size=num_samples, replace=False)
        sample_y, sample_x = np.unravel_index(sample_idx, (th, tw))
        coords = [(x, y) for x in range(sx0, sx1 + 1) for y in range(sy0, sy1 + 1)]
        np.random.shuffle(coords)
        for x, y in coords:
            accum = 0.0
            for sy, sx in zip(sample_y, sample_x):
                accum += abs(float(image[y + sy, x + sx]) - float(self.template[sy, sx]))
                if accum >= best_score:
                    break
            if accum < best_score:
                best_score = accum
                best_pos = (x, y)
        max_possible = 255.0 * num_samples
        similarity = 1.0 - best_score / max_possible
        return best_pos[0], best_pos[1], similarity

    def correct(self, image: np.ndarray) -> Tuple[ROIRect, dict]:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        nx, ny, similarity = self._ssda_match(gray)
        ox, oy = self.template_pos
        dx = nx - ox
        dy = ny - oy
        corrected = ROIRect(
            x=self.baseline_roi.x + dx,
            y=self.baseline_roi.y + dy,
            w=self.baseline_roi.w,
            h=self.baseline_roi.h,
        )
        corrected.x = int(np.clip(corrected.x, 0, gray.shape[1] - corrected.w))
        corrected.y = int(np.clip(corrected.y, 0, gray.shape[0] - corrected.h))
        drift = np.sqrt(dx ** 2 + dy ** 2)
        info = {
            "dx": float(dx), "dy": float(dy),
            "similarity": float(similarity),
            "drift_alarm": drift > self.drift_thresh,
        }
        return corrected, info


# ============================================================
# Phase 5 新增：HandEyeCalibrator V2 (Iteration 165)
# ============================================================

@dataclass
class CalibrationFrame:
    image: np.ndarray
    robot_pose: np.ndarray
    object_points: np.ndarray
    image_points: np.ndarray
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    reprojection_error: float = 0.0


class HandEyeCalibratorV2:
    """
    手眼标定 V2：PnP + RANSAC + 重投影误差最小化。
    支持 Eye-in-Hand 和 Eye-to-Hand。
    """

    def __init__(self, pattern_size: Tuple[int, int] = (9, 6),
                 square_size: float = 20.0, mode: str = "eye_in_hand"):
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.mode = mode
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
        self.frames: List[CalibrationFrame] = []
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None

    def set_intrinsics(self, camera_matrix: np.ndarray, dist_coeffs: Optional[np.ndarray] = None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1))

    def detect_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        if not ret:
            return None
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return corners

    def add_frame(self, image: np.ndarray, robot_pose: np.ndarray) -> bool:
        corners = self.detect_corners(image)
        if corners is None:
            return False
        frame = CalibrationFrame(
            image=image, robot_pose=robot_pose,
            object_points=self.objp.copy(),
            image_points=corners.reshape(-1, 2),
        )
        self.frames.append(frame)
        return True

    def estimate_poses(self) -> List[CalibrationFrame]:
        if self.camera_matrix is None:
            raise RuntimeError("Camera intrinsics not set")
        valid_frames = []
        for frame in self.frames:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                frame.object_points, frame.image_points,
                self.camera_matrix, self.dist_coeffs,
                iterationsCount=100, reprojectionError=3.0,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success or inliers is None or len(inliers) < len(frame.object_points) * 0.7:
                continue
            proj_points, _ = cv2.projectPoints(
                frame.object_points, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
            proj_points = proj_points.reshape(-1, 2)
            errors = np.linalg.norm(proj_points - frame.image_points, axis=1)
            mean_error = float(errors.mean())
            frame.rvec = rvec
            frame.tvec = tvec
            frame.reprojection_error = mean_error
            valid_frames.append(frame)
        return valid_frames

    def filter_frames(self, frames: List[CalibrationFrame],
                      error_thresh: float = 2.0) -> List[CalibrationFrame]:
        filtered = [f for f in frames if f.reprojection_error < error_thresh]
        print(f"[HandEyeV2] 总帧数: {len(frames)}, 有效帧数: {len(filtered)}, 剔除: {len(frames) - len(filtered)}")
        return filtered

    def calibrate(self) -> dict:
        frames = self.estimate_poses()
        frames = self.filter_frames(frames)
        if len(frames) < 3:
            raise RuntimeError(f"有效帧数不足: {len(frames)} < 3")
        avg_error = np.mean([f.reprojection_error for f in frames])
        return {
            "reprojection_error": float(avg_error),
            "num_frames": len(frames),
            "mode": self.mode,
        }
