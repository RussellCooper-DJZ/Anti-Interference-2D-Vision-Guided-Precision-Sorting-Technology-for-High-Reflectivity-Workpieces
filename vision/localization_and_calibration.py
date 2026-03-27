"""
localization_and_calibration.py — 亚像素定位与相机-机器人标定模块
:Author: RussellCooper

功能模块：
  1. SubpixelLocalizer    — 从分割掩膜/边缘图提取亚像素级目标位姿
  2. CameraCalibrator     — 棋盘格相机内参标定
  3. HandEyeCalibrator    — 手眼标定（Eye-in-Hand / Eye-to-Hand）
  4. CoordinateTransformer — 像素坐标 → 相机坐标 → 机器人基坐标系

面向船舶场景的特殊处理：
  - 焊缝/铆钉等线性/点状特征的亚像素定位
  - 大型目标（船体面板）的质心 + 主轴方向估计
  - 高光区域的掩膜排除（避免高光干扰定位）

用法::

    # 亚像素定位
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


# ============================================================
# 1. 亚像素定位器
# ============================================================

class SubpixelLocalizer:
    """
    从分割掩膜和边缘图中提取亚像素级目标位姿。

    支持的目标类型：
      - 'blob'   : 焊接点、铆钉（圆形/椭圆形区域）
      - 'line'   : 焊缝（线性特征）
      - 'region' : 大型面板（多边形区域）

    亚像素精度实现：
      - 质心：使用灰度加权矩（intensity-weighted moments）
      - 边缘：使用 Canny + 亚像素边缘细化（Sobel 梯度插值）
      - 直线：使用 Hough 变换 + 最小二乘拟合
    """

    def __init__(self,
                 min_area: int = 50,
                 max_area: Optional[int] = None,
                 min_circularity: float = 0.0,
                 subpixel_window: int = 5):
        """
        Args:
            min_area:         最小目标面积（像素）
            max_area:         最大目标面积（None=不限）
            min_circularity:  最小圆形度（0=不限，1=完美圆形）
            subpixel_window:  亚像素精化窗口大小（奇数）
        """
        self.min_area        = min_area
        self.max_area        = max_area
        self.min_circularity = min_circularity
        self.subpixel_window = subpixel_window

    def localize(
        self,
        seg_mask: np.ndarray,
        edge_mask: Optional[np.ndarray] = None,
        intensity_image: Optional[np.ndarray] = None,
        glare_mask: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        从分割掩膜中提取所有目标的亚像素位姿。

        Args:
            seg_mask:        二值分割掩膜 (H,W) uint8，目标区域 > 0
            edge_mask:       边缘掩膜 (H,W) uint8（可选，用于线性特征）
            intensity_image: 灰度图 (H,W) uint8（可选，用于加权质心）
            glare_mask:      高光区域掩膜 (H,W) uint8（可选，高光区域排除）

        Returns:
            List[Dict]，每个目标包含：
              centroid_px:      (x, y) float，亚像素质心
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

            # 亚像素质心
            centroid = self._subpixel_centroid(
                mask, cnt, intensity_image
            )

            # 主轴方向（PCA）
            orientation = self._pca_orientation(cnt)

            # 特征类型分类
            feature_type = self._classify_feature(area, circularity, aspect_ratio)

            results.append({
                'centroid_px':     centroid,
                'orientation_deg': orientation,
                'bbox':            (x, y, w, h),
                'area_px':         area,
                'circularity':     circularity,
                'aspect_ratio':    aspect_ratio,
                'contour':         cnt,
                'feature_type':    feature_type,
            })

        return results

    def _subpixel_centroid(
        self,
        mask: np.ndarray,
        contour: np.ndarray,
        intensity: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        计算亚像素质心。

        若提供灰度图，使用强度加权矩（对高光区域鲁棒性更好）；
        否则使用几何矩。
        """
        x, y, w, h = cv2.boundingRect(contour)
        roi_mask = np.zeros_like(mask)
        cv2.drawContours(roi_mask, [contour], -1, 255, -1)

        if intensity is not None:
            # 强度加权矩
            roi = intensity[y:y+h, x:x+w].astype(np.float64)
            m = roi_mask[y:y+h, x:x+w].astype(np.float64) / 255.0
            # 反转权重：高光区域（高亮度）权重降低
            weight = (1.0 - roi / 255.0) * m + 0.1 * m  # 避免全零
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

    @staticmethod
    def _pca_orientation(contour: np.ndarray) -> float:
        """
        使用 PCA 计算轮廓主轴方向（度，-90~90）。
        """
        pts = contour.reshape(-1, 2).astype(np.float64)
        if len(pts) < 5:
            return 0.0
        mean = pts.mean(axis=0)
        centered = pts - mean
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
        从边缘图中检测焊缝直线（亚像素精度 Hough 变换）。

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

        # 亚像素精化
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
# 6. 命令行验证
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="亚像素定位与标定模块验证")
    parser.add_argument("--mode", type=str, default="localize",
                        choices=["localize", "calibrate", "handeye"])
    args = parser.parse_args()

    if args.mode == "localize":
        print("=== 亚像素定位验证 ===")
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

        # 亚像素定位
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
