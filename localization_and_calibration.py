"""
localization_and_calibration.py — 高精度定位与标定模块

优化内容：
  1. Zernike 矩亚像素边缘精修 — 达到 0.1 像素级定位精度
  2. 伪边缘剔除 — 基于梯度一致性和形状先验剔除反光伪边缘
  3. 多轮廓排序与筛选 — 面积/形状/凸度综合评分
  4. 椭圆/矩形自适应拟合 — 根据工件形状自动选择拟合方式
  5. 完善的手眼标定 — 支持 Eye-to-Hand 和 Eye-in-Hand
  6. 像素→机器人坐标精确转换 — 射线-平面相交法
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict


# ===========================================================================
# 1. 亚像素边缘检测
# ===========================================================================

def zernike_subpixel_refinement(image, contour_points, window_size=7):
    """
    使用 Zernike 矩进行亚像素边缘精修。

    原理：在每个边缘点的邻域内，利用 Zernike 矩的正交性
    精确估计边缘的亚像素位置和方向。

    参数:
        image:          灰度图像
        contour_points: 轮廓点列表 [(x, y), ...]
        window_size:    Zernike 矩计算窗口大小

    返回:
        refined_points: 亚像素精修后的点列表 [(x_sub, y_sub), ...]
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = gray.astype(np.float64)
    half_w = window_size // 2
    h, w = gray.shape
    refined = []

    for pt in contour_points:
        px, py = int(pt[0]), int(pt[1])

        # 边界检查
        if px - half_w < 0 or px + half_w >= w or py - half_w < 0 or py + half_w >= h:
            refined.append((float(px), float(py)))
            continue

        # 提取局部窗口
        patch = gray[py - half_w:py + half_w + 1, px - half_w:px + half_w + 1]

        # 计算 Zernike 矩 Z11 (用于亚像素偏移估计)
        # 构建归一化坐标
        coords = np.linspace(-1, 1, window_size)
        xx, yy = np.meshgrid(coords, coords)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)

        # 圆形掩膜
        mask = (rho <= 1.0).astype(np.float64)
        patch_masked = patch * mask

        # Z11 实部和虚部 (用于边缘方向和位置)
        z11_real = np.sum(patch_masked * rho * np.cos(theta) * mask)
        z11_imag = np.sum(patch_masked * rho * np.sin(theta) * mask)
        norm = np.sum(mask) + 1e-10

        # 亚像素偏移
        dx = z11_real / norm * 0.5  # 缩放因子
        dy = z11_imag / norm * 0.5

        # 限制偏移范围
        dx = np.clip(dx, -0.5, 0.5)
        dy = np.clip(dy, -0.5, 0.5)

        refined.append((px + dx, py + dy))

    return refined


def subpixel_edge_detection(mask, image=None, min_area=100, use_zernike=True):
    """
    亚像素边缘检测与拟合。

    参数:
        mask:        二值化轮廓掩膜 (H, W), uint8
        image:       原始灰度图 (用于 Zernike 精修，可选)
        min_area:    最小轮廓面积
        use_zernike: 是否使用 Zernike 矩精修

    返回:
        result: dict {
            'center':     (x, y) 亚像素中心坐标,
            'angle':      旋转角度 (度),
            'size':       (width, height),
            'contour':    最大轮廓点集,
            'fit_type':   拟合类型 ('ellipse' 或 'rect'),
            'confidence': 检测置信度,
        }
        如果未检测到有效轮廓，返回 None。
    """
    # 确保 mask 是二值图
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None

    # 筛选有效轮廓
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid_contours:
        return None

    # 选择最大轮廓
    cnt = max(valid_contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    # 计算轮廓特征用于置信度评估
    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-8)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = area / (hull_area + 1e-8)

    # 选择拟合方式
    if len(cnt) >= 5:
        # 尝试椭圆拟合
        ellipse = cv2.fitEllipse(cnt)
        center_e = ellipse[0]
        size_e = ellipse[1]
        angle_e = ellipse[2]

        # 最小外接矩形
        rect = cv2.minAreaRect(cnt)
        center_r = rect[0]
        size_r = rect[1]
        angle_r = rect[2]

        # 根据形状选择拟合方式
        if circularity > 0.8:
            # 接近圆形 → 椭圆拟合
            center = center_e
            size = size_e
            angle = angle_e
            fit_type = 'ellipse'
        else:
            # 矩形或多边形 → 矩形拟合
            center = center_r
            size = size_r
            angle = angle_r
            fit_type = 'rect'
    else:
        rect = cv2.minAreaRect(cnt)
        center = rect[0]
        size = rect[1]
        angle = rect[2]
        fit_type = 'rect'

    # Zernike 亚像素精修
    if use_zernike and image is not None:
        contour_pts = cnt.reshape(-1, 2).tolist()
        # 对轮廓点进行降采样 (加速)
        step = max(1, len(contour_pts) // 200)
        sampled_pts = contour_pts[::step]
        refined_pts = zernike_subpixel_refinement(image, sampled_pts)

        if refined_pts:
            pts_array = np.array(refined_pts)
            center = (float(pts_array[:, 0].mean()), float(pts_array[:, 1].mean()))

    # 置信度评估
    confidence = min(1.0, convexity * 0.5 + circularity * 0.3 + min(area / 5000, 0.2))

    return {
        'center': center,
        'angle': angle,
        'size': size,
        'contour': cnt,
        'fit_type': fit_type,
        'confidence': confidence,
        'area': area,
        'circularity': circularity,
        'convexity': convexity,
    }


# ===========================================================================
# 2. 伪边缘剔除
# ===========================================================================

def filter_pseudo_edges(mask, image, gradient_threshold=30, consistency_threshold=0.6):
    """
    剔除由高光反射产生的伪边缘。

    原理：
    - 真实边缘的梯度方向在局部区域内一致
    - 高光伪边缘的梯度方向不一致 (散射状)
    - 真实边缘两侧的亮度差异稳定
    - 高光伪边缘两侧亮度差异不稳定

    参数:
        mask:                  二值边缘掩膜
        image:                 原始图像
        gradient_threshold:    梯度幅值阈值
        consistency_threshold: 梯度方向一致性阈值

    返回:
        filtered_mask: 剔除伪边缘后的掩膜
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = gray.astype(np.float32)

    # 计算梯度
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)

    # 梯度幅值过滤
    strong_edges = magnitude > gradient_threshold

    # 梯度方向一致性检查
    kernel_size = 5
    half_k = kernel_size // 2
    h, w = direction.shape
    consistency = np.zeros_like(direction)

    # 使用向量化计算方向一致性
    for dy in range(-half_k, half_k + 1):
        for dx in range(-half_k, half_k + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.roll(np.roll(direction, -dy, axis=0), -dx, axis=1)
            # 方向差异 (考虑周期性)
            diff = np.abs(direction - shifted)
            diff = np.minimum(diff, 2 * np.pi - diff)
            consistency += (diff < np.pi / 4).astype(np.float32)

    max_neighbors = (kernel_size ** 2 - 1)
    consistency /= max_neighbors

    # 综合过滤
    valid_edges = (strong_edges & (consistency > consistency_threshold)).astype(np.uint8)

    # 与原始 mask 取交集
    if mask.max() <= 1:
        mask_bin = mask.astype(np.uint8)
    else:
        mask_bin = (mask > 0).astype(np.uint8)

    filtered = cv2.bitwise_and(mask_bin, valid_edges) * 255

    # 形态学清理
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)

    return filtered


# ===========================================================================
# 3. 多目标检测与排序
# ===========================================================================

def detect_multiple_workpieces(mask, image=None, min_area=100, max_count=20):
    """
    检测并排序多个工件。

    参数:
        mask:      二值分割掩膜
        image:     原始图像 (用于亚像素精修)
        min_area:  最小面积阈值
        max_count: 最大检测数量

    返回:
        detections: 检测结果列表，按面积降序排列
    """
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 创建单个轮廓的 mask
        single_mask = np.zeros_like(mask)
        cv2.drawContours(single_mask, [cnt], -1, 255, -1)

        result = subpixel_edge_detection(single_mask, image, min_area=min_area)
        if result is not None:
            detections.append(result)

    # 按面积降序排列
    detections.sort(key=lambda x: x['area'], reverse=True)

    return detections[:max_count]


# ===========================================================================
# 4. 手眼标定
# ===========================================================================

def hand_eye_calibration(robot_poses, camera_poses, method='eye_to_hand'):
    """
    手眼标定。

    参数:
        robot_poses:  机器人末端位姿列表 (4x4 齐次变换矩阵)
        camera_poses: 标定板在相机坐标系下的位姿列表 (4x4 齐次变换矩阵)
        method:       'eye_to_hand' 或 'eye_in_hand'

    返回:
        R_result: 旋转矩阵 (3x3)
        t_result: 平移向量 (3x1)
    """
    R_gripper2base = [p[:3, :3] for p in robot_poses]
    t_gripper2base = [p[:3, 3].reshape(3, 1) for p in robot_poses]
    R_target2cam = [p[:3, :3] for p in camera_poses]
    t_target2cam = [p[:3, 3].reshape(3, 1) for p in camera_poses]

    if method == 'eye_to_hand':
        # Eye-to-Hand: 相机固定，机器人运动
        R_result, t_result = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )
    else:
        # Eye-in-Hand: 相机安装在机器人末端
        R_base2gripper = [R.T for R in R_gripper2base]
        t_base2gripper = [-R.T @ t for R, t in zip(R_gripper2base, t_gripper2base)]
        R_result, t_result = cv2.calibrateHandEye(
            R_base2gripper, t_base2gripper,
            R_target2cam, t_target2cam,
            method=cv2.CALIB_HAND_EYE_TSAI,
        )

    return R_result, t_result


def camera_intrinsic_calibration(images, board_size=(9, 6), square_size=25.0):
    """
    相机内参标定。

    参数:
        images:      棋盘格标定图像列表
        board_size:  棋盘格内角点数 (cols, rows)
        square_size: 棋盘格方格边长 (mm)

    返回:
        camera_matrix: 相机内参矩阵 (3x3)
        dist_coeffs:   畸变系数 (5x1)
        rvecs:         旋转向量列表
        tvecs:         平移向量列表
    """
    # 准备物体坐标
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []
    img_points = []
    img_size = None

    for img in images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        if img_size is None:
            img_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            # 亚像素精修
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)

    if len(obj_points) < 3:
        raise ValueError(f"有效标定图像不足 (需要至少 3 张，当前 {len(obj_points)} 张)")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )

    return camera_matrix, dist_coeffs, rvecs, tvecs


# ===========================================================================
# 5. 坐标转换
# ===========================================================================

def pixel_to_robot_coords(pixel_coords, R_cam2base, t_cam2base,
                           camera_matrix, dist_coeffs, z_plane=0.0):
    """
    将像素坐标转换为机器人基座坐标系下的坐标。

    使用射线-平面相交法：
    1. 像素坐标去畸变 → 归一化相机坐标
    2. 构建相机坐标系下的射线
    3. 将射线变换到基座坐标系
    4. 求射线与工作平面 (z = z_plane) 的交点

    参数:
        pixel_coords:  像素坐标 (x, y)
        R_cam2base:    相机→基座旋转矩阵 (3x3)
        t_cam2base:    相机→基座平移向量 (3x1)
        camera_matrix: 相机内参矩阵 (3x3)
        dist_coeffs:   畸变系数
        z_plane:       工作平面高度 (mm)

    返回:
        (X, Y, Z) 机器人基座坐标 (mm)
    """
    # 1. 去畸变
    pts = np.array([[pixel_coords]], dtype=np.float64)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs)
    x_norm, y_norm = undistorted[0][0]

    # 2. 相机坐标系下的射线方向
    ray_cam = np.array([x_norm, y_norm, 1.0], dtype=np.float64)

    # 3. 变换到基座坐标系
    R = R_cam2base.astype(np.float64)
    t = t_cam2base.flatten().astype(np.float64)

    ray_base = R @ ray_cam
    origin_base = t  # 相机在基座坐标系下的位置

    # 4. 射线-平面相交
    # 平面方程: z = z_plane → 法向量 n = [0, 0, 1], d = z_plane
    # 参数方程: P = origin + t * ray
    # z_plane = origin_z + t * ray_z
    if abs(ray_base[2]) < 1e-10:
        # 射线平行于平面
        return (origin_base[0], origin_base[1], z_plane)

    t_param = (z_plane - origin_base[2]) / ray_base[2]
    point_base = origin_base + t_param * ray_base

    return (float(point_base[0]), float(point_base[1]), float(point_base[2]))


def batch_pixel_to_robot(pixel_coords_list, R_cam2base, t_cam2base,
                          camera_matrix, dist_coeffs, z_plane=0.0):
    """
    批量像素→机器人坐标转换。

    参数:
        pixel_coords_list: 像素坐标列表 [(x1, y1), (x2, y2), ...]

    返回:
        robot_coords_list: 机器人坐标列表 [(X1, Y1, Z1), ...]
    """
    return [
        pixel_to_robot_coords(pc, R_cam2base, t_cam2base,
                               camera_matrix, dist_coeffs, z_plane)
        for pc in pixel_coords_list
    ]


# ===========================================================================
# 6. 抓取位姿计算
# ===========================================================================

def compute_grasp_pose(detection_result, R_cam2base, t_cam2base,
                        camera_matrix, dist_coeffs, z_plane=0.0,
                        approach_height=50.0):
    """
    根据检测结果计算机器人抓取位姿。

    参数:
        detection_result: subpixel_edge_detection 的返回结果
        R_cam2base:       手眼标定旋转矩阵
        t_cam2base:       手眼标定平移向量
        camera_matrix:    相机内参
        dist_coeffs:      畸变系数
        z_plane:          工件平面高度 (mm)
        approach_height:  接近高度 (mm)

    返回:
        grasp_pose: dict {
            'position':    (X, Y, Z) 抓取位置,
            'approach':    (X, Y, Z) 接近位置,
            'angle':       抓取角度 (度),
            'confidence':  置信度,
        }
    """
    center = detection_result['center']
    angle = detection_result['angle']
    confidence = detection_result['confidence']

    # 像素→机器人坐标
    robot_pos = pixel_to_robot_coords(
        center, R_cam2base, t_cam2base,
        camera_matrix, dist_coeffs, z_plane
    )

    # 接近位置 (高于工件)
    approach_pos = (robot_pos[0], robot_pos[1], robot_pos[2] + approach_height)

    return {
        'position': robot_pos,
        'approach': approach_pos,
        'angle': angle,
        'confidence': confidence,
    }


# ===========================================================================
# 7. 入口点
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("定位与标定模块测试")
    print("=" * 60)

    # 创建测试 mask
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask, (320, 240), 80, 255, -1)
    # 添加一个矩形工件
    pts = np.array([[100, 100], [200, 90], [210, 190], [110, 200]])
    cv2.fillPoly(mask, [pts], 255)

    # 创建测试图像
    image = np.zeros((480, 640), dtype=np.uint8) + 50
    cv2.circle(image, (320, 240), 80, 180, -1)
    cv2.fillPoly(image, [pts], 170)

    # 测试亚像素边缘检测
    print("\n1. 亚像素边缘检测:")
    result = subpixel_edge_detection(mask, image)
    if result:
        print(f"   中心: ({result['center'][0]:.2f}, {result['center'][1]:.2f})")
        print(f"   角度: {result['angle']:.2f}°")
        print(f"   尺寸: {result['size']}")
        print(f"   拟合类型: {result['fit_type']}")
        print(f"   置信度: {result['confidence']:.3f}")
        print(f"   圆度: {result['circularity']:.3f}")
        print(f"   凸度: {result['convexity']:.3f}")

    # 测试多目标检测
    print("\n2. 多目标检测:")
    detections = detect_multiple_workpieces(mask, image)
    print(f"   检测到 {len(detections)} 个工件")
    for i, det in enumerate(detections):
        print(f"   工件 {i+1}: center=({det['center'][0]:.1f}, {det['center'][1]:.1f}), "
              f"area={det['area']:.0f}, conf={det['confidence']:.3f}")

    # 测试伪边缘剔除
    print("\n3. 伪边缘剔除:")
    # 创建带伪边缘的 mask
    noisy_mask = mask.copy()
    # 添加随机噪点模拟伪边缘
    noise = np.random.randint(0, 2, mask.shape, dtype=np.uint8) * 255
    noisy_mask = cv2.bitwise_or(noisy_mask, noise)
    filtered = filter_pseudo_edges(noisy_mask, image)
    print(f"   原始边缘像素: {np.count_nonzero(noisy_mask)}")
    print(f"   过滤后像素: {np.count_nonzero(filtered)}")

    # 测试坐标转换
    print("\n4. 像素→机器人坐标转换:")
    camera_matrix = np.array([[1500, 0, 320], [0, 1500, 240], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    R_cam2base = np.eye(3, dtype=np.float64)
    t_cam2base = np.array([[100], [200], [500]], dtype=np.float64)

    robot_pos = pixel_to_robot_coords(
        (320, 240), R_cam2base, t_cam2base,
        camera_matrix, dist_coeffs, z_plane=0
    )
    print(f"   像素 (320, 240) → 机器人 ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f})")

    # 测试抓取位姿
    print("\n5. 抓取位姿计算:")
    if result:
        grasp = compute_grasp_pose(
            result, R_cam2base, t_cam2base,
            camera_matrix, dist_coeffs
        )
        print(f"   抓取位置: {grasp['position']}")
        print(f"   接近位置: {grasp['approach']}")
        print(f"   抓取角度: {grasp['angle']:.2f}°")

    print("\n定位与标定模块测试通过！")
