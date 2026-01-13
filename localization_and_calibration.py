import cv2
import numpy as np

def subpixel_edge_detection(mask):
    """
    亚像素边缘检测与拟合。
    输入：二值化的轮廓掩膜。
    输出：拟合后的中心点 (x, y) 和角度 theta。

    Sub-pixel edge detection and fitting.
    Input: Binarized contour mask.
    Output: Fitted center point (x, y) and angle theta.

    Subpixel-Kantenerkennung und -Anpassung.
    Eingabe: Binarisierte Konturmaske.
    Ausgabe: Angepasster Mittelpunkt (x, y) und Winkel Theta.
    """
    # 1. 提取轮廓
    # 1. Extract contours
    # 1. Konturen extrahieren
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    
    # 假设最大的轮廓是目标工件
    # Assume the largest contour is the target workpiece
    # Angenommen, die größte Kontur ist das Zielwerkstück
    cnt = max(contours, key=cv2.contourArea)
    
    # 2. 亚像素精修 (使用 Zernike 矩或简单的重心法，这里展示最小外接矩形拟合)
    # 对于更高精度，可以对轮廓点进行多项式拟合
    # 2. Sub-pixel refinement (using Zernike moments or simple centroid method, here showing minimum bounding rectangle fitting)
    # For higher precision, polynomial fitting can be applied to contour points.
    # 2. Subpixel-Verfeinerung (unter Verwendung von Zernike-Momenten oder einfacher Schwerpunktmethode, hier wird die Anpassung des minimalen umschließenden Rechtecks gezeigt)
    # Für höhere Präzision kann eine Polynomanpassung auf Konturpunkte angewendet werden.
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    center = rect[0]  # (x, y)
    size = rect[1]    # (width, height)
    angle = rect[2]   # angle
    
    return center, angle

def hand_eye_calibration(robot_poses, charuco_poses):
    """
    手眼标定 (Eye-to-Hand 示例)。
    robot_poses: 机器人末端在基座坐标系下的位姿 (List of 4x4 matrices)
    charuco_poses: 标定板在相机坐标系下的位姿 (List of 4x4 matrices)

    Hand-eye calibration (Eye-to-Hand example).
    robot_poses: Poses of the robot end-effector in the base coordinate system (List of 4x4 matrices)
    charuco_poses: Poses of the calibration board in the camera coordinate system (List of 4x4 matrices)

    Hand-Auge-Kalibrierung (Eye-to-Hand Beispiel).
    robot_poses: Posen des Roboterendeffektors im Basiskoordinatensystem (Liste von 4x4 Matrizen)
    charuco_poses: Posen des Kalibrierbretts im Kamerakoordinatensystem (Liste von 4x4 Matrizen)
    """
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    
    for pose in robot_poses:
        R_gripper2base.append(pose[:3, :3])
        t_gripper2base.append(pose[:3, 3])
        
    for pose in charuco_poses:
        R_target2cam.append(pose[:3, :3])
        t_target2cam.append(pose[:3, 3])
        
    # 使用 Tsai-Lenz 算法进行标定
    # Perform calibration using Tsai-Lenz algorithm
    # Führt die Kalibrierung mit dem Tsai-Lenz-Algorithmus durch
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )
    
    return R_cam2base, t_cam2base

def pixel_to_robot_coords(pixel_coords, R_cam2base, t_cam2base, camera_matrix, dist_coeffs, z_plane=0):
    """
    将像素坐标转换为机器人基座坐标系下的 2D 坐标。
    假设工件位于已知的平面 z_plane 上。

    Converts pixel coordinates to 2D coordinates in the robot base coordinate system.
    Assumes the workpiece is located on a known plane z_plane.

    Konvertiert Pixelkoordinaten in 2D-Koordinaten im Roboterbasiskoordinatensystem.
    Geht davon aus, dass sich das Werkstück auf einer bekannten Ebene z_plane befindet.
    """
    # 1. 图像坐标去畸变
    # 1. Undistort image coordinates
    # 1. Bildkoordinaten entzerren
    undistorted_points = cv2.undistortPoints(np.array([pixel_coords], dtype=np.float32), camera_matrix, dist_coeffs)
    
    # 2. 转换为相机坐标系下的归一化坐标
    # 2. Convert to normalized coordinates in the camera coordinate system
    # 2. Konvertiert in normalisierte Koordinaten im Kamerakoordinatensystem
    x_norm, y_norm = undistorted_points[0][0]
    
    # 3. 射线-平面相交计算 (简化版)
    # 在实际应用中，需要根据相机外参和内参计算射线，并求与工作平面的交点
    # 这里提供一个逻辑框架
    # 3. Ray-plane intersection calculation (simplified version)
    # In actual applications, the ray needs to be calculated based on camera extrinsic and intrinsic parameters, and the intersection with the working plane needs to be found.
    # This provides a logical framework.
    # 3. Strahl-Ebenen-Schnittpunktberechnung (vereinfachte Version)
    # In realen Anwendungen muss der Strahl basierend auf den externen und internen Kameraparametern berechnet und der Schnittpunkt mit der Arbeitsebene gefunden werden.
    # Dies bietet einen logischen Rahmen.
    point_cam = np.array([x_norm, y_norm, 1.0])
    # 转换到基座坐标系
    # Convert to base coordinate system
    # Konvertiert in das Basiskoordinatensystem
    point_base = R_cam2base @ point_cam + t_cam2base.flatten()
    
    return point_base[:2] # 返回 (X, Y) / Returns (X, Y) / Gibt (X, Y) zurück

if __name__ == "__main__":
    print("Localization and Calibration Module Loaded.")
    # 示例：定义相机内参 (需实际标定获得)
    # Example: Define camera intrinsic parameters (to be obtained through actual calibration)
    # Beispiel: Kameraintrinsikparameter definieren (durch tatsächliche Kalibrierung zu erhalten)
    camera_matrix = np.array([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((5, 1))
    
    # 示例：像素坐标
    # Example: Pixel coordinates
    # Beispiel: Pixelkoordinaten
    pixel_pos = (640, 480)
    # 假设已获得手眼矩阵 R_cam2base, t_cam2base
    # Assume hand-eye matrix R_cam2base, t_cam2base has been obtained
    # Angenommen, die Hand-Auge-Matrix R_cam2base, t_cam2base wurde erhalten
    # robot_pos = pixel_to_robot_coords(pixel_pos, np.eye(3), np.zeros((3,1)), camera_matrix, dist_coeffs)
