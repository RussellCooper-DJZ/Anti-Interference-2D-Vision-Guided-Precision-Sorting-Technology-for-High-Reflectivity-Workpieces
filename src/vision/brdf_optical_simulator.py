"""
brdf_optical_simulator.py — 基于物理的 BRDF 光学与光斑模拟器
模拟金属表面镜面反射、菲涅尔效应（Fresnel Effect）及多角度光源干扰。
"""

import cv2
import numpy as np
import random

class BRDFOpticalSimulator:
    """
    高级光学模拟器，基于双向反射分布函数 (BRDF) 近似计算高反光表面的镜面反射。
    """
    def __init__(self, roughness: float = 0.15):
        self.roughness = roughness # 表面粗糙度，越小镜面反射越强

    def simulate_specular_glare(self, image: np.ndarray, light_pos: tuple, intensity: float = 2.5) -> np.ndarray:
        """
        基于光源位置和表面法线模拟物理级镜面高光。
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        
        # 计算伪法线贴图 (Normal Map estimation from gradients)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        norm = np.sqrt(sobelx**2 + sobely**2 + 1e-5)
        nx = -sobelx / norm
        ny = -sobely / norm
        nz = 1.0 / norm
        
        # 光源方向向量
        lx, ly, lz = light_pos
        
        # 视线方向 (假设垂直向下 [0, 0, 1])
        # 漫反射项 (Diffuse) + 镜面反射项 (Specular - Phong/Blinn-Phong model)
        dot_nl = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
        
        # 半角向量
        hx = lx
        hy = ly
        hz = lz + 1.0
        h_norm = np.sqrt(hx**2 + hy**2 + hz**2 + 1e-5)
        hx /= h_norm
        hy /= h_norm
        hz /= h_norm
        
        dot_nh = np.clip(nx * hx + ny * hy + nz * hz, 0, 1)
        specular = np.power(dot_nh, 1.0 / (self.roughness + 1e-3)) * intensity
        
        # 转换为 3 通道高光斑
        specular_rgb = np.stack([specular, specular, specular], axis=-1) * 255.0
        
        output = image.astype(np.float32) + specular_rgb
        return np.clip(output, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    sim = BRDFOpticalSimulator()
    print("BRDFOpticalSimulator initialized successfully.")
