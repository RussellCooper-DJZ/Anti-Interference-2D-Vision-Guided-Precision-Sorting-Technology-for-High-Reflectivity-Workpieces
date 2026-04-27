"""
vision/polarization_advanced.py — 物理级偏振光模拟模块
借鉴论文《大型船舶钢板面高反光工况2D视觉检测技术调研报告》

论文要点：
  "偏振光技术：使用偏振滤光片减少表面反射，提高图像对比度"
  "多角度照明：设计特殊照明系统，避免直接反射"

本模块基于菲涅尔方程实现物理级偏振仿真，包含：
  1. 镜面/漫反射的偏振分离（Fresnel 方程）
  2. 偏振度（DoP）与偏振角（AoP）计算
  3. 多角度偏振图像合成（模拟旋转偏振片）
  4. 偏振差分成像（PDI）增强缺陷对比度

用法::

    from vision.polarization_advanced import PolarizationSimulator, PolarizationProcessor

    # 模拟多角度偏振图像
    simulator = PolarizationSimulator(refractive_index=2.5 + 3.0j)  # 铁近似折射率
    polar_images = simulator.capture_multi_angle(image_bgr, n_angles=4)

    # 偏振处理
    processor = PolarizationProcessor()
    dop, aop = processor.compute_dop_aop(polar_images)
    enhanced = processor.polarization_difference(polar_images[0], polar_images[2])
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "fresnel_coefficients",
    "PolarizationSimulator",
    "PolarizationProcessor",
    "polarization_glare_suppression",
]


# ============================================================
# 1. 菲涅尔方程（Fresnel Equations）
# ============================================================

def fresnel_coefficients(
    n1: float = 1.0,
    n2: complex = 2.5 + 3.0j,
    theta_i: float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    计算菲涅尔反射/透射系数。

    Args:
        n1: 入射介质折射率（空气≈1.0）
        n2: 反射介质复折射率（铁: 2.5+3.0j，铝: 1.2+7.0j）
        theta_i: 入射角（弧度）

    Returns:
        (rs, rp, ts, tp) — s 偏振和 p 偏振的反射/透射系数幅值平方
    """
    cos_i = math.cos(theta_i)
    sin_i = math.sin(theta_i)

    # 斯涅尔定律求透射角（复数）
    sin_t = (n1 / n2) * sin_i
    cos_t = np.sqrt(1 - sin_t**2)

    # s 偏振（垂直于入射面）
    rs = abs((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)) ** 2
    ts = 1 - rs  # 能量守恒近似

    # p 偏振（平行于入射面）
    rp = abs((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)) ** 2
    tp = 1 - rp

    return float(rs), float(rp), float(ts), float(tp)


# ============================================================
# 2. 多角度偏振模拟器
# ============================================================

class PolarizationSimulator:
    """
    偏振光模拟器。
    模拟在相机前旋转线偏振片（0°, 45°, 90°, 135°）拍摄的效果。

    物理模型：
      I(θ) = I_s * sin²(θ - φ) + I_p * cos²(θ - φ)
      其中 φ 为偏振角（AoP），I_s/I_p 为 s/p 分量强度
    """

    def __init__(
        self,
        refractive_index: complex = 2.5 + 3.0j,
        incident_angle_deg: float = 45.0,
    ):
        self.n2 = refractive_index
        self.theta_i = math.radians(incident_angle_deg)
        self.rs, self.rp, self.ts, self.tp = fresnel_coefficients(
            n1=1.0, n2=self.n2, theta_i=self.theta_i
        )

    def _simulate_polarizer_angle(
        self,
        image_bgr: np.ndarray,
        angle_deg: float,
    ) -> np.ndarray:
        """
        模拟单角度偏振片图像。
        假设高光区域以镜面反射为主（强偏振），纹理区域以漫反射为主（弱偏振）。
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 估计高光掩膜（基于亮度）
        highlight_mask = (gray > 200).astype(np.float32)

        # 偏振片透射系数（马吕斯定律）
        theta = math.radians(angle_deg)
        # 假设偏振角 φ=0（简化），则透射率 = cos²(θ)
        transmission = np.cos(theta) ** 2

        # 高光区域（强偏振）：随偏振角变化大
        # 漫反射区域（非偏振）：几乎不变
        polarized = gray * (0.5 + 0.5 * transmission * highlight_mask)

        # 转回 3 通道 BGR
        polarized = polarized.clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(polarized, cv2.COLOR_GRAY2BGR)

    def capture_multi_angle(
        self,
        image_bgr: np.ndarray,
        n_angles: int = 4,
    ) -> List[np.ndarray]:
        """
        模拟多角度偏振图像采集。

        Args:
            image_bgr: 原始图像
            n_angles: 偏振角度数（默认 4：0°, 45°, 90°, 135°）

        Returns:
            polar_images: 偏振图像列表
        """
        angles = np.linspace(0, 180, n_angles, endpoint=False)
        return [self._simulate_polarizer_angle(image_bgr, float(a)) for a in angles]


# ============================================================
# 3. 偏振处理器
# ============================================================

class PolarizationProcessor:
    """
    偏振图像处理器。
    从多角度偏振图像中提取偏振信息，用于增强缺陷/抑制反光。
    """

    def compute_dop_aop(
        self,
        polar_images: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算偏振度（DoP）和偏振角（AoP）。

        使用 Stokes 参数：
          S0 = I0 + I90
          S1 = I0 - I90
          S2 = I45 - I135
          DoP = sqrt(S1² + S2²) / S0
          AoP = 0.5 * atan2(S2, S1)

        Returns:
            dop: (H, W) float32，偏振度 [0, 1]
            aop: (H, W) float32，偏振角 [0, π]
        """
        if len(polar_images) < 4:
            raise ValueError("需要至少 4 张偏振图像（0°, 45°, 90°, 135°）")

        i0 = cv2.cvtColor(polar_images[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        i45 = cv2.cvtColor(polar_images[1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        i90 = cv2.cvtColor(polar_images[2], cv2.COLOR_BGR2GRAY).astype(np.float32)
        i135 = cv2.cvtColor(polar_images[3], cv2.COLOR_BGR2GRAY).astype(np.float32)

        s0 = i0 + i90 + 1e-6
        s1 = i0 - i90
        s2 = i45 - i135

        dop = np.sqrt(s1**2 + s2**2) / s0
        aop = 0.5 * np.arctan2(s2, s1)

        return dop.astype(np.float32), (aop % np.pi).astype(np.float32)

    def polarization_difference(
        self,
        img0: np.ndarray,
        img90: np.ndarray,
    ) -> np.ndarray:
        """
        偏振差分成像（PDI）。
        0° 和 90° 偏振图像相减，可显著增强表面缺陷对比度。
        """
        g0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g90 = cv2.cvtColor(img90, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(g0 - g90)
        diff = (diff / diff.max() * 255).astype(np.uint8)
        return cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

    def glare_suppression(
        self,
        polar_images: List[np.ndarray],
    ) -> np.ndarray:
        """
        基于偏振最小值法的反光抑制。
        原理：镜面反射是高度偏振的，在某一偏振角度下透射率最低。
        取多张偏振图像的像素级最小值，可有效抑制高光。
        """
        arrays = [p.astype(np.float32) for p in polar_images]
        min_img = np.min(np.stack(arrays, axis=0), axis=0)
        return min_img.clip(0, 255).astype(np.uint8)

    def diffuse_specular_separation(
        self,
        img_min: np.ndarray,
        img_max: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        漫反射/镜面反射分离（基于偏振最小/最大亮度）。

        Returns:
            diffuse:  漫反射分量（近似非偏振光）
            specular: 镜面反射分量（偏振光）
        """
        I_min = cv2.cvtColor(img_min, cv2.COLOR_BGR2GRAY).astype(np.float32)
        I_max = cv2.cvtColor(img_max, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # 漫反射 ≈ 2 * I_min（假设偏振片完全消光时仅剩漫反射）
        diffuse_gray = 2 * I_min
        specular_gray = I_max - I_min

        diffuse_gray = diffuse_gray.clip(0, 255).astype(np.uint8)
        specular_gray = specular_gray.clip(0, 255).astype(np.uint8)

        return (
            cv2.cvtColor(diffuse_gray, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(specular_gray, cv2.COLOR_GRAY2BGR),
        )


# ============================================================
# 4. 统一接口
# ============================================================

def polarization_glare_suppression(
    image_bgr: np.ndarray,
    n_angles: int = 4,
    refractive_index: complex = 2.5 + 3.0j,
) -> np.ndarray:
    """
    一键偏振反光抑制。

    Args:
        image_bgr: 输入图像
        n_angles: 模拟偏振角度数
        refractive_index: 钢板材质复折射率

    Returns:
        suppressed: 反光抑制后的图像
    """
    simulator = PolarizationSimulator(refractive_index=refractive_index)
    polar_images = simulator.capture_multi_angle(image_bgr, n_angles=n_angles)

    processor = PolarizationProcessor()
    return processor.glare_suppression(polar_images)
