"""
data_augmentation.py — 船舶大型金属高光面专项数据增强管线
:Author: RussellCooper

针对以下场景特点设计：
  - 大面积镜面高光（钢板、甲板、上层建筑）
  - 水面动态反射（从下方打光）
  - 港口强侧光 / 阴天漫射 / 夜间泛光灯
  - 焊缝、铆钉、舷窗等细小结构边缘
  - 锈蚀、油污、水渍等表面退化

增强策略分类：
  A. 几何变换   — 翻转、旋转、仿射、透视（模拟不同拍摄角度）
  B. 光照扰动   — 随机亮度/对比度、Gamma、局部高光注入、水面反射模拟
  C. 高光专项   — 随机高光椭圆注入、高光区域局部过曝、镜面条带
  D. 噪声/模糊  — 高斯噪声、运动模糊、散焦模糊、JPEG 压缩
  E. 颜色扰动   — HSV 扰动、颜色通道偏移、灰度化
  F. 遮挡/擦除  — Random Erasing、CutOut（模拟遮挡物）
  G. 掩膜一致性 — 所有几何变换同步作用于 mask 和 edge

用法::

    aug = ShipHullAugPipeline(p=0.8)
    image_aug, mask_aug, edge_aug = aug(image, mask, edge)

    # 或单独使用某个增强
    image_aug = random_sun_glare(image, n_glares=3)
"""

import math
import random
from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "random_flip",
    "random_rotate",
    "random_scale_crop",
    "random_perspective",
    "random_brightness_contrast",
    "random_gamma",
    "random_shadow",
    "random_water_reflection_band",
    "random_sun_glare",
    "random_specular_stripe",
    "simulate_overexposure_region",
    "random_gaussian_noise",
    "random_motion_blur",
    "random_defocus_blur",
    "random_jpeg_compression",
    "random_hsv_jitter",
    "random_channel_shift",
    "random_grayscale",
    "random_erasing",
    "random_fog",
    "generate_edge_from_mask",
    "generate_sobel_edge",
    "ShipHullAugPipeline",
    "cutmix",
    "mixup",
    "apply_cutmix_batch",
    "apply_mixup_batch",
]



# ============================================================
# 工具函数
# ============================================================

def _rng_uniform(lo: float, hi: float) -> float:
    return random.uniform(lo, hi)


def _rng_int(lo: int, hi: int) -> int:
    return random.randint(lo, hi - 1)


# ============================================================
# A. 几何变换
# ============================================================

def random_flip(image: np.ndarray, mask: np.ndarray, edge: np.ndarray,
                p_h: float = 0.5, p_v: float = 0.2
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """随机水平/垂直翻转（船体结构左右对称，垂直翻转概率低）。"""
    if random.random() < p_h:
        image = cv2.flip(image, 1)
        mask  = cv2.flip(mask,  1)
        edge  = cv2.flip(edge,  1)
    if random.random() < p_v:
        image = cv2.flip(image, 0)
        mask  = cv2.flip(mask,  0)
        edge  = cv2.flip(edge,  0)
    return image, mask, edge


def random_rotate(image: np.ndarray, mask: np.ndarray, edge: np.ndarray,
                  max_angle: float = 15.0, p: float = 0.5
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """随机小角度旋转（模拟相机倾斜）。"""
    if random.random() >= p:
        return image, mask, edge
    h, w = image.shape[:2]
    angle = _rng_uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    mask  = cv2.warpAffine(mask,  M, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    edge  = cv2.warpAffine(edge,  M, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, mask, edge


def random_scale_crop(image: np.ndarray, mask: np.ndarray, edge: np.ndarray,
                      scale_range: Tuple[float, float] = (0.7, 1.3),
                      p: float = 0.5
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """随机缩放后中心裁剪回原尺寸（模拟不同拍摄距离）。"""
    if random.random() >= p:
        return image, mask, edge
    h, w = image.shape[:2]
    scale = _rng_uniform(*scale_range)
    new_h, new_w = int(h * scale), int(w * scale)
    image_s = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_s  = cv2.resize(mask,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    edge_s  = cv2.resize(edge,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    def _center_crop_pad(img, th, tw):
        ih, iw = img.shape[:2]
        y0 = max(0, (ih - th) // 2)
        x0 = max(0, (iw - tw) // 2)
        cropped = img[y0:y0 + min(ih, th), x0:x0 + min(iw, tw)]
        ch, cw = cropped.shape[:2]
        if ch < th or cw < tw:
            if img.ndim == 3:
                out = np.zeros((th, tw, img.shape[2]), dtype=img.dtype)
            else:
                out = np.zeros((th, tw), dtype=img.dtype)
            py, px = (th - ch) // 2, (tw - cw) // 2
            out[py:py + ch, px:px + cw] = cropped
            return out
        return cropped

    return (_center_crop_pad(image_s, h, w),
            _center_crop_pad(mask_s,  h, w),
            _center_crop_pad(edge_s,  h, w))


def random_perspective(image: np.ndarray, mask: np.ndarray, edge: np.ndarray,
                       distortion: float = 0.1, p: float = 0.3
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """随机透视变换（模拟斜角拍摄船体）。"""
    if random.random() >= p:
        return image, mask, edge
    h, w = image.shape[:2]
    d = distortion
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [_rng_uniform(0, d * w),   _rng_uniform(0, d * h)],
        [_rng_uniform((1-d)*w, w), _rng_uniform(0, d * h)],
        [_rng_uniform((1-d)*w, w), _rng_uniform((1-d)*h, h)],
        [_rng_uniform(0, d * w),   _rng_uniform((1-d)*h, h)],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
    mask  = cv2.warpPerspective(mask,  M, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT)
    edge  = cv2.warpPerspective(edge,  M, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT)
    return image, mask, edge


# ============================================================
# B. 光照扰动
# ============================================================

def random_brightness_contrast(image: np.ndarray,
                                brightness_range: Tuple[float, float] = (-40, 40),
                                contrast_range: Tuple[float, float] = (0.7, 1.4),
                                p: float = 0.8) -> np.ndarray:
    """随机亮度偏移 + 对比度缩放。"""
    if random.random() >= p:
        return image
    alpha = _rng_uniform(*contrast_range)
    beta  = _rng_uniform(*brightness_range)
    return np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def random_gamma(image: np.ndarray,
                 gamma_range: Tuple[float, float] = (0.4, 2.5),
                 p: float = 0.6) -> np.ndarray:
    """
    随机 Gamma 校正（模拟不同曝光/相机响应曲线）。
    gamma < 1 增亮（过曝），gamma > 1 压暗（欠曝）。
    """
    if random.random() >= p:
        return image
    gamma = _rng_uniform(*gamma_range)
    lut = np.array(
        [np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lut)


def random_shadow(image: np.ndarray,
                  n_shadows: int = 2,
                  shadow_dim: float = 0.5,
                  p: float = 0.4) -> np.ndarray:
    """随机多边形阴影（模拟港口建筑、吊机遮挡产生的阴影）。"""
    if random.random() >= p:
        return image
    h, w = image.shape[:2]
    result = image.astype(np.float32)
    for _ in range(random.randint(1, n_shadows)):
        x1 = _rng_int(0, w)
        y1 = _rng_int(0, h // 2)
        x2 = _rng_int(0, w)
        pts = np.array([[x1, y1], [x2, h],
                        [x2 + _rng_int(-w//4, w//4), h],
                        [x1 + _rng_int(-w//4, w//4), y1]], dtype=np.int32)
        smask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(smask, [pts], 1.0)
        smask = cv2.GaussianBlur(smask, (0, 0), 20.0)
        result *= (1.0 - smask[:, :, np.newaxis] * _rng_uniform(shadow_dim * 0.5, shadow_dim))
    return np.clip(result, 0, 255).astype(np.uint8)


def random_water_reflection_band(image: np.ndarray, p: float = 0.4) -> np.ndarray:
    """在图像下半部分注入水面反射光带（蓝白色波纹高光）。"""
    if random.random() >= p:
        return image
    h, w = image.shape[:2]
    result = image.astype(np.float32)
    xs = np.linspace(0, _rng_uniform(4, 12) * math.pi, w, dtype=np.float32)
    wave = (np.sin(xs + _rng_uniform(0, math.pi)) + 1.0) / 2.0
    y_weight = np.linspace(0.0, 1.0, h, dtype=np.float32)
    y_weight[:h // 2] = 0.0
    intensity = wave[np.newaxis, :] * y_weight[:, np.newaxis] * _rng_uniform(0.3, 0.8)
    result[:, :, 0] += intensity * 180
    result[:, :, 1] += intensity * 165
    result[:, :, 2] += intensity * 130
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# C. 高光专项增强
# ============================================================

def random_sun_glare(image: np.ndarray,
                     n_glares: int = 4,
                     p: float = 0.6) -> np.ndarray:
    """
    随机注入椭圆形镜面高光斑（模拟大型钢板镜面反射）。
    每个高光斑：随机位置、大小、方向、亮度，高斯软边缘。
    """
    if random.random() >= p:
        return image
    h, w = image.shape[:2]
    result = image.astype(np.float32)
    for _ in range(_rng_int(1, n_glares + 1)):
        cx = _rng_int(w // 6, 5 * w // 6)
        cy = _rng_int(h // 6, 5 * h // 6)
        ax = _rng_int(20, min(w // 3, 150))
        ay = _rng_int(10, min(h // 4, 80))
        rot = _rng_uniform(0, 180)
        bright = _rng_uniform(180, 255)
        glare = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(glare, (cx, cy), (ax, ay), rot, 0, 360, bright, -1, cv2.LINE_AA)
        glare = cv2.GaussianBlur(glare, (0, 0), max(ax, ay) * _rng_uniform(0.3, 0.6))
        result[:, :, 0] += glare * _rng_uniform(0.85, 1.0)
        result[:, :, 1] += glare * _rng_uniform(0.85, 1.0)
        result[:, :, 2] += glare * _rng_uniform(0.80, 0.95)
    return np.clip(result, 0, 255).astype(np.uint8)


def random_specular_stripe(image: np.ndarray, p: float = 0.3) -> np.ndarray:
    """随机镜面条带高光（模拟圆柱形管道、船舷弧面的线性高光）。"""
    if random.random() >= p:
        return image
    h, w = image.shape[:2]
    result = image.astype(np.float32)
    for _ in range(_rng_int(1, 4)):
        x0, y0 = _rng_int(0, w), _rng_int(0, h)
        x1, y1 = _rng_int(0, w), _rng_int(0, h)
        width = _rng_int(3, 20)
        bright = _rng_uniform(160, 255)
        stripe = np.zeros((h, w), dtype=np.float32)
        cv2.line(stripe, (x0, y0), (x1, y1), bright, width, cv2.LINE_AA)
        stripe = cv2.GaussianBlur(stripe, (0, 0), width * 0.5)
        result[:, :, 0] += stripe * 0.95
        result[:, :, 1] += stripe * 0.95
        result[:, :, 2] += stripe * 0.90
    return np.clip(result, 0, 255).astype(np.uint8)


def simulate_overexposure_region(image: np.ndarray, p: float = 0.3) -> np.ndarray:
    """模拟局部过曝（像素值饱和到 255，丢失纹理细节）。"""
    if random.random() >= p:
        return image
    h, w = image.shape[:2]
    cx, cy = _rng_int(w // 5, 4 * w // 5), _rng_int(h // 5, 4 * h // 5)
    rx, ry = _rng_int(15, w // 4), _rng_int(10, h // 4)
    oe_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(oe_mask, (cx, cy), (rx, ry), _rng_uniform(0, 180),
                0, 360, 1.0, -1, cv2.LINE_AA)
    oe_mask = cv2.GaussianBlur(oe_mask, (0, 0), max(rx, ry) * 0.4)
    result = image.astype(np.float32) + oe_mask[:, :, np.newaxis] * _rng_uniform(80, 180)
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# D. 噪声与模糊
# ============================================================

def random_gaussian_noise(image: np.ndarray,
                           std_range: Tuple[float, float] = (2.0, 18.0),
                           p: float = 0.5) -> np.ndarray:
    """高斯噪声（传感器热噪声，夜间/长曝光场景更强）。"""
    if random.random() >= p:
        return image
    std = _rng_uniform(*std_range)
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_motion_blur(image: np.ndarray, max_ksize: int = 21, p: float = 0.2) -> np.ndarray:
    """运动模糊（船体振动 / 相机抖动）。"""
    if random.random() >= p:
        return image
    ksize = _rng_int(5, max_ksize + 1) | 1  # 保证奇数
    angle = _rng_uniform(0, 180)
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    cx = cy = ksize // 2
    rad = math.radians(angle)
    for i in range(ksize):
        t = i - cx
        px = int(round(cx + t * math.cos(rad)))
        py = int(round(cy + t * math.sin(rad)))
        if 0 <= px < ksize and 0 <= py < ksize:
            kernel[py, px] = 1.0
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(image, -1, kernel)


def random_defocus_blur(image: np.ndarray, max_radius: int = 8, p: float = 0.15) -> np.ndarray:
    """散焦模糊（圆形核，模拟景深外区域）。"""
    if random.random() >= p:
        return image
    r = _rng_int(2, max_radius + 1)
    ksize = 2 * r + 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    cv2.circle(kernel, (r, r), r, 1.0, -1)
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def random_jpeg_compression(image: np.ndarray,
                             quality_range: Tuple[int, int] = (55, 90),
                             p: float = 0.25) -> np.ndarray:
    """JPEG 压缩伪影（模拟视频流传输损失）。"""
    if random.random() >= p:
        return image
    q = _rng_int(*quality_range)
    _, enc = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# ============================================================
# E. 颜色扰动
# ============================================================

def random_hsv_jitter(image: np.ndarray,
                      hue_shift: float = 10.0,
                      sat_scale: Tuple[float, float] = (0.6, 1.4),
                      val_scale: Tuple[float, float] = (0.7, 1.3),
                      p: float = 0.6) -> np.ndarray:
    """HSV 空间随机扰动（色相偏移 + 饱和度/亮度缩放）。"""
    if random.random() >= p:
        return image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + _rng_uniform(-hue_shift, hue_shift)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * _rng_uniform(*sat_scale), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * _rng_uniform(*val_scale), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_channel_shift(image: np.ndarray,
                          shift_range: float = 20.0,
                          p: float = 0.3) -> np.ndarray:
    """随机通道偏移（模拟不同色温光源）。"""
    if random.random() >= p:
        return image
    result = image.astype(np.float32)
    for c in range(3):
        result[:, :, c] += _rng_uniform(-shift_range, shift_range)
    return np.clip(result, 0, 255).astype(np.uint8)


def random_grayscale(image: np.ndarray, p: float = 0.05) -> np.ndarray:
    """随机转为灰度三通道（模拟黑白相机）。"""
    if random.random() >= p:
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ============================================================
# F. 遮挡 / 环境
# ============================================================

def random_erasing(image: np.ndarray,
                   n_patches: int = 3,
                   patch_ratio: float = 0.1,
                   p: float = 0.3) -> np.ndarray:
    """Random Erasing / CutOut（模拟遮挡物：绳索、设备、人员）。"""
    if random.random() >= p:
        return image
    h, w = image.shape[:2]
    result = image.copy()
    for _ in range(_rng_int(1, n_patches + 1)):
        ph = int(h * _rng_uniform(0.02, patch_ratio))
        pw = int(w * _rng_uniform(0.02, patch_ratio))
        y0 = _rng_int(0, h - ph)
        x0 = _rng_int(0, w - pw)
        result[y0:y0 + ph, x0:x0 + pw] = (
            _rng_int(0, 256), _rng_int(0, 256), _rng_int(0, 256)
        )
    return result


def random_fog(image: np.ndarray, p: float = 0.15) -> np.ndarray:
    """随机雾/霾效果（港口常见海雾，降低能见度，高光扩散）。"""
    if random.random() >= p:
        return image
    h, w = image.shape[:2]
    fog_intensity = _rng_uniform(0.2, 0.6)
    fog_color = np.array([220, 225, 230], dtype=np.float32)
    result = image.astype(np.float32)
    y_weight = np.linspace(fog_intensity, fog_intensity * 0.3, h,
                           dtype=np.float32)[:, np.newaxis]
    result = result * (1 - y_weight) + fog_color * y_weight
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# G. 边缘标注辅助函数
# ============================================================

def generate_edge_from_mask(mask: np.ndarray, edge_width: int = 3) -> np.ndarray:
    """
    从分割掩膜自动生成边缘 ground truth（膨胀 - 腐蚀）。

    Args:
        mask:       二值分割掩膜 (H,W) uint8
        edge_width: 边缘宽度（像素），默认3（更薄的边缘用于精确边缘检测）
    Returns:
        edge_mask: 边缘掩膜 (H,W) uint8，边缘处为 255
    """
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width, edge_width))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded  = cv2.erode(mask,  kernel, iterations=1)
    return (dilated.astype(np.int16) - eroded.astype(np.int16)).clip(0, 255).astype(np.uint8)


def generate_sobel_edge(image: np.ndarray) -> np.ndarray:
    """
    Sobel 边缘先验图（用于训练时辅助监督）。

    Returns:
        float32 [0,1] 归一化边缘强度图
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    if mag.max() > 0:
        mag /= mag.max()
    return mag.astype(np.float32)


# ============================================================
# H. CutMix / MixUp 批增强（训练时使用）
# ============================================================

import torch


def cutmix(
    image1: torch.Tensor,
    mask1: torch.Tensor,
    edge1: torch.Tensor,
    image2: torch.Tensor,
    mask2: torch.Tensor,
    edge2: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CutMix：随机裁剪一块区域从 sample2 粘贴到 sample1。

    混合策略：
      - 从分布 Beta(alpha, alpha) 采样混合比例
      - 在图像上随机选择矩形裁剪区域
      - 交换选中区域

    对分割任务效果优于 MixUp，尤其适合高反光/遮挡场景。

    Args:
        image1, mask1, edge1: 样本 A (B,C,H,W) / (B,1,H,W)
        image2, mask2, edge2: 样本 B (B,C,H,W) / (B,1,H,W)
        alpha: Beta 分布参数

    Returns:
        混合后的 (image, mask, edge)
    """
    lam = np.random.beta(alpha, alpha)
    b, c, h, w = image1.shape

    # 采样裁剪框
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)

    y1 = max(0, cy - cut_h // 2)
    y2 = min(h, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(w, cx + cut_w // 2)

    # 混合
    image_mixed = image1.clone()
    mask_mixed = mask1.clone()
    edge_mixed = edge1.clone()

    image_mixed[:, :, y1:y2, x1:x2] = image2[:, :, y1:y2, x1:x2]
    mask_mixed[:, :, y1:y2, x1:x2] = mask2[:, :, y1:y2, x1:x2]
    edge_mixed[:, :, y1:y2, x1:x2] = edge2[:, :, y1:y2, x1:x2]

    return image_mixed, mask_mixed, edge_mixed


def mixup(
    image1: torch.Tensor,
    mask1: torch.Tensor,
    edge1: torch.Tensor,
    image2: torch.Tensor,
    mask2: torch.Tensor,
    edge2: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MixUp：两张图像按比例线性混合。

    混合公式：M = λ*I1 + (1-λ)*I2

    对小物体和难样本有很好增强效果，可提升模型泛化能力。

    Args:
        alpha: Beta 分布参数（越小混合越偏向端点）

    Returns:
        混合后的 (image, mask, edge)
    """
    lam = np.random.beta(alpha, alpha)
    image_mixed = lam * image1 + (1 - lam) * image2
    mask_mixed = lam * mask1 + (1 - lam) * mask2
    edge_mixed = lam * edge1 + (1 - lam) * edge2
    return image_mixed, mask_mixed, edge_mixed


def apply_cutmix_batch(
    batch_images: torch.Tensor,
    batch_masks: torch.Tensor,
    batch_edges: torch.Tensor,
    alpha: float = 1.0,
    p: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对一个 batch 随机应用 CutMix。

    在 batch 内随机配对样本进行 CutMix 操作。
    """
    if np.random.random() > p:
        return batch_images, batch_masks, batch_edges

    b = batch_images.shape[0]
    # 打乱顺序配对
    indices = torch.randperm(b)
    img2 = batch_images[indices]
    msk2 = batch_masks[indices]
    edg2 = batch_edges[indices]

    img_mixed, msk_mixed, edg_mixed = cutmix(
        batch_images, batch_masks, batch_edges,
        img2, msk2, edg2,
        alpha=alpha,
    )
    return img_mixed, msk_mixed, edg_mixed


def apply_mixup_batch(
    batch_images: torch.Tensor,
    batch_masks: torch.Tensor,
    batch_edges: torch.Tensor,
    alpha: float = 0.4,
    p: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对一个 batch 随机应用 MixUp。
    """
    if np.random.random() > p:
        return batch_images, batch_masks, batch_edges

    b = batch_images.shape[0]
    indices = torch.randperm(b)
    img2 = batch_images[indices]
    msk2 = batch_masks[indices]
    edg2 = batch_edges[indices]

    img_mixed, msk_mixed, edg_mixed = mixup(
        batch_images, batch_masks, batch_edges,
        img2, msk2, edg2,
        alpha=alpha,
    )
    return img_mixed, msk_mixed, edg_mixed


# ============================================================
# I. 完整增强管线
# ============================================================

class ShipHullAugPipeline:
    """
    船舶船体视觉检测专用数据增强管线。

    所有几何变换同步作用于 image / mask / edge，
    光照/噪声/颜色扰动只作用于 image。

    Args:
        p: 整体增强概率（0=不增强，1=每次都增强）

    Examples::

        aug = ShipHullAugPipeline(p=0.9)
        img_a, msk_a, edg_a = aug(image, mask, edge)
    """

    def __init__(self, p: float = 0.85):
        self.p = p

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        edge: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:

        if random.random() >= self.p:
            return image, mask, edge

        h, w = image.shape[:2]
        _mask = mask if mask is not None else np.zeros((h, w), dtype=np.uint8)
        _edge = edge if edge is not None else np.zeros((h, w), dtype=np.uint8)

        # ---- A. 几何变换（image + mask + edge 同步）----
        image, _mask, _edge = random_flip(image, _mask, _edge)
        image, _mask, _edge = random_rotate(image, _mask, _edge, max_angle=12, p=0.4)
        image, _mask, _edge = random_scale_crop(image, _mask, _edge, p=0.4)
        image, _mask, _edge = random_perspective(image, _mask, _edge, distortion=0.08, p=0.25)

        # ---- B. 光照扰动（仅 image）----
        image = random_brightness_contrast(image, p=0.75)
        image = random_gamma(image, gamma_range=(0.45, 2.2), p=0.55)
        image = random_shadow(image, p=0.35)
        image = random_water_reflection_band(image, p=0.35)

        # ---- C. 高光专项（仅 image）----
        image = random_sun_glare(image, n_glares=4, p=0.55)
        image = random_specular_stripe(image, p=0.25)
        image = simulate_overexposure_region(image, p=0.25)

        # ---- D. 噪声/模糊（仅 image）----
        image = random_gaussian_noise(image, p=0.5)
        image = random_motion_blur(image, p=0.18)
        image = random_defocus_blur(image, p=0.12)
        image = random_jpeg_compression(image, p=0.22)

        # ---- E. 颜色扰动（仅 image）----
        image = random_hsv_jitter(image, p=0.55)
        image = random_channel_shift(image, p=0.25)
        image = random_grayscale(image, p=0.04)

        # ---- F. 遮挡/环境（仅 image）----
        image = random_erasing(image, p=0.25)
        image = random_fog(image, p=0.12)

        out_mask = _mask if mask is not None else None
        out_edge = _edge if edge is not None else None
        return image, out_mask, out_edge


# ============================================================
# 命令行入口（可视化验证）
# ============================================================

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="数据增强管线可视化验证")
    parser.add_argument("--input",  type=str, default=None)
    parser.add_argument("--output", type=str, default="./aug_preview")
    parser.add_argument("--n",      type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.input and os.path.exists(args.input):
        image = cv2.imread(args.input)
        mask  = np.zeros(image.shape[:2], dtype=np.uint8)
        edge  = np.zeros(image.shape[:2], dtype=np.uint8)
    else:
        from synth_dataset_generator import synthesize_one_sample
        print("[aug] 使用合成船舶场景图像")
        sample = synthesize_one_sample(h=512, w=512)
        image, mask, edge = sample['image'], sample['mask'], sample['edge']

    aug = ShipHullAugPipeline(p=1.0)

    thumb = 256
    cols = 4
    rows = math.ceil(args.n / cols)
    grid = np.zeros((rows * thumb, cols * thumb, 3), dtype=np.uint8)

    for i in range(args.n):
        img_a, msk_a, edg_a = aug(image.copy(), mask.copy(), edge.copy())
        vis = img_a.copy()
        if msk_a is not None:
            vis[msk_a > 100] = (vis[msk_a > 100].astype(np.float32) * 0.6 +
                                 np.array([0, 200, 0]) * 0.4).astype(np.uint8)
        if edg_a is not None:
            vis[edg_a > 128] = [0, 0, 255]
        r, c = divmod(i, cols)
        grid[r * thumb:(r + 1) * thumb, c * thumb:(c + 1) * thumb] = \
            cv2.resize(vis, (thumb, thumb))

    out_path = os.path.join(args.output, "aug_grid.png")
    cv2.imwrite(out_path, grid)
    print(f"[aug] 增强预览已保存: {out_path}")
