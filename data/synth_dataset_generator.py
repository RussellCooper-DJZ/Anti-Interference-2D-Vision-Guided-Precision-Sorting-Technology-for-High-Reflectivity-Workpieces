"""
synth_dataset_generator.py — 船舶大型金属高光面合成训练数据集生成器
:Author: RussellCooper

面向场景：
  - 船体钢板、舷侧、甲板、焊缝、铆钉、舷窗、管道等结构
  - 港口强侧光、水面镜面反射、阴天漫射光、夜间泛光灯等复杂环境光
  - 油漆剥落、锈蚀、水渍、油污等表面退化

生成内容（每个样本）：
  image.png  — RGB 合成图像（含高光、噪声、退化）
  mask.png   — 二值语义掩膜（船体结构区域 = 255）
  edge.png   — 边缘掩膜（结构边界 = 255）
  meta.json  — 场景参数（光照、材质、结构类型等）

用法：
  python3 synth_dataset_generator.py --count 500 --output ./datasets/synth_ship --preview
  python3 synth_dataset_generator.py --count 1 --output ./datasets/synth_ship --preview --seed 42
"""

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 全局随机数生成器（支持可复现 seed）
# ============================================================
RNG = np.random.default_rng()


def set_seed(seed: int) -> None:
    global RNG
    RNG = np.random.default_rng(seed)
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# 1. 基础材质与纹理生成
# ============================================================

def _perlin_noise(h: int, w: int, scale: float = 0.05,
                  octaves: int = 4) -> np.ndarray:
    """
    生成 [0,1] 范围的 Perlin 风格噪声（用叠加正弦波近似）。
    scale 越小噪声频率越低（大块），octaves 越多细节越丰富。
    """
    noise = np.zeros((h, w), dtype=np.float32)
    amp, freq = 1.0, scale
    for _ in range(octaves):
        xs = np.linspace(0, freq * w, w, dtype=np.float32)
        ys = np.linspace(0, freq * h, h, dtype=np.float32)
        px = RNG.uniform(0, 2 * math.pi)
        py = RNG.uniform(0, 2 * math.pi)
        noise += amp * (
            np.sin(xs[np.newaxis, :] + px) *
            np.sin(ys[:, np.newaxis] + py)
        )
        amp *= 0.5
        freq *= 2.0
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise.astype(np.float32)


def make_steel_plate_texture(h: int, w: int,
                              base_color: Tuple[int, int, int] = (140, 145, 155),
                              roughness: float = 0.15) -> np.ndarray:
    """
    生成船体钢板纹理（冷轧/热轧钢，带轻微各向异性拉丝纹）。

    Args:
        h, w:        图像尺寸
        base_color:  基础 BGR 颜色
        roughness:   表面粗糙度 0-1（越大高光越散）
    Returns:
        BGR uint8 纹理图
    """
    # 基础颜色层
    tex = np.ones((h, w, 3), dtype=np.float32)
    for c, v in enumerate(base_color):
        tex[:, :, c] = v / 255.0

    # 各向异性拉丝纹（水平方向）
    grain_h = _perlin_noise(h, w, scale=0.002, octaves=3)  # 低频
    grain_v = _perlin_noise(h, w, scale=0.08, octaves=2)   # 高频竖纹
    grain = grain_h * 0.6 + grain_v * 0.4
    grain = (grain - 0.5) * roughness * 0.4

    # 轻微颜色变化（金属光泽不均匀）
    color_var = _perlin_noise(h, w, scale=0.02, octaves=2) * 0.08 - 0.04

    for c in range(3):
        tex[:, :, c] = np.clip(tex[:, :, c] + grain + color_var, 0.0, 1.0)

    return (tex * 255.0).astype(np.uint8)


def make_rust_texture(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成锈蚀纹理及其 alpha 掩膜。

    Returns:
        (rust_bgr, rust_alpha): 锈蚀颜色图和透明度图（均为 float32 [0,1]）
    """
    # 锈蚀颜色：橙红到深棕
    rust_colors = [
        (0.10, 0.25, 0.55),   # 深锈红 BGR
        (0.08, 0.20, 0.65),   # 橙锈
        (0.05, 0.15, 0.45),   # 暗锈
    ]
    noise1 = _perlin_noise(h, w, scale=0.04, octaves=5)
    noise2 = _perlin_noise(h, w, scale=0.08, octaves=3)
    blend = noise1 * 0.7 + noise2 * 0.3

    rust = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        c0 = rust_colors[0][c]
        c1 = rust_colors[1][c]
        c2 = rust_colors[2][c]
        rust[:, :, c] = (blend < 0.4) * c0 + \
                        ((blend >= 0.4) & (blend < 0.7)) * c1 + \
                        (blend >= 0.7) * c2

    # alpha：只在高噪声区域出现锈蚀
    alpha_noise = _perlin_noise(h, w, scale=0.06, octaves=4)
    alpha = np.clip((alpha_noise - 0.55) * 3.0, 0.0, 1.0)
    return rust, alpha


def make_paint_chip_texture(h: int, w: int,
                             paint_color: Tuple[float, float, float] = (0.1, 0.12, 0.15)
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成油漆剥落纹理（船体常见黑灰色防腐漆）。

    Returns:
        (paint_bgr, paint_alpha): float32 [0,1]
    """
    noise = _perlin_noise(h, w, scale=0.05, octaves=4)
    paint = np.stack([
        np.full((h, w), paint_color[c], dtype=np.float32) for c in range(3)
    ], axis=2)
    alpha = np.clip((noise - 0.35) * 2.5, 0.0, 1.0)
    return paint, alpha


def make_water_stain_texture(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """生成水渍/盐渍纹理（船体水线附近常见）。"""
    noise = _perlin_noise(h, w, scale=0.03, octaves=3)
    stain = np.ones((h, w, 3), dtype=np.float32) * 0.85  # 浅灰白
    alpha = np.clip((noise - 0.6) * 4.0, 0.0, 1.0) * 0.5
    return stain, alpha


# ============================================================
# 2. 船体结构元素绘制
# ============================================================

def draw_weld_seam(canvas: np.ndarray, mask: np.ndarray,
                   x0: int, y0: int, x1: int, y1: int,
                   width: int = 6) -> None:
    """
    绘制焊缝（直线型，带轻微不规则纹理）。
    同时更新 mask（焊缝为结构边界）。
    """
    h, w = canvas.shape[:2]
    # 焊缝颜色：略深于钢板，带氧化色
    weld_color = (int(RNG.integers(80, 110)),
                  int(RNG.integers(75, 105)),
                  int(RNG.integers(70, 100)))
    # 主焊缝线
    cv2.line(canvas, (x0, y0), (x1, y1), weld_color, width, cv2.LINE_AA)
    # 焊缝纹理（叠加小噪点模拟鱼鳞纹）
    length = int(math.hypot(x1 - x0, y1 - y0))
    if length > 0:
        dx, dy = (x1 - x0) / length, (y1 - y0) / length
        for i in range(0, length, width):
            px = int(x0 + dx * i + RNG.uniform(-1, 1))
            py = int(y0 + dy * i + RNG.uniform(-1, 1))
            r = int(width * 0.6 + RNG.uniform(-1, 1))
            cv2.circle(canvas, (px, py), max(1, r), weld_color, -1, cv2.LINE_AA)
    # 更新边缘掩膜
    cv2.line(mask, (x0, y0), (x1, y1), 255, max(1, width // 2), cv2.LINE_AA)


def draw_rivet_row(canvas: np.ndarray, mask: np.ndarray,
                   cx_list: List[int], cy: int, radius: int = 8) -> None:
    """绘制一排铆钉。"""
    for cx in cx_list:
        # 铆钉本体（圆形凸起，中心亮边缘暗）
        for dr in range(radius, 0, -1):
            t = dr / radius
            v = int(180 * t + 100 * (1 - t))
            cv2.circle(canvas, (cx, cy), dr, (v, v, v + 5), -1, cv2.LINE_AA)
        # 铆钉高光点
        hx = cx - radius // 3
        hy = cy - radius // 3
        cv2.circle(canvas, (hx, hy), max(1, radius // 3), (230, 235, 240), -1)
        # 边缘掩膜
        cv2.circle(mask, (cx, cy), radius + 1, 255, 1, cv2.LINE_AA)


def draw_porthole(canvas: np.ndarray, mask: np.ndarray,
                  cx: int, cy: int, r: int = 30) -> None:
    """绘制舷窗（圆形，带金属框）。"""
    # 玻璃区域（深蓝灰）
    glass_color = (int(RNG.integers(40, 70)),
                   int(RNG.integers(50, 80)),
                   int(RNG.integers(60, 90)))
    cv2.circle(canvas, (cx, cy), r, glass_color, -1, cv2.LINE_AA)
    # 金属框（多圈）
    for ring_r, ring_w in [(r, 4), (r + 5, 3)]:
        frame_v = int(RNG.integers(100, 140))
        cv2.circle(canvas, (cx, cy), ring_r, (frame_v, frame_v, frame_v + 5),
                   ring_w, cv2.LINE_AA)
    # 玻璃高光（斜向椭圆）
    hl_cx = cx - r // 3
    hl_cy = cy - r // 3
    cv2.ellipse(canvas, (hl_cx, hl_cy), (r // 3, r // 5), -30,
                0, 360, (200, 210, 220), -1, cv2.LINE_AA)
    # 掩膜
    cv2.circle(mask, (cx, cy), r + 5, 255, 2, cv2.LINE_AA)


def draw_pipe(canvas: np.ndarray, mask: np.ndarray,
              x0: int, y0: int, x1: int, y1: int,
              radius: int = 12) -> None:
    """绘制管道（圆柱投影，带高光条）。"""
    pipe_v = int(RNG.integers(110, 150))
    cv2.line(canvas, (x0, y0), (x1, y1), (pipe_v, pipe_v, pipe_v + 5),
             radius * 2, cv2.LINE_AA)
    # 高光条（偏上方）
    length = int(math.hypot(x1 - x0, y1 - y0))
    if length > 0:
        dx, dy = (x1 - x0) / length, (y1 - y0) / length
        nx, ny = -dy, dx  # 法向量
        offset = radius // 2
        hx0 = int(x0 + nx * offset)
        hy0 = int(y0 + ny * offset)
        hx1 = int(x1 + nx * offset)
        hy1 = int(y1 + ny * offset)
        cv2.line(canvas, (hx0, hy0), (hx1, hy1), (200, 205, 215),
                 max(1, radius // 3), cv2.LINE_AA)
    cv2.line(mask, (x0, y0), (x1, y1), 255, radius * 2 + 2, cv2.LINE_AA)


def draw_hull_panel(canvas: np.ndarray, mask: np.ndarray,
                    pts: np.ndarray, texture: np.ndarray) -> None:
    """
    绘制一块船体钢板（多边形区域，贴钢板纹理）。

    Args:
        canvas:  目标图像
        mask:    语义掩膜
        pts:     多边形顶点 (N,2) int32
        texture: 钢板纹理图（与 canvas 同尺寸）
    """
    h, w = canvas.shape[:2]
    panel_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(panel_mask, [pts], 255)

    # 将纹理贴到 canvas 的对应区域
    alpha = (panel_mask[:, :, np.newaxis] / 255.0).astype(np.float32)
    canvas[:] = (canvas.astype(np.float32) * (1 - alpha) +
                 texture.astype(np.float32) * alpha).astype(np.uint8)

    # 更新语义掩膜
    cv2.fillPoly(mask, [pts], 200)
    # 绘制面板边缘
    cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=2)


# ============================================================
# 3. 光照系统
# ============================================================

class LightingSystem:
    """
    船舶场景复杂光照模拟器。

    支持以下光照类型：
      - 'direct_sun'   : 港口强侧光（单方向平行光，产生硬阴影和强高光）
      - 'overcast'     : 阴天漫射光（均匀柔和，高光弱）
      - 'water_reflect': 水面镜面反射（动态波纹高光，从下方打光）
      - 'floodlight'   : 夜间泛光灯（多个点光源，强局部高光）
      - 'mixed'        : 混合光照（日光 + 水面反射）
    """

    def __init__(self, mode: str = 'mixed', seed: Optional[int] = None):
        self.mode = mode
        if seed is not None:
            set_seed(seed)

    def apply(self, image: np.ndarray) -> np.ndarray:
        """将光照效果叠加到图像上，返回 BGR uint8。"""
        if self.mode == 'direct_sun':
            return self._direct_sun(image)
        elif self.mode == 'overcast':
            return self._overcast(image)
        elif self.mode == 'water_reflect':
            return self._water_reflect(image)
        elif self.mode == 'floodlight':
            return self._floodlight(image)
        elif self.mode == 'mixed':
            img = self._direct_sun(image)
            img = self._water_reflect(img)
            return img
        else:
            return image.copy()

    # ------ 各光照模式实现 ------

    def _direct_sun(self, image: np.ndarray) -> np.ndarray:
        """港口强侧光：方向性平行光 + 硬高光椭圆。"""
        h, w = image.shape[:2]
        result = image.astype(np.float32)

        # 方向光渐变（模拟侧光阴影）
        angle_deg = float(RNG.uniform(20, 160))
        angle_rad = math.radians(angle_deg)
        xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        xg, yg = np.meshgrid(xs, ys)
        directional = (math.cos(angle_rad) * xg + math.sin(angle_rad) * yg)
        directional = (directional - directional.min()) / \
                      (directional.max() - directional.min() + 1e-8)
        intensity = 0.7 + 0.5 * directional  # [0.7, 1.2]
        result *= intensity[:, :, np.newaxis]

        # 若干强高光椭圆（大型金属面镜面反射）
        n_glare = int(RNG.integers(2, 6))
        for _ in range(n_glare):
            cx = int(RNG.integers(w // 4, 3 * w // 4))
            cy = int(RNG.integers(h // 4, 3 * h // 4))
            ax = int(RNG.integers(30, 120))
            ay = int(RNG.integers(15, 60))
            rot = float(RNG.uniform(0, 180))
            bright = float(RNG.uniform(220, 255))
            glare_layer = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(glare_layer, (cx, cy), (ax, ay), rot,
                        0, 360, bright, -1, cv2.LINE_AA)
            glare_layer = cv2.GaussianBlur(glare_layer, (0, 0), ax * 0.4)
            glare_layer = (glare_layer / 255.0) * float(RNG.uniform(0.6, 1.0))
            result += glare_layer[:, :, np.newaxis] * 120.0

        return np.clip(result, 0, 255).astype(np.uint8)

    def _overcast(self, image: np.ndarray) -> np.ndarray:
        """阴天漫射光：整体亮度略降，加轻微云层遮蔽渐变。"""
        h, w = image.shape[:2]
        result = image.astype(np.float32)

        cloud = _perlin_noise(h, w, scale=0.015, octaves=4)
        cloud_factor = 0.75 + 0.25 * cloud  # [0.75, 1.0]
        result *= cloud_factor[:, :, np.newaxis]

        # 轻微蓝偏（阴天色温偏冷）
        result[:, :, 0] = np.clip(result[:, :, 0] * 1.05, 0, 255)  # B
        result[:, :, 2] = np.clip(result[:, :, 2] * 0.95, 0, 255)  # R

        return np.clip(result, 0, 255).astype(np.uint8)

    def _water_reflect(self, image: np.ndarray) -> np.ndarray:
        """
        水面镜面反射：从图像下方投射动态波纹高光。
        模拟水面将阳光反射到船体下部的效果。
        """
        h, w = image.shape[:2]
        result = image.astype(np.float32)

        # 波纹高光图（下半部分更强）
        wave_noise = _perlin_noise(h, w, scale=0.06, octaves=3)
        # 垂直衰减：越靠近底部水面反射越强
        y_weight = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
        wave_map = wave_noise * y_weight

        # 阈值化：只有高亮区域才形成反射高光
        threshold = float(RNG.uniform(0.55, 0.75))
        glare = np.clip((wave_map - threshold) / (1.0 - threshold), 0.0, 1.0)
        glare = cv2.GaussianBlur(glare, (0, 0), 3.0)

        # 水面反射偏蓝白色
        result[:, :, 0] += glare * 180  # B
        result[:, :, 1] += glare * 170  # G
        result[:, :, 2] += glare * 140  # R

        return np.clip(result, 0, 255).astype(np.uint8)

    def _floodlight(self, image: np.ndarray) -> np.ndarray:
        """夜间泛光灯：2-4 个点光源，强局部高光，环境暗。"""
        h, w = image.shape[:2]
        # 先压暗整体（夜间环境）
        result = image.astype(np.float32) * float(RNG.uniform(0.25, 0.45))

        n_lights = int(RNG.integers(2, 5))
        for _ in range(n_lights):
            lx = int(RNG.integers(0, w))
            ly = int(RNG.integers(0, h // 2))  # 灯光在上方
            radius = int(RNG.integers(80, 250))
            intensity = float(RNG.uniform(1.5, 3.0))

            xs = np.arange(w, dtype=np.float32)
            ys = np.arange(h, dtype=np.float32)
            xg, yg = np.meshgrid(xs, ys)
            dist = np.sqrt((xg - lx) ** 2 + (yg - ly) ** 2)
            falloff = np.exp(-dist / radius) * intensity

            # 灯光偏暖黄色
            result[:, :, 0] += falloff * 200 * 0.85  # B
            result[:, :, 1] += falloff * 200 * 0.95  # G
            result[:, :, 2] += falloff * 200 * 1.00  # R

        return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# 4. 表面退化叠加
# ============================================================

def apply_surface_degradation(
    image: np.ndarray,
    rust_prob: float = 0.5,
    paint_chip_prob: float = 0.4,
    water_stain_prob: float = 0.6,
    oil_stain_prob: float = 0.3,
) -> np.ndarray:
    """
    叠加船体表面退化效果（锈蚀、油漆剥落、水渍、油污）。

    Args:
        image:           BGR uint8 输入图像
        rust_prob:       锈蚀出现概率
        paint_chip_prob: 油漆剥落出现概率
        water_stain_prob:水渍出现概率
        oil_stain_prob:  油污出现概率
    Returns:
        叠加退化后的 BGR uint8 图像
    """
    h, w = image.shape[:2]
    result = image.astype(np.float32) / 255.0

    if RNG.random() < rust_prob:
        rust, alpha = make_rust_texture(h, w)
        strength = float(RNG.uniform(0.3, 0.8))
        a = alpha[:, :, np.newaxis] * strength
        result = result * (1 - a) + rust * a

    if RNG.random() < paint_chip_prob:
        paint, alpha = make_paint_chip_texture(h, w)
        strength = float(RNG.uniform(0.4, 0.9))
        a = alpha[:, :, np.newaxis] * strength
        result = result * (1 - a) + paint * a

    if RNG.random() < water_stain_prob:
        stain, alpha = make_water_stain_texture(h, w)
        strength = float(RNG.uniform(0.2, 0.5))
        a = alpha[:, :, np.newaxis] * strength
        result = result * (1 - a) + stain * a

    if RNG.random() < oil_stain_prob:
        # 油污：深色半透明斑块
        oil_noise = _perlin_noise(h, w, scale=0.04, octaves=3)
        oil_color = np.array([0.05, 0.06, 0.07], dtype=np.float32)
        alpha = np.clip((oil_noise - 0.65) * 4.0, 0.0, 1.0)
        strength = float(RNG.uniform(0.3, 0.7))
        a = alpha[:, :, np.newaxis] * strength
        result = result * (1 - a) + oil_color * a

    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


# ============================================================
# 5. 相机噪声与成像缺陷
# ============================================================

def apply_camera_noise(
    image: np.ndarray,
    gaussian_std: float = 5.0,
    jpeg_quality: Optional[int] = None,
    motion_blur_prob: float = 0.15,
    lens_flare_prob: float = 0.2,
) -> np.ndarray:
    """
    模拟相机成像缺陷：高斯噪声、JPEG 压缩、运动模糊、镜头光晕。

    Args:
        image:            BGR uint8 输入图像
        gaussian_std:     高斯噪声标准差
        jpeg_quality:     JPEG 质量（None=不压缩，60-85=中等压缩）
        motion_blur_prob: 运动模糊出现概率
        lens_flare_prob:  镜头光晕出现概率
    Returns:
        添加噪声后的 BGR uint8 图像
    """
    result = image.astype(np.float32)

    # 高斯噪声（模拟传感器热噪声）
    if gaussian_std > 0:
        noise = RNG.normal(0, gaussian_std, result.shape).astype(np.float32)
        result += noise

    # 运动模糊（船体振动或相机抖动）
    if RNG.random() < motion_blur_prob:
        angle = float(RNG.uniform(0, 180))
        ksize = int(RNG.integers(5, 20))
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        cx, cy = ksize // 2, ksize // 2
        rad = math.radians(angle)
        for i in range(ksize):
            t = i - cx
            px = int(cx + t * math.cos(rad))
            py = int(cy + t * math.sin(rad))
            if 0 <= px < ksize and 0 <= py < ksize:
                kernel[py, px] = 1.0
        s = kernel.sum()
        if s > 0:
            kernel /= s
        result = cv2.filter2D(result, -1, kernel)

    result = np.clip(result, 0, 255).astype(np.uint8)

    # 镜头光晕（强光源产生的光斑链）
    if RNG.random() < lens_flare_prob:
        result = _add_lens_flare(result)

    # JPEG 压缩伪影
    if jpeg_quality is not None:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, enc = cv2.imencode('.jpg', result, encode_param)
        result = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return result


def _add_lens_flare(image: np.ndarray) -> np.ndarray:
    """添加镜头光晕（沿光源-图像中心连线排列的光斑）。"""
    h, w = image.shape[:2]
    result = image.astype(np.float32)
    cx, cy = w // 2, h // 2

    # 光源位置（通常在图像边缘附近）
    src_x = int(RNG.integers(0, w))
    src_y = int(RNG.integers(0, h // 3))

    n_flares = int(RNG.integers(3, 7))
    for i in range(n_flares):
        t = float(i) / n_flares
        fx = int(src_x + (cx - src_x) * t * 2)
        fy = int(src_y + (cy - src_y) * t * 2)
        r = int(RNG.integers(5, 40))
        bright = float(RNG.uniform(0.3, 0.8))
        flare = np.zeros((h, w), dtype=np.float32)
        cv2.circle(flare, (fx, fy), r, bright, -1, cv2.LINE_AA)
        flare = cv2.GaussianBlur(flare, (0, 0), r * 0.5)
        # 光晕偏蓝白
        result[:, :, 0] += flare * 200
        result[:, :, 1] += flare * 190
        result[:, :, 2] += flare * 160

    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# 6. 单样本合成
# ============================================================

LIGHTING_MODES = ['direct_sun', 'overcast', 'water_reflect', 'floodlight', 'mixed']

STRUCTURE_TYPES = [
    'hull_side',      # 舷侧大面积钢板
    'deck_surface',   # 甲板水平面
    'bow_section',    # 船头斜面
    'stern_section',  # 船尾
    'superstructure', # 上层建筑（驾驶台等）
]

# 各结构类型的基础颜色（BGR）
HULL_COLORS = {
    'hull_side':      [(120, 125, 130), (80, 85, 90), (60, 40, 35)],   # 灰/深灰/红底漆
    'deck_surface':   [(90, 95, 100), (70, 75, 80), (50, 55, 60)],
    'bow_section':    [(130, 135, 140), (100, 105, 110)],
    'stern_section':  [(115, 120, 125), (85, 90, 95)],
    'superstructure': [(200, 205, 210), (180, 185, 190), (160, 165, 170)],  # 白/浅灰
}


def synthesize_one_sample(
    h: int = 512,
    w: int = 512,
    structure_type: Optional[str] = None,
    lighting_mode: Optional[str] = None,
    roughness: Optional[float] = None,
) -> Dict:
    """
    合成一个完整训练样本。

    Returns:
        {
          'image':  BGR uint8 (H,W,3),
          'mask':   uint8 (H,W)，船体结构区域 200，边界 255，背景 0,
          'edge':   uint8 (H,W)，结构边界 255，其余 0,
          'meta':   dict，场景参数,
        }
    """
    if structure_type is None:
        structure_type = random.choice(STRUCTURE_TYPES)
    if lighting_mode is None:
        lighting_mode = random.choice(LIGHTING_MODES)
    if roughness is None:
        roughness = float(RNG.uniform(0.05, 0.35))

    # --- 基础钢板纹理 ---
    color_choices = HULL_COLORS.get(structure_type, [(130, 135, 140)])
    base_color = random.choice(color_choices)
    # 随机微调颜色
    base_color = tuple(
        int(np.clip(c + RNG.integers(-15, 15), 20, 240))
        for c in base_color
    )
    texture = make_steel_plate_texture(h, w, base_color, roughness)

    canvas = texture.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    edge = np.zeros((h, w), dtype=np.uint8)

    # --- 绘制船体面板分割线（模拟钢板拼接） ---
    n_panels_h = int(RNG.integers(1, 4))
    n_panels_v = int(RNG.integers(1, 4))
    panel_boundaries_x = sorted(
        [0] + [int(RNG.integers(w // 6, 5 * w // 6)) for _ in range(n_panels_v - 1)] + [w]
    )
    panel_boundaries_y = sorted(
        [0] + [int(RNG.integers(h // 6, 5 * h // 6)) for _ in range(n_panels_h - 1)] + [h]
    )

    for i in range(len(panel_boundaries_y) - 1):
        for j in range(len(panel_boundaries_x) - 1):
            x0 = panel_boundaries_x[j]
            x1 = panel_boundaries_x[j + 1]
            y0 = panel_boundaries_y[i]
            y1 = panel_boundaries_y[i + 1]
            pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32)
            # 每块面板颜色略有差异
            panel_color = tuple(
                int(np.clip(c + RNG.integers(-20, 20), 20, 240))
                for c in base_color
            )
            panel_tex = make_steel_plate_texture(h, w, panel_color, roughness)
            draw_hull_panel(canvas, mask, pts, panel_tex)

    # --- 绘制焊缝 ---
    n_welds = int(RNG.integers(2, 8))
    for _ in range(n_welds):
        if RNG.random() < 0.5:  # 水平焊缝
            y = int(RNG.integers(h // 8, 7 * h // 8))
            x0 = int(RNG.integers(0, w // 4))
            x1 = int(RNG.integers(3 * w // 4, w))
            draw_weld_seam(canvas, edge, x0, y, x1, y,
                           width=int(RNG.integers(4, 10)))
        else:  # 垂直焊缝
            x = int(RNG.integers(w // 8, 7 * w // 8))
            y0 = int(RNG.integers(0, h // 4))
            y1 = int(RNG.integers(3 * h // 4, h))
            draw_weld_seam(canvas, edge, x, y0, x, y1,
                           width=int(RNG.integers(4, 10)))

    # --- 绘制铆钉排 ---
    if RNG.random() < 0.6:
        n_rows = int(RNG.integers(1, 4))
        for _ in range(n_rows):
            cy_r = int(RNG.integers(h // 6, 5 * h // 6))
            spacing = int(RNG.integers(25, 60))
            x_start = int(RNG.integers(10, 40))
            cx_list = list(range(x_start, w - 10, spacing))
            r_size = int(RNG.integers(5, 14))
            draw_rivet_row(canvas, edge, cx_list, cy_r, r_size)

    # --- 绘制舷窗 ---
    if structure_type in ('hull_side', 'superstructure') and RNG.random() < 0.5:
        n_ports = int(RNG.integers(1, 4))
        for _ in range(n_ports):
            px = int(RNG.integers(w // 6, 5 * w // 6))
            py = int(RNG.integers(h // 4, 3 * h // 4))
            pr = int(RNG.integers(20, 45))
            draw_porthole(canvas, edge, px, py, pr)

    # --- 绘制管道 ---
    if RNG.random() < 0.4:
        n_pipes = int(RNG.integers(1, 3))
        for _ in range(n_pipes):
            if RNG.random() < 0.5:
                y_p = int(RNG.integers(h // 4, 3 * h // 4))
                draw_pipe(canvas, edge, 0, y_p, w, y_p,
                          radius=int(RNG.integers(8, 20)))
            else:
                x_p = int(RNG.integers(w // 4, 3 * w // 4))
                draw_pipe(canvas, edge, x_p, 0, x_p, h,
                          radius=int(RNG.integers(8, 20)))

    # --- 表面退化 ---
    canvas = apply_surface_degradation(canvas)

    # --- 光照 ---
    lighting = LightingSystem(mode=lighting_mode)
    canvas = lighting.apply(canvas)

    # --- 相机噪声 ---
    jpeg_q = int(RNG.integers(70, 95)) if RNG.random() < 0.3 else None
    canvas = apply_camera_noise(
        canvas,
        gaussian_std=float(RNG.uniform(2.0, 10.0)),
        jpeg_quality=jpeg_q,
        motion_blur_prob=0.12,
        lens_flare_prob=0.18,
    )

    # --- 整理 edge 掩膜（细化为 1-2 像素宽） ---
    edge_final = np.zeros_like(edge)
    # 从 mask 中提取边界
    mask_bin = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(edge_final, contours, -1, 255, 1)
    # 合并手工绘制的焊缝/铆钉边缘
    edge_final = cv2.bitwise_or(edge_final, edge)
    # 细化
    edge_final = cv2.dilate(edge_final, np.ones((2, 2), np.uint8), iterations=1)

    meta = {
        'structure_type': structure_type,
        'lighting_mode': lighting_mode,
        'roughness': roughness,
        'base_color_bgr': list(base_color),
        'image_size': [h, w],
    }

    return {
        'image': canvas,
        'mask': mask,
        'edge': edge_final,
        'meta': meta,
    }


# ============================================================
# 7. 数据集批量生成
# ============================================================

def generate_dataset(
    count: int,
    output_dir: str,
    img_h: int = 512,
    img_w: int = 512,
    seed: Optional[int] = None,
    preview: bool = False,
    preview_count: int = 8,
) -> None:
    """
    批量生成合成数据集并保存到磁盘。

    目录结构::

        output_dir/
          images/   0000.png, 0001.png, ...
          masks/    0000.png, ...
          edges/    0000.png, ...
          meta/     0000.json, ...
          preview/  grid_0000.png, ...  (若 preview=True)

    Args:
        count:         生成样本数量
        output_dir:    输出根目录
        img_h, img_w:  图像尺寸
        seed:          随机种子（None=随机）
        preview:       是否生成预览网格图
        preview_count: 每张预览网格包含的样本数
    """
    if seed is not None:
        set_seed(seed)

    root = Path(output_dir)
    for sub in ('images', 'masks', 'edges', 'meta'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    if preview:
        (root / 'preview').mkdir(exist_ok=True)

    preview_buffer = []

    print(f"[synth] 开始生成 {count} 个样本 → {output_dir}")
    for idx in range(count):
        sample = synthesize_one_sample(h=img_h, w=img_w)

        name = f"{idx:05d}"
        cv2.imwrite(str(root / 'images' / f"{name}.png"), sample['image'])
        cv2.imwrite(str(root / 'masks'  / f"{name}.png"), sample['mask'])
        cv2.imwrite(str(root / 'edges'  / f"{name}.png"), sample['edge'])
        with open(root / 'meta' / f"{name}.json", 'w') as f:
            json.dump(sample['meta'], f, indent=2, ensure_ascii=False)

        if preview:
            preview_buffer.append(sample)
            if len(preview_buffer) == preview_count or idx == count - 1:
                _save_preview_grid(
                    preview_buffer,
                    str(root / 'preview' / f"grid_{idx:05d}.png"),
                )
                preview_buffer = []

        if (idx + 1) % max(1, count // 10) == 0 or idx == count - 1:
            print(f"  [{idx + 1:5d}/{count}] 完成")

    print(f"[synth] 数据集生成完毕: {output_dir}")
    print(f"  images/  {count} 张")
    print(f"  masks/   {count} 张")
    print(f"  edges/   {count} 张")
    print(f"  meta/    {count} 个")


def _save_preview_grid(samples: List[Dict], output_path: str,
                       thumb_size: int = 256) -> None:
    """将一批样本拼成 image|mask|edge 三列预览网格。"""
    n = len(samples)
    cols = 3  # image, mask, edge
    cell = thumb_size
    grid = np.zeros((n * cell, cols * cell, 3), dtype=np.uint8)

    for row, s in enumerate(samples):
        img = cv2.resize(s['image'], (cell, cell))
        msk = cv2.resize(s['mask'], (cell, cell))
        edg = cv2.resize(s['edge'], (cell, cell))

        msk_vis = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
        edg_vis = cv2.cvtColor(edg, cv2.COLOR_GRAY2BGR)
        # 叠加掩膜到原图
        overlay = img.copy()
        overlay[msk > 100] = (overlay[msk > 100].astype(np.float32) * 0.6 +
                               np.array([0, 200, 0]) * 0.4).astype(np.uint8)
        overlay[edg > 128] = [0, 0, 255]

        y0 = row * cell
        grid[y0:y0 + cell, 0:cell] = img
        grid[y0:y0 + cell, cell:2 * cell] = msk_vis
        grid[y0:y0 + cell, 2 * cell:3 * cell] = edg_vis

        # 标注场景信息
        meta = s['meta']
        label = f"{meta['structure_type']} | {meta['lighting_mode']}"
        cv2.putText(grid, label, (4, y0 + cell - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 100), 1, cv2.LINE_AA)

    # 列标题
    for ci, title in enumerate(['Image', 'Mask', 'Edge']):
        cv2.putText(grid, title, (ci * cell + 4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, grid)


# ============================================================
# 8. 命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="船舶大型金属高光面合成训练数据集生成器"
    )
    parser.add_argument("--count",   type=int,   default=20,
                        help="生成样本数量（默认 20）")
    parser.add_argument("--output",  type=str,   default="./datasets/synth_ship",
                        help="输出目录")
    parser.add_argument("--height",  type=int,   default=512)
    parser.add_argument("--width",   type=int,   default=512)
    parser.add_argument("--seed",    type=int,   default=None,
                        help="随机种子（不设置则每次不同）")
    parser.add_argument("--preview", action="store_true",
                        help="生成预览网格图")
    args = parser.parse_args()

    generate_dataset(
        count=args.count,
        output_dir=args.output,
        img_h=args.height,
        img_w=args.width,
        seed=args.seed,
        preview=args.preview,
    )
