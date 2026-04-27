"""
synth_national_scenes.py — 全国大型金属高光面场景合成训练数据集生成器
:Author: RussellCooper

覆盖中国典型工业/基础设施场景中的大型金属高光面与复杂环境光：

  场景分类（SceneType）：
    1. SHIPYARD      — 船厂/造船（船体钢板、焊缝、肋骨框架）
    2. STEEL_MILL    — 钢铁厂（热轧钢卷、钢坯、连铸坯）
    3. BRIDGE        — 桥梁工程（钢箱梁、斜拉索锚具、桥面钢板）
    4. PORT_CRANE    — 港口起重机（吊具、集装箱顶盖、钢结构）
    5. RAILWAY       — 铁路/高铁（轨道、车体铝合金蒙皮、转向架）
    6. CURTAIN_WALL  — 建筑幕墙（玻璃幕墙钢框、铝板、不锈钢装饰）
    7. PIPELINE      — 管道/储罐（石化管道、LNG 储罐、压力容器）
    8. WIND_TURBINE  — 风电（塔筒、叶片根部法兰、机舱盖）

  环境光类型（LightType）：
    - 强侧光（港口/工厂侧窗）
    - 正顶光（厂房天窗）
    - 水面/地面镜面反射
    - 阴天漫射光
    - 夜间泛光灯（高色温 LED）
    - 焊接弧光干扰
    - 混合多光源

用法::

    # 生成 200 张混合场景训练图像
    python3 synth_national_scenes.py --n 200 --output ./synth_national --preview

    # 生成指定场景
    python3 synth_national_scenes.py --scene STEEL_MILL --n 50 --output ./synth_steel

    # 作为模块调用
    from synth_national_scenes import NationalSceneGenerator
    gen = NationalSceneGenerator(h=512, w=512)
    sample = gen.generate(scene='BRIDGE', light='SIDE_SUN')
    cv2.imwrite('sample.png', sample['image'])
"""

import argparse
import math
import os
import random
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# 枚举定义
# ============================================================

class SceneType(str, Enum):
    SHIPYARD     = "SHIPYARD"
    STEEL_MILL   = "STEEL_MILL"
    BRIDGE       = "BRIDGE"
    PORT_CRANE   = "PORT_CRANE"
    RAILWAY      = "RAILWAY"
    CURTAIN_WALL = "CURTAIN_WALL"
    PIPELINE     = "PIPELINE"
    WIND_TURBINE = "WIND_TURBINE"


class LightType(str, Enum):
    SIDE_SUN     = "SIDE_SUN"      # 强侧光（晴天）
    TOP_SKYLIGHT = "TOP_SKYLIGHT"  # 正顶光（天窗）
    WATER_REFL   = "WATER_REFL"    # 水面/地面镜面反射
    OVERCAST     = "OVERCAST"      # 阴天漫射光
    NIGHT_LED    = "NIGHT_LED"     # 夜间 LED 泛光灯
    WELD_ARC     = "WELD_ARC"      # 焊接弧光干扰
    MIXED        = "MIXED"         # 混合多光源


# ============================================================
# 基础绘制工具
# ============================================================

class DrawUtils:
    """基础图形绘制工具集。"""

    @staticmethod
    def draw_metal_plate(
        canvas: np.ndarray,
        x: int, y: int, w: int, h: int,
        base_color: Tuple[int, int, int],
        roughness: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """绘制金属板（含表面粗糙度噪声）。"""
        if rng is None:
            rng = np.random.default_rng()
        plate = canvas.copy()
        # 基础颜色
        plate[y:y+h, x:x+w] = base_color
        # 实际裁剪区域（防止越界）
        H_canvas, W_canvas = plate.shape[:2]
        y1_clip = min(y + h, H_canvas)
        x1_clip = min(x + w, W_canvas)
        actual_h = y1_clip - y
        actual_w = x1_clip - x
        if actual_h <= 0 or actual_w <= 0:
            return plate
        # 表面粗糙度噪声
        noise = rng.normal(0, roughness * 255, (actual_h, actual_w, 3)).astype(np.int16)
        region = plate[y:y1_clip, x:x1_clip].astype(np.int16) + noise
        plate[y:y1_clip, x:x1_clip] = np.clip(region, 0, 255).astype(np.uint8)
        # 轻微渐变（模拟曲面反射）
        grad = np.linspace(0.85, 1.0, actual_w, dtype=np.float32)
        plate[y:y1_clip, x:x1_clip] = np.clip(
            plate[y:y1_clip, x:x1_clip].astype(np.float32) * grad[np.newaxis, :, np.newaxis],
            0, 255
        ).astype(np.uint8)
        return plate

    @staticmethod
    def draw_weld_seam(
        canvas: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        width: int = 4,
        color: Tuple[int, int, int] = (80, 70, 60),
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """绘制焊缝（含鱼鳞纹理）。"""
        if rng is None:
            rng = np.random.default_rng()
        result = canvas.copy()
        # 主焊缝线
        cv2.line(result, (x1, y1), (x2, y2), color, width)
        # 鱼鳞纹（沿焊缝方向绘制小椭圆）
        length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length < 1:
            return result
        dx = (x2 - x1) / length
        dy = (y2 - y1) / length
        step = width * 2
        n_scales = int(length / step)
        for i in range(n_scales):
            cx = int(x1 + dx * i * step)
            cy = int(y1 + dy * i * step)
            offset = rng.integers(-1, 2)
            cv2.ellipse(result, (cx + offset, cy + offset),
                        (width, width // 2), math.degrees(math.atan2(dy, dx)),
                        0, 180, color, 1)
        return result

    @staticmethod
    def draw_rivet_pattern(
        canvas: np.ndarray,
        region: Tuple[int, int, int, int],
        spacing: int = 30,
        radius: int = 5,
        color: Tuple[int, int, int] = (100, 95, 90),
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """绘制铆钉阵列。"""
        if rng is None:
            rng = np.random.default_rng()
        result = canvas.copy()
        x0, y0, x1, y1 = region
        for ry in range(y0 + spacing // 2, y1, spacing):
            for rx in range(x0 + spacing // 2, x1, spacing):
                jx = int(rng.integers(-3, 4))
                jy = int(rng.integers(-3, 4))
                cx, cy = rx + jx, ry + jy
                # 铆钉本体（深色圆）
                cv2.circle(result, (cx, cy), radius, color, -1)
                # 高光点（模拟凸起反射）
                hx = cx - radius // 3
                hy = cy - radius // 3
                cv2.circle(result, (hx, hy), max(1, radius // 3),
                           (220, 215, 210), -1)
        return result

    @staticmethod
    def draw_specular_highlight(
        canvas: np.ndarray,
        cx: int, cy: int,
        rx: int, ry: int,
        intensity: float = 1.0,
        angle: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """绘制镜面高光（椭圆形，高斯衰减）。"""
        if rng is None:
            rng = np.random.default_rng()
        h, w = canvas.shape[:2]
        result = canvas.copy().astype(np.float32)

        # 生成高斯高光掩膜
        Y, X = np.ogrid[:h, :w]
        # 旋转坐标
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        dx = (X - cx) * cos_a + (Y - cy) * sin_a
        dy = -(X - cx) * sin_a + (Y - cy) * cos_a
        dist2 = (dx / max(rx, 1)) ** 2 + (dy / max(ry, 1)) ** 2
        mask = np.exp(-dist2 * 2.0) * intensity

        # 叠加高光（趋向白色）
        for c in range(3):
            result[:, :, c] = np.clip(
                result[:, :, c] + mask * (255 - result[:, :, c]),
                0, 255
            )
        return result.astype(np.uint8)

    @staticmethod
    def add_rust_texture(
        canvas: np.ndarray,
        region: Tuple[int, int, int, int],
        intensity: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """添加锈蚀纹理（棕红色噪声）。"""
        if rng is None:
            rng = np.random.default_rng()
        result = canvas.copy()
        x0, y0, x1, y1 = region
        # 裁剪到画布范围内
        H_canvas, W_canvas = result.shape[:2]
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(W_canvas, x1); y1 = min(H_canvas, y1)
        h, w = y1 - y0, x1 - x0
        if h <= 0 or w <= 0:
            return result

        # 生成 Perlin-like 噪声（用多尺度高斯近似）
        noise = np.zeros((h, w), dtype=np.float32)
        for scale in [4, 8, 16, 32]:
            small = rng.random((max(1, h // scale), max(1, w // scale)))
            resized = cv2.resize(small.astype(np.float32), (w, h),
                                 interpolation=cv2.INTER_LINEAR)
            noise += resized / scale

        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        rust_mask = (noise > 0.6).astype(np.float32) * intensity

        # 锈蚀颜色（棕红色）
        rust_color = np.array([30, 60, 140], dtype=np.float32)  # BGR
        region_f = result[y0:y1, x0:x1].astype(np.float32)
        for c in range(3):
            region_f[:, :, c] = (
                region_f[:, :, c] * (1 - rust_mask) +
                rust_color[c] * rust_mask
            )
        result[y0:y1, x0:x1] = np.clip(region_f, 0, 255).astype(np.uint8)
        return result


# ============================================================
# 场景生成器基类
# ============================================================

class BaseSceneGenerator:
    """所有场景生成器的基类。"""

    # 典型金属颜色（BGR）
    STEEL_GRAY    = (140, 140, 145)
    STEEL_BRIGHT  = (180, 182, 185)
    ALUMINUM      = (195, 200, 205)
    PAINTED_STEEL = (100, 110, 120)
    HOT_STEEL     = (60, 100, 200)   # 热轧钢（偏红橙）
    GALVANIZED    = (160, 165, 160)  # 镀锌钢

    def __init__(self, h: int = 512, w: int = 512,
                 rng: Optional[np.random.Generator] = None):
        self.h   = h
        self.w   = w
        self.rng = rng if rng is not None else np.random.default_rng()
        self.draw = DrawUtils()

    def generate_background(self, light_type: LightType) -> np.ndarray:
        """生成背景（含环境光）。"""
        bg = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        if light_type == LightType.OVERCAST:
            # 阴天：均匀灰白
            val = int(self.rng.integers(160, 200))
            bg[:] = (val, val, val)
        elif light_type == LightType.NIGHT_LED:
            # 夜间：深蓝黑背景
            bg[:] = (20, 15, 10)
        elif light_type == LightType.WATER_REFL:
            # 水面反射：深蓝绿渐变
            for i in range(self.h):
                t = i / self.h
                bg[i] = (int(40 + t * 30), int(60 + t * 20), int(80 + t * 40))
        elif light_type == LightType.WELD_ARC:
            # 焊接弧光：暗背景
            bg[:] = (15, 12, 10)
        else:
            # 晴天/侧光/顶光：浅灰
            val = int(self.rng.integers(140, 180))
            bg[:] = (val, val, val)
            # 添加轻微渐变
            grad_x = np.linspace(0.9, 1.1, self.w, dtype=np.float32)
            bg = np.clip(
                bg.astype(np.float32) * grad_x[np.newaxis, :, np.newaxis],
                0, 255
            ).astype(np.uint8)

        return bg

    def apply_lighting(
        self,
        image: np.ndarray,
        light_type: LightType,
        n_highlights: int = 3,
    ) -> np.ndarray:
        """在图像上叠加环境光效果。"""
        result = image.copy()

        if light_type == LightType.SIDE_SUN:
            # 强侧光：左侧或右侧强光，产生大面积高光
            side = self.rng.choice(['left', 'right'])
            cx = int(self.rng.integers(0, self.w // 4)) if side == 'left' \
                 else int(self.rng.integers(3 * self.w // 4, self.w))
            cy = int(self.rng.integers(self.h // 4, 3 * self.h // 4))
            for _ in range(n_highlights):
                rx = int(self.rng.integers(self.w // 4, self.w // 2))
                ry = int(self.rng.integers(self.h // 8, self.h // 3))
                intensity = float(self.rng.uniform(0.6, 1.0))
                angle = float(self.rng.uniform(-20, 20))
                result = DrawUtils.draw_specular_highlight(
                    result, cx + int(self.rng.integers(-50, 51)),
                    cy + int(self.rng.integers(-50, 51)),
                    rx, ry, intensity, angle, self.rng
                )

        elif light_type == LightType.TOP_SKYLIGHT:
            # 顶光：从上方均匀照射，中央亮
            for _ in range(n_highlights):
                cx = int(self.rng.integers(self.w // 4, 3 * self.w // 4))
                cy = int(self.rng.integers(0, self.h // 3))
                rx = int(self.rng.integers(self.w // 6, self.w // 3))
                ry = int(self.rng.integers(self.h // 4, self.h // 2))
                intensity = float(self.rng.uniform(0.5, 0.9))
                result = DrawUtils.draw_specular_highlight(
                    result, cx, cy, rx, ry, intensity, 0.0, self.rng
                )

        elif light_type == LightType.WATER_REFL:
            # 水面反射：下方多个动态高光斑
            for _ in range(n_highlights + 2):
                cx = int(self.rng.integers(0, self.w))
                cy = int(self.rng.integers(self.h // 2, self.h))
                rx = int(self.rng.integers(20, 80))
                ry = int(self.rng.integers(10, 40))
                intensity = float(self.rng.uniform(0.4, 0.9))
                angle = float(self.rng.uniform(-45, 45))
                result = DrawUtils.draw_specular_highlight(
                    result, cx, cy, rx, ry, intensity, angle, self.rng
                )

        elif light_type == LightType.NIGHT_LED:
            # 夜间 LED：多个圆形强光点
            n_lamps = int(self.rng.integers(2, 5))
            for _ in range(n_lamps):
                cx = int(self.rng.integers(0, self.w))
                cy = int(self.rng.integers(0, self.h // 3))
                rx = int(self.rng.integers(30, 100))
                ry = int(self.rng.integers(30, 100))
                intensity = float(self.rng.uniform(0.7, 1.0))
                result = DrawUtils.draw_specular_highlight(
                    result, cx, cy, rx, ry, intensity, 0.0, self.rng
                )

        elif light_type == LightType.WELD_ARC:
            # 焊接弧光：局部极亮蓝白光
            cx = int(self.rng.integers(self.w // 4, 3 * self.w // 4))
            cy = int(self.rng.integers(self.h // 4, 3 * self.h // 4))
            # 弧光核心（极亮）
            result = DrawUtils.draw_specular_highlight(
                result, cx, cy, 30, 30, 1.0, 0.0, self.rng
            )
            # 弧光晕（蓝白色）
            result_f = result.astype(np.float32)
            Y, X = np.ogrid[:self.h, :self.w]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            halo = np.exp(-dist / 80) * 0.6
            result_f[:, :, 0] = np.clip(result_f[:, :, 0] + halo * 200, 0, 255)
            result_f[:, :, 1] = np.clip(result_f[:, :, 1] + halo * 220, 0, 255)
            result_f[:, :, 2] = np.clip(result_f[:, :, 2] + halo * 255, 0, 255)
            result = result_f.astype(np.uint8)

        elif light_type == LightType.MIXED:
            # 混合光：随机叠加 2~3 种光源
            sub_lights = self.rng.choice(
                [LightType.SIDE_SUN, LightType.TOP_SKYLIGHT,
                 LightType.WATER_REFL, LightType.NIGHT_LED],
                size=int(self.rng.integers(2, 4)), replace=False
            )
            for sl in sub_lights:
                result = self.apply_lighting(result, sl,
                                             n_highlights=int(self.rng.integers(1, 3)))

        return result

    def make_mask_and_edge(
        self, canvas: np.ndarray, fg_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """从前景掩膜生成分割掩膜和边缘图。"""
        mask = (fg_mask > 0).astype(np.uint8) * 255
        # Canny 边缘
        edge = cv2.Canny(mask, 50, 150)
        # 膨胀边缘（2px）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge = cv2.dilate(edge, kernel, iterations=1)
        return mask, edge

    def add_noise_and_degradation(
        self, image: np.ndarray, light_type: LightType
    ) -> np.ndarray:
        """添加噪声、模糊、压缩伪影等退化效果。"""
        result = image.copy()

        # 高斯噪声
        sigma = float(self.rng.uniform(2, 12))
        noise = self.rng.normal(0, sigma, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 轻微运动模糊（模拟相机抖动）
        if self.rng.random() < 0.3:
            angle = float(self.rng.uniform(0, 180))
            length = int(self.rng.integers(3, 8))
            kernel = np.zeros((length, length))
            kernel[length // 2, :] = 1.0 / length
            M = cv2.getRotationMatrix2D((length // 2, length // 2), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (length, length))
            result = cv2.filter2D(result, -1, kernel)

        # JPEG 压缩伪影
        if self.rng.random() < 0.2:
            quality = int(self.rng.integers(60, 90))
            _, buf = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, quality])
            result = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        return result


# ============================================================
# 各场景专用生成器
# ============================================================

class ShipyardSceneGenerator(BaseSceneGenerator):
    """船厂/造船场景：船体钢板、焊缝、肋骨框架、船坞水面反射。"""

    SCENE_NAME = "SHIPYARD"
    DESCRIPTION = "船厂造船 — 船体钢板、焊缝、肋骨框架、船坞水面反射"

    def generate(self, light_type: LightType) -> Dict:
        bg = self.generate_background(light_type)
        canvas = bg.copy()
        fg_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # 船体钢板（大面积）
        plate_x = int(self.rng.integers(20, 80))
        plate_y = int(self.rng.integers(50, 150))
        plate_w = int(self.rng.integers(self.w // 2, 3 * self.w // 4))
        plate_h = int(self.rng.integers(self.h // 2, 3 * self.h // 4))

        # 钢板颜色（船用钢：深灰/防锈漆）
        base_col = (
            int(self.rng.integers(90, 140)),
            int(self.rng.integers(90, 140)),
            int(self.rng.integers(95, 145)),
        )
        canvas = DrawUtils.draw_metal_plate(
            canvas, plate_x, plate_y, plate_w, plate_h, base_col,
            roughness=0.04, rng=self.rng
        )
        fg_mask[plate_y:plate_y+plate_h, plate_x:plate_x+plate_w] = 255

        # 焊缝（水平 + 垂直）
        n_h_seams = int(self.rng.integers(2, 5))
        for i in range(n_h_seams):
            sy = plate_y + int((i + 1) * plate_h / (n_h_seams + 1))
            canvas = DrawUtils.draw_weld_seam(
                canvas, plate_x, sy, plate_x + plate_w, sy,
                width=int(self.rng.integers(3, 6)), rng=self.rng
            )

        n_v_seams = int(self.rng.integers(1, 4))
        for i in range(n_v_seams):
            sx = plate_x + int((i + 1) * plate_w / (n_v_seams + 1))
            canvas = DrawUtils.draw_weld_seam(
                canvas, sx, plate_y, sx, plate_y + plate_h,
                width=int(self.rng.integers(3, 6)), rng=self.rng
            )

        # 肋骨框架（T 型钢）
        n_ribs = int(self.rng.integers(2, 5))
        for i in range(n_ribs):
            rib_x = plate_x + int((i + 0.5) * plate_w / n_ribs)
            rib_color = (
                int(self.rng.integers(80, 120)),
                int(self.rng.integers(80, 120)),
                int(self.rng.integers(85, 125)),
            )
            cv2.rectangle(canvas,
                          (rib_x - 4, plate_y),
                          (rib_x + 4, plate_y + plate_h),
                          rib_color, -1)

        # 铆钉
        if self.rng.random() < 0.5:
            canvas = DrawUtils.draw_rivet_pattern(
                canvas,
                (plate_x, plate_y, plate_x + plate_w, plate_y + plate_h),
                spacing=int(self.rng.integers(25, 45)),
                radius=int(self.rng.integers(3, 6)),
                rng=self.rng
            )

        # 锈蚀（老旧船体）
        if self.rng.random() < 0.4:
            canvas = DrawUtils.add_rust_texture(
                canvas,
                (plate_x, plate_y, plate_x + plate_w, plate_y + plate_h),
                intensity=float(self.rng.uniform(0.1, 0.4)),
                rng=self.rng
            )

        # 环境光
        canvas = self.apply_lighting(canvas, light_type)
        canvas = self.add_noise_and_degradation(canvas, light_type)
        mask, edge = self.make_mask_and_edge(canvas, fg_mask)

        return {
            'image': canvas, 'mask': mask, 'edge': edge,
            'scene': self.SCENE_NAME, 'light': light_type.value,
            'description': self.DESCRIPTION,
        }


class SteelMillSceneGenerator(BaseSceneGenerator):
    """钢铁厂场景：热轧钢卷、钢坯、连铸坯、高温辐射光。"""

    SCENE_NAME = "STEEL_MILL"
    DESCRIPTION = "钢铁厂 — 热轧钢卷、钢坯、连铸坯、高温辐射光"

    def generate(self, light_type: LightType) -> Dict:
        bg = self.generate_background(light_type)
        canvas = bg.copy()
        fg_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # 钢卷（圆形截面）
        n_coils = int(self.rng.integers(1, 4))
        for i in range(n_coils):
            cx = int(self.rng.integers(self.w // 4, 3 * self.w // 4))
            cy = int(self.rng.integers(self.h // 4, 3 * self.h // 4))
            outer_r = int(self.rng.integers(60, 120))
            inner_r = int(self.rng.integers(20, 40))

            # 热轧钢颜色（偏蓝灰，含氧化层）
            is_hot = self.rng.random() < 0.3
            if is_hot:
                coil_color = (
                    int(self.rng.integers(40, 80)),
                    int(self.rng.integers(80, 130)),
                    int(self.rng.integers(180, 230)),
                )
            else:
                coil_color = (
                    int(self.rng.integers(120, 160)),
                    int(self.rng.integers(120, 160)),
                    int(self.rng.integers(125, 165)),
                )

            # 绘制钢卷（同心圆）
            n_rings = int(self.rng.integers(8, 20))
            for r_idx in range(n_rings):
                r = inner_r + (outer_r - inner_r) * r_idx // n_rings
                ring_color = tuple(
                    int(np.clip(c + self.rng.integers(-15, 16), 0, 255))
                    for c in coil_color
                )
                cv2.circle(canvas, (cx, cy), r, ring_color, 2)

            # 填充掩膜
            cv2.circle(fg_mask, (cx, cy), outer_r, 255, -1)
            cv2.circle(fg_mask, (cx, cy), inner_r, 0, -1)

            # 热辐射光晕
            if is_hot:
                canvas_f = canvas.astype(np.float32)
                Y, X = np.ogrid[:self.h, :self.w]
                dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
                halo = np.exp(-(dist - outer_r) ** 2 / (2 * 30 ** 2)) * 0.5
                halo = np.maximum(halo, 0)
                canvas_f[:, :, 2] = np.clip(canvas_f[:, :, 2] + halo * 180, 0, 255)
                canvas_f[:, :, 1] = np.clip(canvas_f[:, :, 1] + halo * 80, 0, 255)
                canvas = canvas_f.astype(np.uint8)

        # 钢坯（矩形）
        n_billets = int(self.rng.integers(1, 3))
        for _ in range(n_billets):
            bx = int(self.rng.integers(0, self.w // 2))
            by = int(self.rng.integers(self.h // 3, 2 * self.h // 3))
            bw = int(self.rng.integers(self.w // 3, 2 * self.w // 3))
            bh = int(self.rng.integers(40, 100))
            bcolor = (
                int(self.rng.integers(100, 150)),
                int(self.rng.integers(100, 150)),
                int(self.rng.integers(105, 155)),
            )
            canvas = DrawUtils.draw_metal_plate(
                canvas, bx, by, bw, bh, bcolor, roughness=0.06, rng=self.rng
            )
            fg_mask[by:by+bh, bx:bx+bw] = 255

        canvas = self.apply_lighting(canvas, light_type)
        canvas = self.add_noise_and_degradation(canvas, light_type)
        mask, edge = self.make_mask_and_edge(canvas, fg_mask)

        return {
            'image': canvas, 'mask': mask, 'edge': edge,
            'scene': self.SCENE_NAME, 'light': light_type.value,
            'description': self.DESCRIPTION,
        }


class BridgeSceneGenerator(BaseSceneGenerator):
    """桥梁工程场景：钢箱梁、斜拉索锚具、桥面钢板。"""

    SCENE_NAME = "BRIDGE"
    DESCRIPTION = "桥梁工程 — 钢箱梁、斜拉索锚具、桥面钢板"

    def generate(self, light_type: LightType) -> Dict:
        bg = self.generate_background(light_type)
        canvas = bg.copy()
        fg_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # 钢箱梁（大型矩形截面）
        beam_x = int(self.rng.integers(0, self.w // 6))
        beam_y = int(self.rng.integers(self.h // 4, self.h // 3))
        beam_w = self.w - 2 * beam_x
        beam_h = int(self.rng.integers(self.h // 4, self.h // 3))

        beam_color = (
            int(self.rng.integers(110, 150)),
            int(self.rng.integers(110, 150)),
            int(self.rng.integers(115, 155)),
        )
        canvas = DrawUtils.draw_metal_plate(
            canvas, beam_x, beam_y, beam_w, beam_h,
            beam_color, roughness=0.03, rng=self.rng
        )
        fg_mask[beam_y:beam_y+beam_h, beam_x:beam_x+beam_w] = 255

        # 加劲肋（竖向）
        n_ribs = int(self.rng.integers(4, 10))
        for i in range(n_ribs):
            rx = beam_x + int((i + 1) * beam_w / (n_ribs + 1))
            rib_color = tuple(
                int(np.clip(c - 20, 0, 255)) for c in beam_color
            )
            cv2.rectangle(canvas,
                          (rx - 3, beam_y),
                          (rx + 3, beam_y + beam_h),
                          rib_color, -1)

        # 焊缝（水平）
        n_seams = int(self.rng.integers(1, 4))
        for i in range(n_seams):
            sy = beam_y + int((i + 1) * beam_h / (n_seams + 1))
            canvas = DrawUtils.draw_weld_seam(
                canvas, beam_x, sy, beam_x + beam_w, sy,
                width=5, rng=self.rng
            )

        # 斜拉索锚具（圆形）
        n_anchors = int(self.rng.integers(2, 6))
        for i in range(n_anchors):
            ax = beam_x + int((i + 0.5) * beam_w / n_anchors)
            ay = beam_y - int(self.rng.integers(20, 60))
            ar = int(self.rng.integers(15, 30))
            anchor_color = (
                int(self.rng.integers(130, 170)),
                int(self.rng.integers(130, 170)),
                int(self.rng.integers(135, 175)),
            )
            cv2.circle(canvas, (ax, ay), ar, anchor_color, -1)
            cv2.circle(fg_mask, (ax, ay), ar, 255, -1)
            # 锚具高光
            canvas = DrawUtils.draw_specular_highlight(
                canvas, ax - ar // 3, ay - ar // 3,
                ar // 2, ar // 2, 0.7, 0.0, self.rng
            )

        # 桥面钢板（顶部）
        deck_h = int(self.rng.integers(30, 60))
        canvas = DrawUtils.draw_metal_plate(
            canvas, beam_x, beam_y - deck_h, beam_w, deck_h,
            self.GALVANIZED, roughness=0.05, rng=self.rng
        )
        fg_mask[beam_y - deck_h:beam_y, beam_x:beam_x + beam_w] = 255

        canvas = self.apply_lighting(canvas, light_type)
        canvas = self.add_noise_and_degradation(canvas, light_type)
        mask, edge = self.make_mask_and_edge(canvas, fg_mask)

        return {
            'image': canvas, 'mask': mask, 'edge': edge,
            'scene': self.SCENE_NAME, 'light': light_type.value,
            'description': self.DESCRIPTION,
        }


class PortCraneSceneGenerator(BaseSceneGenerator):
    """港口起重机场景：吊具、集装箱顶盖、钢结构桁架。"""

    SCENE_NAME = "PORT_CRANE"
    DESCRIPTION = "港口起重机 — 吊具、集装箱顶盖、钢结构桁架"

    def generate(self, light_type: LightType) -> Dict:
        bg = self.generate_background(light_type)
        canvas = bg.copy()
        fg_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # 集装箱顶盖（矩形，波纹钢板）
        n_containers = int(self.rng.integers(1, 4))
        for i in range(n_containers):
            cx = int(self.rng.integers(0, self.w - 150))
            cy = int(self.rng.integers(self.h // 3, 2 * self.h // 3))
            cw = int(self.rng.integers(120, 200))
            ch = int(self.rng.integers(60, 100))
            # 集装箱颜色（红/蓝/绿/黄）
            container_colors = [
                (50, 50, 180), (180, 50, 50), (50, 150, 50), (50, 180, 180)
            ]
            cc = container_colors[int(self.rng.integers(0, 4))]
            canvas = DrawUtils.draw_metal_plate(
                canvas, cx, cy, cw, ch, cc, roughness=0.04, rng=self.rng
            )
            fg_mask[cy:cy+ch, cx:cx+cw] = 255

            # 波纹（水平线）
            n_corrugations = int(self.rng.integers(5, 12))
            for j in range(n_corrugations):
                wy = cy + int(j * ch / n_corrugations)
                cv2.line(canvas, (cx, wy), (cx + cw, wy),
                         tuple(int(c * 0.85) for c in cc), 1)

        # 桁架结构（斜向钢杆）
        n_trusses = int(self.rng.integers(3, 7))
        for _ in range(n_trusses):
            tx1 = int(self.rng.integers(0, self.w))
            ty1 = int(self.rng.integers(0, self.h // 3))
            tx2 = int(self.rng.integers(0, self.w))
            ty2 = int(self.rng.integers(self.h // 3, self.h))
            truss_color = (
                int(self.rng.integers(100, 140)),
                int(self.rng.integers(100, 140)),
                int(self.rng.integers(105, 145)),
            )
            cv2.line(canvas, (tx1, ty1), (tx2, ty2), truss_color,
                     int(self.rng.integers(4, 10)))
            # 桁架掩膜
            cv2.line(fg_mask, (tx1, ty1), (tx2, ty2), 255,
                     int(self.rng.integers(4, 10)))

        canvas = self.apply_lighting(canvas, light_type)
        canvas = self.add_noise_and_degradation(canvas, light_type)
        mask, edge = self.make_mask_and_edge(canvas, fg_mask)

        return {
            'image': canvas, 'mask': mask, 'edge': edge,
            'scene': self.SCENE_NAME, 'light': light_type.value,
            'description': self.DESCRIPTION,
        }


class RailwaySceneGenerator(BaseSceneGenerator):
    """铁路/高铁场景：轨道、车体铝合金蒙皮、转向架。"""

    SCENE_NAME = "RAILWAY"
    DESCRIPTION = "铁路/高铁 — 轨道、车体铝合金蒙皮、转向架"

    def generate(self, light_type: LightType) -> Dict:
        bg = self.generate_background(light_type)
        canvas = bg.copy()
        fg_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # 车体铝合金蒙皮（大面积，高反光）
        body_y = int(self.rng.integers(self.h // 4, self.h // 3))
        body_h = int(self.rng.integers(self.h // 3, self.h // 2))
        canvas = DrawUtils.draw_metal_plate(
            canvas, 0, body_y, self.w, body_h,
            self.ALUMINUM, roughness=0.02, rng=self.rng
        )
        fg_mask[body_y:body_y+body_h, :] = 255

        # 车窗（暗色矩形）
        n_windows = int(self.rng.integers(3, 7))
        win_w = int(self.rng.integers(50, 80))
        win_h = int(self.rng.integers(35, 55))
        win_y = body_y + (body_h - win_h) // 2
        for i in range(n_windows):
            win_x = int((i + 0.3) * self.w / n_windows)
            win_color = (
                int(self.rng.integers(30, 60)),
                int(self.rng.integers(40, 70)),
                int(self.rng.integers(50, 80)),
            )
            cv2.rectangle(canvas, (win_x, win_y),
                          (win_x + win_w, win_y + win_h), win_color, -1)
            # 窗框
            cv2.rectangle(canvas, (win_x - 3, win_y - 3),
                          (win_x + win_w + 3, win_y + win_h + 3),
                          (160, 160, 165), 3)

        # 轨道（底部）
        rail_y = body_y + body_h + int(self.rng.integers(10, 30))
        for rail_offset in [-self.w // 6, self.w // 6]:
            rail_cx = self.w // 2 + rail_offset
            rail_color = (100, 100, 105)
            cv2.rectangle(canvas,
                          (rail_cx - 8, rail_y),
                          (rail_cx + 8, min(self.h, rail_y + 30)),
                          rail_color, -1)
            fg_mask[rail_y:min(self.h, rail_y + 30),
                    rail_cx - 8:rail_cx + 8] = 255

        # 转向架（车底复杂结构）
        bogie_y = body_y + body_h - 20
        bogie_color = (80, 80, 85)
        for bx in [self.w // 4, 3 * self.w // 4]:
            cv2.ellipse(canvas, (bx, bogie_y + 20),
                        (40, 20), 0, 0, 360, bogie_color, -1)
            cv2.ellipse(fg_mask, (bx, bogie_y + 20),
                        (40, 20), 0, 0, 360, 255, -1)

        canvas = self.apply_lighting(canvas, light_type)
        canvas = self.add_noise_and_degradation(canvas, light_type)
        mask, edge = self.make_mask_and_edge(canvas, fg_mask)

        return {
            'image': canvas, 'mask': mask, 'edge': edge,
            'scene': self.SCENE_NAME, 'light': light_type.value,
            'description': self.DESCRIPTION,
        }


class CurtainWallSceneGenerator(BaseSceneGenerator):
    """建筑幕墙场景：玻璃幕墙钢框、铝板、不锈钢装饰条。"""

    SCENE_NAME = "CURTAIN_WALL"
    DESCRIPTION = "建筑幕墙 — 玻璃幕墙钢框、铝板、不锈钢装饰条"

    def generate(self, light_type: LightType) -> Dict:
        bg = self.generate_background(light_type)
        canvas = bg.copy()
        fg_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # 幕墙网格
        n_cols = int(self.rng.integers(3, 7))
        n_rows = int(self.rng.integers(3, 6))
        frame_w = 8
        frame_color = (
            int(self.rng.integers(150, 200)),
            int(self.rng.integers(150, 200)),
            int(self.rng.integers(155, 205)),
        )

        cell_w = self.w // n_cols
        cell_h = self.h // n_rows

        for row in range(n_rows):
            for col in range(n_cols):
                x0 = col * cell_w
                y0 = row * cell_h
                x1 = x0 + cell_w
                y1 = y0 + cell_h

                # 玻璃填充（深蓝绿，含天空反射）
                glass_color = (
                    int(self.rng.integers(60, 100)),
                    int(self.rng.integers(80, 120)),
                    int(self.rng.integers(100, 140)),
                )
                cv2.rectangle(canvas, (x0 + frame_w, y0 + frame_w),
                              (x1 - frame_w, y1 - frame_w), glass_color, -1)

                # 铝合金框架
                cv2.rectangle(canvas, (x0, y0), (x1, y1), frame_color, frame_w)
                fg_mask[y0:y1, x0:x1] = 255

                # 玻璃反射高光（随机）
                if self.rng.random() < 0.4:
                    hcx = int(self.rng.integers(x0 + frame_w, x1 - frame_w))
                    hcy = int(self.rng.integers(y0 + frame_w, y1 - frame_w))
                    canvas = DrawUtils.draw_specular_highlight(
                        canvas, hcx, hcy,
                        int(self.rng.integers(10, 40)),
                        int(self.rng.integers(5, 20)),
                        float(self.rng.uniform(0.4, 0.9)),
                        float(self.rng.uniform(-30, 30)),
                        self.rng
                    )

        # 不锈钢装饰条（水平）
        n_strips = int(self.rng.integers(1, 4))
        for i in range(n_strips):
            sy = int(self.rng.integers(0, self.h))
            strip_color = (
                int(self.rng.integers(170, 210)),
                int(self.rng.integers(170, 210)),
                int(self.rng.integers(175, 215)),
            )
            cv2.rectangle(canvas, (0, sy), (self.w, sy + 6), strip_color, -1)
            fg_mask[sy:sy + 6, :] = 255

        canvas = self.apply_lighting(canvas, light_type)
        canvas = self.add_noise_and_degradation(canvas, light_type)
        mask, edge = self.make_mask_and_edge(canvas, fg_mask)

        return {
            'image': canvas, 'mask': mask, 'edge': edge,
            'scene': self.SCENE_NAME, 'light': light_type.value,
            'description': self.DESCRIPTION,
        }


class PipelineSceneGenerator(BaseSceneGenerator):
    """管道/储罐场景：石化管道、LNG 储罐、压力容器。"""

    SCENE_NAME = "PIPELINE"
    DESCRIPTION = "管道/储罐 — 石化管道、LNG储罐、压力容器"

    def generate(self, light_type: LightType) -> Dict:
        bg = self.generate_background(light_type)
        canvas = bg.copy()
        fg_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # 大型储罐（圆柱体正视图）
        tank_cx = int(self.rng.integers(self.w // 4, 3 * self.w // 4))
        tank_cy = int(self.rng.integers(self.h // 3, 2 * self.h // 3))
        tank_rx = int(self.rng.integers(80, 150))
        tank_ry = int(self.rng.integers(60, 120))
        tank_color = (
            int(self.rng.integers(150, 200)),
            int(self.rng.integers(150, 200)),
            int(self.rng.integers(155, 205)),
        )

        # 储罐本体
        cv2.ellipse(canvas, (tank_cx, tank_cy), (tank_rx, tank_ry),
                    0, 0, 360, tank_color, -1)
        cv2.ellipse(fg_mask, (tank_cx, tank_cy), (tank_rx, tank_ry),
                    0, 0, 360, 255, -1)

        # 储罐高光（圆柱镜面反射）
        canvas = DrawUtils.draw_specular_highlight(
            canvas,
            tank_cx - tank_rx // 3, tank_cy - tank_ry // 3,
            tank_rx // 3, tank_ry // 2,
            float(self.rng.uniform(0.6, 0.95)),
            float(self.rng.uniform(-20, 20)),
            self.rng
        )

        # 管道（水平/竖向）
        n_pipes = int(self.rng.integers(2, 6))
        for _ in range(n_pipes):
            is_horizontal = self.rng.random() < 0.6
            pipe_r = int(self.rng.integers(10, 30))
            pipe_color = (
                int(self.rng.integers(120, 170)),
                int(self.rng.integers(120, 170)),
                int(self.rng.integers(125, 175)),
            )
            if is_horizontal:
                py = int(self.rng.integers(0, self.h))
                cv2.rectangle(canvas, (0, py - pipe_r), (self.w, py + pipe_r),
                              pipe_color, -1)
                fg_mask[max(0, py - pipe_r):min(self.h, py + pipe_r), :] = 255
                # 管道高光线
                canvas = DrawUtils.draw_specular_highlight(
                    canvas, self.w // 2, py - pipe_r // 2,
                    self.w // 3, pipe_r // 3,
                    float(self.rng.uniform(0.5, 0.9)), 0.0, self.rng
                )
            else:
                px = int(self.rng.integers(0, self.w))
                cv2.rectangle(canvas, (px - pipe_r, 0), (px + pipe_r, self.h),
                              pipe_color, -1)
                fg_mask[:, max(0, px - pipe_r):min(self.w, px + pipe_r)] = 255

        # 法兰（圆形连接件）
        n_flanges = int(self.rng.integers(1, 4))
        for _ in range(n_flanges):
            fx = int(self.rng.integers(50, self.w - 50))
            fy = int(self.rng.integers(50, self.h - 50))
            fr = int(self.rng.integers(20, 45))
            flange_color = (
                int(self.rng.integers(130, 170)),
                int(self.rng.integers(130, 170)),
                int(self.rng.integers(135, 175)),
            )
            cv2.circle(canvas, (fx, fy), fr, flange_color, -1)
            cv2.circle(canvas, (fx, fy), fr - 5, (80, 80, 85), -1)
            cv2.circle(fg_mask, (fx, fy), fr, 255, -1)

        canvas = self.apply_lighting(canvas, light_type)
        canvas = self.add_noise_and_degradation(canvas, light_type)
        mask, edge = self.make_mask_and_edge(canvas, fg_mask)

        return {
            'image': canvas, 'mask': mask, 'edge': edge,
            'scene': self.SCENE_NAME, 'light': light_type.value,
            'description': self.DESCRIPTION,
        }


class WindTurbineSceneGenerator(BaseSceneGenerator):
    """风电场景：塔筒、叶片根部法兰、机舱盖。"""

    SCENE_NAME = "WIND_TURBINE"
    DESCRIPTION = "风电 — 塔筒、叶片根部法兰、机舱盖"

    def generate(self, light_type: LightType) -> Dict:
        bg = self.generate_background(light_type)
        canvas = bg.copy()
        fg_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # 塔筒（锥形，底宽顶窄）
        tower_top_w = int(self.rng.integers(40, 70))
        tower_bot_w = int(self.rng.integers(80, 130))
        tower_cx = self.w // 2
        tower_top_y = int(self.rng.integers(0, self.h // 4))
        tower_bot_y = self.h

        pts = np.array([
            [tower_cx - tower_top_w // 2, tower_top_y],
            [tower_cx + tower_top_w // 2, tower_top_y],
            [tower_cx + tower_bot_w // 2, tower_bot_y],
            [tower_cx - tower_bot_w // 2, tower_bot_y],
        ], dtype=np.int32)

        tower_color = (
            int(self.rng.integers(200, 240)),
            int(self.rng.integers(200, 240)),
            int(self.rng.integers(200, 240)),
        )
        cv2.fillPoly(canvas, [pts], tower_color)
        cv2.fillPoly(fg_mask, [pts], 255)

        # 塔筒高光（竖向高光线）
        canvas = DrawUtils.draw_specular_highlight(
            canvas,
            tower_cx - tower_top_w // 4,
            (tower_top_y + tower_bot_y) // 2,
            tower_top_w // 4,
            (tower_bot_y - tower_top_y) // 2,
            float(self.rng.uniform(0.4, 0.8)),
            90.0, self.rng
        )

        # 机舱盖（顶部椭圆）
        nacelle_cx = tower_cx
        nacelle_cy = tower_top_y + int(self.rng.integers(20, 50))
        nacelle_rx = int(self.rng.integers(50, 80))
        nacelle_ry = int(self.rng.integers(25, 40))
        nacelle_color = (
            int(self.rng.integers(180, 220)),
            int(self.rng.integers(180, 220)),
            int(self.rng.integers(180, 220)),
        )
        cv2.ellipse(canvas, (nacelle_cx, nacelle_cy),
                    (nacelle_rx, nacelle_ry), 0, 0, 360, nacelle_color, -1)
        cv2.ellipse(fg_mask, (nacelle_cx, nacelle_cy),
                    (nacelle_rx, nacelle_ry), 0, 0, 360, 255, -1)

        # 叶片根部法兰（大圆）
        hub_r = int(self.rng.integers(30, 50))
        cv2.circle(canvas, (nacelle_cx, nacelle_cy), hub_r,
                   (160, 160, 165), -1)
        cv2.circle(fg_mask, (nacelle_cx, nacelle_cy), hub_r, 255, -1)

        # 叶片（3 片，120° 间隔）
        blade_len = int(self.rng.integers(100, 180))
        for angle_deg in [90, 210, 330]:
            angle_rad = math.radians(angle_deg)
            bx = int(nacelle_cx + blade_len * math.cos(angle_rad))
            by = int(nacelle_cy + blade_len * math.sin(angle_rad))
            blade_pts = self._make_blade(
                nacelle_cx, nacelle_cy, bx, by,
                width=int(self.rng.integers(12, 20))
            )
            blade_color = (
                int(self.rng.integers(200, 240)),
                int(self.rng.integers(200, 240)),
                int(self.rng.integers(200, 240)),
            )
            cv2.fillPoly(canvas, [blade_pts], blade_color)
            cv2.fillPoly(fg_mask, [blade_pts], 255)

        canvas = self.apply_lighting(canvas, light_type)
        canvas = self.add_noise_and_degradation(canvas, light_type)
        mask, edge = self.make_mask_and_edge(canvas, fg_mask)

        return {
            'image': canvas, 'mask': mask, 'edge': edge,
            'scene': self.SCENE_NAME, 'light': light_type.value,
            'description': self.DESCRIPTION,
        }

    @staticmethod
    def _make_blade(x1, y1, x2, y2, width):
        """生成叶片多边形顶点。"""
        dx = x2 - x1
        dy = y2 - y1
        length = max(math.sqrt(dx**2 + dy**2), 1)
        nx = -dy / length * width / 2
        ny = dx / length * width / 2
        return np.array([
            [int(x1 + nx), int(y1 + ny)],
            [int(x1 - nx), int(y1 - ny)],
            [int(x2), int(y2)],
        ], dtype=np.int32)


# ============================================================
# 主生成器（统一入口）
# ============================================================

class NationalSceneGenerator:
    """
    全国大型金属高光面场景统一生成器。

    用法::

        gen = NationalSceneGenerator(h=512, w=512, seed=42)

        # 生成指定场景
        sample = gen.generate(scene='BRIDGE', light='SIDE_SUN')

        # 随机生成
        sample = gen.generate_random()

        # 批量生成
        gen.generate_dataset(n=500, output_dir='./synth_national')
    """

    # 场景权重（反映实际工业需求频率）
    SCENE_WEIGHTS = {
        SceneType.SHIPYARD:     0.18,
        SceneType.STEEL_MILL:   0.15,
        SceneType.BRIDGE:       0.15,
        SceneType.PORT_CRANE:   0.12,
        SceneType.RAILWAY:      0.15,
        SceneType.CURTAIN_WALL: 0.10,
        SceneType.PIPELINE:     0.10,
        SceneType.WIND_TURBINE: 0.05,
    }

    # 光照权重
    LIGHT_WEIGHTS = {
        LightType.SIDE_SUN:     0.25,
        LightType.TOP_SKYLIGHT: 0.15,
        LightType.WATER_REFL:   0.15,
        LightType.OVERCAST:     0.20,
        LightType.NIGHT_LED:    0.10,
        LightType.WELD_ARC:     0.05,
        LightType.MIXED:        0.10,
    }

    _GENERATORS = {
        SceneType.SHIPYARD:     ShipyardSceneGenerator,
        SceneType.STEEL_MILL:   SteelMillSceneGenerator,
        SceneType.BRIDGE:       BridgeSceneGenerator,
        SceneType.PORT_CRANE:   PortCraneSceneGenerator,
        SceneType.RAILWAY:      RailwaySceneGenerator,
        SceneType.CURTAIN_WALL: CurtainWallSceneGenerator,
        SceneType.PIPELINE:     PipelineSceneGenerator,
        SceneType.WIND_TURBINE: WindTurbineSceneGenerator,
    }

    def __init__(self, h: int = 512, w: int = 512, seed: Optional[int] = None):
        self.h    = h
        self.w    = w
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

        # 实例化所有场景生成器
        self._gens = {
            scene: cls(h=h, w=w, rng=np.random.default_rng(
                seed + i if seed is not None else None
            ))
            for i, (scene, cls) in enumerate(self._GENERATORS.items())
        }

    def generate(
        self,
        scene: Optional[str] = None,
        light: Optional[str] = None,
    ) -> Dict:
        """
        生成单张训练样本。

        Args:
            scene: 场景名称（None = 按权重随机）
            light: 光照类型（None = 按权重随机）

        Returns:
            Dict 包含：
              image:       (H, W, 3) BGR uint8
              mask:        (H, W) uint8 分割掩膜
              edge:        (H, W) uint8 边缘图
              scene:       场景名称字符串
              light:       光照类型字符串
              description: 场景描述
        """
        # 随机选择场景
        if scene is None:
            scenes = list(self.SCENE_WEIGHTS.keys())
            weights = list(self.SCENE_WEIGHTS.values())
            scene_type = self._py_rng.choices(scenes, weights=weights, k=1)[0]
        else:
            scene_type = SceneType(scene)

        # 随机选择光照
        if light is None:
            lights = list(self.LIGHT_WEIGHTS.keys())
            weights = list(self.LIGHT_WEIGHTS.values())
            light_type = self._py_rng.choices(lights, weights=weights, k=1)[0]
        else:
            light_type = LightType(light)

        return self._gens[scene_type].generate(light_type)

    def generate_random(self) -> Dict:
        """完全随机生成一张样本。"""
        return self.generate()

    def generate_dataset(
        self,
        n: int,
        output_dir: str,
        preview: bool = False,
        verbose: bool = True,
    ) -> Dict:
        """
        批量生成训练数据集。

        目录结构::

            output_dir/
              images/    *.png  原始图像
              masks/     *.png  分割掩膜（0/255）
              edges/     *.png  边缘图（0/255）
              labels/    *.json 元数据（场景/光照/描述）
              preview/   *.png  可视化预览（可选）

        Args:
            n:          生成样本数量
            output_dir: 输出目录
            preview:    是否保存可视化预览
            verbose:    是否打印进度

        Returns:
            统计信息 Dict
        """
        out = Path(output_dir)
        for sub in ['images', 'masks', 'edges', 'labels']:
            (out / sub).mkdir(parents=True, exist_ok=True)
        if preview:
            (out / 'preview').mkdir(exist_ok=True)

        stats = {s.value: 0 for s in SceneType}
        stats_light = {l.value: 0 for l in LightType}

        for i in range(n):
            sample = self.generate_random()
            stem = f"{i:06d}_{sample['scene']}_{sample['light']}"

            cv2.imwrite(str(out / 'images' / f"{stem}.png"), sample['image'])
            cv2.imwrite(str(out / 'masks'  / f"{stem}.png"), sample['mask'])
            cv2.imwrite(str(out / 'edges'  / f"{stem}.png"), sample['edge'])

            import json
            meta = {
                'id':          stem,
                'scene':       sample['scene'],
                'light':       sample['light'],
                'description': sample['description'],
                'h': self.h, 'w': self.w,
            }
            with open(str(out / 'labels' / f"{stem}.json"), 'w',
                      encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            if preview:
                vis = self._make_preview(sample)
                cv2.imwrite(str(out / 'preview' / f"{stem}.png"), vis)

            stats[sample['scene']] += 1
            stats_light[sample['light']] += 1

            if verbose and (i + 1) % 20 == 0:
                print(f"  [{i+1:4d}/{n}] 已生成 {i+1} 张")

        if verbose:
            print(f"\n数据集生成完成 → {output_dir}")
            print(f"  总计: {n} 张")
            print("  场景分布:")
            for k, v in stats.items():
                print(f"    {k:15s}: {v:4d} 张 ({100*v/n:.1f}%)")
            print("  光照分布:")
            for k, v in stats_light.items():
                print(f"    {k:15s}: {v:4d} 张 ({100*v/n:.1f}%)")

        return {'total': n, 'scene_stats': stats, 'light_stats': stats_light}

    @staticmethod
    def _make_preview(sample: Dict) -> np.ndarray:
        """生成 2x2 预览图（原图 + 掩膜 + 边缘 + 叠加）。"""
        img  = sample['image']
        mask = cv2.cvtColor(sample['mask'], cv2.COLOR_GRAY2BGR)
        edge = cv2.cvtColor(sample['edge'], cv2.COLOR_GRAY2BGR)

        # 叠加图
        overlay = img.copy()
        overlay[sample['mask'] > 0] = (
            overlay[sample['mask'] > 0] * 0.6 +
            np.array([0, 200, 0]) * 0.4
        ).astype(np.uint8)
        overlay[sample['edge'] > 0] = [0, 0, 255]

        # 添加标签
        def put_label(im, text):
            cv2.putText(im, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            return im

        img_l  = put_label(img.copy(),     f"Scene: {sample['scene']}")
        mask_l = put_label(mask.copy(),    f"Light: {sample['light']}")
        edge_l = put_label(edge.copy(),    "Edge Map")
        over_l = put_label(overlay.copy(), "Overlay")

        top    = np.hstack([img_l, mask_l])
        bottom = np.hstack([edge_l, over_l])
        return np.vstack([top, bottom])


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="全国大型金属高光面场景合成训练数据集生成器 | @author RussellCooper"
    )
    parser.add_argument("--n",       type=int,   default=50,
                        help="生成样本数量（默认 50）")
    parser.add_argument("--output",  type=str,   default="./synth_national",
                        help="输出目录")
    parser.add_argument("--scene",   type=str,   default=None,
                        choices=[s.value for s in SceneType],
                        help="指定场景（默认随机）")
    parser.add_argument("--light",   type=str,   default=None,
                        choices=[l.value for l in LightType],
                        help="指定光照类型（默认随机）")
    parser.add_argument("--h",       type=int,   default=512)
    parser.add_argument("--w",       type=int,   default=512)
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--preview", action="store_true",
                        help="保存可视化预览图")
    args = parser.parse_args()

    print("=" * 60)
    print("  全国大型金属高光面场景合成数据集生成器")
    print(f"  @author RussellCooper")
    print(f"  场景: {args.scene or '随机'}  光照: {args.light or '随机'}")
    print(f"  分辨率: {args.h}x{args.w}  数量: {args.n}")
    print("=" * 60)

    gen = NationalSceneGenerator(h=args.h, w=args.w, seed=args.seed)
    gen.generate_dataset(
        n=args.n,
        output_dir=args.output,
        preview=args.preview,
        verbose=True,
    )


if __name__ == "__main__":
    main()
