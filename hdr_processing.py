"""
hdr_processing.py — 高反光工件专用 HDR 处理与图像增强模块
:Author: RussellCooper

完整可运行实现，覆盖：
  1. 多重曝光融合  — Mertens (无需曝光时间) + Debevec (需曝光时间)
  2. 合成多曝光生成 — 从单张图像模拟多 EV 档位（测试/数据增强用）
  3. 高光检测与修复 — 多通道过曝检测 + OpenCV inpaint / 软混合修复
  4. 偏振模拟       — 多帧最小值法 + 双边滤波镜面/漫反射分离
  5. 自适应增强     — CLAHE (LAB 空间) + 引导滤波 + Unsharp Mask
  6. AntiGlarePipeline — 统一管线，支持单帧/多帧输入

依赖: opencv-contrib-python>=4.5, numpy>=1.21
运行: python3 hdr_processing.py [--input img.jpg] [--output ./out] [--debug]
"""

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. 多重曝光融合
# ---------------------------------------------------------------------------

def exposure_fusion_mertens(
    images: List[np.ndarray],
    contrast_weight: float = 1.0,
    saturation_weight: float = 1.0,
    exposure_weight: float = 1.0,
) -> np.ndarray:
    """
    Mertens 多重曝光融合（无需曝光时间元数据）。

    Args:
        images:            BGR uint8 图像列表
        contrast_weight:   对比度权重（增大可增强边缘）
        saturation_weight: 饱和度权重
        exposure_weight:   曝光适度权重
    Returns:
        融合后的 BGR uint8 图像
    """
    if not images:
        raise ValueError("images 列表不能为空")
    if len(images) == 1:
        return images[0].copy()

    h0, w0 = images[0].shape[:2]
    aligned = []
    for img in images:
        if img.shape[:2] != (h0, w0):
            img = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_LINEAR)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        aligned.append(img)

    merger = cv2.createMergeMertens(contrast_weight, saturation_weight, exposure_weight)
    fused_f = merger.process(aligned)
    return np.clip(fused_f * 255.0, 0, 255).astype(np.uint8)


def exposure_fusion_debevec(
    images: List[np.ndarray],
    exposure_times: List[float],
    tonemap_gamma: float = 1.5,
) -> np.ndarray:
    """
    Debevec HDR 重建 + Reinhard 色调映射（需要曝光时间）。

    Args:
        images:         BGR uint8 图像列表
        exposure_times: 对应曝光时间列表（秒）
        tonemap_gamma:  Reinhard gamma 值
    Returns:
        融合后的 BGR uint8 图像
    """
    if len(images) != len(exposure_times):
        raise ValueError("images 与 exposure_times 长度必须一致")

    aligned = []
    for img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        aligned.append(img)

    times = np.array(exposure_times, dtype=np.float32)

    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(aligned, times)

    merge_hdr = cv2.createMergeDebevec()
    hdr = merge_hdr.process(aligned, times, response)

    tonemap = cv2.createTonemapReinhard(
        gamma=tonemap_gamma, intensity=0.0, light_adapt=0.8, color_adapt=0.0
    )
    ldr = tonemap.process(hdr)
    return np.clip(ldr * 255.0, 0, 255).astype(np.uint8)


def generate_synthetic_exposures(
    image: np.ndarray,
    ev_stops: List[float] = (-2.0, 0.0, 2.0),
) -> Tuple[List[np.ndarray], List[float]]:
    """
    从单张图像合成多曝光序列（用于无真实多曝光数据时的测试/增强）。

    通过 gamma 变换模拟不同 EV 档位，并加入符合实际的泊松噪声。

    Args:
        image:    BGR uint8 图像
        ev_stops: EV 档位列表，0 = 正常曝光
    Returns:
        (images_list, exposure_times_list)
    """
    base_time = 0.01  # 10 ms 基准曝光
    imgs_out, times_out = [], []
    img_f = image.astype(np.float32) / 255.0

    for ev in ev_stops:
        scale = 2.0 ** ev
        adj = np.clip(img_f * scale, 0.0, 1.0)
        # 模拟相机 gamma 响应 (sRGB ≈ 2.2)
        adj_gamma = np.power(np.clip(adj, 1e-6, 1.0), 1.0 / 2.2)
        # 加入轻微泊松噪声（模拟真实传感器）
        noisy = np.random.poisson(adj_gamma * 255.0).astype(np.float32) / 255.0
        noisy = np.clip(noisy, 0.0, 1.0)
        imgs_out.append((noisy * 255.0).astype(np.uint8))
        times_out.append(float(base_time * scale))

    return imgs_out, times_out


# ---------------------------------------------------------------------------
# 2. 高光检测与修复
# ---------------------------------------------------------------------------

def detect_highlight_mask(
    image: np.ndarray,
    threshold: int = 240,
    dilate_iters: int = 2,
) -> np.ndarray:
    """
    检测过曝（高光）区域，返回二值掩膜。

    任意通道超过 threshold 即标记为高光，并膨胀以覆盖过渡区。

    Args:
        image:        BGR uint8 图像
        threshold:    亮度阈值 0-255
        dilate_iters: 膨胀迭代次数
    Returns:
        highlight_mask: uint8 掩膜，高光区域为 255
    """
    b, g, r = cv2.split(image)
    mask = ((b.astype(np.int32) > threshold) |
            (g.astype(np.int32) > threshold) |
            (r.astype(np.int32) > threshold)).astype(np.uint8) * 255

    # 形态学：先闭运算填孔，再膨胀扩边
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    if dilate_iters > 0:
        mask = cv2.dilate(mask, k5, iterations=dilate_iters)
    return mask


def repair_highlight_regions(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    threshold: int = 240,
    method: str = 'telea',
    inpaint_radius: int = 5,
) -> np.ndarray:
    """
    修复高光区域，用周边纹理填充。

    Args:
        image:          BGR uint8 图像
        mask:           高光掩膜（None 则自动检测）
        threshold:      自动检测阈值
        method:         'telea' | 'ns' | 'blend'
        inpaint_radius: inpaint 修复半径
    Returns:
        修复后的 BGR uint8 图像
    """
    if mask is None:
        mask = detect_highlight_mask(image, threshold)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.max() == 0:
        return image.copy()

    if method == 'blend':
        # 软混合：用高斯模糊版本替代高光区域，边缘平滑过渡
        blurred = cv2.GaussianBlur(image, (31, 31), 0)
        alpha = cv2.GaussianBlur(
            (mask / 255.0).astype(np.float32), (21, 21), 0
        )[:, :, np.newaxis]
        result = image.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
        return np.clip(result, 0, 255).astype(np.uint8)
    else:
        flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
        return cv2.inpaint(image, mask, inpaint_radius, flag)


# ---------------------------------------------------------------------------
# 3. 偏振模拟 — 镜面反射抑制
# ---------------------------------------------------------------------------

def polarization_min_method(images: List[np.ndarray]) -> np.ndarray:
    """
    多帧逐像素最小值法模拟偏振去反光。

    原理：镜面反射强度随角度剧烈变化，漫反射相对稳定；
    取多帧最小值可有效压制高光峰值。

    Args:
        images: BGR uint8 图像列表（不同角度或曝光）
    Returns:
        去反光后的 BGR uint8 图像
    """
    if not images:
        raise ValueError("images 不能为空")
    if len(images) == 1:
        return images[0].copy()
    stack = np.stack([img.astype(np.float32) for img in images], axis=0)
    return np.min(stack, axis=0).astype(np.uint8)


def specular_diffuse_separation(
    image: np.ndarray,
    bilateral_d: int = 15,
    sigma_color: float = 40.0,
    sigma_space: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    双边滤波镜面/漫反射分离。

    漫反射 = 双边滤波低频成分；镜面反射 = 原图 - 漫反射（偏移 0.5 防负值）。

    Args:
        image:        BGR uint8 图像
        bilateral_d:  双边滤波邻域直径
        sigma_color:  颜色空间 sigma
        sigma_space:  坐标空间 sigma
    Returns:
        (diffuse, specular) 均为 BGR uint8
    """
    img_f = image.astype(np.float32) / 255.0
    diffuse_f = cv2.bilateralFilter(img_f, bilateral_d, sigma_color / 255.0, sigma_space)
    specular_f = np.clip(img_f - diffuse_f + 0.5, 0.0, 1.0)
    diffuse = (diffuse_f * 255.0).astype(np.uint8)
    specular = (specular_f * 255.0).astype(np.uint8)
    return diffuse, specular


# ---------------------------------------------------------------------------
# 4. 自适应图像增强
# ---------------------------------------------------------------------------

def apply_clahe_lab(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    在 LAB 颜色空间 L 通道上应用 CLAHE，避免颜色失真。

    clip_limit 根据图像对比度自动微调：低对比度场景自动上调。
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # 自适应 clip_limit：低对比度时增强更多
    std = float(l.std())
    auto_clip = clip_limit * max(0.5, min(2.0, 60.0 / (std + 1e-3)))

    clahe = cv2.createCLAHE(clipLimit=auto_clip, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
    return result


def guided_filter_opencv(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 8,
    eps: float = 100.0,
) -> np.ndarray:
    """
    边缘保持引导滤波（优先使用 ximgproc，回退到双边滤波）。

    Args:
        guide:  引导图像 BGR uint8
        src:    待滤波图像 BGR uint8
        radius: 滤波半径
        eps:    正则化参数（越大越平滑，典型值 100-10000）
    Returns:
        滤波后的 BGR uint8 图像
    """
    g_f = guide.astype(np.float32) / 255.0
    s_f = src.astype(np.float32) / 255.0
    try:
        out_f = cv2.ximgproc.guidedFilter(g_f, s_f, radius, eps / (255.0 ** 2))
    except AttributeError:
        # ximgproc 不可用时退回双边滤波
        out_f = cv2.bilateralFilter(s_f, radius * 2 + 1,
                                    eps / 255.0, float(radius))
    return np.clip(out_f * 255.0, 0, 255).astype(np.uint8)


def unsharp_mask(
    image: np.ndarray,
    ksize: int = 5,
    sigma: float = 1.0,
    amount: float = 1.5,
    threshold: int = 8,
) -> np.ndarray:
    """
    非锐化掩膜（Unsharp Mask）增强边缘细节。

    只对差值超过 threshold 的像素进行锐化，避免噪声放大。
    """
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)
    diff = image.astype(np.int16) - blurred.astype(np.int16)
    strong = (np.abs(diff) > threshold).astype(np.float32)
    sharpened = image.astype(np.float32) + amount * diff * strong
    return np.clip(sharpened, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 5. 完整反光抑制管线
# ---------------------------------------------------------------------------

class AntiGlarePipeline:
    """
    高反光工件图像处理管线。

    处理顺序：
        多曝光合成 → HDR 融合 → 高光修复 → 镜面分离 → CLAHE → 引导滤波 → 锐化

    Examples::

        pipeline = AntiGlarePipeline()
        # 单张图像（自动合成多曝光）
        result = pipeline.process_single(bgr_image)
        # 真实多曝光图像
        result = pipeline.process_multi(images, exposure_times)
    """

    def __init__(
        self,
        highlight_threshold: int = 235,
        repair_method: str = 'telea',
        clahe_clip: float = 2.0,
        guided_radius: int = 8,
        guided_eps: float = 100.0,
        sharpen_amount: float = 1.2,
        specular_blend: float = 0.7,
    ):
        self.highlight_threshold = highlight_threshold
        self.repair_method = repair_method
        self.clahe_clip = clahe_clip
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps
        self.sharpen_amount = sharpen_amount
        self.specular_blend = specular_blend

    def process_single(
        self,
        image: np.ndarray,
        ev_stops: List[float] = (-2.0, 0.0, 2.0),
    ) -> np.ndarray:
        """单张图像输入（自动合成多曝光序列后处理）。"""
        imgs, times = generate_synthetic_exposures(image, ev_stops)
        return self.process_multi(imgs, times)

    def process_multi(
        self,
        images: List[np.ndarray],
        exposure_times: Optional[List[float]] = None,
    ) -> np.ndarray:
        """多张图像输入。"""
        # Step 1: HDR 融合
        if exposure_times and len(exposure_times) == len(images):
            fused = exposure_fusion_debevec(images, exposure_times)
        else:
            fused = exposure_fusion_mertens(images)

        # Step 2: 高光修复
        hl_mask = detect_highlight_mask(fused, self.highlight_threshold)
        if hl_mask.max() > 0:
            fused = repair_highlight_regions(fused, hl_mask, method=self.repair_method)

        # Step 3: 镜面反射抑制（漫反射 + 原图软混合）
        diffuse, _ = specular_diffuse_separation(fused)
        result = cv2.addWeighted(fused, 1.0 - self.specular_blend,
                                 diffuse, self.specular_blend, 0)

        # Step 4: CLAHE 对比度增强
        result = apply_clahe_lab(result, self.clahe_clip)

        # Step 5: 引导滤波（保边去噪）
        result = guided_filter_opencv(result, result,
                                      self.guided_radius, self.guided_eps)

        # Step 6: 非锐化掩膜增强边缘
        result = unsharp_mask(result, amount=self.sharpen_amount)

        return result

    def get_debug_stages(
        self,
        image: np.ndarray,
        ev_stops: List[float] = (-2.0, 0.0, 2.0),
    ) -> Dict[str, np.ndarray]:
        """返回各处理阶段中间结果，用于调试可视化。"""
        stages: Dict[str, np.ndarray] = {'00_original': image.copy()}

        imgs, times = generate_synthetic_exposures(image, ev_stops)
        for i, (ev, img) in enumerate(zip(ev_stops, imgs)):
            stages[f'01_ev{ev:+.0f}'] = img.copy()

        fused = exposure_fusion_mertens(imgs)
        stages['02_hdr_fused'] = fused.copy()

        hl_mask = detect_highlight_mask(fused, self.highlight_threshold)
        stages['03_highlight_mask'] = cv2.cvtColor(hl_mask, cv2.COLOR_GRAY2BGR)

        repaired = repair_highlight_regions(fused, hl_mask, method=self.repair_method)
        stages['04_repaired'] = repaired.copy()

        diffuse, specular = specular_diffuse_separation(repaired)
        stages['05_diffuse'] = diffuse.copy()
        stages['06_specular'] = specular.copy()

        clahe_out = apply_clahe_lab(repaired, self.clahe_clip)
        stages['07_clahe'] = clahe_out.copy()

        stages['08_final'] = self.process_single(image, ev_stops)
        return stages


# ---------------------------------------------------------------------------
# 6. 调试可视化工具
# ---------------------------------------------------------------------------

def save_debug_grid(
    stages: Dict[str, np.ndarray],
    output_path: str,
    thumb_w: int = 320,
    thumb_h: int = 240,
    max_cols: int = 4,
) -> None:
    """将各阶段图像拼成网格保存。"""
    names = sorted(stages.keys())
    n = len(names)
    cols = min(n, max_cols)
    rows = math.ceil(n / cols)

    cell_h = thumb_h + 28
    grid = np.zeros((rows * cell_h, cols * thumb_w, 3), dtype=np.uint8)

    for idx, name in enumerate(names):
        r, c = divmod(idx, cols)
        img = stages[name]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        thumb = cv2.resize(img, (thumb_w, thumb_h))
        y0, x0 = r * cell_h, c * thumb_w
        grid[y0:y0 + thumb_h, x0:x0 + thumb_w] = thumb
        cv2.putText(grid, name, (x0 + 4, y0 + thumb_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 200), 1,
                    cv2.LINE_AA)

    cv2.imwrite(output_path, grid)
    print(f"[hdr] 调试网格已保存: {output_path}")


# ---------------------------------------------------------------------------
# 7. 命令行入口
# ---------------------------------------------------------------------------

def _make_test_image() -> np.ndarray:
    """生成带高光的合成金属工件测试图像。"""
    img = np.full((480, 640, 3), 55, dtype=np.uint8)
    # 工件主体
    cv2.rectangle(img, (80, 90), (560, 390), (170, 175, 185), -1)
    cv2.rectangle(img, (80, 90), (560, 390), (90, 95, 105), 3)
    # 模拟高光斑（多个，不同强度）
    for (cx, cy, r, bright) in [(220, 170, 50, 255), (400, 280, 70, 248),
                                  (310, 220, 35, 252), (480, 150, 25, 250)]:
        for dr in range(r, 0, -4):
            a = 1.0 - dr / r
            v = int(bright * a + 170 * (1 - a))
            cv2.circle(img, (cx, cy), dr, (v, v, min(v + 10, 255)), -1)
    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高反光工件 HDR 处理管线")
    parser.add_argument("--input", type=str, default=None,
                        help="输入图像路径（不提供则使用合成测试图）")
    parser.add_argument("--output", type=str, default="./hdr_output",
                        help="输出目录")
    parser.add_argument("--debug", action="store_true",
                        help="保存各阶段调试网格图")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.input and os.path.exists(args.input):
        image = cv2.imread(args.input)
        print(f"[hdr] 加载图像: {args.input}  shape={image.shape}")
    else:
        print("[hdr] 使用合成测试图像")
        image = _make_test_image()

    pipeline = AntiGlarePipeline()
    result = pipeline.process_single(image)

    out_img = os.path.join(args.output, "result.png")
    cv2.imwrite(out_img, result)
    print(f"[hdr] 处理结果: {out_img}")

    if args.debug:
        stages = pipeline.get_debug_stages(image)
        save_debug_grid(stages, os.path.join(args.output, "debug_stages.png"))

    print("[hdr] 完成。")
