"""
hdr_processing.py — 高反光工件专用 HDR 处理与图像增强模块

优化内容：
  1. 多重曝光融合 — Mertens 算法 + 自适应权重调节
  2. 高光检测与修复 — 自动检测过曝区域并修复
  3. 偏振模拟 — 多帧最小值法抑制镜面反射
  4. 自适应图像增强 — CLAHE + 引导滤波 + 自适应参数
  5. 反光抑制管线 — 整合所有处理步骤的统一接口
"""

import cv2
import numpy as np


# ===========================================================================
# 1. 多重曝光融合
# ===========================================================================

def exposure_fusion(images, contrast_weight=1.0, saturation_weight=1.0,
                    exposure_weight=1.0):
    """
    使用 Mertens 算法进行多重曝光融合，支持自定义权重。

    参数:
        images:            多张不同曝光的 BGR 图像列表
        contrast_weight:   对比度权重 (增大可增强边缘)
        saturation_weight: 饱和度权重
        exposure_weight:   曝光适度权重

    返回:
        融合后的 8-bit BGR 图像
    """
    if len(images) == 0:
        raise ValueError("输入图像列表不能为空")

    if len(images) == 1:
        return images[0].copy()

    # 确保所有图像尺寸一致
    target_shape = images[0].shape
    aligned_images = []
    for img in images:
        if img.shape != target_shape:
            img = cv2.resize(img, (target_shape[1], target_shape[0]))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        aligned_images.append(img)

    merge_mertens = cv2.createMergeMertens(
        contrast_weight, saturation_weight, exposure_weight
    )
    fused = merge_mertens.process(aligned_images)

    # 转换为 8-bit 并处理溢出
    fused_8bit = np.clip(fused * 255, 0, 255).astype(np.uint8)
    return fused_8bit


def weighted_exposure_fusion(images, exposure_times=None):
    """
    加权多重曝光融合 — 根据曝光时间自动调整融合权重。

    如果提供了曝光时间，使用 Debevec 方法估计 HDR 辐照度图后再色调映射；
    否则退回到 Mertens 融合。

    参数:
        images:         多张不同曝光的 BGR 图像列表
        exposure_times: 对应的曝光时间列表 (秒)，可选

    返回:
        融合后的 8-bit BGR 图像
    """
    if exposure_times is None or len(exposure_times) != len(images):
        return exposure_fusion(images)

    # 确保图像为 BGR 格式
    aligned = []
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        aligned.append(img)

    times = np.array(exposure_times, dtype=np.float32)

    # Debevec 方法估计相机响应函数
    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(aligned, times)

    # 合并为 HDR 图像
    merge = cv2.createMergeDebevec()
    hdr = merge.process(aligned, times, response)

    # Reinhard 色调映射
    tonemap = cv2.createTonemapReinhard(gamma=1.5, intensity=0, light_adapt=0.8,
                                         color_adapt=0.0)
    ldr = tonemap.process(hdr)
    ldr_8bit = np.clip(ldr * 255, 0, 255).astype(np.uint8)

    return ldr_8bit


# ===========================================================================
# 2. 高光检测与修复
# ===========================================================================

def detect_specular_regions(image, threshold=240, min_area=50):
    """
    检测图像中的高光 (过曝) 区域。

    参数:
        image:     BGR 图像
        threshold: 亮度阈值 (0-255)
        min_area:  最小高光区域面积 (像素)

    返回:
        glare_mask: 高光区域二值掩膜 (H, W), uint8
        glare_info: 高光区域信息列表 [{center, area, bbox}]
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 多通道高光检测 (任一通道过曝即为高光)
    if len(image.shape) == 3:
        b, g, r = cv2.split(image)
        glare_mask = ((b > threshold) | (g > threshold) | (r > threshold)).astype(np.uint8) * 255
    else:
        glare_mask = (gray > threshold).astype(np.uint8) * 255

    # 形态学处理：闭运算填充小孔，开运算去除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, kernel)

    # 提取连通域
    contours, _ = cv2.findContours(glare_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glare_info = []
    filtered_mask = np.zeros_like(glare_mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0
            x, y, w, h = cv2.boundingRect(cnt)
            glare_info.append({
                'center': (cx, cy),
                'area': area,
                'bbox': (x, y, w, h),
            })
            cv2.drawContours(filtered_mask, [cnt], -1, 255, -1)

    return filtered_mask, glare_info


def repair_specular_regions(image, glare_mask, method='inpaint'):
    """
    修复高光区域。

    参数:
        image:      BGR 图像
        glare_mask: 高光区域掩膜
        method:     修复方法 ('inpaint', 'interpolate', 'blend')

    返回:
        修复后的图像
    """
    if glare_mask.max() == 0:
        return image.copy()

    if method == 'inpaint':
        # OpenCV 图像修复 (Navier-Stokes 方法)
        repaired = cv2.inpaint(image, glare_mask, inpaintRadius=5,
                               flags=cv2.INPAINT_NS)

    elif method == 'interpolate':
        # 使用周围像素的中值进行插值修复
        repaired = image.copy()
        # 膨胀高光区域获取边界
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dilated = cv2.dilate(glare_mask, kernel)
        border = dilated - glare_mask

        # 计算边界区域的平均颜色
        border_pixels = image[border > 0]
        if len(border_pixels) > 0:
            mean_color = border_pixels.mean(axis=0).astype(np.uint8)
            repaired[glare_mask > 0] = mean_color

    elif method == 'blend':
        # 高斯模糊混合 — 用模糊版本替代高光区域
        blurred = cv2.GaussianBlur(image, (31, 31), 0)
        mask_float = (glare_mask / 255.0).astype(np.float32)
        # 软边缘混合
        mask_soft = cv2.GaussianBlur(mask_float, (15, 15), 0)
        mask_3ch = np.stack([mask_soft] * 3, axis=-1)
        repaired = (image.astype(np.float32) * (1 - mask_3ch) +
                    blurred.astype(np.float32) * mask_3ch)
        repaired = np.clip(repaired, 0, 255).astype(np.uint8)

    else:
        repaired = image.copy()

    return repaired


# ===========================================================================
# 3. 偏振模拟
# ===========================================================================

def simulate_polarization_effect(images, method='min'):
    """
    模拟偏振效果抑制镜面反射。

    参数:
        images: 多张图像列表 (不同光照/角度)
        method: 'min' (最小值), 'median' (中值), 'robust_min' (鲁棒最小值)

    返回:
        抑制反射后的图像
    """
    if len(images) == 0:
        raise ValueError("输入图像列表不能为空")
    if len(images) == 1:
        return images[0].copy()

    imgs_float = [img.astype(np.float32) for img in images]

    if method == 'min':
        result = np.minimum.reduce(imgs_float)

    elif method == 'median':
        stacked = np.stack(imgs_float, axis=0)
        result = np.median(stacked, axis=0)

    elif method == 'robust_min':
        # 鲁棒最小值：去掉最大值和最小值后取平均
        stacked = np.stack(imgs_float, axis=0)
        if len(images) >= 3:
            sorted_stack = np.sort(stacked, axis=0)
            # 去掉最大和最小
            result = sorted_stack[1:-1].mean(axis=0)
        else:
            result = np.minimum.reduce(imgs_float)

    else:
        result = np.minimum.reduce(imgs_float)

    return np.clip(result, 0, 255).astype(np.uint8)


# ===========================================================================
# 4. 自适应图像增强
# ===========================================================================

def adaptive_image_enhancement(image, clip_limit=None, tile_size=None,
                                guided_radius=10, guided_eps=100):
    """
    自适应图像增强：CLAHE + 引导滤波/双边滤波。

    优化：根据图像亮度统计自动调节 CLAHE 参数。

    参数:
        image:         BGR 图像
        clip_limit:    CLAHE 裁剪限制 (None=自动)
        tile_size:     CLAHE 块大小 (None=自动)
        guided_radius: 引导滤波半径
        guided_eps:    引导滤波正则化参数

    返回:
        增强后的图像
    """
    # 转换到 LAB 色彩空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 自适应参数计算
    if clip_limit is None:
        # 根据亮度通道的标准差自动调节
        l_std = l_channel.std()
        if l_std < 30:
            clip_limit = 3.0   # 低对比度场景，增强更多
        elif l_std < 60:
            clip_limit = 2.0   # 正常场景
        else:
            clip_limit = 1.5   # 高对比度场景 (可能有强反光)

    if tile_size is None:
        h, w = l_channel.shape
        tile_size = (max(4, w // 40), max(4, h // 40))

    # CLAHE 增强
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l_channel)

    # 合并回 LAB 并转换回 BGR
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # 引导滤波 / 双边滤波 (保边去噪)
    try:
        from cv2.ximgproc import guidedFilter
        enhanced = guidedFilter(
            guide=enhanced, src=enhanced,
            radius=guided_radius, eps=guided_eps
        )
    except ImportError:
        enhanced = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    return enhanced


def enhance_for_edge_detection(image):
    """
    专门为边缘检测优化的增强流程。

    步骤：
    1. 高光检测与修复
    2. CLAHE 增强对比度
    3. 锐化边缘
    4. 降噪保边

    返回:
        增强后的图像 (适合边缘检测)
    """
    # 1. 检测并修复高光
    glare_mask, _ = detect_specular_regions(image, threshold=235)
    if glare_mask.max() > 0:
        image = repair_specular_regions(image, glare_mask, method='blend')

    # 2. CLAHE 增强
    image = adaptive_image_enhancement(image)

    # 3. 锐化
    kernel_sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    sharpened = cv2.filter2D(image, -1, kernel_sharpen)
    # 混合：70% 锐化 + 30% 原图
    image = cv2.addWeighted(sharpened, 0.7, image, 0.3, 0)

    # 4. 轻微降噪
    image = cv2.GaussianBlur(image, (3, 3), 0.5)

    return image


# ===========================================================================
# 5. 反光抑制统一管线
# ===========================================================================

class AntiGlarePipeline:
    """
    反光抑制统一管线 — 整合所有 HDR 处理步骤。

    用法:
        pipeline = AntiGlarePipeline()
        result = pipeline.process(raw_images)
        # result['enhanced'] — 增强后的图像
        # result['glare_mask'] — 高光区域掩膜
        # result['edge_ready'] — 适合边缘检测的图像
    """

    def __init__(self, glare_threshold=240, repair_method='blend',
                 use_polarization=True, polarization_method='min'):
        self.glare_threshold = glare_threshold
        self.repair_method = repair_method
        self.use_polarization = use_polarization
        self.polarization_method = polarization_method

    def process(self, images):
        """
        处理多张输入图像。

        参数:
            images: BGR 图像列表 (多重曝光或多帧)

        返回:
            dict: {
                'fused':      HDR 融合结果,
                'enhanced':   增强后的图像,
                'glare_mask': 高光区域掩膜,
                'glare_info': 高光区域信息,
                'edge_ready': 适合边缘检测的图像,
                'polarized':  偏振模拟结果 (如果启用),
            }
        """
        result = {}

        # 1. 偏振模拟 (如果有多帧)
        if self.use_polarization and len(images) > 1:
            polarized = simulate_polarization_effect(
                images, method=self.polarization_method
            )
            result['polarized'] = polarized
        else:
            result['polarized'] = None

        # 2. HDR 融合
        fused = exposure_fusion(images)
        result['fused'] = fused

        # 3. 高光检测
        glare_mask, glare_info = detect_specular_regions(
            fused, threshold=self.glare_threshold
        )
        result['glare_mask'] = glare_mask
        result['glare_info'] = glare_info

        # 4. 高光修复
        repaired = repair_specular_regions(fused, glare_mask, method=self.repair_method)

        # 5. 自适应增强
        enhanced = adaptive_image_enhancement(repaired)
        result['enhanced'] = enhanced

        # 6. 边缘检测专用增强
        edge_ready = enhance_for_edge_detection(repaired)
        result['edge_ready'] = edge_ready

        return result

    def process_single(self, image):
        """处理单张图像 (无 HDR 融合)。"""
        return self.process([image])


# ===========================================================================
# 6. 入口点
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HDR 处理与反光抑制模块测试")
    print("=" * 60)

    # 创建模拟测试图像
    h, w = 480, 640
    base = np.zeros((h, w, 3), dtype=np.uint8) + 80

    # 添加工件
    cv2.circle(base, (320, 240), 100, (180, 180, 190), -1)

    # 模拟不同曝光
    img_under = np.clip(base.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)
    img_normal = base.copy()
    img_over = np.clip(base.astype(np.float32) * 1.8, 0, 255).astype(np.uint8)
    # 添加高光斑到过曝图像
    cv2.circle(img_over, (300, 220), 30, (255, 255, 255), -1)

    images = [img_under, img_normal, img_over]

    # 测试各功能
    print("\n1. 测试曝光融合...")
    fused = exposure_fusion(images)
    print(f"   融合结果: {fused.shape}, dtype={fused.dtype}")

    print("\n2. 测试高光检测...")
    glare_mask, glare_info = detect_specular_regions(img_over)
    print(f"   高光区域: {len(glare_info)} 个")
    for info in glare_info:
        print(f"   - 中心: {info['center']}, 面积: {info['area']}")

    print("\n3. 测试高光修复...")
    for method in ['inpaint', 'interpolate', 'blend']:
        repaired = repair_specular_regions(img_over, glare_mask, method=method)
        print(f"   {method}: {repaired.shape}")

    print("\n4. 测试偏振模拟...")
    for method in ['min', 'median', 'robust_min']:
        pol = simulate_polarization_effect(images, method=method)
        print(f"   {method}: mean={pol.mean():.1f}")

    print("\n5. 测试自适应增强...")
    enhanced = adaptive_image_enhancement(fused)
    print(f"   增强结果: {enhanced.shape}")

    print("\n6. 测试反光抑制管线...")
    pipeline = AntiGlarePipeline()
    result = pipeline.process(images)
    for key, val in result.items():
        if isinstance(val, np.ndarray):
            print(f"   {key}: {val.shape}")
        else:
            print(f"   {key}: {val}")

    print("\nHDR 处理模块测试通过！")
