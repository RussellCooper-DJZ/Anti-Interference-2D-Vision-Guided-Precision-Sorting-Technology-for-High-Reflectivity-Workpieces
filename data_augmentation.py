"""
data_augmentation.py — 高反光工件专用数据增强管线

核心功能：
  1. 高光模拟 (Specular Glare Synthesis) — 在训练图像上添加逼真的高光斑
  2. 油污/指纹模拟 — 模拟工业环境中金属表面的油污和指纹
  3. 多光照条件模拟 — 模拟不同光源方向和强度
  4. 金属表面纹理增强 — 模拟拉丝/磨砂/镜面等金属质感
  5. 标准几何/颜色增强 — 翻转、旋转、亮度、对比度等
  6. 边缘标注自动生成 — 从分割 mask 自动生成边缘 ground truth

所有增强均同步作用于图像和 mask，保证标注一致性。
"""

import cv2
import numpy as np
import os
from pathlib import Path

try:
    import albumentations as A
    from albumentations.core.transforms_interface import ImageOnlyTransform
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Warning: albumentations not installed. Using basic augmentation only.")


# ===========================================================================
# 1. 高反光专用增强变换
# ===========================================================================

class SpecularGlareSynthesis:
    """
    在图像上合成逼真的高光斑，模拟金属表面镜面反射。

    原理：
    - 随机生成椭圆形高光区域
    - 使用高斯模糊产生自然的光晕效果
    - 中心区域过曝 (饱和)，边缘渐变
    - 支持多个高光斑叠加
    """

    def __init__(self, max_glares=5, intensity_range=(0.3, 1.0),
                 size_range=(20, 150), p=0.7):
        self.max_glares = max_glares
        self.intensity_range = intensity_range
        self.size_range = size_range
        self.p = p

    def __call__(self, image):
        if np.random.random() > self.p:
            return image

        result = image.copy().astype(np.float32)
        h, w = result.shape[:2]
        n_glares = np.random.randint(1, self.max_glares + 1)

        for _ in range(n_glares):
            # 随机位置
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)

            # 随机大小和形状
            size = np.random.randint(self.size_range[0], self.size_range[1])
            aspect = np.random.uniform(0.5, 2.0)
            angle = np.random.uniform(0, 360)

            # 创建高光掩膜
            glare_mask = np.zeros((h, w), dtype=np.float32)
            axes = (int(size * aspect), size)
            cv2.ellipse(glare_mask, (cx, cy), axes, angle, 0, 360, 1.0, -1)

            # 高斯模糊产生自然渐变
            blur_size = max(size * 2 + 1, 3)
            if blur_size % 2 == 0:
                blur_size += 1
            glare_mask = cv2.GaussianBlur(glare_mask, (blur_size, blur_size), 0)

            # 随机强度
            intensity = np.random.uniform(*self.intensity_range)
            glare_mask *= intensity

            # 叠加高光 (加法混合)
            glare_color = np.random.uniform(200, 255, 3).astype(np.float32)
            for c in range(3):
                result[:, :, c] += glare_mask * glare_color[c]

        return np.clip(result, 0, 255).astype(np.uint8)


class OilStainSynthesis:
    """
    模拟金属表面油污和指纹。

    原理：
    - 使用 Perlin 噪声或随机多边形生成不规则形状
    - 降低局部对比度和饱和度
    - 添加轻微的颜色偏移 (偏黄/偏暗)
    """

    def __init__(self, max_stains=3, opacity_range=(0.1, 0.4), p=0.5):
        self.max_stains = max_stains
        self.opacity_range = opacity_range
        self.p = p

    def __call__(self, image):
        if np.random.random() > self.p:
            return image

        result = image.copy().astype(np.float32)
        h, w = result.shape[:2]
        n_stains = np.random.randint(1, self.max_stains + 1)

        for _ in range(n_stains):
            # 生成随机多边形油污区域
            n_points = np.random.randint(5, 12)
            cx, cy = np.random.randint(0, w), np.random.randint(0, h)
            radius = np.random.randint(30, min(h, w) // 3)
            angles = np.sort(np.random.uniform(0, 2 * np.pi, n_points))
            radii = radius * np.random.uniform(0.5, 1.5, n_points)
            points = np.array([
                [int(cx + r * np.cos(a)), int(cy + r * np.sin(a))]
                for a, r in zip(angles, radii)
            ])

            stain_mask = np.zeros((h, w), dtype=np.float32)
            cv2.fillPoly(stain_mask, [points], 1.0)
            stain_mask = cv2.GaussianBlur(stain_mask, (21, 21), 0)

            opacity = np.random.uniform(*self.opacity_range)
            stain_mask *= opacity

            # 油污效果：降低亮度 + 偏黄色
            stain_color = np.array([0.85, 0.90, 0.75], dtype=np.float32)  # BGR 偏黄暗
            for c in range(3):
                result[:, :, c] = result[:, :, c] * (1 - stain_mask) + \
                                  result[:, :, c] * stain_color[c] * stain_mask

        return np.clip(result, 0, 255).astype(np.uint8)


class DirectionalLighting:
    """
    模拟不同方向的光照条件。

    原理：
    - 创建线性渐变光照图
    - 模拟从不同角度照射的效果
    - 可叠加环境光和点光源
    """

    def __init__(self, intensity_range=(0.5, 1.5), p=0.6):
        self.intensity_range = intensity_range
        self.p = p

    def __call__(self, image):
        if np.random.random() > self.p:
            return image

        result = image.copy().astype(np.float32)
        h, w = result.shape[:2]

        # 随机光照方向
        angle = np.random.uniform(0, 2 * np.pi)
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)

        # 方向性光照渐变
        gradient = np.cos(angle) * xx + np.sin(angle) * yy
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min() + 1e-8)

        # 映射到强度范围
        low, high = self.intensity_range
        light_map = low + gradient * (high - low)

        # 可选：添加随机点光源
        if np.random.random() > 0.5:
            px, py = np.random.randint(0, w), np.random.randint(0, h)
            dist = np.sqrt((xx * w / 2 - px + w / 2) ** 2 + (yy * h / 2 - py + h / 2) ** 2)
            point_light = np.exp(-dist ** 2 / (2 * (min(h, w) * 0.3) ** 2))
            light_map += point_light * np.random.uniform(0.2, 0.5)

        for c in range(3):
            result[:, :, c] *= light_map

        return np.clip(result, 0, 255).astype(np.uint8)


class MetalTextureSynthesis:
    """
    模拟金属表面纹理 (拉丝/磨砂效果)。

    原理：
    - 生成方向性噪声模拟拉丝纹理
    - 叠加细微的高频噪声模拟磨砂质感
    """

    def __init__(self, intensity_range=(0.02, 0.08), p=0.4):
        self.intensity_range = intensity_range
        self.p = p

    def __call__(self, image):
        if np.random.random() > self.p:
            return image

        result = image.copy().astype(np.float32)
        h, w = result.shape[:2]
        intensity = np.random.uniform(*self.intensity_range)

        # 拉丝纹理：方向性噪声
        if np.random.random() > 0.5:
            angle = np.random.uniform(0, 180)
            noise = np.random.randn(h, w).astype(np.float32)
            # 方向性模糊
            ksize = np.random.choice([15, 21, 31])
            kernel = np.zeros((ksize, ksize), dtype=np.float32)
            kernel[ksize // 2, :] = 1.0 / ksize
            M = cv2.getRotationMatrix2D((float(ksize // 2), float(ksize // 2)), angle, 1.0)
            kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
            noise = cv2.filter2D(noise, -1, kernel)
        else:
            # 磨砂纹理：均匀细噪声
            noise = np.random.randn(h, w).astype(np.float32) * 0.5

        noise *= intensity * 255
        for c in range(3):
            result[:, :, c] += noise

        return np.clip(result, 0, 255).astype(np.uint8)


# ===========================================================================
# 2. 边缘标注自动生成
# ===========================================================================

def generate_edge_from_mask(mask, edge_width=3):
    """
    从分割 mask 自动生成边缘 ground truth。

    参数:
        mask:       二值分割掩膜 (H, W), uint8, 值 0/255
        edge_width: 边缘宽度 (像素)

    返回:
        edge_mask:  边缘掩膜 (H, W), uint8, 值 0/255
    """
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)

    # 膨胀 - 腐蚀 = 边缘
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width, edge_width))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    edge = dilated - eroded

    return edge


def generate_sobel_edge(image):
    """
    使用 Sobel 算子生成边缘先验图 (用于训练时的辅助监督)。

    返回:
        sobel_edge: 归一化的 Sobel 边缘图 (H, W), float32, 值 [0, 1]
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # 归一化到 [0, 1]
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    return magnitude.astype(np.float32)


# ===========================================================================
# 3. 组合增强管线
# ===========================================================================

class HighReflectivityAugPipeline:
    """
    高反光工件专用数据增强管线。

    整合所有自定义增强和标准增强，提供统一的调用接口。
    同步处理图像、分割 mask 和边缘 mask。

    用法:
        pipeline = HighReflectivityAugPipeline(mode='train', img_size=256)
        augmented = pipeline(image=img, mask=seg_mask)
        # augmented['image'], augmented['mask'], augmented['edge_mask']
    """

    def __init__(self, mode='train', img_size=256):
        self.mode = mode
        self.img_size = img_size

        # 高反光专用增强
        self.specular = SpecularGlareSynthesis(max_glares=5, p=0.7)
        self.oil_stain = OilStainSynthesis(max_stains=3, p=0.5)
        self.lighting = DirectionalLighting(p=0.6)
        self.metal_texture = MetalTextureSynthesis(p=0.4)

    def __call__(self, image, mask):
        """
        参数:
            image: BGR 图像 (H, W, 3), uint8
            mask:  分割掩膜 (H, W), uint8, 值 0/255

        返回:
            dict: {
                'image': 增强后的图像,
                'mask':  增强后的分割掩膜,
                'edge_mask': 自动生成的边缘掩膜
            }
        """
        if self.mode == 'train':
            result = self._train_augment(image, mask)
        else:
            result = self._val_augment(image, mask)

        # 自动生成边缘标注
        result['edge_mask'] = generate_edge_from_mask(result['mask'])

        return result

    def _train_augment(self, image, mask):
        """训练阶段增强：包含所有增强操作。"""

        # --- 几何增强 (同步作用于 image 和 mask) ---
        # 随机水平翻转
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # 随机垂直翻转
        if np.random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # 随机旋转 (0/90/180/270 度)
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        # 随机仿射变换 (小幅度)
        if np.random.random() > 0.5:
            h, w = image.shape[:2]
            angle = np.random.uniform(-15, 15)
            scale = np.random.uniform(0.85, 1.15)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            mask = cv2.warpAffine(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT)

        # 随机裁剪 + 缩放
        image, mask = self._random_crop_resize(image, mask)

        # --- 高反光专用增强 (仅作用于 image) ---
        image = self.specular(image)
        image = self.oil_stain(image)
        image = self.lighting(image)
        image = self.metal_texture(image)

        # --- 颜色增强 ---
        # 随机亮度
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.7, 1.3)
            image = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # 随机对比度
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.7, 1.3)
            mean = image.mean()
            image = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

        # 随机高斯噪声
        if np.random.random() > 0.5:
            sigma = np.random.uniform(3, 15)
            noise = np.random.randn(*image.shape) * sigma
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 随机高斯模糊
        if np.random.random() > 0.3:
            ksize = np.random.choice([3, 5, 7])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        return {'image': image, 'mask': mask}

    def _val_augment(self, image, mask):
        """验证阶段增强：仅缩放到目标尺寸。"""
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        return {'image': image, 'mask': mask}

    def _random_crop_resize(self, image, mask):
        """随机裁剪并缩放到目标尺寸。"""
        h, w = image.shape[:2]
        target = self.img_size

        # 随机裁剪比例
        crop_ratio = np.random.uniform(0.7, 1.0)
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)

        # 随机裁剪位置
        y = np.random.randint(0, max(h - crop_h, 1))
        x = np.random.randint(0, max(w - crop_w, 1))

        image = image[y:y + crop_h, x:x + crop_w]
        mask = mask[y:y + crop_h, x:x + crop_w]

        # 缩放到目标尺寸
        image = cv2.resize(image, (target, target))
        mask = cv2.resize(mask, (target, target), interpolation=cv2.INTER_NEAREST)

        return image, mask


# ===========================================================================
# 4. 合成数据生成器 (用于无真实数据时的冷启动训练)
# ===========================================================================

class SyntheticMetalWorkpieceGenerator:
    """
    合成高反光金属工件训练数据生成器。

    在没有真实数据时，可以生成合成训练数据进行冷启动训练。
    生成的数据包含：
    - 各种形状的金属工件 (圆形、矩形、多边形、环形)
    - 金属质感的表面纹理
    - 随机高光斑和反射
    - 对应的分割 mask 和边缘 mask
    """

    def __init__(self, img_size=256, min_objects=1, max_objects=5):
        self.img_size = img_size
        self.min_objects = min_objects
        self.max_objects = max_objects

    def generate(self):
        """
        生成一组合成训练数据。

        返回:
            dict: {
                'image': 合成图像 (H, W, 3), uint8
                'mask':  分割掩膜 (H, W), uint8
                'edge_mask': 边缘掩膜 (H, W), uint8
            }
        """
        size = self.img_size
        image = self._create_background(size)
        mask = np.zeros((size, size), dtype=np.uint8)

        n_objects = np.random.randint(self.min_objects, self.max_objects + 1)

        for _ in range(n_objects):
            obj_img, obj_mask = self._create_metal_workpiece(size)
            # 叠加到场景中
            blend_mask = obj_mask.astype(np.float32) / 255.0
            for c in range(3):
                image[:, :, c] = (image[:, :, c] * (1 - blend_mask) +
                                  obj_img[:, :, c] * blend_mask).astype(np.uint8)
            mask = np.maximum(mask, obj_mask)

        # 添加高光和环境效果
        image = SpecularGlareSynthesis(max_glares=3, p=0.8)(image)
        image = DirectionalLighting(p=0.7)(image)
        image = MetalTextureSynthesis(p=0.5)(image)

        # 添加噪声
        noise = np.random.randn(*image.shape) * np.random.uniform(3, 10)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        edge_mask = generate_edge_from_mask(mask)

        return {'image': image, 'mask': mask, 'edge_mask': edge_mask}

    def _create_background(self, size):
        """创建工业传送带/工作台背景。"""
        bg_type = np.random.choice(['uniform', 'gradient', 'textured'])

        if bg_type == 'uniform':
            base_val = np.random.randint(30, 80)
            bg = np.full((size, size, 3), base_val, dtype=np.uint8)
            # 添加轻微颜色变化
            bg = bg.astype(np.int16)
            bg[:, :, 0] += np.random.randint(-10, 10)
            bg[:, :, 1] += np.random.randint(-10, 10)
            bg[:, :, 2] += np.random.randint(-10, 10)
            bg = np.clip(bg, 0, 255).astype(np.uint8)
        elif bg_type == 'gradient':
            bg = np.zeros((size, size, 3), dtype=np.uint8)
            for c in range(3):
                start = np.random.randint(30, 70)
                end = np.random.randint(50, 90)
                if np.random.random() > 0.5:
                    gradient = np.linspace(start, end, size).reshape(1, -1)
                else:
                    gradient = np.linspace(start, end, size).reshape(-1, 1)
                bg[:, :, c] = np.broadcast_to(gradient, (size, size)).astype(np.uint8)
        else:
            bg = np.random.randint(40, 70, (size, size, 3), dtype=np.uint8)
            bg = cv2.GaussianBlur(bg, (11, 11), 0)

        return np.clip(bg, 0, 255).astype(np.uint8)

    def _create_metal_workpiece(self, size):
        """创建单个金属工件。"""
        obj_img = np.zeros((size, size, 3), dtype=np.uint8)
        obj_mask = np.zeros((size, size), dtype=np.uint8)

        # 随机选择工件形状
        shape = np.random.choice(['circle', 'rectangle', 'polygon', 'ring', 'ellipse'])

        # 随机位置和大小
        cx = np.random.randint(size // 4, 3 * size // 4)
        cy = np.random.randint(size // 4, 3 * size // 4)
        obj_size = np.random.randint(size // 6, size // 3)

        # 金属颜色 (银灰色系)
        base_color = np.random.randint(140, 220)
        color_var = np.random.randint(-15, 15, 3)
        metal_color = np.clip([base_color + v for v in color_var], 0, 255).tolist()

        if shape == 'circle':
            cv2.circle(obj_img, (cx, cy), obj_size, metal_color, -1)
            cv2.circle(obj_mask, (cx, cy), obj_size, 255, -1)

        elif shape == 'rectangle':
            angle = np.random.uniform(0, 360)
            w_half = obj_size
            h_half = int(obj_size * np.random.uniform(0.4, 1.0))
            rect = cv2.boxPoints(((cx, cy), (w_half * 2, h_half * 2), angle))
            rect = rect.astype(np.intp)
            cv2.fillPoly(obj_img, [rect], metal_color)
            cv2.fillPoly(obj_mask, [rect], 255)

        elif shape == 'polygon':
            n_sides = np.random.randint(5, 9)
            angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
            angles += np.random.uniform(0, 2 * np.pi)
            radii = obj_size * np.random.uniform(0.7, 1.0, n_sides)
            pts = np.array([
                [int(cx + r * np.cos(a)), int(cy + r * np.sin(a))]
                for a, r in zip(angles, radii)
            ])
            cv2.fillPoly(obj_img, [pts], metal_color)
            cv2.fillPoly(obj_mask, [pts], 255)

        elif shape == 'ring':
            outer_r = obj_size
            inner_r = int(obj_size * np.random.uniform(0.3, 0.7))
            cv2.circle(obj_img, (cx, cy), outer_r, metal_color, -1)
            cv2.circle(obj_mask, (cx, cy), outer_r, 255, -1)
            cv2.circle(obj_img, (cx, cy), inner_r, (0, 0, 0), -1)
            cv2.circle(obj_mask, (cx, cy), inner_r, 0, -1)

        elif shape == 'ellipse':
            axes = (obj_size, int(obj_size * np.random.uniform(0.4, 0.8)))
            angle = np.random.uniform(0, 360)
            cv2.ellipse(obj_img, (cx, cy), axes, angle, 0, 360, metal_color, -1)
            cv2.ellipse(obj_mask, (cx, cy), axes, angle, 0, 360, 255, -1)

        # 添加金属质感
        if obj_mask.max() > 0:
            # 轻微的亮度渐变模拟金属反射
            y_coords, x_coords = np.where(obj_mask > 0)
            if len(y_coords) > 0:
                gradient = np.random.uniform(0.8, 1.2, len(y_coords))
                for c in range(3):
                    vals = obj_img[y_coords, x_coords, c].astype(np.float32)
                    obj_img[y_coords, x_coords, c] = np.clip(vals * gradient, 0, 255).astype(np.uint8)

        return obj_img, obj_mask

    def generate_batch(self, batch_size, save_dir=None):
        """
        批量生成合成数据。

        参数:
            batch_size: 生成数量
            save_dir:   保存目录 (可选)

        返回:
            list of dict
        """
        batch = []
        for i in range(batch_size):
            sample = self.generate()
            batch.append(sample)

            if save_dir:
                save_path = Path(save_dir)
                (save_path / 'images').mkdir(parents=True, exist_ok=True)
                (save_path / 'masks').mkdir(parents=True, exist_ok=True)
                (save_path / 'edges').mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(save_path / 'images' / f'syn_{i:05d}.png'), sample['image'])
                cv2.imwrite(str(save_path / 'masks' / f'syn_{i:05d}.png'), sample['mask'])
                cv2.imwrite(str(save_path / 'edges' / f'syn_{i:05d}.png'), sample['edge_mask'])

        return batch


# ===========================================================================
# 5. 入口点
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("高反光工件数据增强管线测试")
    print("=" * 60)

    # 测试合成数据生成
    print("\n1. 测试合成数据生成器...")
    gen = SyntheticMetalWorkpieceGenerator(img_size=256, min_objects=1, max_objects=3)
    sample = gen.generate()
    print(f"   图像尺寸: {sample['image'].shape}")
    print(f"   掩膜尺寸: {sample['mask'].shape}")
    print(f"   边缘尺寸: {sample['edge_mask'].shape}")
    print(f"   掩膜非零像素: {np.count_nonzero(sample['mask'])}")
    print(f"   边缘非零像素: {np.count_nonzero(sample['edge_mask'])}")

    # 测试增强管线
    print("\n2. 测试增强管线...")
    pipeline = HighReflectivityAugPipeline(mode='train', img_size=256)
    augmented = pipeline(image=sample['image'], mask=sample['mask'])
    print(f"   增强后图像: {augmented['image'].shape}")
    print(f"   增强后掩膜: {augmented['mask'].shape}")
    print(f"   增强后边缘: {augmented['edge_mask'].shape}")

    # 测试批量生成并保存
    print("\n3. 测试批量生成 (5 张)...")
    save_dir = '/tmp/synthetic_test'
    gen.generate_batch(5, save_dir=save_dir)
    print(f"   保存到: {save_dir}")
    for subdir in ['images', 'masks', 'edges']:
        files = os.listdir(os.path.join(save_dir, subdir))
        print(f"   {subdir}/: {len(files)} 个文件")

    print("\n数据增强管线测试通过！")
