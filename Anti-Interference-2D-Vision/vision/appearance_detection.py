"""
appearance_detection.py — 外观缺陷检测模块：光度立体、划痕检测、边缘缺陷
:Author: RussellCooper

基于Smart3智能视觉系统用户手册算法实现：

1. 光度立体 (Photometric Stereo)
   - 从多张不同光照方向的图像重建表面法线
   - 增强高光抑制，恢复被高光遮蔽的细节

2. 划痕检测 (Scratch Detection)
   - 使用形态学滤波增强细线结构
   - 结合边缘检测和纹理分析识别划痕

3. 边缘缺陷检测 (Edge Defect Detection)
   - 比较实际边缘与理想边缘的偏差
   - 测量边缘轮廓度

依赖: opencv-contrib-python>=4.5, numpy>=1.21
"""

import argparse
import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.distance import cdist

__all__ = [
    # 光度立体
    "PhotometricStereo",
    "estimate_albedo_normal",
    "reconstruct_surface",
    # 划痕检测
    "ScratchDetector",
    "detect_scratches",
    # 边缘缺陷检测
    "EdgeDefectDetector",
    "detect_edge_defects",
    "compute_contour_roughness",
]



# ---------------------------------------------------------------------------
# 1. 光度立体 (Photometric Stereo)
# ---------------------------------------------------------------------------

class PhotometricStereo:
    """
    光度立体视觉算法 - 从多张已知光照方向的图像重建表面法线。

    原理：
      对于漫反射表面：E(x,y) = ρ(x,y) * L · N(x,y)
      其中 E 是像素亮度，ρ 是反照率，L 是光源方向，N 是法线

    通过最小二乘法求解：N = (L^T L)^{-1} L^T E

    光照条件：
      - 需要 3+ 张不同光照方向的图像
      - 光源方向需已知或可标定
      - 假设朗伯漫反射表面

    用法::

        ps = PhotometricStereo(light_directions)
        albedo, normal = ps.process(image_list)
        depth = ps.reconstruct_depth(normal)
    """

    def __init__(
        self,
        light_directions: Optional[List[np.ndarray]] = None,
        gamma: float = 1.0,
        shadow_threshold: float = 0.05,
    ):
        """
        Args:
            light_directions: List of (3,) arrays, each being the light source
                              direction vector (not necessarily normalized).
                              If None, will use auto-detected default directions.
            gamma:            Gamma correction value (>1 brightens, <1 darkens)
            shadow_threshold:  Minimum intensity to consider a pixel as lit (not in shadow)
        """
        self.light_directions = light_directions
        self.gamma = gamma
        self.shadow_threshold = shadow_threshold

        # Default light directions (calibrated for typical setup with 4 lights)
        # These are azimuth (θ) and elevation (φ) angles in degrees
        self._default_lights = [
            (45,  30),   # Light 1: front-right
            (135, 30),   # Light 2: back-right
            (225, 30),   # Light 3: back-left
            (315, 30),   # Light 4: front-left
        ]

    def _get_light_directions(self, n_images: int) -> np.ndarray:
        """获取光源方向矩阵 (n_images, 3)。"""
        if self.light_directions is not None:
            L = np.array(self.light_directions, dtype=np.float64)
            # Ensure shape (n, 3)
            if L.ndim == 1:
                L = L.reshape(1, -1)
            return L

        # 使用默认光源方向
        lights = []
        if n_images == 3:
            for theta, phi in [(45, 30), (135, 30), (225, 30)]:
                theta_rad = math.radians(theta)
                phi_rad = math.radians(phi)
                lx = math.cos(phi_rad) * math.sin(theta_rad)
                ly = math.cos(phi_rad) * math.cos(theta_rad)
                lz = math.sin(phi_rad)
                lights.append([lx, ly, lz])
        elif n_images == 4:
            for theta, phi in self._default_lights:
                theta_rad = math.radians(theta)
                phi_rad = math.radians(phi)
                lx = math.cos(phi_rad) * math.sin(theta_rad)
                ly = math.cos(phi_rad) * math.cos(theta_rad)
                lz = math.sin(phi_rad)
                lights.append([lx, ly, lz])
        else:
            # 均匀分布的球面点
            for i in range(n_images):
                theta = 2 * math.pi * i / n_images
                phi = math.radians(30)
                lx = math.cos(phi) * math.sin(theta)
                ly = math.cos(phi) * math.cos(theta)
                lz = math.sin(phi)
                lights.append([lx, ly, lz])

        return np.array(lights, dtype=np.float64)

    def process(
        self,
        images: List[np.ndarray],
        return_depth: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        处理多张光照图像，重建表面法线和反照率。

        Args:
            images: List of BGR uint8 images (must be same size)
            return_depth: 是否重建深度图

        Returns:
            Dict with keys:
              - 'albedo': (H, W) float64, 反照率 (0-1)
              - 'normal': (H, W, 3) float64, 表面法线 (normalized)
              - 'depth':  (H, W) float64, 深度图 (if return_depth=True)
              - 'gradient_x': (H, W) float64, X方向梯度
              - 'gradient_y': (H, W) float64, Y方向梯度
        """
        if len(images) < 3:
            raise ValueError(f"需要至少3张图像，当前只有{len(images)}张")

        # 检查图像尺寸一致性
        h, w = images[0].shape[:2]
        gray_images = []
        for img in images:
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
            # Gamma校正
            if self.gamma != 1.0:
                gray = np.power(gray / 255.0, 1.0 / self.gamma) * 255.0
            gray_images.append(gray)

        # 获取光源方向 (n_images, 3)
        L = self._get_light_directions(len(gray_images))

        # 计算反照率和法线
        albedo, nx, ny, nz = self._solve_pixel(gray_images, L)

        # 计算深度（通过法线积分）
        gradient_x = nx / (np.abs(nz) + 1e-8)
        gradient_y = ny / (np.abs(nz) + 1e-8)

        result = {
            'albedo': albedo,
            'normal': np.stack([nx, ny, np.abs(nz)], axis=-1),
            'gradient_x': gradient_x,
            'gradient_y': gradient_y,
        }

        if return_depth:
            depth = self._integrate_gradients(gradient_x, gradient_y)
            result['depth'] = depth

        return result

    def _solve_pixel(
        self,
        gray_images: List[np.ndarray],
        L: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对每个像素求解法线。

        Args:
            gray_images: List of (H, W) float64 images
            L: (n_images, 3) light directions

        Returns:
            albedo: (H, W)
            nx, ny: (H, W) surface normal components
        """
        n_images = len(gray_images)
        h, w = gray_images[0].shape

        # Stack images: (H, W, n_images)
        I = np.stack(gray_images, axis=-1)  # (H, W, n_images)
        E = I.reshape(-1, n_images)  # (H*W, n_images)

        # 伪逆求解: N = (L^T L)^{-1} L^T E
        L_T_L = L.T @ L
        L_pinv = np.linalg.inv(L_T_L) @ L.T  # (3, n_images)

        # 求解每个像素
        N = E @ L_pinv.T  # (H*W, 3)

        # 归一化法线
        N_norm = np.linalg.norm(N, axis=1, keepdims=True)
        N_norm = np.maximum(N_norm, 1e-8)
        N_normalized = N / N_norm

        # 反照率 = |N| (因为 E = ρ * L·N, 当 L·N = 1 时，ρ = |N|)
        albedo_flat = N_norm.flatten()

        # 重构法线分量
        nx = N_normalized[:, 0].reshape(h, w)
        ny = N_normalized[:, 1].reshape(h, w)
        nz = N_normalized[:, 2].reshape(h, w)

        return albedo_flat.reshape(h, w), nx, ny, nz

    def _integrate_gradients(
        self,
        grad_x: np.ndarray,
        grad_y: np.ndarray,
        method: str = 'poisson',
    ) -> np.ndarray:
        """
        从梯度场重构深度图（泊松方程求解）。

        Args:
            grad_x: (H, W) gradient in x direction
            grad_y: (H, W) gradient in y direction
            method: 'poisson' or 'frankot'

        Returns:
            depth: (H, W) depth map
        """
        h, w = grad_x.shape

        if method == 'poisson':
            # 泊松方程求解: ∇²z = ∇·g
            # 使用离散泊松方程的快速求解方法
            depth = np.zeros((h, w), dtype=np.float64)

            # 构建稀疏矩阵（简化版本）
            # 使用积分方法近似
            depth[0, 0] = 0
            # 第一行积分
            for j in range(1, w):
                depth[0, j] = depth[0, j-1] + grad_x[0, j]
            # 第一列积分
            for i in range(1, h):
                depth[i, 0] = depth[i-1, 0] + grad_y[i, 0]

            # 剩余部分（平均四种路径）
            for i in range(1, h):
                for j in range(1, w):
                    d1 = depth[i-1, j] + grad_y[i, j]  # 从上
                    d2 = depth[i, j-1] + grad_x[i, j]  # 从左
                    d3 = (depth[i-1, j-1] + grad_y[i-1, j] +
                          grad_x[i, j-1] + grad_x[i, j]) / 2  # 从左上
                    if j < w - 1:
                        d4 = (depth[i-1, j+1] + grad_y[i-1, j] -
                              grad_x[i-1, j] + grad_x[i, j]) / 2  # 从右上（假设）
                        depth[i, j] = (d1 + d2 + d3 + d4) / 4
                    else:
                        # 最后一列：从右上路径越界，用三条路径平均
                        depth[i, j] = (d1 + d2 + d3) / 3

            return depth

        else:  # Frankot-Chellappa (更鲁棒但较慢)
            from numpy.fft import fft2, ifft2

            fx = grad_x
            fy = grad_y

            # 确保满足可积性条件 (df/dy - df/dx = 0 的傅里叶变换)
            rows, cols = h, w
            u, v = np.meshgrid(
                np.arange(cols) - cols // 2,
                np.arange(rows) - rows // 2
            )

            # 频率域计算
            with np.errstate(divide='ignore', invalid='ignore'):
                denom = (u**2 + v**2).astype(np.float64)
                denom[0, 0] = 1  # 避免除零

                F_x = fft2(fx)
                F_y = fft2(fy)
                F_z = (1j * u * F_x + 1j * v * F_y) / denom

            F_z[0, 0] = 0
            z = np.real(ifft2(F_z))

            # 归一化
            z = z - z.min()
            if z.max() > 0:
                z = z / z.max()

            return z


# ---------------------------------------------------------------------------
# 1b. 光度立体神经网络版（专利规避）
# ---------------------------------------------------------------------------

class PhotometricStereoNet:
    """
    光度立体神经网络 - 使用卷积神经网络直接回归表面法线和反照率。

    规避说明：
      传统光度立体使用最小二乘法求解 N = (L^TL)^{-1} L^TE，
      该显式数学公式落入MIT US6,477,268专利保护范围。
      本实现使用神经网络直接学习 I -> {albedo, normal} 的映射，
      属于不同的技术方案，不侵犯该专利。

    原理：
      - 训练一个小型U-Net风格网络学习多光源图像到法线/反照率的映射
      - 网络端到端训练，损失函数为预测与真值的L2损失
      - 推理时无需显式求解光照方程

    用法::

        net = PhotometricStereoNet(model_path='psnet.pth')
        net.eval()
        result = net.predict(image_list)  # 直接输出albedo和normal
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        input_channels: int = 3,
    ):
        """
        Args:
            model_path: 预训练模型路径，None则使用随机初始化
            device: 'cuda' 或 'cpu'
            input_channels: 输入通道数（通常为3*n_lights）
        """
        self.device = device
        self.input_channels = input_channels
        self.model = None
        self._initialized = False

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def _build_model(self, n_lights: int, output_mode: str = 'both'):
        """构建光度立体网络（轻量U-Net风格）。"""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PhotometricStereoNet需要PyTorch: pip install torch")

        class PSNet(nn.Module):
            """光度立体网络：编码器-解码器架构。"""
            def __init__(self, in_channels, out_channels=3):
                super().__init__()
                # 编码器
                self.enc1 = nn.Sequential(
                    nn.Conv2d(in_channels, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                )
                self.enc2 = nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
                self.enc3 = nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )
                # 瓶颈
                self.bottleneck = nn.Sequential(
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
                # 解码器
                self.dec3 = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 2, stride=2),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                )
                self.dec2 = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 2, stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                )
                self.dec1 = nn.Sequential(
                    nn.ConvTranspose2d(64, 32, 2, stride=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                )
                # 输出：albedo(1) + normal(3) = 4
                self.head = nn.Conv2d(32, out_channels, 1)

            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(e1)
                e3 = self.enc3(e2)
                b = self.bottleneck(e3)
                d3 = self.dec3(b)
                d2 = self.dec2(d3)
                d1 = self.dec1(d2)
                out = self.head(d1)
                return out

        return PSNet(n_lights * 3, out_channels=4)  # 4 = albedo(1) + normal(3)

    def _ensure_model(self, n_lights: int):
        """确保模型已初始化。"""
        if not self._initialized:
            self.model = self._build_model(n_lights).to(self.device)
            self._initialized = True

    def predict(
        self,
        images: List[np.ndarray],
        return_depth: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        使用神经网络预测反照率和法线。

        Args:
            images: List of BGR uint8 images (same size, 3+ images)
            return_depth: 是否计算深度图

        Returns:
            Dict with 'albedo', 'normal', optionally 'depth'
        """
        try:
            import torch
        except ImportError:
            raise ImportError("需要PyTorch: pip install torch")

        self._ensure_model(len(images))

        # 预处理
        h, w = images[0].shape[:2]
        gray_images = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256))
            gray_images.append(gray)

        # 合并为3通道堆叠（因为PSNet设计为3通道输入）
        # 如果>3张图，使用前3张；如果<3，补全到3
        while len(gray_images) < 3:
            gray_images.append(gray_images[0])
        stacked = np.stack(gray_images[:3], axis=0)  # (3, 256, 256)

        # 转为tensor
        x = torch.from_numpy(stacked).float().unsqueeze(0).to(self.device) / 255.0

        # 推理
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)  # (1, 4, 256, 256)

        # 解析输出
        pred = out.squeeze(0).cpu().numpy()  # (4, 256, 256)
        albedo = pred[0]
        normal = np.stack([pred[1], pred[2], np.abs(pred[3])], axis=0)

        # 归一化法线
        n_norm = np.linalg.norm(normal, axis=0, keepdims=True)
        n_norm = np.maximum(n_norm, 1e-8)
        normal = normal / n_norm

        # 归一化反照率
        albedo = np.clip(albedo, 0, 1)

        # 调整大小回原图尺寸
        albedo = cv2.resize(albedo, (w, h))
        normal = np.moveaxis(cv2.resize(np.moveaxis(normal, 0, -1), (w, h)), -1, 0)

        result = {
            'albedo': albedo.astype(np.float64),
            'normal': normal.astype(np.float64),
        }

        if return_depth:
            # 通过法线积分计算深度
            ps = PhotometricStereo()
            grad_x = normal[0] / (np.abs(normal[2]) + 1e-8)
            grad_y = normal[1] / (np.abs(normal[2]) + 1e-8)
            result['depth'] = ps._integrate_gradients(grad_x, grad_y)

        return result

    def train_step(
        self,
        images: List[np.ndarray],
        gt_albedo: np.ndarray,
        gt_normal: np.ndarray,
    ) -> Dict[str, float]:
        """
        单步训练（需要真实标签）。

        Args:
            images: 输入图像列表
            gt_albedo: (H, W) 真值反照率
            gt_normal: (H, W, 3) 真值法线

        Returns:
            loss dict
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("需要PyTorch")

        self._ensure_model(len(images))

        # 预处理
        h, w = images[0].shape[:2]
        gray_images = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256))
            gray_images.append(gray)

        while len(gray_images) < 3:
            gray_images.append(gray_images[0])
        stacked = np.stack(gray_images[:3], axis=0)

        x = torch.from_numpy(stacked).float().unsqueeze(0).to(self.device) / 255.0

        # 真值
        gt_a = cv2.resize(gt_albedo, (256, 256))
        gt_n = np.moveaxis(cv2.resize(np.moveaxis(gt_normal, -1, 0), (256, 256)), 0, -1)
        y_albedo = torch.from_numpy(gt_a).float().unsqueeze(0).unsqueeze(0).to(self.device)
        y_normal = torch.from_numpy(gt_n).permute(0, 3, 1, 2).to(self.device)

        # 前向传播
        self.model.train()
        out = self.model(x)
        pred_albedo = out[:, 0:1]
        pred_normal = out[:, 1:4]

        # 损失：Albedo L2 + Normal余弦相似度
        loss_albedo = nn.functional.mse_loss(pred_albedo, y_albedo)
        loss_normal = 1 - torch.mean(torch.sum(pred_normal * y_normal, dim=1) /
                                      (torch.norm(pred_normal, dim=1) + 1e-8))
        loss = loss_albedo + loss_normal

        # 反向传播
        loss.backward()

        return {'loss': loss.item(), 'albedo': loss_albedo.item(), 'normal': loss_normal.item()}

    def save_model(self, path: str):
        """保存模型。"""
        if self.model is not None:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        """加载模型。"""
        import torch.nn as nn
        if self.model is None:
            self.model = self._build_model(3).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self._initialized = True


def estimate_albedo_normal(
    images: List[np.ndarray],
    light_directions: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    便捷函数：估计反照率和法线。

    Args:
        images: List of BGR uint8 images
        light_directions: Optional list of (3,) light direction vectors

    Returns:
        (albedo, normal) tuples
    """
    ps = PhotometricStereo(light_directions=light_direections)
    result = ps.process(images, return_depth=False)
    return result['albedo'], result['normal']


def reconstruct_surface(
    normal: np.ndarray,
    method: str = 'poisson',
) -> np.ndarray:
    """
    从法线图重构表面深度。

    Args:
        normal: (H, W, 3) surface normals (normalized)
        method: 'poisson' or 'frankot'

    Returns:
        depth: (H, W) depth map
    """
    ps = PhotometricStereo()
    grad_x = normal[..., 0] / (np.abs(normal[..., 2]) + 1e-8)
    grad_y = normal[..., 1] / (np.abs(normal[..., 2]) + 1e-8)
    return ps._integrate_gradients(grad_x, grad_y, method=method)


# ---------------------------------------------------------------------------
# 2. 划痕检测 (Scratch Detection)
# ---------------------------------------------------------------------------

class ScratchDetector:
    """
    划痕检测器 - 检测物体表面的细线状缺陷。

    算法原理：
      1. 使用形态学顶帽变换提取亮/暗细线
      2. 结合边缘检测增强划痕边缘
      3. 使用纹理分析区分划痕与背景纹理
      4. 自适应阈值分割

    用法::

        detector = ScratchDetector()
        scratches = detector.detect(image)
        result = detector.visualize(image, scratches)
    """

    def __init__(
        self,
        line_width_range: Tuple[int, int] = (1, 10),
        min_length: int = 20,
        threshold_ratio: float = 0.3,
        use_clahe: bool = True,
    ):
        """
        Args:
            line_width_range: 划痕宽度范围（像素）
            min_length: 最小划痕长度（像素）
            threshold_ratio: 阈值比例（用于自适应阈值）
            use_clahe: 是否使用CLAHE增强对比度
        """
        self.line_width_range = line_width_range
        self.min_length = min_length
        self.threshold_ratio = threshold_ratio
        self.use_clahe = use_clahe

    def detect(
        self,
        image: np.ndarray,
        return_mask: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        检测图像中的划痕。

        Args:
            image: BGR uint8 image
            return_mask: 是否返回二值掩膜

        Returns:
            Dict with keys:
              - 'scratch_mask': (H, W) uint8, 划痕区域掩膜
              - 'scratch_lines': List of dicts with line parameters
              - 'confidence': (H, W) float, 划痕置信度图
        """
        # 预处理
        if self.use_clahe:
            gray = self._enhance_contrast(image)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 顶帽变换提取细线结构
        line_mask = self._topHat_lines(gray)

        # 边缘增强
        edge_enhanced = self._enhance_edges(gray)

        # 结合顶帽和边缘信息
        combined = cv2.addWeighted(
            line_mask.astype(np.float32) / 255.0,
            0.6,
            edge_enhanced.astype(np.float32) / 255.0,
            0.4,
            0
        )

        # 自适应阈值分割
        scratch_confidence = self._adaptive_threshold(combined)

        # 形态学后处理
        scratch_mask = self._postprocess_scratches(scratch_confidence)

        # 提取划痕线段
        lines = self._extract_lines(gray, scratch_mask)

        result = {
            'scratch_mask': scratch_mask,
            'scratch_lines': lines,
            'confidence': (scratch_confidence * 255).astype(np.uint8),
        }

        return result

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """CLAHE增强对比度。"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)

        result = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
        return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    def _topHat_lines(self, gray: np.ndarray) -> np.ndarray:
        """
        使用顶帽变换提取细线结构。

        顶帽变换 = 原图 - 开运算
        开运算 = 腐蚀 + 膨胀（去除小亮点）
        顶帽结果 = 保留比结构元素更小的亮结构
        """
        min_w, max_w = self.line_width_range

        # 多尺度结构元素
        results = []
        for width in range(min_w, max_w + 1, 2):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (width, width)
            )
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            results.append(tophat)

        # 取最大值（多尺度融合）
        line_mask = np.maximum.reduce(results)

        # 归一化
        if line_mask.max() > 0:
            line_mask = (line_mask.astype(np.float32) / line_mask.max() * 255).astype(np.uint8)

        return line_mask

    def _enhance_edges(self, gray: np.ndarray) -> np.ndarray:
        """
        使用Canny和形态学增强边缘。

        原理：划痕通常在边缘处有梯度突变
        """
        # Canny边缘检测
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # 形态学闭运算连接断开的边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 骨架化（提取中心线）
        skeleton = self._skeletonize(edges_closed)

        # 膨胀回原尺寸
        enhanced = cv2.dilate(skeleton, kernel, iterations=1)

        return enhanced

    def _skeletonize(self, binary: np.ndarray) -> np.ndarray:
        """骨架化提取中心线。"""
        skeleton = np.zeros_like(binary)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        done = False

        temp = binary.copy()
        while not done:
            eroded = cv2.erode(temp, element)
            opened = cv2.dilate(eroded, element)
            temp_diff = cv2.subtract(temp, opened)
            skeleton = cv2.bitwise_or(skeleton, temp_diff)
            done = (cv2.countNonZero(temp) == 0)
            temp = eroded.copy()

        return skeleton

    def _adaptive_threshold(self, combined: np.ndarray) -> np.ndarray:
        """
        自适应阈值分割。

        使用局部均值作为阈值，增强对比度
        """
        # 计算局部均值
        local_mean = cv2.blur(combined * 255, (21, 21)) / 255.0

        # 阈值 = 局部均值 * threshold_ratio
        threshold = local_mean * self.threshold_ratio

        # 分割
        scratch_mask = (combined > threshold).astype(np.float32)

        return scratch_mask

    def _postprocess_scratches(self, mask: np.ndarray) -> np.ndarray:
        """形态学后处理。"""
        mask_u8 = (mask * 255).astype(np.uint8)

        # 去噪（去除小区域）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

        # 闭运算（连接断开的划痕）
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 骨架化（细化为单像素线）
        skeleton = self._skeletonize(closed)

        return skeleton

    def _extract_lines(
        self,
        gray: np.ndarray,
        mask: np.ndarray,
    ) -> List[Dict]:
        """
        从掩膜中提取划痕线段。

        Returns:
            List of dicts with keys: 'start', 'end', 'length', 'angle'
        """
        # 概率Hough变换
        lines = cv2.HoughLinesP(
            mask,
            rho=1,
            theta=np.pi / 180,
            threshold=max(10, self.min_length // 5),
            minLineLength=self.min_length,
            maxLineGap=5,
        )

        result = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = math.hypot(x2 - x1, y2 - y1)
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

                result.append({
                    'start': (float(x1), float(y1)),
                    'end': (float(x2), float(y2)),
                    'length': float(length),
                    'angle': float(angle),
                    'midpoint': ((x1 + x2) / 2, (y1 + y2) / 2),
                })

        return result

    def visualize(
        self,
        image: np.ndarray,
        detection_result: Dict,
        line_color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        可视化划痕检测结果。

        Returns:
            BGR image with detected scratches overlaid
        """
        vis = image.copy()
        mask = detection_result['scratch_mask']
        lines = detection_result['scratch_lines']

        # 叠加掩膜（半透明）
        mask_color = np.zeros_like(image)
        mask_color[mask > 0] = [0, 0, 255]  # 红色
        vis = cv2.addWeighted(vis, 0.8, mask_color, 0.2, 0)

        # 绘制线段
        for line in lines:
            pt1 = (int(line['start'][0]), int(line['start'][1]))
            pt2 = (int(line['end'][0]), int(line['end'][1]))
            cv2.line(vis, pt1, pt2, line_color, thickness)
            # 绘制端点
            cv2.circle(vis, pt1, 3, (0, 255, 0), -1)
            cv2.circle(vis, pt2, 3, (0, 255, 0), -1)

        return vis


def detect_scratches(
    image: np.ndarray,
    line_width_range: Tuple[int, int] = (1, 10),
    min_length: int = 20,
) -> Dict[str, np.ndarray]:
    """
    便捷函数：检测划痕。

    Args:
        image: BGR uint8 image
        line_width_range: 划痕宽度范围
        min_length: 最小长度

    Returns:
        detection result dict
    """
    detector = ScratchDetector(
        line_width_range=line_width_range,
        min_length=min_length,
    )
    return detector.detect(image)


# ---------------------------------------------------------------------------
# 3. 边缘缺陷检测 (Edge Defect Detection)
# ---------------------------------------------------------------------------

class EdgeDefectDetector:
    """
    边缘缺陷检测器 - 检测边缘区域的缺陷。

    算法原理：
      1. 提取理想边缘（模板或几何拟合）
      2. 比较实际边缘与理想边缘的偏差
      3. 计算边缘轮廓度（粗糙度）

    用法::

        detector = EdgeDefectDetector()
        defects = detector.detect(image, reference_edge)
        result = detector.visualize(image, defects)
    """

    def __init__(
        self,
        max_defect_width: int = 10,
        max_defect_depth: float = 5.0,
        roughness_threshold: float = 2.0,
    ):
        """
        Args:
            max_defect_width: 最大缺陷宽度（像素）
            max_defect_depth: 最大缺陷深度（像素偏离）
            roughness_threshold: 粗糙度阈值
        """
        self.max_defect_width = max_defect_width
        self.max_defect_depth = max_defect_depth
        self.roughness_threshold = roughness_threshold

    def detect(
        self,
        image: np.ndarray,
        reference_edge: np.ndarray,
        current_edge: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        检测边缘缺陷。

        Args:
            image: BGR uint8 image
            reference_edge: 参考边缘点坐标 (N, 1-2) 格式或轮廓
            current_edge: 当前边缘（如果不提供，使用Canny自动检测）

        Returns:
            Dict with keys:
              - 'defect_mask': (H, W) uint8, 缺陷区域掩膜
              - 'defect_points': List of defect locations
              - 'roughness_map': (H, W) float, 粗糙度图
              - 'edge_deviation': 边缘偏差图
        """
        # 如果没有提供当前边缘，使用Canny检测
        if current_edge is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            current_edge_points = self._extract_edge_points(edges)
        else:
            current_edge_points = self._extract_edge_points(current_edge)

        # 计算边缘偏差
        deviation_map = self._compute_deviation(
            reference_edge, current_edge_points
        )

        # 检测缺陷点
        h, w = image.shape[:2]
        defect_mask, defect_points = self._detect_defect_points(
            deviation_map, current_edge_points, h, w
        )

        # 计算粗糙度
        roughness_map = self._compute_roughness_map(
            deviation_map, current_edge_points, h, w
        )

        return {
            'defect_mask': defect_mask,
            'defect_points': defect_points,
            'roughness_map': roughness_map,
            'edge_deviation': deviation_map,
        }

    def _extract_edge_points(self, edge_input) -> np.ndarray:
        """从边缘图或轮廓提取边缘点。"""
        if isinstance(edge_input, np.ndarray) and edge_input.ndim == 2:
            # 可能是边缘图
            if edge_input.max() > 1:  # 可能是二值图像
                coords = np.where(edge_input > 0)
                if len(coords[0]) > 0:
                    points = np.column_stack([coords[1], coords[0]])  # (x, y) format
                    return points

        # 假设是轮廓点数组
        return np.array(edge_input).reshape(-1, 2)

    def _compute_deviation(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> np.ndarray:
        """
        计算当前边缘相对于参考边缘的偏差。

        Args:
            reference: 参考边缘点 (N, 2) 或轮廓
            current: 当前边缘点 (M, 2)

        Returns:
            deviation_map: 每个当前点到最近参考点的距离
        """
        if len(current) == 0:
            return np.array([])

        # 计算每个当前点到最近参考点的距离
        dist_matrix = cdist(current, reference)
        min_distances = np.min(dist_matrix, axis=1)

        return min_distances

    def _detect_defect_points(
        self,
        deviations: np.ndarray,
        edge_points: np.ndarray,
        h: int,
        w: int,
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        从偏差中检测缺陷点。

        Args:
            deviations: 偏差数组
            edge_points: 边缘点坐标
            h, w: 图像尺寸

        Returns:
            (defect_mask, defect_points_list)
        """
        defect_mask = np.zeros((h, w), dtype=np.uint8)
        defect_list = []

        threshold = self.max_defect_depth

        for i, (point, deviation) in enumerate(zip(edge_points, deviations)):
            if deviation > threshold:
                x, y = int(point[0]), int(point[1])
                if 0 <= x < w and 0 <= y < h:
                    defect_mask[y, x] = 255
                    defect_list.append({
                        'location': (float(x), float(y)),
                        'deviation': float(deviation),
                        'index': i,
                    })

        return defect_mask, defect_list

    def _compute_roughness_map(
        self,
        deviations: np.ndarray,
        edge_points: np.ndarray,
        h: int,
        w: int,
    ) -> np.ndarray:
        """
        计算边缘粗糙度图。

        粗糙度 = 局部邻域内偏差的标准差
        """
        roughness = np.zeros((h, w), dtype=np.float32)

        # 使用高斯加权计算局部粗糙度
        # 简化实现：直接使用偏差作为粗糙度指标
        for point, dev in zip(edge_points, deviations):
            x, y = int(point[0]), int(point[1])
            if 0 <= x < w and 0 <= y < h:
                roughness[y, x] = dev

        return roughness

    def compute_contour_roughness(
        self,
        contour: np.ndarray,
        window_size: int = 10,
    ) -> float:
        """
        计算轮廓的整体粗糙度。

        Args:
            contour: 轮廓点 (N, 1, 2) 或 (N, 2)
            window_size: 平滑窗口大小

        Returns:
            average_roughness: 平均粗糙度
        """
        if isinstance(contour, np.ndarray) and contour.ndim == 3:
            pts = contour.reshape(-1, 2)
        else:
            pts = np.array(contour).reshape(-1, 2)

        if len(pts) < window_size:
            return 0.0

        # 计算相邻点间距
        distances = []
        for i in range(len(pts) - 1):
            d = math.hypot(pts[i+1, 0] - pts[i, 0], pts[i+1, 1] - pts[i, 1])
            distances.append(d)

        # 计算局部均值的标准差作为粗糙度
        distances = np.array(distances)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        return float(std_dist)

    def visualize(
        self,
        image: np.ndarray,
        detection_result: Dict,
        defect_color: Tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        """可视化缺陷检测结果。"""
        vis = image.copy()
        defect_mask = detection_result['defect_mask']
        defect_points = detection_result['defect_points']

        # 叠加缺陷掩膜
        mask_color = np.zeros_like(image)
        mask_color[defect_mask > 0] = defect_color
        vis = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)

        # 标记缺陷点
        for pt in defect_points:
            x, y = int(pt['location'][0]), int(pt['location'][1])
            cv2.circle(vis, (x, y), 3, defect_color, -1)

        return vis


def detect_edge_defects(
    image: np.ndarray,
    reference_edge: np.ndarray,
    current_edge: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    便捷函数：检测边缘缺陷。

    Args:
        image: BGR uint8 image
        reference_edge: 参考边缘
        current_edge: 当前边缘（可选）

    Returns:
        detection result dict
    """
    detector = EdgeDefectDetector()
    return detector.detect(image, reference_edge, current_edge)


def compute_contour_roughness(
    contour: np.ndarray,
    window_size: int = 10,
) -> float:
    """便捷函数：计算轮廓粗糙度。"""
    detector = EdgeDefectDetector()
    return detector.compute_contour_roughness(contour, window_size)


# ---------------------------------------------------------------------------
# 4. 命令行入口
# ---------------------------------------------------------------------------

def _make_test_image_scratch() -> np.ndarray:
    """生成带划痕的测试图像。"""
    img = np.full((480, 640, 3), 180, dtype=np.uint8)  # 浅灰色背景

    # 添加一些圆形工件
    for cx, cy, r in [(200, 200, 60), (400, 300, 80)]:
        cv2.circle(img, (cx, cy), r, (150, 155, 160), -1)
        cv2.circle(img, (cx, cy), r, (80, 85, 90), 2)

    # 添加划痕（细线）
    cv2.line(img, (100, 150), (300, 160), (200, 200, 200), 1)
    cv2.line(img, (350, 100), (360, 350), (210, 210, 210), 1)

    # 添加一些噪声模拟真实场景
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def _make_test_image_edge_defect() -> Tuple[np.ndarray, np.ndarray]:
    """生成带边缘缺陷的测试图像和参考边缘。"""
    img = np.full((480, 640, 3), 150, dtype=np.uint8)

    # 绘制理想圆形
    cv2.circle(img, (320, 240), 100, (100, 100, 100), 2)
    reference_edge = np.array([[320 + 100 * math.cos(a), 240 + 100 * math.sin(a)]
                                for a in np.linspace(0, 2*math.pi, 100)])

    # 绘制有缺陷的圆形
    defective_img = img.copy()
    for i, (x, y) in enumerate(reference_edge):
        # 在某些位置添加缺陷
        if 20 < i < 30 or 60 < i < 70:
            offset = 5
        else:
            offset = 0
        px, py = int(x + offset), int(y + offset)
        cv2.circle(defective_img, (px, py), 2, (80, 80, 80), -1)

    return defective_img, reference_edge


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="外观缺陷检测模块")
    parser.add_argument("--mode", type=str, default="scratch",
                        choices=["scratch", "photometric", "edge"])
    parser.add_argument("--output_dir", type=str, default="./appearance_output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "scratch":
        print("=== 划痕检测测试 ===")
        test_img = _make_test_image_scratch()
        cv2.imwrite(os.path.join(args.output_dir, "test_scratch_input.png"), test_img)

        detector = ScratchDetector()
        result = detector.detect(test_img)
        vis = detector.visualize(test_img, result)
        cv2.imwrite(os.path.join(args.output_dir, "test_scratch_result.png"), vis)

        print(f"检测到划痕数: {len(result['scratch_lines'])}")
        for i, line in enumerate(result['scratch_lines'][:5]):
            print(f"  [{i}] 长度={line['length']:.1f}px, 角度={line['angle']:.1f}°")

    elif args.mode == "photometric":
        print("=== 光度立体测试 ===")
        print("注意：光度立体需要多张不同光照方向的图像")
        print("使用合成数据进行演示...")

        # 模拟4张不同光照的图像
        base_img = np.full((256, 256, 3), 128, dtype=np.uint8)
        cv2.circle(base_img, (128, 128), 50, (180, 180, 180), -1)

        images = []
        for angle in [0, 45, 90, 135]:
            img_rot = base_img.copy()
            # 简单模拟光照变化（实际应该用真实多光源图像）
            factor = 0.7 + 0.3 * abs(math.sin(math.radians(angle)))
            img_adj = np.clip(base_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            images.append(img_adj)

        ps = PhotometricStereo()
        result = ps.process(images, return_depth=True)

        # 可视化反照率和法线
        albedo_vis = (result['albedo'] * 255).astype(np.uint8)
        normal_vis = ((result['normal'] + 1) / 2 * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(args.output_dir, "photometric_albedo.png"), albedo_vis)
        cv2.imwrite(os.path.join(args.output_dir, "photometric_normal.png"), normal_vis)
        print(f"反照率范围: {result['albedo'].min():.2f} - {result['albedo'].max():.2f}")

    elif args.mode == "edge":
        print("=== 边缘缺陷检测测试 ===")
        test_img, ref_edge = _make_test_image_edge_defect()
        cv2.imwrite(os.path.join(args.output_dir, "test_edge_input.png"), test_img)

        detector = EdgeDefectDetector()
        result = detector.detect(test_img, ref_edge)
        vis = detector.visualize(test_img, result)
        cv2.imwrite(os.path.join(args.output_dir, "test_edge_result.png"), vis)

        roughness = detector.compute_contour_roughness(ref_edge)
        print(f"参考轮廓粗糙度: {roughness:.3f}")
        print(f"检测到缺陷点数: {len(result['defect_points'])}")

    print(f"\n结果已保存到: {args.output_dir}")
    print("完成。")
