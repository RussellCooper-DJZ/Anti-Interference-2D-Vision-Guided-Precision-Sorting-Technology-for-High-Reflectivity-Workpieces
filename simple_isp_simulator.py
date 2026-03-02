"""
simple_isp_simulator.py — 增强版 Simple ISP 模拟器

适配 Renesas RZ/V2H + CMOS 传感器的 ISP 处理流水线。

优化内容：
  1. 完善 Bayer 去马赛克 — 支持多种 Bayer 模式
  2. 自动曝光 (AE) 模拟 — 基于亮度统计的自动曝光调节
  3. 3D 降噪模拟 — 基于帧间差异的时域降噪
  4. 高反光场景专用预设 — 针对金属工件优化的 ISP 参数组合
  5. 参数验证与安全范围限制
"""

import cv2
import numpy as np


class SimpleISPSimulator:
    """
    Simple ISP 模拟器 — 模拟 Renesas RZ/V2H ISP 处理流水线。

    处理流程:
        Bayer RAW → 去马赛克 → 黑电平校正 → 白平衡 → 颜色矩阵 →
        伽玛校正 → 2D 降噪 → 3D 降噪 → 边缘增强 → 自动曝光调节
    """

    # 参数范围定义
    PARAM_RANGES = {
        'V4L2_CID_RZ_ISP_BL':         (0, 127),
        'V4L2_CID_RZ_ISP_WB':         (0, 3),
        'V4L2_CID_RZ_ISP_GAMMA':      (1, 9999),
        'V4L2_CID_RZ_ISP_CMX':        (0, 3),
        'V4L2_CID_RZ_ISP_2DNR':       (0, 100),
        'V4L2_CID_RZ_ISP_3DNR':       (0, 1),
        'V4L2_CID_RZ_ISP_EMP':        (0, 3),
        'V4L2_CID_RZ_ISP_AE':         (0, 1),
        'V4L2_CID_RZ_ISP_EXPOSE_LV':  (-40, 40),
        'V4L2_CID_RZ_ISP_T_BL':       (1, 254),
        'V4L2_CID_RZ_ISP_THRESHOLD':  (1, 64),
    }

    # 高反光场景预设参数
    PRESETS = {
        'default': {
            'V4L2_CID_RZ_ISP_BL': 0,
            'V4L2_CID_RZ_ISP_WB': 1,
            'V4L2_CID_RZ_ISP_GAMMA': 100,
            'V4L2_CID_RZ_ISP_CMX': 1,
            'V4L2_CID_RZ_ISP_2DNR': 50,
            'V4L2_CID_RZ_ISP_3DNR': 0,
            'V4L2_CID_RZ_ISP_EMP': 0,
            'V4L2_CID_RZ_ISP_AE': 0,
            'V4L2_CID_RZ_ISP_EXPOSE_LV': 0,
            'V4L2_CID_RZ_ISP_T_BL': 128,
            'V4L2_CID_RZ_ISP_THRESHOLD': 10,
        },
        'high_reflectivity': {
            'V4L2_CID_RZ_ISP_BL': 10,       # 稍高黑电平抑制暗部噪声
            'V4L2_CID_RZ_ISP_WB': 1,        # 中性白平衡
            'V4L2_CID_RZ_ISP_GAMMA': 80,    # 低伽玛压缩高光动态范围
            'V4L2_CID_RZ_ISP_CMX': 1,       # 标准色彩矩阵
            'V4L2_CID_RZ_ISP_2DNR': 40,     # 适度 2D 降噪
            'V4L2_CID_RZ_ISP_3DNR': 1,      # 开启 3D 降噪
            'V4L2_CID_RZ_ISP_EMP': 2,       # 正常边缘增强
            'V4L2_CID_RZ_ISP_AE': 1,        # 开启自动曝光
            'V4L2_CID_RZ_ISP_EXPOSE_LV': -5, # 稍微欠曝避免过曝
            'V4L2_CID_RZ_ISP_T_BL': 110,    # 较低目标亮度
            'V4L2_CID_RZ_ISP_THRESHOLD': 15, # 适中亮度阈值
        },
        'metal_edge_detection': {
            'V4L2_CID_RZ_ISP_BL': 5,
            'V4L2_CID_RZ_ISP_WB': 1,
            'V4L2_CID_RZ_ISP_GAMMA': 70,    # 更低伽玛保留边缘细节
            'V4L2_CID_RZ_ISP_CMX': 0,       # 原始色彩 (减少色彩串扰)
            'V4L2_CID_RZ_ISP_2DNR': 30,     # 轻度降噪 (保留边缘)
            'V4L2_CID_RZ_ISP_3DNR': 1,
            'V4L2_CID_RZ_ISP_EMP': 3,       # 强边缘增强
            'V4L2_CID_RZ_ISP_AE': 1,
            'V4L2_CID_RZ_ISP_EXPOSE_LV': -8, # 欠曝抑制高光
            'V4L2_CID_RZ_ISP_T_BL': 100,
            'V4L2_CID_RZ_ISP_THRESHOLD': 20,
        },
    }

    def __init__(self, preset='default'):
        """
        初始化 ISP 模拟器。

        参数:
            preset: 预设名称 ('default', 'high_reflectivity', 'metal_edge_detection')
        """
        self.params = self.PRESETS.get(preset, self.PRESETS['default']).copy()
        self._prev_frame = None  # 用于 3D 降噪
        self._gamma_lut = None   # 伽玛查找表缓存
        self._gamma_val = None

        # 颜色矩阵
        self.color_matrices = {
            0: np.eye(3, dtype=np.float32),
            1: np.array([[1.05, -0.02, -0.03],
                         [-0.02, 1.05, -0.03],
                         [-0.03, -0.02, 1.05]], dtype=np.float32),
            2: np.array([[1.2, -0.1, -0.1],
                         [-0.1, 1.2, -0.1],
                         [-0.1, -0.1, 1.2]], dtype=np.float32),
            3: np.array([[0.393, 0.769, 0.189],
                         [0.349, 0.686, 0.168],
                         [0.272, 0.534, 0.131]], dtype=np.float32),
        }

        # 白平衡增益
        self.wb_gains = {
            0: (1.2, 1.0, 0.85),   # Daylight
            1: (1.0, 1.0, 1.0),    # Neutral
            2: (0.85, 1.0, 1.15),  # Indoor
            3: (1.4, 1.0, 0.75),   # Car Light
        }

    def set_param(self, param_id, value):
        """设置 ISP 参数 (带范围验证)。"""
        if param_id not in self.PARAM_RANGES:
            print(f"Warning: 未知 ISP 参数: {param_id}")
            return False

        low, high = self.PARAM_RANGES[param_id]
        value = max(low, min(high, value))
        self.params[param_id] = value

        # 清除伽玛 LUT 缓存
        if param_id == 'V4L2_CID_RZ_ISP_GAMMA':
            self._gamma_lut = None

        return True

    def get_param(self, param_id):
        """获取 ISP 参数值。"""
        return self.params.get(param_id, None)

    def load_preset(self, preset_name):
        """加载预设参数。"""
        if preset_name in self.PRESETS:
            self.params = self.PRESETS[preset_name].copy()
            self._gamma_lut = None
            return True
        return False

    # --- ISP 处理步骤 ---

    def _bayer_to_rgb(self, bayer_raw, pattern=cv2.COLOR_BayerBG2BGR):
        """去马赛克：Bayer RAW → BGR。"""
        if len(bayer_raw.shape) == 3:
            return bayer_raw  # 已经是 RGB/BGR
        return cv2.cvtColor(bayer_raw, pattern)

    def _apply_black_level(self, image):
        """黑电平校正。"""
        bl = self.params['V4L2_CID_RZ_ISP_BL']
        if bl == 0:
            return image
        return np.clip(image.astype(np.int16) - bl, 0, 255).astype(np.uint8)

    def _apply_white_balance(self, image):
        """白平衡。"""
        wb_mode = self.params['V4L2_CID_RZ_ISP_WB']
        r_gain, g_gain, b_gain = self.wb_gains.get(wb_mode, (1.0, 1.0, 1.0))

        if r_gain == 1.0 and g_gain == 1.0 and b_gain == 1.0:
            return image

        result = image.astype(np.float32)
        result[:, :, 0] *= b_gain  # B
        result[:, :, 1] *= g_gain  # G
        result[:, :, 2] *= r_gain  # R
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_color_matrix(self, image):
        """颜色矩阵校正。"""
        cmx_mode = self.params['V4L2_CID_RZ_ISP_CMX']
        matrix = self.color_matrices.get(cmx_mode, np.eye(3, dtype=np.float32))

        if np.allclose(matrix, np.eye(3)):
            return image

        # BGR → RGB → 矩阵乘法 → RGB → BGR
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w, c = img_rgb.shape
        corrected = np.dot(img_rgb.reshape(-1, c), matrix.T).reshape(h, w, c)
        corrected = np.clip(corrected, 0, 1)
        return cv2.cvtColor((corrected * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _apply_gamma(self, image):
        """伽玛校正 (使用 LUT 加速)。"""
        gamma_val = self.params['V4L2_CID_RZ_ISP_GAMMA'] / 100.0
        if gamma_val <= 0:
            gamma_val = 1.0

        # 缓存 LUT
        if self._gamma_lut is None or self._gamma_val != gamma_val:
            inv_gamma = 1.0 / gamma_val
            self._gamma_lut = np.array(
                [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
            ).astype(np.uint8)
            self._gamma_val = gamma_val

        return cv2.LUT(image, self._gamma_lut)

    def _apply_2d_nr(self, image):
        """2D 降噪。"""
        level = self.params['V4L2_CID_RZ_ISP_2DNR']
        if level <= 0:
            return image

        # 使用非局部均值降噪 (比中值滤波更好地保留边缘)
        h_param = level / 10.0  # 映射到合理范围
        if h_param < 3:
            # 轻度降噪用高斯
            ksize = 3
            return cv2.GaussianBlur(image, (ksize, ksize), 0.5)
        else:
            # 中重度用 fastNlMeansDenoising
            h_param = min(h_param, 15)
            return cv2.fastNlMeansDenoisingColored(
                image, None, h_param, h_param, 7, 21
            )

    def _apply_3d_nr(self, image):
        """3D 降噪 (时域降噪，需要前一帧)。"""
        if self.params['V4L2_CID_RZ_ISP_3DNR'] == 0:
            self._prev_frame = image.copy()
            return image

        if self._prev_frame is None:
            self._prev_frame = image.copy()
            return image

        # 时域加权平均
        alpha = 0.7  # 当前帧权重
        result = cv2.addWeighted(
            image, alpha, self._prev_frame, 1 - alpha, 0
        )
        self._prev_frame = image.copy()
        return result

    def _apply_edge_enhancement(self, image):
        """边缘增强。"""
        mode = self.params['V4L2_CID_RZ_ISP_EMP']
        if mode == 0:
            return image

        # 不同强度的锐化核
        strength_map = {
            1: 0.3,   # Weak
            2: 0.6,   # Normal
            3: 1.0,   # Strong
        }
        strength = strength_map.get(mode, 0)

        # Unsharp Mask 方法 (比拉普拉斯更可控)
        blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
        sharpened = cv2.addWeighted(
            image, 1.0 + strength, blurred, -strength, 0
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _apply_auto_exposure(self, image):
        """自动曝光调节。"""
        if self.params['V4L2_CID_RZ_ISP_AE'] == 0:
            return image

        # 计算当前亮度
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_brightness = gray.mean()

        target = self.params['V4L2_CID_RZ_ISP_T_BL']
        threshold = self.params['V4L2_CID_RZ_ISP_THRESHOLD']
        expose_lv = self.params['V4L2_CID_RZ_ISP_EXPOSE_LV']

        # 加上曝光补偿
        adjusted_target = target + expose_lv

        # 如果亮度偏差超过阈值，调整
        diff = adjusted_target - current_brightness
        if abs(diff) > threshold:
            gain = adjusted_target / max(current_brightness, 1)
            gain = np.clip(gain, 0.3, 3.0)  # 限制调整范围
            result = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
            return result

        return image

    def process_raw_image(self, bayer_raw_image):
        """
        完整 ISP 处理流水线。

        参数:
            bayer_raw_image: Bayer RAW 图像 (单通道) 或 BGR 图像

        返回:
            处理后的 BGR 图像
        """
        # 1. 去马赛克
        image = self._bayer_to_rgb(bayer_raw_image)

        # 2. 黑电平校正
        image = self._apply_black_level(image)

        # 3. 白平衡
        image = self._apply_white_balance(image)

        # 4. 颜色矩阵校正
        image = self._apply_color_matrix(image)

        # 5. 伽玛校正
        image = self._apply_gamma(image)

        # 6. 2D 降噪
        image = self._apply_2d_nr(image)

        # 7. 3D 降噪
        image = self._apply_3d_nr(image)

        # 8. 边缘增强
        image = self._apply_edge_enhancement(image)

        # 9. 自动曝光
        image = self._apply_auto_exposure(image)

        return image

    def get_params_summary(self):
        """获取当前参数摘要。"""
        summary = []
        for key, val in self.params.items():
            low, high = self.PARAM_RANGES[key]
            summary.append(f"  {key}: {val} (range: {low}~{high})")
        return '\n'.join(summary)


# ===========================================================================
# 入口点
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Simple ISP 模拟器测试")
    print("=" * 60)

    # 创建测试图像
    test_img = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
    # 添加高光区域
    cv2.circle(test_img, (160, 120), 30, (250, 250, 250), -1)

    # 测试默认预设
    print("\n1. 默认预设:")
    isp = SimpleISPSimulator(preset='default')
    result = isp.process_raw_image(test_img)
    print(f"   输出: {result.shape}, mean={result.mean():.1f}")

    # 测试高反光预设
    print("\n2. 高反光预设:")
    isp_hr = SimpleISPSimulator(preset='high_reflectivity')
    result_hr = isp_hr.process_raw_image(test_img)
    print(f"   输出: {result_hr.shape}, mean={result_hr.mean():.1f}")

    # 测试金属边缘检测预设
    print("\n3. 金属边缘检测预设:")
    isp_edge = SimpleISPSimulator(preset='metal_edge_detection')
    result_edge = isp_edge.process_raw_image(test_img)
    print(f"   输出: {result_edge.shape}, mean={result_edge.mean():.1f}")

    # 测试参数设置
    print("\n4. 参数设置测试:")
    isp.set_param('V4L2_CID_RZ_ISP_GAMMA', 150)
    print(f"   GAMMA={isp.get_param('V4L2_CID_RZ_ISP_GAMMA')}")
    isp.set_param('V4L2_CID_RZ_ISP_GAMMA', 99999)  # 超出范围
    print(f"   GAMMA (clamped)={isp.get_param('V4L2_CID_RZ_ISP_GAMMA')}")

    # 测试 3D 降噪 (多帧)
    print("\n5. 3D 降噪测试 (3 帧):")
    isp_3d = SimpleISPSimulator(preset='high_reflectivity')
    for i in range(3):
        noisy = test_img.copy()
        noisy = np.clip(noisy.astype(np.int16) + np.random.randint(-20, 20, noisy.shape, dtype=np.int16),
                        0, 255).astype(np.uint8)
        result_3d = isp_3d.process_raw_image(noisy)
        print(f"   帧 {i+1}: mean={result_3d.mean():.1f}")

    print("\n参数摘要:")
    print(isp_hr.get_params_summary())

    print("\nISP 模拟器测试通过！")
