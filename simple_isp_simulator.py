import cv2
import numpy as np

class SimpleISPSimulator:
    def __init__(self):
        # 默认 ISP 参数，对应 V4L2_CID_RZ_ISP_*
        self.params = {
            'V4L2_CID_RZ_ISP_BL': 0,      # Black Level (0 to +127)
            'V4L2_CID_RZ_ISP_WB': 1,      # White Balance (0:Daylight, 1:Neutral, 2:Indoor, 3:CarLight)
            'V4L2_CID_RZ_ISP_GAMMA': 100, # Gamma (1 to 9999, default 100 for 1.00)
            'V4L2_CID_RZ_ISP_CMX': 1,     # Color Matrix (0:Original, 1:Standard, 2:Vivid, 3:Sepia)
            'V4L2_CID_RZ_ISP_2DNR': 100,  # 2D Noise Reduction (0 to 100)
            'V4L2_CID_RZ_ISP_3DNR': 1,    # 3D Noise Reduction (0:Off, 1:On)
            'V4L2_CID_RZ_ISP_EMP': 0,     # Edge Enhancement (0:Off, 1:Weak, 2:Normal, 3:Strong)
            'V4L2_CID_RZ_ISP_AE': 0,      # Auto Exposure (0:Off, 1:On)
            'V4L2_CID_RZ_ISP_EXPOSE_LV': 0, # Exposure Level (-40 to 40 dB)
            'V4L2_CID_RZ_ISP_T_BL': 128,  # Target Brightness (1 to 254)
            'V4L2_CID_RZ_ISP_THRESHOLD': 10 # Brightness Threshold (1 to 64)
        }

        # 预定义的颜色矩阵 (示例，实际值需查阅 ISP 文档)
        self.color_matrices = {
            0: np.eye(3), # Original
            1: np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), # Standard (identity for simplicity)
            2: np.array([[1.2, -0.1, -0.1], [-0.1, 1.2, -0.1], [-0.1, -0.1, 1.2]]), # Vivid (example)
            3: np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]) # Sepia (example)
        }

        # 预定义的白平衡增益 (示例，实际值需查阅 ISP 文档)
        self.white_balance_gains = {
            0: (1.2, 1.0, 0.8), # Daylight (R, G, B gains)
            1: (1.0, 1.0, 1.0), # Neutral
            2: (0.8, 1.0, 1.2), # Indoor (warmer light)
            3: (1.5, 1.0, 0.7)  # Car Light (example)
        }

    def set_param(self, param_id, value):
        if param_id in self.params:
            self.params[param_id] = value
        else:
            print(f"Warning: Unknown ISP parameter ID: {param_id}")

    def _bayer_to_rgb(self, bayer_raw, pattern=cv2.COLOR_BayerBG2BGR):
        # 模拟去马赛克 (ACPI 法在 OpenCV 中通常由 COLOR_BayerXX2BGR 自动处理)
        # Simulate demosaicing (ACPI method is typically handled by COLOR_BayerXX2BGR in OpenCV)
        return cv2.cvtColor(bayer_raw, pattern)

    def _apply_black_level(self, image):
        # 黑色色阶调整
        bl_value = self.params['V4L2_CID_RZ_ISP_BL']
        return np.clip(image.astype(np.int32) - bl_value, 0, 255).astype(np.uint8)

    def _apply_white_balance(self, image):
        # 白平衡调整
        wb_mode = self.params['V4L2_CID_RZ_ISP_WB']
        r_gain, g_gain, b_gain = self.white_balance_gains.get(wb_mode, (1.0, 1.0, 1.0))
        
        # 假设输入是 BGR 格式
        b, g, r = cv2.split(image.astype(np.float32))
        r = np.clip(r * r_gain, 0, 255)
        g = np.clip(g * g_gain, 0, 255)
        b = np.clip(b * b_gain, 0, 255)
        return cv2.merge([b, g, r]).astype(np.uint8)

    def _apply_gamma_correction(self, image):
        # 伽玛校正
        gamma_val = self.params['V4L2_CID_RZ_ISP_GAMMA'] / 100.0 # 100 -> 1.00
        if gamma_val == 0: gamma_val = 1.0 # Avoid division by zero or invalid gamma
        inv_gamma = 1.0 / gamma_val
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _apply_color_matrix(self, image):
        # 颜色矩阵校正
        cmx_mode = self.params['V4L2_CID_RZ_ISP_CMX']
        matrix = self.color_matrices.get(cmx_mode, np.eye(3))
        
        # 转换为浮点型并应用矩阵 (假设 RGB 顺序)
        img_float = image.astype(np.float32) / 255.0
        # OpenCV 默认是 BGR，需要转换到 RGB 进行矩阵乘法，再转回 BGR
        img_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
        
        # Reshape for matrix multiplication: (H*W, 3) @ (3, 3)
        h, w, c = img_rgb.shape
        img_linear = img_rgb.reshape(-1, c)
        corrected_linear = np.dot(img_linear, matrix.T) # Transpose matrix for correct multiplication
        
        corrected_rgb = np.clip(corrected_linear.reshape(h, w, c), 0, 1)
        return cv2.cvtColor((corrected_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def _apply_2d_noise_reduction(self, image):
        # 2D 降噪 (中值滤波模拟)
        nr_level = self.params['V4L2_CID_RZ_ISP_2DNR']
        if nr_level > 0:
            ksize = int(nr_level / 20) * 2 + 1 # 映射到奇数核大小，例如 1, 3, 5, ...
            ksize = max(1, min(ksize, 7)) # 限制核大小
            return cv2.medianBlur(image, ksize)
        return image

    def _apply_edge_enhancement(self, image):
        # 边缘增强 (拉普拉斯滤波模拟)
        emp_mode = self.params['V4L2_CID_RZ_ISP_EMP']
        if emp_mode > 0:
            # 简单的拉普拉斯增强
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            enhanced = image.astype(np.float64) - laplacian * (emp_mode * 0.5) # 强度因子
            return np.clip(enhanced, 0, 255).astype(np.uint8)
        return image

    def process_raw_image(self, bayer_raw_image):
        # 模拟 Simple ISP 的处理流水线
        # 1. 去马赛克
        rgb_image = self._bayer_to_rgb(bayer_raw_image)
        
        # 2. 黑电平校正
        rgb_image = self._apply_black_level(rgb_image)
        
        # 3. 白平衡
        rgb_image = self._apply_white_balance(rgb_image)

        # 4. 颜色矩阵校正
        rgb_image = self._apply_color_matrix(rgb_image)
        
        # 5. 伽玛校正
        rgb_image = self._apply_gamma_correction(rgb_image)
        
        # 6. 2D 降噪
        rgb_image = self._apply_2d_noise_reduction(rgb_image)

        # 7. 边缘增强
        rgb_image = self._apply_edge_enhancement(rgb_image)

        # 3D 降噪和自动曝光/曝光级/目标亮度/亮度阈值需要帧间信息或更复杂的模拟，这里暂时跳过
        # 3D noise reduction and auto exposure/exposure level/target brightness/brightness threshold require inter-frame information or more complex simulation, skipped for now.

        return rgb_image

if __name__ == "__main__":
    print("Simple ISP Simulator Module Loaded.")
    # 示例用法：
    # 假设有一个模拟的 Bayer RAW 图像 (例如，从一个彩色图像转换而来)
    # 创建一个 dummy 彩色图像
    dummy_color_img = np.zeros((240, 320, 3), dtype=np.uint8)
    dummy_color_img[50:150, 50:150] = [0, 0, 255] # 蓝色方块
    dummy_color_img[100:200, 100:200] = [0, 255, 0] # 绿色方块
    dummy_color_img[150:250, 150:250] = [255, 0, 0] # 红色方块

    # 模拟 Bayer RAW 图像 (这里简化为灰度，实际应是特定模式的彩色)
    # 实际的 Bayer RAW 图像是单通道的，每个像素代表一个颜色分量
    # 为了演示，我们先将彩色图转灰度，再模拟 Bayer 模式
    # 这是一个非常简化的模拟，实际 Bayer 转换更复杂
    dummy_bayer_raw = cv2.cvtColor(dummy_color_img, cv2.COLOR_BGR2GRAY) # 仅为演示，非真实 Bayer

    isp = SimpleISPSimulator()
    
    # 设置一些 ISP 参数
    isp.set_param('V4L2_CID_RZ_ISP_GAMMA', 200) # 伽玛值 2.0
    isp.set_param('V4L2_CID_RZ_ISP_2DNR', 50)  # 中等 2D 降噪
    isp.set_param('V4L2_CID_RZ_ISP_CMX', 2)    # 鲜艳色校正
    isp.set_param('V4L2_CID_RZ_ISP_EMP', 2)    # 正常边缘增强

    processed_img = isp.process_raw_image(dummy_bayer_raw)
    
    # 可以显示或保存处理后的图像
    # cv2.imshow("Original (Simulated Bayer)", dummy_bayer_raw)
    # cv2.imshow("Processed by Simple ISP Simulator", processed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("Simulated ISP processing complete. You can uncomment imshow lines to visualize.")
