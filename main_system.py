import time
import cv2
import numpy as np
import torch
from hdr_processing import exposure_fusion, adaptive_image_enhancement
from feature_extraction import SimpleUNet, predict_contour
from localization_and_calibration import subpixel_edge_detection, pixel_to_robot_coords
from simple_isp_simulator import SimpleISPSimulator # 导入 SimpleISPSimulator

class HighReflectiveSortingSystem:
    def __init__(self, model_path=None):
        # 初始化设备
        # Initialize device
        # Gerät initialisieren
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载深度学习模型
        # Load deep learning model
        # Deep-Learning-Modell laden
        self.model = SimpleUNet().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # 初始化 Simple ISP 模拟器
        # Initialize Simple ISP simulator
        # Simple ISP Simulator initialisieren
        self.isp = SimpleISPSimulator()
        # 可以根据需要设置默认 ISP 参数
        # Can set default ISP parameters as needed
        # Standard-ISP-Parameter können bei Bedarf eingestellt werden
        self.isp.set_param("V4L2_CID_RZ_ISP_GAMMA", 150) # 示例：设置伽玛值
        self.isp.set_param("V4L2_CID_RZ_ISP_2DNR", 70)  # 示例：设置 2D 降噪
        
        # 相机参数 (示例值，需实际标定)
        # Camera parameters (example values, require actual calibration)
        # Kameraparameter (Beispielwerte, erfordern tatsächliche Kalibrierung)
        self.camera_matrix = np.array([[1500, 0, 960], [0, 1500, 540], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1))
        
        # 手眼矩阵 (示例值，需实际标定)
        # Hand-eye matrix (example values, require actual calibration)
        # Hand-Auge-Matrix (Beispielwerte, erfordern tatsächliche Kalibrierung)
        self.R_cam2base = np.eye(3)
        self.t_cam2base = np.array([[100], [200], [500]], dtype=np.float32)

    def process_frame(self, bayer_raw_images_list):
        """
        完整处理流程：模拟ISP处理 -> HDR融合 -> 增强 -> 特征提取 -> 定位 -> 坐标转换

        Complete processing flow: Simulate ISP processing -> HDR fusion -> Enhancement -> Feature extraction -> Localization -> Coordinate transformation

        Vollständiger Verarbeitungsprozess: ISP-Verarbeitung simulieren -> HDR-Fusion -> Verbesserung -> Merkmalsextraktion -> Lokalisierung -> Koordinatentransformation
        """
        start_time = time.time()
        
        # 0. 模拟 ISP 处理每个 Bayer RAW 图像
        # 0. Simulate ISP processing for each Bayer RAW image
        # 0. ISP-Verarbeitung für jedes Bayer-RAW-Bild simulieren
        processed_rgb_images = []
        for raw_img in bayer_raw_images_list:
            processed_rgb_images.append(self.isp.process_raw_image(raw_img))

        # 1. HDR 融合
        # 1. HDR Fusion
        # 1. HDR-Fusion
        hdr_img = exposure_fusion(processed_rgb_images)
        
        # 2. 图像增强
        # 2. Image Enhancement
        # 2. Bildverbesserung
        enhanced_img = adaptive_image_enhancement(hdr_img)
        
        # 3. 深度学习特征提取 (轮廓预测)
        # 3. Deep Learning Feature Extraction (Contour Prediction)
        # 3. Deep-Learning-Merkmalsextraktion (Konturvorhersage)
        contour_mask = predict_contour(self.model, enhanced_img, self.device)
        
        # 4. 亚像素定位
        # 4. Sub-pixel Localization
        # 4. Subpixel-Lokalisierung
        result = subpixel_edge_detection(contour_mask)
        if result is None:
            return None
        
        pixel_center, angle = result
        
        # 5. 坐标转换 (像素 -> 机器人)
        # 5. Coordinate Transformation (Pixel -> Robot)
        # 5. Koordinatentransformation (Pixel -> Roboter)
        robot_coords = pixel_to_robot_coords(
            pixel_center, self.R_cam2base, self.t_cam2base, 
            self.camera_matrix, self.dist_coeffs
        )
        
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        return {
            "robot_coords": robot_coords,
            "angle": angle,
            "processing_time_ms": processing_time_ms
        }

def performance_test():
    """
    性能测试函数，验证处理节拍是否满足 <= 300ms 的要求。

    Performance test function to verify if the processing cycle meets the <= 300ms requirement.

    Leistungstestfunktion zur Überprüfung, ob der Verarbeitungszyklus die Anforderung von <= 300 ms erfüllt.
    """
    system = HighReflectiveSortingSystem()
    
    # 模拟三张不同曝光的 Bayer RAW 图像 (1920x1080)
    # Simulate three Bayer RAW images with different exposures (1920x1080)
    # Simuliert drei Bayer-RAW-Bilder mit unterschiedlichen Belichtungen (1920x1080)
    # 注意：这里为了简化模拟，仍然使用灰度图作为“Bayer RAW”输入，实际应是单通道的 Bayer 模式图像
    # Note: For simplified simulation, grayscale images are still used as "Bayer RAW" input here; actual Bayer mode images should be single-channel.
    # Hinweis: Zur vereinfachten Simulation werden hier weiterhin Graustufenbilder als "Bayer RAW"-Eingabe verwendet; tatsächliche Bayer-Modus-Bilder sollten einkanalig sein.
    dummy_bayer_raw_imgs = [np.random.randint(0, 255, (1080, 1920), dtype=np.uint8) for _ in range(3)]
    
    print("Starting performance test...")
    # 预热
    # Warm-up
    # Aufwärmen
    _ = system.process_frame(dummy_bayer_raw_imgs)
    
    # 测试 10 次取平均
    # Test 10 times and take the average
    # 10 Mal testen und den Durchschnitt nehmen
    times = []
    for i in range(10):
        res = system.process_frame(dummy_bayer_raw_imgs)
        times.append(res["processing_time_ms"])
        print(f"Iteration {i+1}: {res["processing_time_ms"]:.2f} ms")
    
    avg_time = sum(times) / len(times)
    print(f"\nAverage Processing Time: {avg_time:.2f} ms")
    
    if avg_time <= 300:
        print("Performance Check: PASSED (<= 300ms)")
    else:
        print("Performance Check: FAILED (> 300ms)")

if __name__ == "__main__":
    performance_test()
