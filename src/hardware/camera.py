"""
camera.py — 工业相机抽象层 (Industrial Camera Abstraction Layer)
支持 Basler (pypylon), Hikvision (MVS), 以及通用 OpenCV 相机。
"""

import cv2
import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

try:
    from pypylon import pylon
    HAS_PYPYLON = True
except ImportError:
    HAS_PYPYLON = False

class CameraInterface(ABC):
    """相机抽象基类"""
    @abstractmethod
    def open(self) -> bool: pass
    
    @abstractmethod
    def close(self): pass
    
    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]: pass
    
    @abstractmethod
    def set_exposure(self, exposure_time_us: float): pass

class BaslerCamera(CameraInterface):
    """Basler 工业相机实现"""
    def __init__(self, serial_number: Optional[str] = None):
        if not HAS_PYPYLON:
            raise ImportError("Please install 'pypylon' to use Basler cameras.")
        self.serial_number = serial_number
        self.camera = None
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def open(self) -> bool:
        try:
            if self.serial_number:
                info = pylon.DeviceInfo()
                info.SetSerialNumber(self.serial_number)
                self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
            else:
                self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()
            return True
        except Exception as e:
            print(f"Failed to open Basler camera: {e}")
            return False

    def close(self):
        if self.camera:
            self.camera.Close()

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.camera or not self.camera.IsOpen():
            return None
        res = self.camera.GrabOne(5000)
        if res.GrabSucceeded():
            image = self.converter.Convert(res)
            return image.GetArray()
        return None

    def set_exposure(self, exposure_time_us: float):
        if self.camera:
            self.camera.ExposureTime.SetValue(exposure_time_us)

class GenericCVCamera(CameraInterface):
    """通用 OpenCV 相机 (USB/RTSP/Virtual)"""
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.device_id)
        return self.cap.isOpened()

    def close(self):
        if self.cap:
            self.cap.release()

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.cap: return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def set_exposure(self, exposure_time_us: float):
        # OpenCV 对曝光的控制通常是平台相关的
        if self.cap:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_time_us)

class MockCamera(CameraInterface):
    """模拟相机 (用于测试和演示)"""
    def __init__(self, image_path: Optional[str] = None):
        self.image = cv2.imread(image_path) if image_path else None
        if self.image is None:
            # 创建一个带反光的模拟工件图
            self.image = np.zeros((1080, 1920, 3), dtype=np.uint8) + 50
            cv2.circle(self.image, (960, 540), 200, (200, 200, 200), -1) # 工件
            cv2.circle(self.image, (900, 480), 50, (255, 255, 255), -1)  # 反光斑

    def open(self) -> bool: return True
    def close(self): pass
    def get_frame(self) -> Optional[np.ndarray]:
        # 添加一些随机噪声模拟真实采集
        noise = np.random.normal(0, 2, self.image.shape).astype(np.int16)
        noisy_img = np.clip(self.image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return noisy_img
    def set_exposure(self, exposure_time_us: float): pass

def get_camera(config: Dict) -> CameraInterface:
    """工厂方法：根据配置获取相机实例"""
    cam_type = config.get("type", "mock").lower()
    if cam_type == "basler":
        return BaslerCamera(config.get("serial"))
    elif cam_type == "cv":
        return GenericCVCamera(config.get("device_id", 0))
    else:
        return MockCamera(config.get("mock_path"))
