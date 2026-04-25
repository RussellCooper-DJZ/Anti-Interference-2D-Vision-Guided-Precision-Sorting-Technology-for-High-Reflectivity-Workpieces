"""
vision — 视觉算法核心包

包含以下子模块：
  feature_extraction           : AGEANet / AGEANetLite 模型架构
  hdr_processing               : HDR 融合与反光抑制管线
  localization_and_calibration : 亚像素定位与手眼标定
"""

__all__ = [
    "AGEANet",
    "AGEANetLite",
    "predict",
    "get_model_info",
    "AntiGlarePipeline",
    "SubpixelLocalizer",
    "CameraCalibrator",
    "HandEyeCalibrator",
    "CoordinateTransformer",
]
