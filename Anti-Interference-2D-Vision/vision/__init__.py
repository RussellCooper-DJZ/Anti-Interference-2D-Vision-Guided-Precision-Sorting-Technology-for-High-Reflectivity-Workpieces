"""
vision — 视觉算法核心包

包含以下子模块：
  feature_extraction           : FLARE / FLARELite 模型架构
  hdr_processing               : HDR 融合与反光抑制管线
  localization_and_calibration : 像素级定位与手眼标定
  existence_checking           : 有无检测（Blob/灰度匹配/特征匹配/轮廓匹配）
  appearance_detection         : 光度立体、划痕检测、边缘缺陷检测
  measurement                  : 卡尺、间隙测量、几何关系、轮廓操作
"""

__all__ = [
    # 特征提取 / 深度学习模型
    "FLARE",
    "FLARELite",
    "predict",
    "get_model_info",
    # HDR 与反光抑制
    "AntiGlarePipeline",
    # 定位与标定
    "SubpixelLocalizer",
    "CameraCalibrator",
    "HandEyeCalibrator",
    "CoordinateTransformer",
    "detect_glare_regions",
    # 角点检测
    "CornerDetector",
    "detect_corners",
    # 高斯线提取
    "GaussianLineExtractor",
    "extract_gaussian_line",
    # 霍夫圆/直线检测
    "HoughCircleDetector",
    "HoughLineDetector",
    # ROI校正
    "ROICorrection",
    "correct_roi_affine",
    # ROI工具
    "ROIType",
    "ROI",
    "ROIGenerator",
    "generate_roi_from_contour",
    "generate_roi_from_binary",
    "generate_array_roi",
    "ROICorrector",
    "correct_roi",
    "AutoROI",
    "AutoROIMode",
    "select_roi_adaptive",
    # 九点标定
    "NinePointCalibrator",
    "calibrate_nine_point",
    # 旋转中心标定
    "RotationCenterCalibrator",
    # 自动贴合
    "FeatureType",
    "Feature2D",
    "FeatureMatcher",
    "PoseEstimator",
    "AutoFitter",
    "auto_fit",
    # 外观检测
    "PhotometricStereo",
    "ScratchDetector",
    "EdgeDefectDetector",
    "detect_scratches",
    "detect_edge_defects",
    "compute_contour_roughness",
    # 测量与匹配
    "CaliperMeasurement",
    "measure_caliper",
    "GapMeasurement",
    "measure_gap",
    "GeometricRelations",
    "compute_geometric_relations",
    "ContourOperations",
    "filter_contours",
    "split_contour",
    "connect_collinear_contours",
    "smooth_contour",
    "ImageSharpness",
    "compute_sharpness",
    "BlobAnalyzer",
    "analyze_blobs",
    # 有无检测
    "BlobDetector",
    "analyze_blob",
    "GrayMatcher",
    "match_gray",
    "FeaturePointMatcher",
    "match_features",
    "ContourMatcher",
    "match_contours",
]
