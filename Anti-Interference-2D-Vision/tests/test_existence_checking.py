"""
tests/test_existence_checking.py — 有无检测模块测试

覆盖 BlobDetector、GrayMatcher、FeaturePointMatcher、ContourMatcher
及对应的便捷函数。
"""

import math

import cv2
import numpy as np
import pytest

from vision.existence_checking import (
    BlobDetector,
    analyze_blob,
    GrayMatcher,
    match_gray,
    FeaturePointMatcher,
    match_features,
    ContourMatcher,
    match_contours,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def blank_image():
    return np.zeros((300, 300, 3), dtype=np.uint8)


@pytest.fixture
def circle_image(blank_image):
    cv2.circle(blank_image, (150, 150), 50, (255, 255, 255), -1)
    return blank_image


@pytest.fixture
def multi_shape_image(blank_image):
    cv2.circle(blank_image, (80, 80), 30, (255, 255, 255), -1)
    cv2.rectangle(blank_image, (180, 120), (260, 200), (255, 255, 255), -1)
    return blank_image


@pytest.fixture
def template_and_search():
    """生成模板和搜索图像（用于匹配测试）。"""
    tpl = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(tpl, (25, 25), (75, 75), (200, 200, 200), -1)
    search = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(search, (50, 50), (100, 100), (200, 200, 200), -1)
    cv2.rectangle(search, (180, 180), (230, 230), (200, 200, 200), -1)
    return tpl, search


# ============================================================
# BlobDetector
# ============================================================

class TestBlobDetector:
    def test_detect_single_circle(self, circle_image):
        detector = BlobDetector(threshold_method="manual", threshold_low=50)
        blobs = detector.detect(circle_image)
        assert len(blobs) >= 1
        blob = blobs[0]
        assert "area" in blob
        assert "circularity" in blob
        assert blob["area"] > 0

    def test_detect_multiple_shapes(self, multi_shape_image):
        detector = BlobDetector(threshold_method="manual", threshold_low=50)
        blobs = detector.detect(multi_shape_image)
        assert len(blobs) >= 2

    def test_roi_filter(self, multi_shape_image):
        detector = BlobDetector(threshold_method="manual", threshold_low=50)
        blobs = detector.detect(multi_shape_image, roi=(0, 0, 120, 120))
        assert len(blobs) >= 1
        for b in blobs:
            assert b["centroid_x"] < 120
            assert b["centroid_y"] < 120

    def test_threshold_methods(self, circle_image):
        for method in ["manual", "otsu"]:
            detector = BlobDetector(threshold_method=method, threshold_low=50)
            blobs = detector.detect(circle_image)
            assert len(blobs) >= 1

    def test_morphology(self, circle_image):
        detector = BlobDetector(
            threshold_method="manual",
            threshold_low=50,
            morphology_method="close",
            morphology_kernel=5,
        )
        blobs = detector.detect(circle_image)
        assert len(blobs) >= 1

    def test_ignore_boundary(self, circle_image):
        detector = BlobDetector(
            threshold_method="manual",
            threshold_low=50,
            ignore_boundary_blobs=True,
        )
        blobs = detector.detect(circle_image)
        # 圆心在(150,150)，不接触边界，应被保留
        assert len(blobs) >= 1

    def test_feature_fields(self, circle_image):
        detector = BlobDetector(threshold_method="manual", threshold_low=50)
        blobs = detector.detect(circle_image)
        assert len(blobs) >= 1
        b = blobs[0]
        expected_keys = [
            "centroid_x", "centroid_y", "area", "perimeter",
            "circularity", "rectangularity", "compactness",
            "aspect_ratio", "hu_moments", "symmetry",
            "mean_distance", "solidity",
        ]
        for k in expected_keys:
            assert k in b

    def test_empty_image(self, blank_image):
        detector = BlobDetector(threshold_method="manual", threshold_low=50)
        blobs = detector.detect(blank_image)
        assert blobs == []

    def test_analyze_blob_convenience(self, circle_image):
        blobs = analyze_blob(circle_image, threshold_method="manual", threshold_low=50)
        assert len(blobs) >= 1


# ============================================================
# GrayMatcher
# ============================================================

class TestGrayMatcher:
    def test_create_template_and_match(self, template_and_search):
        tpl, search = template_and_search
        matcher = GrayMatcher(angle_step=45.0)
        matcher.create_template(tpl)
        results = matcher.match(search, num_matches=2, min_score=30.0)
        assert isinstance(results, list)
        if len(results) > 0:
            assert "score" in results[0]
            assert "center" in results[0]
            assert "bounding_box" in results[0]

    def test_match_without_template_raises(self):
        matcher = GrayMatcher()
        search = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="create_template"):
            matcher.match(search)

    def test_match_gray_convenience(self, template_and_search):
        tpl, search = template_and_search
        results = match_gray(search, tpl, num_matches=1)
        assert isinstance(results, list)

    def test_non_max_suppression(self, template_and_search):
        tpl, search = template_and_search
        matcher = GrayMatcher(angle_step=90.0)
        matcher.create_template(tpl)
        results = matcher.match(search, num_matches=5, min_score=20.0, overlap_threshold=10.0)
        # 重叠抑制后结果数应 <= 原始结果数
        assert len(results) <= 5


# ============================================================
# FeaturePointMatcher
# ============================================================

class TestFeaturePointMatcher:
    def test_orb_match(self, template_and_search):
        tpl, search = template_and_search
        matcher = FeaturePointMatcher(method="orb")
        matcher.create_template(tpl)
        results = matcher.match(search)
        assert isinstance(results, list)
        if len(results) > 0:
            r = results[0]
            assert "homography" in r
            assert "inliers" in r
            assert "score" in r
            assert r["inliers"] >= 0

    def test_create_template_empty_raises(self):
        matcher = FeaturePointMatcher(method="orb")
        blank = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="未能检测到足够特征点"):
            matcher.create_template(blank)

    def test_match_no_features_returns_empty(self):
        matcher = FeaturePointMatcher(method="orb")
        tpl = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(tpl, (30, 30), (70, 70), (200, 200, 200), -1)
        matcher.create_template(tpl)
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        results = matcher.match(blank)
        assert results == []

    def test_match_features_convenience(self, template_and_search):
        tpl, search = template_and_search
        results = match_features(search, tpl, method="orb")
        assert isinstance(results, list)

    def test_different_methods(self, template_and_search):
        tpl, search = template_and_search
        for method in ["orb", "akaze"]:
            matcher = FeaturePointMatcher(method=method)
            matcher.create_template(tpl)
            results = matcher.match(search)
            assert isinstance(results, list)


# ============================================================
# ContourMatcher
# ============================================================

class TestContourMatcher:
    def test_create_template_and_match(self, template_and_search):
        tpl, search = template_and_search
        matcher = ContourMatcher(angle_step=45.0)
        matcher.create_template(tpl)
        results = matcher.match(search)
        assert isinstance(results, list)
        if len(results) > 0:
            assert "score" in results[0]
            assert "center" in results[0]
            assert "angle" in results[0]

    def test_match_convenience(self, template_and_search):
        tpl, search = template_and_search
        results = match_contours(search, tpl)
        assert isinstance(results, list)

    def test_empty_search(self, template_and_search):
        tpl, _ = template_and_search
        matcher = ContourMatcher()
        matcher.create_template(tpl)
        blank = np.zeros((200, 200, 3), dtype=np.uint8)
        results = matcher.match(blank)
        assert results == []

    def test_roi_search(self, template_and_search):
        tpl, search = template_and_search
        matcher = ContourMatcher(angle_step=45.0)
        matcher.create_template(tpl)
        results = matcher.match(search, roi=(40, 40, 80, 80))
        assert isinstance(results, list)

    def test_nms_in_match(self, template_and_search):
        tpl, search = template_and_search
        matcher = ContourMatcher(angle_step=30.0, match_threshold=20.0)
        matcher.create_template(tpl)
        results = matcher.match(search)
        # NMS 去重后不应有距离过近的结果
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                x1, y1 = results[i]["center"]
                x2, y2 = results[j]["center"]
                dist = math.hypot(x1 - x2, y1 - y2)
                assert dist >= 20.0 - 1e-6


# ============================================================
# 边界与异常测试
# ============================================================

class TestEdgeCases:
    def test_blob_grayscale_input(self, circle_image):
        gray = cv2.cvtColor(circle_image, cv2.COLOR_BGR2GRAY)
        detector = BlobDetector(threshold_method="manual", threshold_low=50)
        blobs = detector.detect(gray)
        assert len(blobs) >= 1

    def test_gray_matcher_grayscale_template(self, template_and_search):
        tpl, search = template_and_search
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        matcher = GrayMatcher(angle_step=45.0)
        matcher.create_template(tpl_gray)
        results = matcher.match(search, min_score=20.0)
        assert isinstance(results, list)

    def test_blob_dark_object(self, blank_image):
        # 在白色背景上画黑色圆
        blank_image[:] = 255
        cv2.circle(blank_image, (150, 150), 40, (0, 0, 0), -1)
        detector = BlobDetector(threshold_method="manual", threshold_low=50, object_type="dark")
        blobs = detector.detect(blank_image)
        assert len(blobs) >= 1

    def test_blob_fill_holes(self, blank_image):
        cv2.circle(blank_image, (150, 150), 50, (255, 255, 255), -1)
        cv2.circle(blank_image, (150, 150), 20, (0, 0, 0), -1)
        detector = BlobDetector(threshold_method="manual", threshold_low=50, fill_holes=True)
        blobs = detector.detect(blank_image)
        # 填充后应只有一个 Blob
        assert len(blobs) >= 1
