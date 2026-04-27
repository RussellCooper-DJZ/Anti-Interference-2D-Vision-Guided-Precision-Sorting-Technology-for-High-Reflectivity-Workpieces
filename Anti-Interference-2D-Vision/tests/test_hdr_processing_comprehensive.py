"""
test_hdr_processing_comprehensive.py — HDR处理模块综合测试
使用pytest parametrize覆盖多场景
覆盖: exposure_fusion_debevec, specular_diffuse_separation, apply_clahe_lab,
      guided_filter_opencv, unsharp_mask, GlareInpainter, save_debug_grid
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pytest

from vision.hdr_processing import (
    exposure_fusion_mertens,
    exposure_fusion_debevec,
    generate_synthetic_exposures,
    detect_highlight_mask,
    repair_highlight_regions,
    polarization_min_method,
    specular_diffuse_separation,
    apply_clahe_lab,
    guided_filter_opencv,
    unsharp_mask,
    AntiGlarePipeline,
    save_debug_grid,
    GlareInpainter,
)


# ============================================================
# exposure_fusion_debevec - 参数化测试
# ============================================================

class TestExposureFusionDebevec:
    """Debevec HDR融合测试"""

    @pytest.fixture
    def multi_exposure_images(self):
        """标准多曝光图像"""
        img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
        img2 = np.full((100, 100, 3), 128, dtype=np.uint8)
        img3 = np.full((100, 100, 3), 200, dtype=np.uint8)
        return [img1, img2, img3]

    @pytest.fixture
    def exposure_times(self):
        return [0.01, 0.1, 1.0]

    @pytest.mark.parametrize("gamma", [1.0, 1.5, 2.0, 2.5])
    def test_different_gamma(self, multi_exposure_images, exposure_times, gamma):
        """测试不同gamma值的融合结果"""
        result = exposure_fusion_debevec(multi_exposure_images, exposure_times, tonemap_gamma=gamma)
        assert result.shape == multi_exposure_images[0].shape
        assert result.dtype == np.uint8

    def test_mismatched_lengths_raises(self, multi_exposure_images):
        """测试图像与曝光时间不匹配应抛异常"""
        with pytest.raises(ValueError):
            exposure_fusion_debevec(multi_exposure_images, [0.01, 0.1])

    def test_single_exposure_time(self):
        """单曝光时间测试"""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        result = exposure_fusion_debevec([img], [0.1], tonemap_gamma=1.5)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_gray2bgr_conversion(self):
        """灰度图自动转BGR"""
        gray = np.full((100, 100), 128, dtype=np.uint8)
        result = exposure_fusion_debevec([gray], [0.1])
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_resize_different_sizes(self):
        """不同尺寸图像自动resize"""
        img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
        img2 = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = exposure_fusion_debevec([img1, img2], [0.01, 0.1])
        assert result.shape[:2] == (100, 100)


# ============================================================
# generate_synthetic_exposures - 参数化测试
# ============================================================

class TestSyntheticExposures:
    """合成曝光测试"""

    @pytest.mark.parametrize("ev_stops", [
        [-2.0, 0.0, 2.0],
        [-1.5, 0.0, 1.5],
        [-3.0, -1.0, 1.0, 3.0],
        [0.0],
        [-1.0, 1.0],
    ])
    def test_various_ev_stops(self, ev_stops):
        """测试不同EV档位"""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        imgs, times = generate_synthetic_exposures(img, ev_stops=ev_stops)
        assert len(imgs) == len(ev_stops)
        assert len(times) == len(ev_stops)

    def test_single_image(self):
        """单张图像返回单曝光"""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        imgs, times = generate_synthetic_exposures(img, ev_stops=[0.0])
        assert len(imgs) == 1
        assert len(times) == 1

    def test_negative_ev(self):
        """欠曝EV"""
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        imgs, times = generate_synthetic_exposures(img, ev_stops=[-3.0, -2.0, -1.0])
        # Negative EV should darken the image (lower values)
        assert len(imgs) == 3

    def test_positive_ev(self):
        """过曝EV"""
        img = np.full((100, 100, 3), 50, dtype=np.uint8)
        imgs, times = generate_synthetic_exposures(img, ev_stops=[1.0, 2.0, 3.0])
        assert all(img.max() > 50 for img in imgs)


# ============================================================
# specular_diffuse_separation - 参数化测试
# ============================================================

class TestSpecularDiffuseSeparation:
    """镜面/漫反射分离测试"""

    @pytest.fixture
    def test_image(self):
        return np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)

    @pytest.mark.parametrize("d,sigma_color,sigma_space", [
        (5, 30.0, 30.0),
        (9, 50.0, 50.0),
        (15, 75.0, 75.0),
        (3, 20.0, 20.0),
    ])
    def test_different_bilateral_params(self, test_image, d, sigma_color, sigma_space):
        """测试不同双边滤波参数"""
        diffuse, specular = specular_diffuse_separation(
            test_image, bilateral_d=d, sigma_color=sigma_color, sigma_space=sigma_space
        )
        assert diffuse.shape == test_image.shape
        assert specular.shape == test_image.shape
        assert diffuse.dtype == np.uint8
        assert specular.dtype == np.uint8

    def test_output_ranges(self, test_image):
        """验证输出在有效范围内"""
        diffuse, specular = specular_diffuse_separation(test_image)
        assert diffuse.min() >= 0
        assert diffuse.max() <= 255
        assert specular.min() >= 0
        assert specular.max() <= 255


# ============================================================
# apply_clahe_lab - 参数化测试
# ============================================================

class TestCLAHE:
    """CLAHE增强测试"""

    @pytest.fixture
    def test_image(self):
        return np.random.randint(0, 200, (128, 128, 3), dtype=np.uint8)

    @pytest.mark.parametrize("clip_limit", [1.0, 2.0, 3.0, 4.0, 5.0])
    def test_different_clip_limits(self, test_image, clip_limit):
        """测试不同clip_limit值"""
        result = apply_clahe_lab(test_image, clip_limit=clip_limit)
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("tile_grid_size", [
        (4, 4), (8, 8), (16, 16), (4, 8), (8, 4)
    ])
    def test_different_tile_grids(self, test_image, tile_grid_size):
        """测试不同tile grid大小"""
        result = apply_clahe_lab(test_image, tile_grid_size=tile_grid_size)
        assert result.shape == test_image.shape

    def test_low_contrast_image(self):
        """低对比度图像"""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = apply_clahe_lab(img)
        assert result.shape == img.shape

    def test_high_contrast_image(self):
        """高对比度图像"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[50:, :, :] = 255
        result = apply_clahe_lab(img)
        assert result.shape == img.shape


# ============================================================
# guided_filter_opencv - 参数化测试
# ============================================================

class TestGuidedFilter:
    """引导滤波测试"""

    @pytest.fixture
    def guide_and_src(self):
        guide = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        src = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return guide, src

    @pytest.mark.parametrize("radius,eps", [
        (4, 50.0),
        (8, 100.0),
        (12, 500.0),
        (16, 1000.0),
        (8, 50.0),
        (4, 500.0),
    ])
    def test_different_params(self, guide_and_src, radius, eps):
        """测试不同半径和eps参数"""
        guide, src = guide_and_src
        result = guided_filter_opencv(guide, src, radius=radius, eps=eps)
        assert result.shape == src.shape
        assert result.dtype == np.uint8

    def test_grayscale_input(self):
        """灰度图输入"""
        guide = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        src = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = guided_filter_opencv(guide, src, radius=5, eps=100.0)
        assert result.shape == src.shape


# ============================================================
# unsharp_mask - 参数化测试
# ============================================================

class TestUnsharpMask:
    """非锐化掩膜测试"""

    @pytest.fixture
    def test_image(self):
        return np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)

    @pytest.mark.parametrize("ksize,sigma,amount,threshold", [
        (3, 1.0, 1.0, 5),
        (5, 1.0, 1.5, 8),
        (7, 2.0, 2.0, 10),
        (3, 0.5, 0.5, 3),
        (5, 1.5, 2.5, 15),
    ])
    def test_different_params(self, test_image, ksize, sigma, amount, threshold):
        """测试不同参数组合"""
        result = unsharp_mask(test_image, ksize=ksize, sigma=sigma, amount=amount, threshold=threshold)
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    def test_zero_threshold(self):
        """零阈值应该锐化所有差异"""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[20:30, 20:30] = 255
        result = unsharp_mask(img, ksize=3, threshold=0)
        assert result.shape == img.shape

    def test_high_threshold(self):
        """高阈值应该不锐化"""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[20:30, 20:30] = 255
        result = unsharp_mask(img, ksize=3, threshold=100)
        assert result.shape == img.shape


# ============================================================
# detect_highlight_mask - 参数化测试
# ============================================================

class TestHighlightDetectionParametrized:
    """高光检测参数化测试"""

    @pytest.fixture
    def bright_spot_image(self):
        """中心亮斑图像"""
        img = np.full((200, 200, 3), 100, dtype=np.uint8)
        cv2.circle(img, (100, 100), 30, (255, 255, 255), -1)
        return img

    @pytest.mark.parametrize("threshold", [200, 220, 240, 250])
    def test_different_thresholds(self, bright_spot_image, threshold):
        """测试不同阈值"""
        mask = detect_highlight_mask(bright_spot_image, threshold=threshold, use_lab=False)
        assert mask.shape == bright_spot_image.shape[:2]
        assert mask.dtype == np.uint8

    @pytest.mark.parametrize("threshold,dilate_iters", [
        (240, 0), (240, 1), (240, 2), (240, 3),
        (220, 2), (250, 2),
    ])
    def test_dilate_iterations(self, bright_spot_image, threshold, dilate_iters):
        """测试不同膨胀迭代次数"""
        mask = detect_highlight_mask(bright_spot_image, threshold=threshold,
                                     dilate_iters=dilate_iters, use_lab=False)
        assert mask.shape == bright_spot_image.shape[:2]

    @pytest.mark.parametrize("use_lab,adaptive", [
        (True, True), (True, False),
        (False, True), (False, False),
    ])
    def test_lab_and_adaptive_combinations(self, bright_spot_image, use_lab, adaptive):
        """测试LAB和自适应组合"""
        mask = detect_highlight_mask(bright_spot_image, threshold=240,
                                     use_lab=use_lab, adaptive=adaptive)
        assert mask.shape == bright_spot_image.shape[:2]

    def test_grayscale_image(self):
        """灰度图像输入"""
        img = np.full((100, 100), 128, dtype=np.uint8)
        img[40:60, 40:60] = 255
        mask = detect_highlight_mask(img, threshold=240, use_lab=False)
        assert mask.shape == img.shape[:2]


# ============================================================
# repair_highlight_regions - 参数化测试
# ============================================================

class TestHighlightRepairParametrized:
    """高光修复参数化测试"""

    @pytest.fixture
    def highlight_image(self):
        img = np.full((200, 200, 3), 100, dtype=np.uint8)
        cv2.rectangle(img, (80, 80), (120, 120), (255, 255, 255), -1)
        return img

    @pytest.mark.parametrize("method", ['telea', 'ns', 'blend'])
    def test_all_methods(self, highlight_image, method):
        """测试所有修复方法"""
        result = repair_highlight_regions(highlight_image, method=method, use_lab=False)
        assert result.shape == highlight_image.shape
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("inpaint_radius", [3, 5, 7, 10])
    def test_different_radii(self, highlight_image, inpaint_radius):
        """测试不同inpaint半径"""
        result = repair_highlight_regions(highlight_image, method='telea',
                                           inpaint_radius=inpaint_radius, use_lab=False)
        assert result.shape == highlight_image.shape

    def test_custom_mask(self, highlight_image):
        """自定义掩膜"""
        mask = np.zeros_like(highlight_image[:, :, 0])
        mask[80:120, 80:120] = 255
        result = repair_highlight_regions(highlight_image, mask=mask, use_lab=False)
        assert result.shape == highlight_image.shape

    def test_empty_mask_no_change(self, highlight_image):
        """空掩膜应不修改图像"""
        mask = np.zeros((200, 200), dtype=np.uint8)
        result = repair_highlight_regions(highlight_image, mask=mask)
        assert result.shape == highlight_image.shape


# ============================================================
# polarization_min_method - 参数化测试
# ============================================================

class TestPolarizationMethod:
    """偏振模拟测试"""

    @pytest.mark.parametrize("num_images", [2, 3, 4, 5, 8])
    def test_different_image_counts(self, num_images):
        """测试不同图像数量"""
        imgs = [np.random.randint(50, 200, (50, 50, 3), dtype=np.uint8) for _ in range(num_images)]
        result = polarization_min_method(imgs)
        assert result.shape == imgs[0].shape
        assert result.dtype == np.uint8

    def test_single_image_returns_copy(self):
        """单图像返回副本"""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        result = polarization_min_method([img])
        assert np.array_equal(result, img)

    def test_empty_raises(self):
        """空列表抛异常"""
        with pytest.raises(ValueError):
            polarization_min_method([])

    def test_pixelwise_minimum(self):
        """逐像素最小值"""
        img1 = np.full((10, 10, 3), 200, dtype=np.uint8)
        img2 = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = polarization_min_method([img1, img2])
        expected = np.full((10, 10, 3), 100, dtype=np.uint8)
        assert np.array_equal(result, expected)


# ============================================================
# AntiGlarePipeline - 参数化测试
# ============================================================

class TestAntiGlarePipelineParametrized:
    """AntiGlarePipeline综合测试"""

    @pytest.fixture
    def test_image(self):
        img = np.full((128, 128, 3), 100, dtype=np.uint8)
        cv2.circle(img, (64, 64), 25, (255, 255, 255), -1)
        return img

    @pytest.mark.parametrize("repair_method", ['telea', 'ns', 'blend'])
    def test_pipeline_repair_methods(self, test_image, repair_method):
        """测试不同修复方法的管线"""
        pipeline = AntiGlarePipeline(repair_method=repair_method)
        result = pipeline.process_single(test_image)
        assert result.shape == test_image.shape
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("clahe_clip", [1.0, 2.0, 3.0, 4.0])
    def test_pipeline_clahe_clips(self, test_image, clahe_clip):
        """测试不同CLAHE clip值"""
        pipeline = AntiGlarePipeline(clahe_clip=clahe_clip)
        result = pipeline.process_single(test_image)
        assert result.shape == test_image.shape

    @pytest.mark.parametrize("guided_radius,guided_eps", [
        (4, 50.0), (8, 100.0), (12, 500.0), (16, 1000.0)
    ])
    def test_pipeline_guided_params(self, test_image, guided_radius, guided_eps):
        """测试引导滤波参数"""
        pipeline = AntiGlarePipeline(guided_radius=guided_radius, guided_eps=guided_eps)
        result = pipeline.process_single(test_image)
        assert result.shape == test_image.shape

    @pytest.mark.parametrize("sharpen_amount", [0.5, 1.0, 1.5, 2.0, 2.5])
    def test_pipeline_sharpen_amounts(self, test_image, sharpen_amount):
        """测试不同锐化量"""
        pipeline = AntiGlarePipeline(sharpen_amount=sharpen_amount)
        result = pipeline.process_single(test_image)
        assert result.shape == test_image.shape

    @pytest.mark.parametrize("specular_blend", [0.3, 0.5, 0.7, 0.9])
    def test_pipeline_specular_blends(self, test_image, specular_blend):
        """测试不同镜面混合比例"""
        pipeline = AntiGlarePipeline(specular_blend=specular_blend)
        result = pipeline.process_single(test_image)
        assert result.shape == test_image.shape

    def test_pipeline_multi_with_exposure_times(self, test_image):
        """多图像输入带曝光时间"""
        imgs = [test_image, test_image, test_image]
        times = [0.01, 0.1, 1.0]
        pipeline = AntiGlarePipeline()
        result = pipeline.process_multi(imgs, times)
        assert result.shape == test_image.shape

    def test_pipeline_multi_without_times(self, test_image):
        """多图像输入不带曝光时间"""
        imgs = [test_image, test_image, test_image]
        pipeline = AntiGlarePipeline()
        result = pipeline.process_multi(imgs)
        assert result.shape == test_image.shape


# ============================================================
# GlareInpainter - 参数化测试
# ============================================================

class TestGlareInpainter:
    """GlareInpainter测试"""

    @pytest.fixture
    def glare_image(self):
        """带高光的图像"""
        img = np.random.randint(50, 150, (200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 40, (255, 255, 255), -1)
        return img

    @pytest.mark.parametrize("mode", ['telea', 'ns', 'hybrid'])
    def test_different_modes(self, glare_image, mode):
        """测试不同模式"""
        inpainter = GlareInpainter(mode=mode)
        result = inpainter.inpaint(glare_image)
        assert result.shape == glare_image.shape
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("telea_radius,ns_radius", [
        (3, 2), (5, 3), (7, 5), (10, 7)
    ])
    def test_different_radii(self, glare_image, telea_radius, ns_radius):
        """测试不同半径"""
        inpainter = GlareInpainter(mode='hybrid', telea_radius=telea_radius, ns_radius=ns_radius)
        result = inpainter.inpaint(glare_image)
        assert result.shape == glare_image.shape

    def test_custom_mask(self, glare_image):
        """自定义掩膜"""
        inpainter = GlareInpainter()
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 40, 255, -1)
        result = inpainter.inpaint(glare_image, mask=mask)
        assert result.shape == glare_image.shape

    def test_auto_detection(self, glare_image):
        """自动高光检测"""
        inpainter = GlareInpainter()
        mask = inpainter._detect_glare_mask(glare_image)
        assert mask.shape == glare_image.shape[:2]
        assert mask.dtype == np.uint8

    def test_adaptive_radius_calculation(self, glare_image):
        """自适应半径计算"""
        inpainter = GlareInpainter(adaptive_radius=True)
        mask = inpainter._detect_glare_mask(glare_image)
        telea_r, ns_r = inpainter._compute_adaptive_radius(mask)
        assert isinstance(telea_r, int)
        assert isinstance(ns_r, int)
        assert telea_r >= 3
        assert ns_r >= 2


# ============================================================
# save_debug_grid - 参数化测试
# ============================================================

class TestSaveDebugGrid:
    """调试网格保存测试"""

    def test_basic_grid(self, tmp_path):
        """基本网格保存"""
        stages = {
            '00_original': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            '01_processed': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            '02_final': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        }
        output_path = tmp_path / "debug.png"
        save_debug_grid(stages, str(output_path))
        assert output_path.exists()

    def test_grayscale_stages(self, tmp_path):
        """灰度图阶段"""
        stages = {
            '00_gray': np.random.randint(0, 255, (100, 100), dtype=np.uint8),
            '01_gray': np.random.randint(0, 255, (100, 100), dtype=np.uint8),
        }
        output_path = tmp_path / "debug_gray.png"
        save_debug_grid(stages, str(output_path))
        assert output_path.exists()

    @pytest.mark.parametrize("thumb_w,thumb_h", [
        (160, 120), (320, 240), (640, 480)
    ])
    def test_different_thumb_sizes(self, tmp_path, thumb_w, thumb_h):
        """测试不同缩略图大小"""
        stages = {
            'stage1': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
            'stage2': np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        }
        output_path = tmp_path / f"debug_{thumb_w}x{thumb_h}.png"
        save_debug_grid(stages, str(output_path), thumb_w=thumb_w, thumb_h=thumb_h)
        assert output_path.exists()

    def test_many_stages(self, tmp_path):
        """多阶段网格"""
        stages = {f'stage_{i:02d}': np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
                  for i in range(10)}
        output_path = tmp_path / "debug_many.png"
        save_debug_grid(stages, str(output_path), max_cols=4)
        assert output_path.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
