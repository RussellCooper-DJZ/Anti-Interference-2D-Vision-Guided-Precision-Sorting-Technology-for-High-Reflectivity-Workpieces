"""抗高光视觉核心算法的无硬件单元测试。"""

import random

import numpy as np

from src.vision.glare_simulator import GlareSimulator
from src.vision.hdr_processing import (
    detect_highlight_mask,
    exposure_fusion_mertens,
    generate_synthetic_exposures,
)


def test_glare_simulator_increases_highlight_energy() -> None:
    """合成光斑应让均匀输入中至少部分像素亮度提高。"""
    random.seed(7)
    np.random.seed(7)
    image = np.full((128, 128, 3), 80, dtype=np.uint8)
    workpiece_mask = np.full((128, 128), 255, dtype=np.uint8)

    augmented = GlareSimulator(max_glare_blobs=1, intensity_range=(200, 200), size_range=(0.1, 0.1)).apply(
        image, workpiece_mask
    )

    assert augmented.shape == image.shape
    assert augmented.dtype == np.uint8
    assert int(augmented.max()) > int(image.max())
    assert float(augmented.mean()) > float(image.mean())


def test_multi_exposure_sequence_has_expected_brightness_order() -> None:
    """模拟多曝光序列应输出欠曝、正常、过曝三帧且亮度递增。"""
    random.seed(11)
    image = np.full((96, 128, 3), 70, dtype=np.uint8)

    under, normal, over = GlareSimulator(max_glare_blobs=1).generate_multi_exposure_sim(image)

    assert len((under, normal, over)) == 3
    assert under.shape == normal.shape == over.shape == image.shape
    assert under.mean() < normal.mean() < over.mean()


def test_highlight_mask_detects_saturated_region() -> None:
    """高光掩膜应覆盖人为设置的饱和区域。"""
    image = np.full((128, 128, 3), 80, dtype=np.uint8)
    image[48:80, 48:80] = 255

    highlight_mask = detect_highlight_mask(image, threshold=240, dilate_iters=0)

    assert highlight_mask.dtype == np.uint8
    assert highlight_mask[64, 64] == 255
    assert highlight_mask[10, 10] == 0


def test_mertens_hdr_fusion_returns_valid_color_frame() -> None:
    """Mertens 融合应返回与输入尺寸一致的有效三通道图像。"""
    np.random.seed(17)
    x_gradient = np.tile(np.linspace(15, 230, 96, dtype=np.uint8), (64, 1))
    image = np.dstack([x_gradient, x_gradient, x_gradient])
    exposures, _ = generate_synthetic_exposures(image, ev_stops=(-1.0, 0.0, 1.0))

    fused = exposure_fusion_mertens(exposures)

    assert fused.shape == image.shape
    assert fused.dtype == np.uint8
    assert 0 <= int(fused.min()) <= int(fused.max()) <= 255
    assert float(fused.std()) > 0.0


def test_empty_exposure_list_is_rejected() -> None:
    """输入为空时必须显式报错，避免嵌入式上游调用静默失败。"""
    try:
        exposure_fusion_mertens([])
    except ValueError as error:
        assert "不能为空" in str(error)
    else:
        raise AssertionError("Expected ValueError for an empty exposure sequence")
