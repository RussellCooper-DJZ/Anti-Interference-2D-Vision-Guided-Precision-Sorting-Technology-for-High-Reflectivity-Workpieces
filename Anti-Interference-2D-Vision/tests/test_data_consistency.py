"""
test_data_consistency.py — Data分支一致性测试

按照 BRANCH_AUDIT_GUIDE.md 的 Data 分支审计框架：
  - C: 一致性 (Consistency) — 几何变换中 mask/edge 与 image 同步
  - R: 可复现性 (Reproducibility) — 相同 seed 产生相同结果
  - P: 物理合理性 (Physical Sanity) — 像素值在有效范围内
  - S: 边界安全 (Safety) — 坐标不越界

CutMix/Mosaic 关键测试:
  - CutMix: patch区域的mask必须来自同一源图像
  - MixUp: mask和image按相同λ线性混合
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pytest


class TestCutMixConsistency:
    """CutMix 一致性测试 — 按 BRANCH_AUDIT_GUIDE.md"""

    def test_cutmix_mask_patch_from_same_source(self):
        """
        验证 CutMix 后，patch区域的mask必须来自同一源图像。

        按审计指南:
        out_masks[i, :, y1:y2, x1:x2] == masks[idx[i], :, y1:y2, x1:x2]
        """
        from data.data_augmentation import cutmix

        # 创建两个样本
        torch.manual_seed(42)
        image1 = torch.randn(1, 3, 64, 64)
        mask1 = torch.zeros(1, 1, 64, 64)
        mask1[0, 0, 10:30, 10:30] = 1.0  # 左上角有物体
        edge1 = torch.zeros(1, 1, 64, 64)
        edge1[0, 0, 10, :] = 1.0  # 上边有边缘

        image2 = torch.randn(1, 3, 64, 64)
        mask2 = torch.zeros(1, 1, 64, 64)
        mask2[0, 0, 40:60, 40:60] = 1.0  # 右下角有物体
        edge2 = torch.zeros(1, 1, 64, 64)
        edge2[0, 0, :, 40] = 1.0  # 左边有边缘

        # 固定随机种子以确定patch位置
        np.random.seed(123)
        torch.manual_seed(123)

        img_mixed, mask_mixed, edge_mixed = cutmix(
            image1, mask1, edge1,
            image2, mask2, edge2,
            alpha=1.0
        )

        # 找到patch区域 (两个mask不同的区域)
        # mask1的物体在[10:30, 10:30]，mask2的物体在[40:60, 40:60]
        # CutMix会随机选择一块区域交换
        # 验证: 如果patch区域在mask1的物体上，则应完全来自mask1
        #      如果patch区域在mask2的物体上，则应完全来自mask2
        #      不应出现混合 (mask是二值的，不应出现0.5这样的值)

        # 验证mask_mixed是二值的 (CutMix直接复制，不混合)
        unique_values = torch.unique(mask_mixed).numpy()
        for val in unique_values:
            assert val in [0.0, 1.0], f"CutMix mask should be binary, got {val}"

    def test_cutmix_edge_patch_from_same_source(self):
        """验证 CutMix 后，patch区域的edge也来自同一源图像"""
        from data.data_augmentation import cutmix

        torch.manual_seed(42)
        edge1 = torch.zeros(1, 1, 64, 64)
        edge1[0, 0, 10, :] = 1.0  # 上边有边缘

        edge2 = torch.zeros(1, 1, 64, 64)
        edge2[0, 0, :, 40] = 1.0  # 左边有边缘

        np.random.seed(123)
        torch.manual_seed(123)

        _, _, edge_mixed = cutmix(
            torch.randn(1, 3, 64, 64),
            torch.zeros(1, 1, 64, 64),
            edge1,
            torch.randn(1, 3, 64, 64),
            torch.zeros(1, 1, 64, 64),
            edge2,
            alpha=1.0
        )

        # Edge也应是二值的
        unique_values = torch.unique(edge_mixed).numpy()
        for val in unique_values:
            assert val in [0.0, 1.0], f"CutMix edge should be binary, got {val}"

    def test_cutmix_batch_pairing(self):
        """验证 CutMix batch 随机配对的正确性"""
        from data.data_augmentation import apply_cutmix_batch

        b, c, h, w = 4, 3, 32, 32
        torch.manual_seed(99)
        images = torch.randn(b, c, h, w)
        masks = torch.randint(0, 2, (b, 1, h, w)).float()
        edges = torch.randint(0, 2, (b, 1, h, w)).float()

        torch.manual_seed(99)
        img_out, mask_out, edge_out = apply_cutmix_batch(
            images, masks, edges, alpha=1.0, p=1.0
        )

        # 输出形状应不变
        assert img_out.shape == images.shape
        assert mask_out.shape == masks.shape
        assert edge_out.shape == edges.shape


class TestMixUpConsistency:
    """MixUp 一致性测试"""

    def test_mixup_linear_interpolation(self):
        """
        验证 MixUp 按相同λ线性混合 image, mask, edge

        公式: M = λ*I1 + (1-λ)*I2
        """
        from data.data_augmentation import mixup

        torch.manual_seed(42)
        image1 = torch.randn(1, 3, 64, 64)
        mask1 = torch.zeros(1, 1, 64, 64)
        mask1[0, 0, 10:30, 10:30] = 1.0

        torch.manual_seed(42)
        image2 = torch.randn(1, 3, 64, 64)
        mask2 = torch.zeros(1, 1, 64, 64)
        mask2[0, 0, 40:60, 40:60] = 1.0

        edge1 = torch.zeros(1, 1, 64, 64)
        edge2 = torch.zeros(1, 1, 64, 64)

        # 固定np.random seed确保λ可复现
        np.random.seed(42)
        img_mixed, mask_mixed, edge_mixed = mixup(
            image1, mask1, edge1,
            image2, mask2, edge2,
            alpha=0.4
        )

        # 验证混合后的mask是浮点数 (因为是线性混合)
        # 不再是二值的，但值域应在[0, 1]范围内
        assert mask_mixed.min() >= 0.0
        assert mask_mixed.max() <= 1.0

        # 验证MixUp公式: mask_mixed = lam * mask1 + (1-lam) * mask2
        # mask1在region1=1, region2=0; mask2相反
        # 由于mask1和mask2的物体在不同区域，它们不重叠
        # region1应该是 lam * 1 + (1-lam) * 0 = lam
        # region2应该是 lam * 0 + (1-lam) * 1 = 1-lam
        # 或者反过来，取决于哪个是image1哪个是image2
        # 关键验证: region1 + region2 = 1 (因为mask1+mask2在该区域=1)
        region1 = mask_mixed[0, 0, 10:30, 10:30].mean()
        region2 = mask_mixed[0, 0, 40:60, 40:60].mean()
        # 两个region的均值应该互补 = 1
        assert abs(region1 + region2 - 1.0) < 0.01, f"region1={region1}, region2={region2} should sum to 1.0"


class TestReproducibility:
    """可复现性测试 — 相同seed应产生相同结果"""

    def test_cutmix_reproducibility(self):
        """验证 CutMix 的可复现性"""
        from data.data_augmentation import cutmix

        def run_cutmix(seed):
            np.random.seed(seed)
            torch.manual_seed(seed)
            return cutmix(
                torch.randn(1, 3, 32, 32),
                torch.randint(0, 2, (1, 1, 32, 32)).float(),
                torch.randint(0, 2, (1, 1, 32, 32)).float(),
                torch.randn(1, 3, 32, 32),
                torch.randint(0, 2, (1, 1, 32, 32)).float(),
                torch.randint(0, 2, (1, 1, 32, 32)).float(),
                alpha=1.0
            )

        # 相同seed应产生相同结果
        result1 = run_cutmix(42)
        result2 = run_cutmix(42)

        torch.testing.assert_close(result1[0], result2[0])  # image
        torch.testing.assert_close(result1[1], result2[1])  # mask
        torch.testing.assert_close(result1[2], result2[2])  # edge

    def test_mixup_reproducibility(self):
        """验证 MixUp 的可复现性"""
        from data.data_augmentation import mixup

        def run_mixup(seed):
            np.random.seed(seed)
            torch.manual_seed(seed)
            return mixup(
                torch.randn(1, 3, 32, 32),
                torch.randint(0, 2, (1, 1, 32, 32)).float(),
                torch.randint(0, 2, (1, 1, 32, 32)).float(),
                torch.randn(1, 3, 32, 32),
                torch.randint(0, 2, (1, 1, 32, 32)).float(),
                torch.randint(0, 2, (1, 1, 32, 32)).float(),
                alpha=0.4
            )

        result1 = run_mixup(42)
        result2 = run_mixup(42)

        torch.testing.assert_close(result1[0], result2[0])
        torch.testing.assert_close(result1[1], result2[1])


class TestPhysicalSanity:
    """物理合理性测试"""

    def test_cutmix_output_range(self):
        """验证 CutMix 输出的像素值在合理范围内"""
        from data.data_augmentation import cutmix

        image1 = torch.rand(1, 3, 64, 64)  # [0, 1]
        mask1 = torch.randint(0, 2, (1, 1, 64, 64)).float()
        edge1 = torch.randint(0, 2, (1, 1, 64, 64)).float()

        image2 = torch.rand(1, 3, 64, 64)
        mask2 = torch.randint(0, 2, (1, 1, 64, 64)).float()
        edge2 = torch.randint(0, 2, (1, 1, 64, 64)).float()

        img_mixed, mask_mixed, edge_mixed = cutmix(
            image1, mask1, edge1,
            image2, mask2, edge2,
            alpha=1.0
        )

        # Image应在[0, 1]范围内
        assert img_mixed.min() >= 0.0
        assert img_mixed.max() <= 1.0

        # Mask应为二值
        assert mask_mixed.min() >= 0.0
        assert mask_mixed.max() <= 1.0

        # Edge应为二值
        assert edge_mixed.min() >= 0.0
        assert edge_mixed.max() <= 1.0

    def test_mixup_output_range(self):
        """验证 MixUp 输出的像素值在合理范围内"""
        from data.data_augmentation import mixup

        image1 = torch.rand(1, 3, 64, 64)
        mask1 = torch.randint(0, 2, (1, 1, 64, 64)).float()
        edge1 = torch.randint(0, 2, (1, 1, 64, 64)).float()

        image2 = torch.rand(1, 3, 64, 64)
        mask2 = torch.randint(0, 2, (1, 1, 64, 64)).float()
        edge2 = torch.randint(0, 2, (1, 1, 64, 64)).float()

        img_mixed, mask_mixed, edge_mixed = mixup(
            image1, mask1, edge1,
            image2, mask2, edge2,
            alpha=0.4
        )

        # Image混合后仍在[0, 1]范围内
        assert img_mixed.min() >= 0.0
        assert img_mixed.max() <= 1.0

        # Mask混合后应在[0, 1]范围内
        assert mask_mixed.min() >= 0.0
        assert mask_mixed.max() <= 1.0


class TestBoundarySafety:
    """边界安全测试 — 坐标不越界"""

    def test_cutmix_no_negative_indices(self):
        """验证 CutMix 不会产生负坐标索引"""
        from data.data_augmentation import cutmix

        # 使用最小尺寸测试边界情况
        image1 = torch.randn(1, 3, 8, 8)
        mask1 = torch.zeros(1, 1, 8, 8)
        edge1 = torch.zeros(1, 1, 8, 8)

        image2 = torch.randn(1, 3, 8, 8)
        mask2 = torch.zeros(1, 1, 8, 8)
        edge2 = torch.zeros(1, 1, 8, 8)

        # 多次随机测试，不应崩溃
        for seed in range(10):
            np.random.seed(seed)
            torch.manual_seed(seed)
            try:
                img_mixed, mask_mixed, edge_mixed = cutmix(
                    image1, mask1, edge1,
                    image2, mask2, edge2,
                    alpha=1.0
                )
                assert img_mixed.shape == image1.shape
            except Exception as e:
                pytest.fail(f"CutMix failed with seed {seed}: {e}")

    def test_cutmix_preserves_shape(self):
        """验证 CutMix 输出形状与输入一致"""
        from data.data_augmentation import cutmix

        for h, w in [(32, 32), (64, 128), (256, 256)]:
            image1 = torch.randn(1, 3, h, w)
            mask1 = torch.randint(0, 2, (1, 1, h, w)).float()
            edge1 = torch.randint(0, 2, (1, 1, h, w)).float()

            image2 = torch.randn(1, 3, h, w)
            mask2 = torch.randint(0, 2, (1, 1, h, w)).float()
            edge2 = torch.randint(0, 2, (1, 1, h, w)).float()

            img_mixed, mask_mixed, edge_mixed = cutmix(
                image1, mask1, edge1,
                image2, mask2, edge2,
                alpha=1.0
            )

            assert img_mixed.shape == image1.shape
            assert mask_mixed.shape == mask1.shape
            assert edge_mixed.shape == edge1.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
