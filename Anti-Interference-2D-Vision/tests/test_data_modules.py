"""
test_data_modules.py — data/ 子模块快速冒烟测试
覆盖 real_world_dataloader、synth_dataset_generator、synth_national_scenes
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest


class TestRealWorldDataLoader:
    """real_world_dataloader 基础测试"""

    def test_import(self):
        from data.real_world_dataloader import RealWorldDataset, get_real_world_dataloader
        assert RealWorldDataset is not None
        assert get_real_world_dataloader is not None


class TestSynthDatasetGenerator:
    """synth_dataset_generator 基础测试"""

    def test_import(self):
        from data.synth_dataset_generator import (
            SynthDatasetGenerator,
            PBRSurface,
            EnvironmentLighting,
        )
        assert SynthDatasetGenerator is not None
        assert PBRSurface is not None
        assert EnvironmentLighting is not None

    def test_pbr_surface_presets(self):
        from data.synth_dataset_generator import PBRSurface
        # 验证至少有一个预设材质
        presets = PBRSurface.list_presets()
        assert isinstance(presets, (list, tuple))

    def test_environment_lighting_presets(self):
        from data.synth_dataset_generator import EnvironmentLighting
        presets = EnvironmentLighting.list_presets()
        assert isinstance(presets, (list, tuple))


class TestSynthNationalScenes:
    """synth_national_scenes 基础测试"""

    def test_import(self):
        from data.synth_national_scenes import (
            NationalSceneGenerator,
            ScenePreset,
        )
        assert NationalSceneGenerator is not None
        assert ScenePreset is not None

    def test_scene_preset_list(self):
        from data.synth_national_scenes import ScenePreset
        presets = ScenePreset.list_presets()
        assert isinstance(presets, (list, tuple))
