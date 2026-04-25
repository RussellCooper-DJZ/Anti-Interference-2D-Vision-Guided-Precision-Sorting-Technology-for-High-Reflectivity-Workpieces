"""
data — 数据处理与生成包

包含以下子模块：
  synth_national_scenes  : 全国 8 大场景合成训练图像生成器
  synth_dataset_generator: 基础合成数据集生成器
  data_augmentation      : 高反光专用数据增强管线
  real_world_dataloader  : 真实场景数据加载器与 DataLoader 工厂
"""

__all__ = [
    "SynthShipDataset",
    "RealShipDataset",
    "MixedShipDataset",
    "OnlineSynthDataset",
    "build_dataloaders",
    "ShipHullAugPipeline",
    "generate_edge_from_mask",
]
