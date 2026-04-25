# Anti-Interference 2D Vision Guided Precision Sorting

[![CI](https://github.com/RussellCooper-DJZ/Anti-Interference-2D-Vision-Guided-Precision-Sorting-Technology-for-High-Reflectivity-Workpieces/actions/workflows/python-app.yml/badge.svg)](https://github.com/RussellCooper-DJZ/Anti-Interference-2D-Vision-Guided-Precision-Sorting-Technology-for-High-Reflectivity-Workpieces/actions/workflows/python-app.yml)

## 抗干扰 2D 视觉引导高反光工件精准分拣系统

基于 **Renesas RZ/V2H + RA8P1** 平台，使用深度学习实现高反光金属工件（汽车门板钢、铝合金、大型船舶/桥梁高光面等）的精准边缘识别与机器人抓取。

---

## 🌟 核心特性 (Core Features)

| 特性 | 描述 |
|------|------|
| **AGEANet 模型** | Anti-Glare Edge-Aware U-Net，带 CBAM 注意力机制和边缘感知分支 |
| **反光抑制** | HDR 融合 + 高光检测修复 + 偏振模拟 + 自适应增强 |
| **亚像素定位** | Zernike 矩亚像素边缘精修，达到 0.1 像素级精度 |
| **EdgeVision-C 架构** | 纯 C AI 视觉架构，支持分层运行时、INT8 量化与 Helium 优化 |
| **全国复杂场景合成** | 支持船厂、钢厂、桥梁、高铁等 8 大场景 × 7 种复杂光照的训练数据生成 |
| **ABB 仿真集成** | 内置 ABB RobotStudio TCP/IP 通信接口与 RAPID 服务端，支持闭环仿真联调 |

---

## 📁 项目结构 (Project Structure)

经过模块化重构，当前代码库结构遵循标准开源规范：

```text
.
├── src/                            # 源代码核心目录
│   ├── vision/                     # 视觉算法核心模块
│   │   ├── feature_extraction.py   # AGEANet / AGEANet-Lite 模型架构
│   │   ├── hdr_processing.py       # HDR 融合与反光抑制管线
│   │   └── localization_and_calibration.py # 亚像素定位与手眼标定
│   ├── data/                       # 数据处理与生成模块
│   │   ├── synth_national_scenes.py # 全国 8 大场景合成训练图像生成器
│   │   ├── synth_dataset_generator.py # 基础合成数据集生成器
│   │   ├── data_augmentation.py    # 高反光专用数据增强管线
│   │   └── real_world_dataloader.py # 真实场景数据加载器
│   ├── training/                   # 模型训练模块
│   │   └── train.py                # 完整端到端训练流程
│   ├── robot/                      # 机器人控制与仿真接口
│   │   ├── abb_robotstudio_interface.py # ABB RobotStudio TCP 通信接口
│   │   └── abb_rapid/              # ABB RAPID 控制器代码
│   ├── embedded/                   # 嵌入式端部署模块 (Renesas RA8P1)
│   │   ├── core/                   # EdgeVision-C 纯 C 推理引擎
│   │   ├── ra8p1_main_app.c        # 嵌入式主程序
│   │   └── ra8p1_helium_processing.c # Helium MVE 向量化处理
│   └── main_pipeline.py            # 端到端测试主入口
├── docs/                           # 文档目录
│   ├── guides/                     # 详细操作指南 (Dataset, Annotation, etc.)
│   ├── legal/                      # 专利、审计与合规文档
│   ├── technical/                  # 技术设计与方案说明
│   └── visualization/              # 可视化结果与分析图表
├── scripts/                        # 独立工具与辅助脚本
├── requirements.txt                # Python 依赖清单
├── pyproject.toml                  # 项目配置文件
└── LICENSE                         # 许可证
```

---

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
```bash
pip install -r requirements.txt
```

### 2. 数据生成与模型训练
```bash
# 生成合成数据集
python src/data/synth_national_scenes.py
# 开始训练
python src/training/train.py
```

### 3. 运行主管线
```bash
python src/main_pipeline.py
```

---

## 📚 文档指南 (Documentation)

- [数据集采集与标注指南](docs/guides/DATASET_COLLECTION_AND_ANNOTATION_GUIDE.md)
- [ABB RobotStudio 联调指南](docs/robot/README_RAPID.md)
- [嵌入式部署方案](docs/technical/ra8p1_embedded_vision_solution.md)
- [专利与技术披露](docs/legal/Technical_Disclosure_Document.md)

---

## ⚖️ 许可证 (License)

本项目采用 [MIT License](LICENSE) 授权。
