# Anti-Interference 2D Vision-Guided Precision Sorting for High-Reflectivity Workpieces

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

经过模块化重构，当前代码库结构如下：

```text
repo/
├── vision/                         # 视觉算法核心模块
│   ├── feature_extraction.py       # AGEANet / AGEANet-Lite 模型架构
│   ├── hdr_processing.py           # HDR 融合与反光抑制管线
│   └── localization_and_calibration.py # 亚像素定位与手眼标定
├── data/                           # 数据处理与生成模块
│   ├── synth_national_scenes.py    # 全国 8 大场景合成训练图像生成器
│   ├── synth_dataset_generator.py  # 基础合成数据集生成器
│   ├── data_augmentation.py        # 高反光专用数据增强管线
│   └── real_world_dataloader.py    # 真实场景数据加载器
├── training/                       # 模型训练模块
│   └── train.py                    # 完整端到端训练流程
├── robot/                          # 机器人控制与仿真接口
│   ├── abb_robotstudio_interface.py# ABB RobotStudio TCP 通信接口
│   └── abb_rapid/                  # ABB RAPID 控制器代码
│       ├── abb_server.mod          # 运行于虚拟控制器的服务端程序
│       └── README_RAPID.md         # RobotStudio 联调使用说明
├── embedded/                       # 嵌入式端部署模块 (Renesas RA8P1)
│   ├── core/                       # EdgeVision-C 纯 C 推理引擎
│   ├── ra8p1_main_app.c            # 嵌入式主程序
│   └── ra8p1_helium_processing.c   # Helium MVE 向量化处理
├── scripts/                        # 独立工具脚本
│   ├── generate_visualization.py   # 完整可视化数据生成脚本
│   ├── inspect_dataset.py          # 数据集检查工具
│   └── labelme_to_mask.py          # LabelMe 标注转换工具
├── docs/                           # 文档与可视化数据
│   ├── technical/                  # 详细技术方案与硬件设计文档
│   ├── visualization/              # 生成的场景样本与统计分析图表
│   └── ABB_RobotStudio_Integration_Guide.md # ABB 仿真集成指南
├── main_pipeline.py                # 顶层系统入口与演示程序
└── requirements.txt                # Python 依赖清单
```

---

## 🚀 快速开始 (Quick Start)

> **注意**：本项目已全面采用标准 `logging` 模块替代 `print` 输出，以提供更规范的日志记录。

### 1. 环境安装
```bash
pip install -r requirements.txt
```

### 2. 运行主流水线演示 (包含视觉处理与 ABB 模拟通信)
```bash
python main_pipeline.py --demo
```

### 3. 生成全国复杂场景合成数据集
```bash
python data/synth_national_scenes.py --n 100 --scene SHIPYARD --light WELD_ARC --output ./dataset
```

### 4. 生成可视化分析报告
```bash
python scripts/generate_visualization.py
```

---

## 许可证与专利声明

本项目采用 **Apache License 2.0** 协议开源。

**专利声明**：本项目实现的 `AGEANet` 架构、`EdgeVision-C` 静态内存管理算法及 `Helium` 优化算子受专利保护。访问源代码不代表获得专利许可，商业用途请联系 **RussellCooper**。详情请参阅 [PATENTS](./PATENTS) 文件。

**合规性声明**：本项目遵循 **Clean-room 工程化**标准独立开发，所有核心算子实现均基于数学定义，未参考任何受保护的第三方源代码。详情请参阅 [CLEAN_ROOM_AUDIT.md](./CLEAN_ROOM_AUDIT.md)。
