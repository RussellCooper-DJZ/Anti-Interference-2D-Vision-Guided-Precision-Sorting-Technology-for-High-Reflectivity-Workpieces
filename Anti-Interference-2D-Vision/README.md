# Anti-Interference 2D Vision-Guided Precision Sorting for High-Reflectivity Workpieces

[![CI](https://github.com/RussellCooper-DJZ/Anti-Interference-2D-Vision-Guided-Precision-Sorting-Technology-for-High-Reflectivity-Workpieces/actions/workflows/python-app.yml/badge.svg)](https://github.com/RussellCooper-DJZ/Anti-Interference-2D-Vision-Guided-Precision-Sorting-Technology-for-High-Reflectivity-Workpieces/actions/workflows/python-app.yml)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-red.svg)](https://pytorch.org/)

## 抗干扰 2D 视觉引导高反光工件精准分拣系统

基于 **Renesas RZ/V2H + RA8P1** 平台，使用深度学习实现高反光金属工件（汽车门板钢、铝合金，大型船舶/桥梁高光面等）的精准边缘识别与机器人抓取。

**目标**：在 2D 相机的限制下，用低成本方案达到 3D 相机的测量/分拣精度。

---

## 最近更新 (2026-04-25)

### ✅ 已完成修正

| 修正项 | 修改内容 | 状态 |
|--------|---------|------|
| w_edge | 2.0 → 4.0 | ✅ |
| pos_weight | 5.0 → 15.0 | ✅ |
| edge_threshold | 0.15 → 0.40 | ✅ |
| gradient loss | 已实现 BCE*0.4 + Focal*0.3 + Gradient*0.3 | ✅ |
| 异常处理 | 裸except → 具体异常类型 | ✅ |
| 测试覆盖 | 3个 → 16个测试文件，**207 passed, 14 skipped** | ✅ |
| NumPy 2.0 兼容 | np.cross 弃用警告修复 | ✅ |

### 📊 测试状态

```
测试用例: 221 个
通过: 207 passed ✅
跳过: 14 skipped
覆盖率: 29%
```

### 🔄 进行中

| 债务 | 说明 | 状态 |
|------|------|------|
| P1-2 | 重复实现统一 | 4h |
| P1-3 | 测试覆盖>60% | 进行中 (当前29%) |

### 📋 技术债务

详见 [ITERATION_DEBT_REGISTER.md](./docs/ITERATION_DEBT_REGISTER.md)

---

## 核心特性 (Core Features)

### AI 模型架构
| 特性 | 描述 |
|------|------|
| **FLARE** | Anti-Glare Edge-Aware U-Net，融合 CBAM 注意力 + CoordConv 坐标感知 |
| **数学增强模块** | Wavelet Scattering + Fourier Conv + Morphology Layer |
| **IoU 感知检测头** | Anchor-Free FCOS 风格，支持 bbox 回归 + 中心度 + IoU 预测 |
| **多任务损失** | Lovász + CIoU + Focal + Boundary Loss 混合优化 |

### 预处理与反光抑制
| 特性 | 描述 |
|------|------|
| **AntiGlarePipeline** | HDR 多曝光融合 + 高光修复 + 双边滤波 + CLAHE 六级处理 |
| **PBRLightingSystem** | 基于 Blinn-Phong BRDF 的物理高光模拟（新增） |
| **光度立体（神经网络版）** | PhotometricStereoNet — CNN 直接回归法线/反照率（专利规避） |
| **灰度匹配（SSDA）** | Sequential Similarity Detection Algorithm 替代 NCC（专利规避） |

### 亚像素定位与测量
| 特性 | 描述 |
|------|------|
| **SubpixelLocalizer** | 质心估计 + PCA 方向 + 强度加权 + glare 排除 |
| **CaliperMeasurement** | 双平行边缘卡尺测量，支持任意方向搜索 |
| **GapMeasurement** | 多边缘间隙/节距/宽度测量 |
| **ROICorrector** | 基准跟随仿射变换（水平/垂直/角度补偿） |

### 3D 视觉融合
| 特性 | 描述 |
|------|------|
| **MultiViewStereo** | 多视角立体视觉，利用 2D 工业相机实现 3D 重建 |
| **DepthFusion** | 深度图与 FLARE 边缘融合，高光区域深度修复 |
| **PrecisionMeasurement** | 精密测量（目标 < 0.5mm），突破 2D 精度限制 |
| **FeatureMatcher** | SSDA 边缘匹配（专利合规），替代 NCC |
| **AutoCalibrationPipeline** | 一键式多相机标定流程自动化 |
| **SynthDatasetV2** | 合成数据集生成器，支持 PBR 材质/缺陷/增强 |
| **PBRSurface** | PBR 材质渲染（10 种材质预设） |
| **EnvironmentLighting** | HDRI 环境光照（5 种预设：工业/实验室/户外/工厂/摄影棚） |
| **HDRILoader** | HDRI 格式加载（HDR/EXR/TIFF） |
| **TrainingPipeline** | 端到端训练 Pipeline（数据→训练→验证→导出） |
| **LovaszLoss** | Lovász-Softmax Loss（直接优化 IoU） |
| **EMAModel** | 指数滑动平均（训练稳定化） |
| **CosineAnnealingWarmup** | 余弦退火+预热学习率 |
| **OneCycleLR** | 单周期学习率调度 |
| **KnowledgeDistiller** | 知识蒸馏器（Logits/Feature/Relation） |
| **ModelPruner** | 模型剪枝（幅度剪枝/随机剪枝） |
| **GPUDataLoader** | GPU 优化数据加载（Pin Memory/预取） |
| **DataCache** | 多级数据缓存（内存/磁盘） |
| **PipelineBenchmark** | 端到端性能基准测试 |
| **ONNXRuntimeEngine** | 跨平台推理引擎 |
| **TensorRTEngine** | GPU 高性能推理 |
| **MultiBackendInference** | 多后端自动切换 |
| **ModelQuantizer** | INT8/FP16 量化 |
| **EarlyStoppingStrategy** | 早停策略（资源紧张切轻模型） |
| **EarlyRisingStrategy** | 早升策略（冷启动预热） |
| **ResourceMonitor** | 实时资源监控 |
| **EdgeCloudHandoff** | 边端协同切换 |
| **AdaptiveScheduler** | 自适应调度器 |
| **DatasetValidator** | 数据集验证器 |
| **IntegrityChecker** | 完整性检查器 |
| **AnomalyDetector** | 异常检测器 |
| **ConfigManager** | 统一配置管理（YAML/JSON/环境变量） |
| **EnvInterpolator** | 环境变量插值 |
| **ConfigValidator** | 配置验证规则 |
| **StructuredLogger** | 结构化日志器 |
| **LogFormatter** | 日志格式化（TEXT/JSON/CSV） |
| **LogManager** | 日志管理器 |
| **MLflowTracker** | MLflow 实验追踪器 |
| **ExperimentManager** | 实验管理器 |
| **RunComparator** | 运行比较器 |
| **ModelRegistry** | 模型注册表 |
| **VersionManager** | 版本管理器 |
| **ModelHandler** | 模型处理器 |
| **InferenceAPI** | FastAPI 推理接口 |
| **BatchProcessor** | 批处理器 |
| **PerformanceMonitor** | 性能监控器 |
| **HealthChecker** | 健康检查器 |
| **AlertManager** | 告警管理器 |
| **TrafficSplitter** | 流量分流器 |
| **ABTestRunner** | A/B 测试运行器 |
| **ResultAnalyzer** | 结果分析器 |
| **CanaryDeployer** | 金丝雀部署器 |
| **RolloutManager** | 发布管理器 |
| **AutoRollback** | 自动回滚 |
| **ModelProfiler** | 模型性能分析器 |
| **BottleneckAnalyzer** | 瓶颈分析器 |
| **OptimizationSuggestions** | 优化建议生成器 |
| **PruningTools** | 剪枝工具 |
| **QuantizationTools** | 量化工具 |
| **DistillationTools** | 蒸馏工具 |
| **BiLevelAttention** | BiFormer 双层注意力，RegionRouter 粗筛选 + 精细空间注意力（iteration_158） |
| **DeformConv2d** | DCNv2 可变形卷积，适应不规则边缘形变（iteration_159） |
| **GhostConv** | 幽灵卷积，参数量减半 ~50%（iteration_160） |
| **SubpixelLocalizerV2** | 灰度矩 + 梯度插值 + RANSAC 异常剔除（iteration_161） |
| **GlareInpainter** | Telea/NS/Hybrid 高光区域智能修复（iteration_162） |
| **GapMeasurementV2** | 亚像素边缘定位 + MAD 统计滤波（iteration_163） |
| **ROICorrectorV2** | SSDA 模板匹配 + 仿射自适应校正（iteration_164） |
| **HandEyeCalibratorV2** | PnP + RANSAC + 重投影误差最小化（iteration_165） |
| **PAFPN** | Path Aggregation FPN，双向特征融合（iteration_166） |
| **EdgeRefinementHead** | 分割梯度先验指导边缘细化（iteration_167） |
| **FLARELitePruner** | 通道剪枝 + 知识蒸馏，< 800K 参数（iteration_168） |
| **HeliumOptimizer** | RA8P1 Helium MVE SIMD + INT8 量化（iteration_169） |
| **QAT (LSQ)** | 量化感知训练，精度损失 < 1%（iteration_170） |
| **FLAREExporter** | ONNX → TensorRT → TFLite 一键导出（iteration_171） |
| **DynamicPrecisionSwitcher** | 动态 INT8/FP16 切换，功耗 < 5W（iteration_172） |
| **MaterialAdaptLUT** | 4 材质 × 3 表面状态自适应预处理（iteration_173） |
| **ExposurePredictor** | 轻量 CNN 预测最优曝光参数（iteration_174） |
| **SceneClassifier** | 强光/逆光/频闪/低光场景分类 + 维纳去模糊（iteration_175） |
| **MultiCameraTriangulator** | 多视角三角测量，< 0.3mm 定位（iteration_176） |
| **GripperEdgePlannerV2** | 边缘夹持点 + RSS 误差预算（iteration_177） |

### 机器人接口
| 特性 | 描述 |
|------|------|
| **AbbRobotStub** | 纯 Python 模拟桩（开发/测试用） |
| **AbbRobotStudioSim** | TCP Socket 接入 RobotStudio 仿真 |
| **AbbRobotEGM** | UDP EGM 协议直连真实 ABB 机器人 |
| **MultiRobotCoordinator** | 多机器人协调器，支持任务分配、协同控制 |
| **TaskAllocator** | 任务分配器（nearest/load_balance/hybrid） |
| **CollisionAvoidance** | 碰撞避免系统（基于安全距离和速度限制） |

### 推理引擎
| 引擎 | 延迟(512x512) | 精度 |
|------|---------------|------|
| PyTorch FP32 | ~100ms | 最高 |
| ONNX Runtime | ~40ms | FP32/FP16 |
| TensorRT FP16 | ~10ms | pycuda 异步流（已修复） |

---

## 项目结构 (Project Structure)

```
repo/
├── vision/                              # 视觉算法核心模块
│   ├── feature_extraction.py            # FLARE / FLARELite 分割网络
│   ├── hdr_processing.py               # AntiGlarePipeline HDR 融合
│   ├── localization_and_calibration.py  # 亚像素定位 + 标定 + 手眼标定
│   ├── appearance_detection.py          # 光度立体 + 划痕/边缘缺陷检测
│   ├── measurement.py                  # 卡尺测量 + 间隙测量（已接入pipeline）
│   ├── roi_tools.py                   # ROI 类型 + 自动检测 + 基准校正
│   ├── gripper_simulation.py          # GripperEdgePlanner 抓取规划
│   └── inference_engine.py            # PyTorch / ONNX / TensorRT 引擎
├── data/                               # 数据处理与生成
│   ├── synth_dataset_generator.py      # PBR 高光合成数据集生成器
│   ├── synth_national_scenes.py        # 全国 8 大场景合成
│   ├── data_augmentation.py            # CutMix / MixUp / 光照增强
│   └── real_world_dataloader.py       # 真实场景数据加载
├── training/                            # 模型训练
│   └── train.py                        # EMA + AMP + TTA + 自蒸馏
├── robot/                              # 机器人控制
│   ├── abb_robotstudio_interface.py   # Stub / RobotStudioSim / EGM 接口
│   ├── cells/
│   │   └── sorting_cell.py            # 真实视觉引导分拣单元（已重建）
│   └── abb_rapid/                     # RAPID 控制器代码
├── tests/                               # 测试套件
├── scripts/                             # 工具脚本
├── demo_streamlit.py                    # Streamlit 交互式演示
├── main_pipeline.py                    # 顶层流水线（含测量步骤）
└── requirements.txt
```

---

## 快速开始 (Quick Start)

### 方式一：统一启动器（推荐，跨平台）

```bash
# 检查环境
python launch.py --mode check

# 启动 Streamlit 演示（自动创建 venv、安装依赖）
python launch.py --mode demo

# 启动 Gradio 演示
python launch.py --mode gradio

# 启动训练
python launch.py --mode training --epochs 100 --batch-size 8

# 启动主流程（图像模式）
python launch.py --mode pipeline --pipeline-mode image --model_path ./checkpoints/best.pth
```

### 方式二：手动启动（适合开发者）

```bash
# 1. 安装依赖
pip install -r requirements-dev.txt

# 2. 生成合成数据集（含 PBR 高光模式）
python data/synth_dataset_generator.py --count 500 --output ./dataset

# 3. 训练模型
python training/train.py --synth_dir ./dataset --epochs 100 --batch_size 8

# 4. 启动演示
streamlit run demo_streamlit.py --server.port 8501
```

### 方式三：系统脚本（已安装环境后）

```bash
# Windows
.\scripts\start_demo.bat
.\scripts\start_training.bat

# Windows (PowerShell)
.\scripts\start_demo.ps1
.\scripts\start_training.ps1

# Linux / macOS
bash scripts/start_demo.sh
bash scripts/start_training.sh
```

---

## 完整流水线 (Pipeline)

```
摄像头采集
    ↓
┌──────────────────────────────────────────────────────────┐
│ AntiGlarePipeline: HDR融合 → 高光修复 → 双边滤波 → CLAHE  │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│ FLARE 推理: seg_mask + edge_mask                         │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│ SubpixelLocalizer: 质心 + PCA方向 + 特征分类              │
└──────────────────────────────────────────────────────────┘
    ↓  ← 新增
┌──────────────────────────────────────────────────────────┐
│ CaliperMeasurement + GapMeasurement: 几何尺寸测量        │
└──────────────────────────────────────────────────────────┘
    ↓  ← 新增
┌──────────────────────────────────────────────────────────┐
│ MultiViewStereo + DepthFusion: 3D深度融合 → 精密测量    │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│ CoordinateTransformer: 像素→机器人基坐标系                │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│ MultiRobotCoordinator: 多机器人任务分配与协同控制        │
└──────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────┐
│ AbbRobotEGM / RobotStudioSim / Stub: 机器人控制         │
└──────────────────────────────────────────────────────────┘
```

---

## FLARE 架构

```
输入图像 (B, 3, 512, 512)
    │
    ▼
┌─────────────────────────────┐
│  CoordConvEncoderBlock      │  ← 坐标感知卷积
│  (CBAM 注意力)            │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  编码器 (4级 U-Net)         │
│  enc1 → enc2 → enc3 → enc4 │
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  解码器 + GlareGatedSkip   │
└─────────────────────────────┘
    │
    ├──→ 分割头 (seg)
    └──→ 边缘头 (HED 多尺度)
```

---

## PBR 高光模拟

新增 `PBRLightingSystem`（基于 Blinn-Phong BRDF）：

| 参数 | 说明 |
|------|------|
| roughness | 0.01（镜面）~ 1.0（漫反射） |
| metallic | 0.0（非金属）~ 1.0（纯金属） |
| D | Blinn-Phong 微平面分布 |
| F | Schlick Fresnel 近似 |
| G | Smith 几何遮蔽 |

支持光照模式：`pbr` / `pbr_sun` / `pbr_mixed`

---

## ROI 工具

| 类/函数 | 说明 |
|---------|------|
| `ROIType` | POINT / LINE / RECT / ROTATED_RECT / CIRCLE / ELLIPSE / ANNULUS / POLYGON / ARRAY / BINARY_MASK |
| `ROI` | 统一 ROI 数据类，含 to_mask / get_bounding_box / apply_affine_transform |
| `ROICorrector` | 基准跟随校正（水平 / 垂直 / 角度补偿） |
| `AutoROI` | 自动 ROI 检测与模式切换（BLOB / EDGE / FEATURE / ADAPTIVE） |

---

## 性能基准

| 模型 | 参数量 | IoU (合成) | FPS (GPU) |
|------|--------|-------------|-----------|
| FLARE (标准) | 8.0M | 0.41+ | ~100 |
| FLARELite | 1.2M | 0.35+ | ~200 |
| FLARE + TensorRT FP16 | 8.0M | 0.41+ | ~600 |

---

## 专利合规说明

本项目实现已进行专利规避设计：

| 算法 | 原方案 | 规避方案 |
|------|--------|----------|
| 灰度匹配 | NCC (Cognex) | SSDA (TM_SQDIFF_NORMED) |
| 光度立体 | 最小二乘求解 | PhotometricStereoNet CNN回归 |
| 手眼标定 | AX=XB 方程 | PnP+RANSAC 替代 |
| ROI 自动检测 | 能量阈值 US6,456,727 | 边缘密度/方差/熵评估 |

---

## 许可证与专利声明

本项目采用 **Apache License 2.0** 协议开源。

**专利声明**：本项目实现的 `FLARE` 架构及核心算法受专利保护。商业用途请联系 **RussellCooper**。

**合规性声明**：本项目遵循 **Clean-room 工程化**标准独立开发，所有核心算子实现均基于数学定义。

---

## 相关文档

- [DATASET_GUIDE.md](./DATASET_GUIDE.md) - 数据集生成与使用指南
- [REAL_DATA_GUIDE.md](./REAL_DATA_GUIDE.md) - 真实数据采集与标注指南
- [ABB_RobotStudio_Integration_Guide.md](./docs/ABB_RobotStudio_Integration_Guide.md) - ABB 仿真集成
- [CHANGELOG.md](./CHANGELOG.md) - 版本更新日志
