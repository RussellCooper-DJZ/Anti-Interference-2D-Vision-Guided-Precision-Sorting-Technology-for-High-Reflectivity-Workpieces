# AI 项目上下文 — Anti-Interference-2D

> **用途**：这是 AI 代理启动后的**首要参考文档**。阅读本文件后，你应该能在 5 分钟内理解项目全貌，并知道当前该做什么。
>
> **更新日期**：2026-04-25
> **项目版本**：ageanet v0.5.0
> **测试状态**：207 passed, 14 skipped, 覆盖率 29%

---

## 一、项目身份卡片

```yaml
项目名称: Anti-Interference-2D / AGEANet
一句话定义: 高反光金属工件精准边缘识别 + 机器人视觉引导分拣系统
核心价值: 在复杂工业环境下，对高反光金属工件实现 <0.5mm 精度的边缘识别与机器人精准抓取
部署平台: Renesas RZ/V2H (AI推理) + RA8P1 Cortex-M85 + Helium SIMD (嵌入式预处理)
机器人: ABB (RobotStudio 仿真 / EGM 真实通信)
场景: 船舶大型金属高光面（钢板、甲板、焊缝、铆钉、舷窗）
```

---

## 二、目录全景地图

```
Anti-Interference-2D/                          ← 项目根
│
├── Anti-Interference-2D-Vision/               ← 核心代码仓库 (Monorepo)
│   │
│   ├── vision/                                ← Lib: 核心视觉算法
│   │   ├── feature_extraction.py              │   FLARE/FLARELite + 注意力/卷积/FPN/边缘头
│   │   ├── hdr_processing.py                  │   HDR融合 + 反光抑制 + GlareInpainter
│   │   ├── localization_and_calibration.py    │   亚像素定位 + 手眼标定 + ROI校正 + 坐标变换
│   │   ├── measurement.py                     │   卡尺测量 + 间隙测量
│   │   ├── appearance_detection.py            │   缺陷/划痕/油污检测
│   │   ├── inference_engine.py                │   PyTorch/ONNX/TensorRT 多后端推理
│   │   ├── roi_tools.py                       │   ROI提取与操作
│   │   ├── gripper_simulation.py              │   抓取器仿真
│   │   └── existence_checking.py              │   有无检测
│   │
│   ├── data/                                  ← Lib: 数据层
│   │   ├── data_augmentation.py               │   船舶专项增强（高光/水面/锈蚀模拟）
│   │   ├── synth_dataset_generator.py         │   PBR合成数据集生成（Blinn-Phong BRDF）
│   │   └── real_world_dataloader.py           │   真实数据加载器
│   │
│   ├── robot/                                 ← Lib: 机器人通信
│   │   ├── abb_robotstudio_interface.py       │   ABB TCP/UDP/EGM协议
│   │   └── cells/sorting_cell.py              │   分拣单元控制逻辑
│   │
│   ├── embedded/                              ← Lib: 嵌入式C代码
│   │   ├── ra8p1_helium_processing.c          │   Helium SIMD图像预处理
│   │   └── ra8p1_main_app.c                   │   主程序入口
│   │
│   ├── training/                              ← App: 训练与评估入口
│   │   ├── train.py                           │   训练主程序
│   │   └── evaluate.py                        │   评估主程序
│   │
│   ├── tests/                                 ← App: 单元测试（严重不足！）
│   │   ├── test_feature_extraction.py
│   │   ├── test_gripper_simulation.py
│   │   └── test_har_processing.py
│   │
│   ├── docs/                                  ← 技术文档
│   │   ├── ALGORITHM_AUDIT_GUIDE.md           │   Vision分支三维审计框架
│   │   ├── BRANCH_AUDIT_GUIDE.md              │   六大分支审计框架
│   │   ├── ITERATION_AUDIT_REPORT.md          │   200次迭代全景审查报告
│   │   ├── ITERATION_DEBT_REGISTER.md         │   技术债务登记册
│   │   └── ITERATION_GATE_GUIDE.md            │   迭代价值门控（核心价值六问）
│   │
│   ├── main_pipeline.py                       ← App: 端到端流水线主入口
│   ├── demo_streamlit.py                      ← App: Streamlit演示
│   ├── demo_gradio.py                         ← App: Gradio演示
│   ├── pyproject.toml                         ← 包配置 (name=ageanet, v0.5.0)
│   ├── requirements.txt                       ← 运行时依赖
│   ├── requirements-dev.txt                   ← 开发依赖
│   └── CHANGELOG.md                           ← 变更日志
│
├── results/                                   ← 迭代产物
│   ├── auto_tuning/                           │   iteration_001 ~ iteration_200
│   │   ├── iteration_001/
│   │   │   ├── plan.md                        │   规划
│   │   │   ├── work_*.py                      │   执行产出
│   │   │   ├── review.md                      │   审查
│   │   │   └── compound.md                    │   复利/沉淀
│   │   ├── iteration_002/
│   │   └── ... (共200个)
│   └── iteration_audit_raw.json               │   自动化审计元数据
│
├── archive/                                   ← 归档
│   ├── Anti-ference-2D/                       │   历史版本
│   └── iteration_graveyard/                   │   偏离迭代的孤立代码(60个)
│
├── scripts/                                   ← 根目录级运行脚本
│   ├── auto_optimize.py                       │   自动优化引擎
│   ├── audit_iterations.py                    │   迭代审计脚本
│   ├── analyze_debt.py                        │   债务分析脚本
│   └── fix_iterations.py                      │   迭代修复脚本
│
├── docs/                                      ← 项目级文档
│   └── AI_PROJECT_CONTEXT.md                  │   ← 本文件
│
├── competition/                               ← 比赛资料
├── team/                                      ← 团队提交资料
└── assets/                                    ← 数据资产（测试图/视频）
```

---

## 三、技术栈矩阵

| 层级 | 技术 | 版本 | 用途 |
|------|------|------|------|
| **语言** | Python | 3.8+ | 算法研发 |
| **语言** | C | C11 | 嵌入式 (RA8P1) |
| **深度学习** | PyTorch | ≥2.0 | 模型训练/推理 |
| **CV** | OpenCV | ≥4.5 | 图像处理 |
| **数值** | NumPy | ≥1.21 | 数组运算 |
| **导出** | ONNX | ≥1.14 | 模型部署转换 |
| **包管理** | pip + setuptools | — | `pyproject.toml` |
| **构建** | `python -m build` | — | 生产构建 |
| **代码质量** | black / flake8 / isort / mypy | — | 格式化/检查 |
| **测试** | pytest | ≥7.0 | 单元测试（严重不足） |
| **CI** | GitHub Actions | — | `.github/workflows/python-app.yml` |
| **演示** | Streamlit / Gradio | — | Web交互界面 |
| **可视化** | TensorBoard / matplotlib | — | 训练可视化 |

**⚠️ 关键事实**：本项目 **不使用 npm / bun / yarn / Node.js**。它是一个纯 Python + C 嵌入式项目。

---

## 四、核心模块详细索引

### 4.1 `vision/feature_extraction.py` (~1900行)

**职责**：FLARE/FLARELite 网络定义 + 所有 CNN 模块

| 类/函数 | 来源迭代 | 功能 | 状态 |
|---------|---------|------|------|
| `FLARE` | 早期 | U-Net + CBAM + 双分支(分割+边缘) + 检测头 | ✅ 核心 |
| `FLARELite` | 早期 | 轻量版，深度可分离卷积，~1.2M参数 | ✅ 核心 |
| `BiLevelAttention` | iter 158 | BiFormer风格双层注意力，RegionRouter + CA + SA | ✅ 已整合 |
| `DeformConv2d` | iter 159 | DCNv2可变形卷积，offset+mask+unfold采样 | ✅ 已整合 |
| `CoordDeformConv` | iter 159 | CoordConv + DCNv2融合 | ✅ 已整合 |
| `GhostConv` | iter 160 | GhostNet轻量化，intrinsic+ghost maps | ✅ 已整合 |
| `GhostDoubleConv` | iter 160 | Ghost版DoubleConv | ✅ 已整合 |
| `PAFPN` | iter 166 | Path Aggregation FPN，自顶向下+自底向上 | ✅ 已整合 |
| `FeatureAlignBlock` | iter 166 | 特征对齐块，3×3 conv + SE | ✅ 已整合 |
| `EdgeRefinementHead` | iter 167 | 边缘细化头，Sobel梯度先验+conv精修 | ✅ 已整合 |
| `WaveletScattering` | 早期 | 小波散射变换（已修复shape mismatch） | ✅ 已修复 |
| `FourierConv` | 早期 | 频域卷积（cartesian模式有bug，未默认启用） | ⚠️ 已知问题 |
| `DetectionHead` / `FPN` / `AnchorFreeHead` / `CIoULoss` | 早期 | YOLO风格检测组件 | ✅ 已前移修复 |
| `predict()` | 早期 | 单张图像推理辅助函数 | ✅ 可用 |
| `get_model_info()` | 早期 | 模型参数量统计 | ✅ 可用 |

**已知问题**：
- `FourierConv` 的 cartesian 模式有严重 shape mismatch（未默认启用，不触发）
- `WaveletScattering` 曾用 `np.pi` 但模块未 import numpy（已修复为 `math.pi`）

### 4.2 `vision/localization_and_calibration.py` (~2900行)

**职责**：像素级定位 + 相机标定 + 手眼标定 + 坐标变换

| 类/函数 | 来源迭代 | 功能 | 状态 |
|---------|---------|------|------|
| `SubpixelLocalizer` | 早期 | 质心+PCA方向提取 | ✅ 核心 |
| `SubpixelLocalizerV2` | iter 161 | 灰度矩+梯度插值+RANSAC异常剔除 | ✅ 已整合 |
| `CameraCalibrator` | 早期 | 棋盘格内参标定 | ✅ 核心 |
| `HandEyeCalibrator` | 早期 | 手眼标定（Eye-in-Hand/ Eye-to-Hand） | ✅ 核心 |
| `HandEyeCalibratorV2` | iter 165 | PnP+RANSAC+重投影误差最小化 | ✅ 已整合 |
| `CoordinateTransformer` | 早期 | 像素→相机→机器人坐标变换 | ✅ 核心 |
| `ROICorrection` | 早期 | ROI仿射校正 | ✅ 核心 |
| `ROICorrectorV2` | iter 164 | SSDA模板匹配+漂移监测 | ✅ 已整合 |
| `ROIRect` | iter 164 | ROI数据类 | ✅ 已整合 |
| `CornerDetector` / `GaussianLineExtractor` / `HoughCircleDetector` / `HoughLineDetector` | 早期 | 传统CV检测器 | ✅ 可用 |
| `NinePointCalibrator` / `RotationCenterCalibrator` | 早期 | 专用标定 | ✅ 可用 |
| `FeatureMatcher` / `PoseEstimator` / `AutoFitter` | 早期 | 特征匹配与姿态估计 | ✅ 可用 |

### 4.3 `vision/hdr_processing.py` (~600行)

**职责**：HDR融合 + 反光抑制

| 类/函数 | 来源迭代 | 功能 | 状态 |
|---------|---------|------|------|
| `AntiGlarePipeline` | 早期 | 六级HDR处理管线 | ✅ 核心 |
| `exposure_fusion_mertens()` | 早期 | Mertens曝光融合 | ✅ 可用 |
| `detect_highlight_mask()` | 早期 | 高光区域检测 | ✅ 可用 |
| `repair_highlight_regions()` | 早期 | 高光区域修复 | ✅ 可用 |
| `GlareInpainter` | iter 162 | Telea/NS/Hybrid智能高光修复 | ✅ 已整合 |

### 4.4 `vision/measurement.py`

**职责**：精密测量

| 类 | 功能 |
|----|------|
| `CaliperMeasurement` | 双平行边缘卡尺测量，任意方向搜索 |
| `GapMeasurement` | 多边缘间隙/节距/宽度测量 |

### 4.5 `data/data_augmentation.py` (~760行)

**职责**：船舶专项数据增强

- 几何变换（翻转、旋转、仿射、透视）
- 光照扰动（亮度、对比度、Gamma、局部高光注入、水面反射）
- 高光专项（随机高光椭圆、过曝、镜面条带）
- 噪声/模糊（高斯、运动、散焦、JPEG压缩）
- 颜色扰动（HSV、通道偏移、灰度化）
- 遮挡/擦除（Random Erasing、CutOut）
- mask-图像一致性保证

### 4.6 `data/synth_dataset_generator.py` (~1090行)

**职责**：PBR合成数据集生成

- `PBRLightingSystem`：Blinn-Phong BRDF 物理高光模拟
- `DefectGenerator`：划痕/凹坑/裂纹/污染物
- `DataAugmentor`：几何/颜色/模糊/噪声增强
- `DomainAdapter`：域适应
- `SynthDatasetV2`：批量合成数据生成

### 4.7 `robot/abb_robotstudio_interface.py` (~1190行)

**职责**：ABB机器人三层通信接口

| 类 | 模式 | 协议 |
|----|------|------|
| `AbbRobotStub` | 模拟桩（无需任何软件） | 纯Python |
| `AbbRobotStudioSim` | RobotStudio仿真 | TCP Socket JSON |
| `AbbRobotEGM` | 真实机器人 | UDP EGM |

### 4.8 `training/train.py` (~1150行)

**职责**：训练主程序

- 双头损失：分割（BCE + Dice）+ 边缘（BCE + Focal）
- 学习率调度：Cosine Annealing with Warmup
- 混合精度训练（AMP）
- 检查点保存（best_val_iou）
- TensorBoard日志

### 4.9 `training/evaluate.py` (~394行)

**职责**：评估脚本

- IoU、Dice、边缘F1、Precision、Recall
- TTA（Test Time Augmentation）
- 预测可视化对比图

---

## 五、200次迭代五阶段历史

```
Phase 1 (1–50)   → 算法奠基期          → 质量最高，代码量 562行/iter
Phase 2 (51–100) → MLOps膨胀期         → 开始偏离，代码量 474行/iter
Phase 3 (101–137)→ 流水线失控期        → ❌ 严重偏离，18个缺失Plan
Phase 4 (138–157)→ 平台化幻象期        → ❌ 严重偏离，通用SaaS平台
Phase 5 (158–200)→ 回归聚焦期          → ✅ 回归正轨，代码量 104行/iter
```

**关键数据**：
- 直接视觉算法迭代：仅 23/200 (11.5%)
- 偏离主线的通用基础设施迭代：~60/200 (30%+)
- 完整四件套（Plan+Work+Review+Compound）：157/200 (78.5%)
- 代码总产出：74,872 行

---

## 六、已完成的工作（2026-04-24）

### 6.1 P0 债务偿还
- [x] 标记 60 个偏离迭代的 review.md
- [x] 归档 60 个迭代的孤立 work 文件 → `archive/iteration_graveyard/`
- [x] 补全 18 个缺失 Plan (120–137)
- [x] 补全 25 个缺失 Compound (24, 114–137)

### 6.2 算法整合到主代码库
- [x] `DeformConv2d` / `CoordDeformConv` → `feature_extraction.py`
- [x] `GhostConv` / `GhostDoubleConv` → `feature_extraction.py`
- [x] `PAFPN` / `FeatureAlignBlock` → `feature_extraction.py`
- [x] `EdgeRefinementHead` → `feature_extraction.py`
- [x] `SubpixelLocalizerV2` → `localization_and_calibration.py`
- [x] `ROICorrectorV2` / `ROIRect` → `localization_and_calibration.py`
- [x] `HandEyeCalibratorV2` / `CalibrationFrame` → `localization_and_calibration.py`
- [x] `GlareInpainter` → `hdr_processing.py`

### 6.3 算法错误修正
- [x] `WaveletScattering`: `np.pi` → `math.pi` + shape mismatch修复
- [x] iter 164 SSDA: 重写为真正的 early termination
- [x] iter 167 RCF: 文档明确标注"非完整RCF"
- [x] iter 179 LovászLoss: 重写为标准 hinge loss
- [x] iter 180 CutMix: mask 直接复制替代 `torch.max`
- [x] iter 181 OHEM: batch-wide top-k 替代 threshold-then-topk
- [x] iter 186 Grad-CAM: 正样本区域平均激活替代 `logits.sum()`
- [x] iter 190 AsyncInferencer: 同步点从 `submit()` 移至 `get()`
- [x] iter 193 AutoAugment: 改名 `SimpleAugPolicySearch`

### 6.4 文档与机制
- [x] `docs/ITERATION_AUDIT_REPORT.md` — 200次迭代全景审查
- [x] `docs/ALGORITHM_AUDIT_GUIDE.md` — Vision分支三维审计
- [x] `docs/BRANCH_AUDIT_GUIDE.md` — 六大分支审计
- [x] `docs/ITERATION_GATE_GUIDE.md` — 核心价值六问门控
- [x] `docs/ITERATION_DEBT_REGISTER.md` — 技术债务登记册
- [x] `CHANGELOG.md` — 新增16条变更记录

---

## 七、待偿还债务（⚠️ 当前重点）

### P1 — 严重债务（必须偿还）

| # | 债务 | 影响 | 建议行动 | 预估工时 |
|---|------|------|---------|---------|
| P1-2 | **重复实现**：配置管理、日志、监控、检查点在多个迭代中重复出现 | 维护成本倍增，代码冗余 | ~~提取公共基类，统一实现~~ → ✅ 已完成 (core/metrics.py) | 4h → ✅ |
| P1-3 | **测试覆盖不足**：仅3个测试文件，无法覆盖23个vision模块 | 无法保证核心模块正确性，回归风险高 | 为核心模块补充pytest单元测试，目标覆盖率>60% | 8h |

### P2 — 中等债务（建议偿还）

| # | 债务 | 影响 | 建议行动 | 预估工时 |
|---|------|------|---------|---------|
| P2-1 | **Phase 5代码量偏低**：部分迭代仅30–75行，可能只是概念验证 | 缺乏深度，生产就绪度不足 | 评估是否需要合并或深化 | 2h |
| P2-2 | **179–200 Plan风格突变**：突然全英文"Goal: Improve..." | 与之前中文风格不一致 | 统一文档风格（中文为主，英文术语保留） | 1h |
| P2-3 | **CHANGELOG早期记录不完整**：1–50迭代缺少详细变更 | 无法追溯早期演进 | 补全早期迭代变更记录 | 1h |

### P3 — 轻微债务（低优先级）

| # | 债务 | 建议行动 |
|---|------|---------|
| P3-1 | compound.md 命名不统一（大小写混用） | 统一为小写 `compound.md` |
| P3-2 | work 文件命名风格不一致 | 统一为 `work_<模块>_<描述>.py` |

---

## 八、快速启动命令

### 8.1 环境搭建

```bash
# 进入项目目录
cd Anti-Interference-2D-Vision

# 创建虚拟环境（WSL Ubuntu）
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 开发安装（可编辑模式）
pip install -e .
```

### 8.2 运行测试

```bash
# 语法检查
python -m py_compile vision/feature_extraction.py
python -m py_compile vision/localization_and_calibration.py
python -m py_compile vision/hdr_processing.py

# 运行已有测试
pytest

# 带覆盖率
pytest --cov=vision --cov=data --cov=training --cov=robot
```

### 8.3 运行主流程

```bash
# 演示模式（合成图像，无需硬件）
python main_pipeline.py --mode demo --model_path ./checkpoints/best.pth

# 训练
python training/train.py --synth_dir ./dataset --epochs 50 --batch_size 8

# 评估
python training/evaluate.py --checkpoint checkpoints/best.pth --data dataset_merged

# Streamlit演示
streamlit run demo_streamlit.py --server.port 8501
```

### 8.4 审计脚本

```bash
# 审计全部200次迭代
python scripts/audit_iterations.py

# 分析债务
python scripts/analyze_debt.py
```

---

## 九、常见陷阱（新手必读）

| # | 陷阱 | 场景 | 解决方案 |
|---|------|------|---------|
| 1 | `np` 未导入就用 `np.pi` | 模块级别没有 `import numpy as np` | 检查模块顶部导入，或用 `math.pi` 替代 |
| 2 | 类定义在文件末尾 | `DetectionHead` 原在1362+行，FLARE引用时NameError | 将类定义前移至使用之前 |
| 3 | `torch.cos(float)` 类型错误 | 传入Python float而非Tensor | 用 `math.cos()` 或包装为 `torch.tensor()` |
| 4 | 多尺度特征图尺寸不一致却`torch.cat` | `WaveletScattering`中不同池化核输出不同尺寸 | 池化后`F.interpolate`上采样回原始尺寸 |
| 5 | PowerShell `&&` 和嵌套引号解析失败 | WSL执行命令 | 用 `bash -c '...'` 单层引号包裹，或写临时.py文件 |
| 6 | `logits.sum()` 替代 class score | Grad-CAM目标选择 | 必须对特定类别分数求梯度 |
| 7 | `stream.synchronize()`在`submit()`中 | 异步推理 | 将同步推迟到`get()` |
| 8 | CutMix mask 用 `torch.max` | 数据增强 | 分割任务直接复制patch区域，不要用max |

---

## 十、审计指南索引

| 文档 | 用途 | 何时查阅 |
|------|------|---------|
| `docs/ALGORITHM_AUDIT_GUIDE.md` | Vision算法三维审计（命名/公式/行为） | 新增/修改CNN/注意力/损失函数时 |
| `docs/BRANCH_AUDIT_GUIDE.md` | 六大分支独立审计框架 | 涉及Robot/Data/Training/Embedded/Pipeline时 |
| `docs/ITERATION_GATE_GUIDE.md` | 迭代价值门控（核心价值六问） | 每次迭代Plan阶段 |
| `docs/ITERATION_AUDIT_REPORT.md` | 200次迭代全景审查 | 了解历史债务和偏离模式 |
| `docs/ITERATION_DEBT_REGISTER.md` | 技术债务登记册 | 了解当前待偿还债务 |

---

## 十一、下一步行动建议（按优先级排序）

1. **🔴 补充单元测试**（P1-3）：为核心模块补充 pytest，这是当前最大的技术债务
2. **✅ 统一重复实现**（P1-2）：已通过 core/metrics.py 完成
3. **🟡 深化Phase 5低代码量迭代**（P2-1）：评估179–200中30–75行的迭代是否需要合并
4. **🟢 统一文档风格**（P2-2）：将179–200的英文Plan翻译为中文
5. **🟢 补全CHANGELOG早期记录**（P2-3）

---

*本文件是AI代理的首要参考文档。如果项目中出现本文件未涵盖的新问题，请更新本文件以保持同步。*
