# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **公共基础设施包 `core/`**：统一提取配置管理 (`core/config.py`)、结构化日志 (`core/logging.py`)、检查点管理 (`core/checkpoint.py`)、指标监控 (`core/monitoring.py`)，替代散落在 6+ 个迭代中的重复实现
- **`tests/test_existence_checking.py`**：为 `vision/existence_checking.py` 补充 pytest 单元测试，覆盖 BlobDetector、GrayMatcher、FeaturePointMatcher、ContourMatcher 四大类
- **`tests/test_core_infrastructure.py`**：验证 `core/` 包公共基类的正确性
- **Phase 5 评估报告**：`results/auto_tuning/PHASE5_EVALUATION.md`，统计 iteration 179–200 代码量并给出深化/归档建议
- **CHANGELOG 早期记录补全**：补全 iteration 1–22 的详细变更记录
- **`tests/test_data_modules.py`**：为 `data/real_world_dataloader.py`、`data/synth_dataset_generator.py`、`data/synth_national_scenes.py` 补充快速冒烟测试
- **`tests/test_training_modules.py`**：为 `training/train.py`、`training/evaluate.py` 补充基础 API 测试（EMA、LovászLoss、IoU/Dice 计算等）
- **`launch.py`**：统一跨平台启动器，自动检测 OS、创建虚拟环境、安装依赖、启动服务
- **`gui/` 桌面 GUI**：基于 PyQt6 的原生跨平台桌面应用，支持拖拽图片、模型切换、HDR 开关、分割/边缘/高光可视化、结果导出
- **`scripts/start_demo.sh`** / **`scripts/start_training.sh`**：Linux / macOS 启动脚本
- **`scripts/start_demo.ps1`** / **`scripts/start_training.ps1`**：Windows PowerShell 启动脚本
- **`Dockerfile`** + **`.dockerignore`**：一键构建容器镜像，支持 CPU / CUDA 双版本，`docker run -p 8501:8501 ageanet:latest`

### Changed
- **统一文档风格**：iteration 179–200 的 `compound.md` 统一为中文为主风格（标题/正文）
- **统一文件命名**：`COMPOUND.md` → `compound.md`（127 处），`master_COMPOUND.md` → `master_compound.md`；早期迭代中不符合 `work_<模块>_<描述>` 约定的 work 文件已重命名
- **`core/__init__.py`**：在文档字符串和 `__all__` 中明确标注 Legacy 接口已弃用，并给出 Unified 迁移指南
- **`.github/workflows/python-app.yml`**：Python 版本升级至 3.12；移除全部 `--ignore` 参数，恢复被忽略的核心基础设施测试和扩展测试；依赖安装改为统一使用 `requirements-dev.txt`
- **`pyproject.toml`**：`requires-python` 修正为 `>=3.10`；`classifiers` 和 `tool.black.target-version` 同步更新为 3.10/3.11/3.12
- **`requirements.txt`**：补充 `pyyaml>=6.0` 作为核心依赖；新增 ONNX Runtime、TensorRT、pycuda 的可选依赖注释

### Fixed
- **CI 测试忽略问题**：`test_core_infrastructure.py`、`test_feature_extraction_extended.py`、`test_hdr_processing_extended.py`、`test_unified_logger.py`、`test_unified_config.py`、`test_unified_monitor.py` 共 6 个测试文件不再被 CI 忽略
- **Python 版本不一致**：消除 CI(3.10)、Badge(3.12+)、pyproject.toml(>=3.8) 之间的版本声明冲突

### Deprecated
- 以下迭代中的旧基础设施 work 文件已标记为弃用，顶部添加 `DeprecationWarning`，提示使用 `core.*` 中的统一实现：
  - iter 38 `work_early_stopping_v2.py`（ModelCheckpoint）
  - iter 39 `work_metrics_monitor.py`（MetricsCollector）
  - iter 43 `work_config_manager.py`（ConfigManager）
  - iter 44 `work_logging_system.py`（StructuredLogger）
  - iter 62 `work_monitoring_alerting.py`（AlertManager）
  - iter 195 `work_checkpoint_manager.py`（CheckpointManager）
- **Legacy core 接口**：`core/config.py` 的 `ConfigManager`、`core/logging.py` 的 `StructuredLogger`/`LogManager`、`core/checkpoint.py` 的 `CheckpointManager`、`core/monitoring.py` 的 `MetricsCollector`/`AlertManager` 已标记为 deprecated，将在 v0.7.0 移除。请迁移至 `UnifiedConfig`、`UnifiedLogger`、`UnifiedCheckpoint`、`UnifiedMonitor`。

## [0.6.0] — 2026-04-25

### Added
- **核心视觉模块（iteration 1–22 补全记录）**：
  - `vision/existence_checking.py`：Blob分析、灰度匹配(SSDA)、特征匹配(ORB/AKAZE)、轮廓匹配（iter 1）
  - `vision/feature_extraction.py`：CoordConv编码器、4级U-Net、GlareGatedSkip、HED边缘头（iter 1）
  - `vision/hdr_processing.py`：HDR融合、高光修复、偏振去反光、CLAHE、引导滤波（iter 1）
  - `vision/appearance_detection.py`：光度立体网络 PhotometricStereoNet（CNN回归法线/反照率，规避MIT专利）（iter 8）
  - `vision/localization_and_calibration.py`：Zernike矩亚像素定位、几何哈希定位、多特征融合定位（iter 10）
  - `vision/measurement.py`：卡尺测量 CaliperMeasurement、间隙测量 GapMeasurement（iter 9）
  - `vision/roi_tools.py`：ROI校正、ROI追踪（iter 13）
  - `vision/inference_engine.py`：TensorRT 引擎构建与推理（iter 15）
  - `vision/gripper_simulation.py`：夹爪边缘规划 GripperEdgePlanner（iter 17）
- **数据与训练模块**：
  - `data/data_augmentation.py`：CutMix、MixUp、TTA（iter 1）
  - `training/train.py`：EMA(0.999)、梯度裁剪、Warmup、Lovász+Dice+BCE+Boundary+Focal+EdgeAware 组合损失（iter 1）
  - `data/synth_dataset_generator.py`：物理高光模拟（Blinn-Phong BRDF）（iter 7）
  - `data/real_world_dataloader.py`：真实世界数据加载器（iter 12）
- **应用与部署模块**：
  - `demo_streamlit.py`：模型管理器 ModelManager、Streamlit 模型选择器（iter 5）
  - `demo_streamlit.py`：训练仪表板、后台进程管理 TrainingProcess（iter 6）
  - `training/train.py`：增量学习、EWC缓解遗忘、样本回放策略（iter 20）
  - `main_pipeline.py`：故障恢复框架、健康状态等级、异常检测与恢复策略（iter 22）
- **文档迭代**：算法架构文档、开发指南、用户指南（iter 2）；视频实时处理（iter 4）；量化感知训练 QAT（iter 16）；边缘计算场景适配（iter 18）；自适应阈值与多曝光（iter 19）；多机器人协同（iter 21）

### Fixed
- `vision/existence_checking.py`：GrayMatcher 将 NCC 相关匹配替换为 SSDA（`TM_SQDIFF_NORMED`），规避 Cognex US6,041,139 专利（iter 1）
- `vision/appearance_detection.py`：PhotometricStereoNet 改用 CNN 直接回归法线/反照率，规避 MIT US6,477,268 最小二乘求解专利（iter 8）

### Added
- **200次迭代全面审计**：完成 `docs/ITERATION_AUDIT_REPORT.md`，识别迭代债务、偏离主线、质量趋势
- **算法引用审计指南**：`docs/ALGORITHM_AUDIT_GUIDE.md` — Vision 分支三维审计框架（命名/公式/行为）
- **全分支审计指南**：`docs/BRANCH_AUDIT_GUIDE.md` — 六大分支独立审计维度
- **迭代价值门控**：`docs/ITERATION_GATE_GUIDE.md` — 核心价值六问防止未来偏离
- **DeformConv2d (DCNv2)**：`vision/feature_extraction.py` 新增可变形卷积（iter 159）
- **GhostConv**：`vision/feature_extraction.py` 新增轻量化 Ghost 卷积（iter 160）
- **PAFPN**：`vision/feature_extraction.py` 新增 Path Aggregation FPN（iter 166）
- **EdgeRefinementHead**：`vision/feature_extraction.py` 新增边缘细化头（iter 167）
- **SubpixelLocalizerV2**：`vision/localization_and_calibration.py` 新增灰度矩+RANSAC亚像素定位（iter 161）
- **ROICorrectorV2**：`vision/localization_and_calibration.py` 新增 SSDA模板匹配 ROI校正（iter 164）
- **HandEyeCalibratorV2**：`vision/localization_and_calibration.py` 新增 PnP+RANSAC手眼标定（iter 165）
- **GlareInpainter**：`vision/hdr_processing.py` 新增 Telea/NS/Hybrid 高光修复（iter 162）
- **偏离迭代标记**：60个偏离主线的迭代 review.md 已追加偏离声明
- **缺失 artifact 补全**：18个缺失 Plan + 25个缺失 Compound 已补全
- **孤立代码归档**：60个迭代的偏离 work 文件已移至 `archive/iteration_graveyard/`

### Fixed
- **WaveletScattering**：修复 `np.pi` 未定义 + shape mismatch + `torch.cos(float)` 类型错误
- **LovászLoss (iter 179)**：修复公式为标准的 Lovász hinge loss（hinge error + 累积 IoU）
- **CutMix (iter 180)**：修复 mask 用 `torch.max` → 直接复制 patch 区域
- **OHEM (iter 181)**：修复为 batch-wide top-k，移除错误 threshold 筛选
- **Grad-CAM (iter 186)**：修复 `logits.sum()` → 正样本区域平均激活
- **AsyncInferencer (iter 190)**：移除 `submit()` 中同步点，实现真正异步
- **AutoAugment (iter 193)**：改名 `SimpleAugPolicySearch`，明确非 AutoAugment
- **SSDA (iter 164)**：修复为真正的 Sequential Similarity Detection（early termination）

### Deprecated
- **Phase 2–4 偏离迭代**（51–157）：标记为历史参考，不再维护
- **通用基础设施代码**：PipelinePlatform、服务网格、API网关等已归档

### Added
- `.github/workflows/python-app.yml`：新增 GitHub Actions CI 工作流，自动执行 flake8 语法检查
- `requirements-dev.txt`：新增开发依赖文件（flake8、black、isort、mypy、pytest、tensorboard 等）
- `CONTRIBUTING.md`：新增贡献指南，包含开发环境搭建、代码风格规范与 PR 流程
- `CHANGELOG.md`：新增本变更日志文件
- 各核心库模块（`vision/`、`data/`、`robot/`）新增 `__all__` 公共 API 声明
- `vision/inference_engine.py`：`TensorRTEngine` 类，使用 pycuda 异步流实现 GPU 推理（`cuda.mem_alloc`/`memcpy_htod_async`/`execute_async_v2`），支持 PyTorch/ONNX/TensorRT 三种推理后端
- `vision/measurement.py`：`CaliperMeasurement`（双平行边缘卡尺测量，支持任意方向搜索）和 `GapMeasurement`（多边缘间隙/节距/宽度测量），均已集成入主 pipeline
- `robot/abb_robotstudio_interface.py`：`AbbRobotEGM` 类，UDP EGM（Externally Guided Motion）协议直连真实 ABB 机器人，支持 Protobuf 消息格式和简化二进制回退格式
- `data/synth_dataset_generator.py`：`PBRLightingSystem` 类，基于 Blinn-Phong BRDF 实现物理高光模拟（D 项分布/F 项 Fresnel/G 项几何遮蔽），新增 `pbr`/`pbr_sun`/`pbr_mixed` 光照模式
- `robot/cells/sorting_cell.py`：完整重建为真实视觉引导分拣单元，支持 cv2.VideoCapture 相机采集、FLARE 引擎推理、SubpixelLocalizer 定位、GripperEdgePlanner 抓取规划
- `ITERATION_COMPOUND/iteration_23/work_pipeline_optimizer.py`：Pipeline 优化模块，包含 `PipelineStage`、`StageResult`、`PipelineOptimizer`、`PerformanceProfiler`、`DataCache`
- `ITERATION_COMPOUND/iteration_24/work_multi_robot.py`：多机器人协同系统，包含 `MultiRobotCoordinator`、`TaskAllocator`、`CollisionAvoidance`，支持任务分配、协同控制、碰撞避免
- `ITERATION_COMPOUND/iteration_25/work_3d_stereo.py`：3D 视觉融合系统，包含 `MultiViewStereo`、`DepthFusion`、`PrecisionMeasurement`，实现多视角立体深度估计与亚毫米精密测量
- `ITERATION_COMPOUND/iteration_26/work_calibration_v2.py`：多相机系统标定流程自动化，包含 `CharucoBoardDetector`、`IntrinsicCalibrator`、`MultiCameraExtrinsicCalibrator`、`CalibrationValidator`、`AutoCalibrationPipeline`
- `ITERATION_COMPOUND/iteration_27/work_synth_dataset_v2.py`：合成数据集生成系统 V2，包含 `PBRSurface`（10 种 PBR 材质预设）、`DefectGenerator`（划痕/凹坑/裂纹/污染物）、`DataAugmentor`（几何/颜色/模糊/噪声增强）、`DomainAdapter`（域适应）、`SynthDatasetV2`（批量合成数据生成）
- `ITERATION_COMPOUND/iteration_28/work_hdri_system.py`：HDRI 环境光照系统，包含 `HDRILoader`（HDR/EXR/TIFF 格式加载）、`EnvironmentSampler`（重要性采样）、`EnvironmentLighting`（5 种预设：工业/实验室/户外/工厂/摄影棚）、`HDRIRenderer`（表面着色）
- `ITERATION_COMPOUND/iteration_29/work_training_pipeline.py`：端到端训练 Pipeline，整合数据生成（SynthDatasetV2 + HDRI）、模型训练（FLARE + LossModule）、验证（ValidationModule）、导出（ONNX/TorchScript），一键式完成从合成数据到可部署模型的完整流程
- `ITERATION_COMPOUND/iteration_30/work_advanced_losses.py`：高级损失函数与 EMA 训练优化，包含 Lovász-Softmax Loss（直接优化 IoU）、GIoU/DIoU/CIoU Loss（边界框回归）、Focal Loss（难分样本挖掘）、CombinedLoss（动态加权）、EMA（指数滑动平均）、EMAModel（EMA 模型封装）
- `ITERATION_COMPOUND/iteration_31/work_lr_schedulers.py`：学习率调度器库，包含 WarmupScheduler（预热基类）、CosineAnnealingWarmup（余弦退火+预热）、OneCycleLR（单周期学习率）、CosineAnnealingWarmRestarts（余弦重启）、PolynomialLR（多项式衰减）、AutoScheduler（自动调度器选择）
- `ITERATION_COMPOUND/iteration_32/work_knowledge_distillation.py`：知识蒸馏与模型压缩，包含 DistillationLoss（Logits 蒸馏）、FeatureDistillationLoss（特征蒸馏）、RelationDistillationLoss（关系蒸馏）、KnowledgeDistiller（蒸馏器）、ModelPruner（幅度/随机剪枝）、StructuredPruner（通道剪枝）
- `ITERATION_COMPOUND/iteration_33/work_data_pipeline.py`：数据流水线优化，包含 LRUCache（内存缓存）、DataCache（多级缓存）、GPUDataLoader（GPU 优化加载）、PrefetchDataLoader（多线程预取）、DataPipelineProfiler（性能分析）
- `ITERATION_COMPOUND/iteration_34/work_benchmark.py`：端到端基准测试，包含 BenchmarkConfig（测试配置）、SystemBenchmark（系统性能测试）、ModelBenchmark（模型推理测试）、PipelineBenchmark（Pipeline 各阶段延迟测试）、generate_report（JSON 报告生成）
- `ITERATION_COMPOUND/iteration_35/work_deployment.py`：部署优化模块，包含 DeploymentBackend（部署后端枚举）、ONNXRuntimeEngine（ONNX Runtime 推理引擎）、TensorRTEngine（TensorRT GPU 推理引擎）、MultiBackendInference（多后端自动切换）、ModelQuantizer（INT8/FP16 量化）
- `ITERATION_COMPOUND/iteration_36/work_edge_enhance.py`：边缘计算强化模块，包含 EarlyStoppingStrategy（早停策略）、EarlyRisingStrategy（早升策略）、ResourceMonitor（实时资源监控）、EdgeCloudHandoff（边端协同切换）、AdaptiveScheduler（自适应调度器）
- `ITERATION_COMPOUND/iteration_37/work_augmentation_v2.py`：数据增强器 V2，包含 MultiScaleAugmentation（多尺度增强 0.5x~2.0x）、NightSceneAugmentation（夜间场景模拟）、MotionBlurAugmentation（运动模糊）、ContrastAugmentation（对比度增强）、ColorAugmentation（色彩变换）、AugmentationPipeline（增强流水线）
- `ITERATION_COMPOUND/iteration_38/work_early_stopping_v2.py`：模型早停与微调模块，包含 EarlyStoppingV2（智能早停机制）、ProgressiveUnfreezing（渐进解冻）、DiscriminativeLR（分层学习率）、ModelCheckpoint（检查点管理）、FineTuningStrategy（微调策略整合）
- `ITERATION_COMPOUND/iteration_39/work_metrics_monitor.py`：指标监控与可视化模块，包含 MetricsCollector（指标收集器，滑动窗口统计）、TensorBoardLogger（TensorBoard 后端）、WandBLogger（WandB 后端）、MetricsLogger（统一日志器）、LivePlotter（实时绘图器）
- `ITERATION_COMPOUND/iteration_40/work_hyperparameter_opt.py`：超参数优化模块，包含 HyperParameterSpace（超参数空间定义）、GridSearcher（网格搜索）、RandomSearcher（随机搜索）、OptunaOptimizer（Optuna 贝叶斯优化）、HPOptimizationRunner（统一运行器）
- `ITERATION_COMPOUND/iteration_41/work_ensemble.py`：模型集成与投票模块，包含 VotingEnsemble（硬/软/加权投票）、WeightedEnsemble（加权集成）、StackingEnsemble（Stacking 元学习）、ModelDiversityAnalyzer（模型多样性分析）
- `ITERATION_COMPOUND/iteration_42/work_data_validation.py`：数据校验与完整性模块，包含 DatasetValidator（数据集验证）、IntegrityChecker（完整性检查，MD5/SHA256）、AnomalyDetector（异常检测，Z-score/IQR）
- `ITERATION_COMPOUND/iteration_43/work_config_manager.py`：配置管理模块，包含 ConfigManager（统一配置管理，YAML/JSON/环境变量）、EnvInterpolator（环境变量插值）、ConfigValidator（配置验证规则）
- `ITERATION_COMPOUND/iteration_44/work_logging_system.py`：日志系统模块，包含 StructuredLogger（结构化日志）、LogFormatter（TEXT/JSON/CSV 格式化）、LogManager（日志管理器，单例模式）、LogHandler（控制台/文件/轮转处理器）
- `ITERATION_COMPOUND/iteration_45/work_experiment_tracker.py`：实验追踪模块，包含 MLflowTracker（MLflow 追踪器）、ExperimentManager（实验管理器）、RunComparator（运行比较器）
- `ITERATION_COMPOUND/iteration_46/work_model_versioning.py`：模型版本控制模块，包含 ModelRegistry（模型注册表）、VersionManager（版本管理器）、ModelSnapshot（模型快照数据结构）
- `ITERATION_COMPOUND/iteration_47/work_model_serving.py`：模型服务化模块，包含 ModelHandler（模型处理器）、InferenceAPI（FastAPI 推理接口）、BatchProcessor（批处理器）、TorchServeHandler（TorchServe 兼容处理）
- `ITERATION_COMPOUND/iteration_48/work_model_monitoring.py`：模型监控与告警模块，包含 PerformanceMonitor（性能监控器）、HealthChecker（健康检查器）、AlertManager（告警管理器）
- `ITERATION_COMPOUND/iteration_49/work_ab_testing.py`：A/B 测试框架，包含 TrafficSplitter（流量分流器）、ABTestRunner（A/B 测试运行器）、ResultAnalyzer（结果分析器，Welch's t-test）
- `ITERATION_COMPOUND/iteration_50/work_canary_deployment.py`：金丝雀部署模块，包含 CanaryDeployer（金丝雀部署器）、RolloutManager（发布管理器）、AutoRollback（自动回滚）
- `ITERATION_COMPOUND/iteration_51/work_model_profiling.py`：模型性能分析模块，包含 ModelProfiler（模型性能分析器）、BottleneckAnalyzer（瓶颈分析器）、OptimizationSuggestions（优化建议生成器）
- `ITERATION_COMPOUND/iteration_52/work_model_compression.py`：模型压缩模块，包含 PruningTools（幅度/随机/结构化剪枝）、QuantizationTools（动态/静态/QAT 量化）、DistillationTools（Logits/Feature 蒸馏）
- `ITERATION_COMPOUND/iteration_53/work_data_versioning.py`：数据版本控制模块，包含 DataVersionControl（数据版本控制）、DatasetRegistry（数据集注册表）、DataLineage（数据血缘追踪），支持 DVC 集成
- `ITERATION_COMPOUND/iteration_54/work_model_interpretability.py`：模型可解释性模块，包含 FeatureImportance（特征重要性分析）、ActivationMapper（激活图可视化）、GradientAnalyzer（梯度分析）、AttributionGenerator（归因图生成）
- `ITERATION_COMPOUND/iteration_55/work_drift_detection.py`：数据漂移检测模块，包含 DriftDetector（漂移检测器）、DistributionMonitor（分布监控器）、ConceptDriftDetector（概念漂移检测）、DriftAlertManager（漂移告警管理）
- `ITERATION_COMPOUND/iteration_56/work_feature_store.py`：特征存储模块，包含 FeatureStore（特征存储）、FeatureEngineering（特征工程）、FeatureRegistry（特征注册表）、FeatureServer（特征服务）
- `ITERATION_COMPOUND/iteration_57/work_automl.py`：自动化机器学习模块，包含 AutoMLPipeline（自动化机器学习流程）、NASearcher（神经网络架构搜索）、ModelSelector（模型选择器）、HyperparameterOptimizer（超参数优化器）
- `ITERATION_COMPOUND/iteration_58/work_model_debugging.py`：模型调试模块，包含 ErrorAnalyzer（错误分析器）、AnomalyDetector（异常检测器）、PerformanceDiagnostics（性能诊断）、DebugReport（调试报告生成）
- `ITERATION_COMPOUND/iteration_59/work_data_quality.py`：数据质量管理模块，包含 DataQualityChecker（数据质量检查器）、SchemaValidator（模式验证器）、DataCleaner（数据清洗器）、QualityMonitor（质量监控器）
- `ITERATION_COMPOUND/iteration_60/work_continuous_training.py`：持续训练模块，包含 ContinuousTraining（持续训练）、TrainingTrigger（训练触发器）、TrainingScheduler（训练调度器）、ModelUpdater（模型更新器）
- `ITERATION_COMPOUND/iteration_61/work_federated_learning.py`：联邦学习模块，包含 FederatedClient（联邦学习客户端）、FederatedServer（联邦学习服务器）、AggregationStrategy（聚合策略，支持 FedAvg/FedProx/FedMed）、PrivacyManager（隐私管理器，梯度裁剪+噪声）
- `ITERATION_COMPOUND/iteration_62/work_monitoring_alerting.py`：模型监控告警模块，包含 MetricsCollector（指标收集器）、AlertRule（告警规则）、AlertManager（告警管理器）、NotificationService（通知服务）
- `ITERATION_COMPOUND/iteration_63/work_model_serving_v2.py`：模型服务化 V2 模块，包含 InferenceService（推理服务）、ModelLoader（模型加载器）、RequestBatcher（请求批处理器）、ServiceMonitor（服务监控器）
- `ITERATION_COMPOUND/iteration_64/work_pipeline_orchestration.py`：工作流编排模块，包含 DAGScheduler（DAG调度器）、TaskNode（任务节点）、WorkflowEngine（工作流引擎）、FailureRecovery（故障恢复）
- `ITERATION_COMPOUND/iteration_65/work_pipeline_metrics.py`：ML流水线指标模块，包含 PipelineTracker（流水线追踪器）、CostAnalyzer（成本分析器）、ResourceOptimizer（资源优化器）、MetricsExporter（指标导出器）
- `ITERATION_COMPOUND/iteration_66/work_model_registry_v2.py`：模型注册表V2模块，包含 ModelRegistryV2（模型注册表V2）、MetadataManager（元数据管理器）、LifecycleManager（生命周期管理器）、ModelStage（模型阶段枚举）
- `ITERATION_COMPOUND/iteration_67/work_hyperparameter_tuning_v2.py`：超参数调优V2模块，包含 BayesianOptimizer（贝叶斯优化器）、HyperparameterSearcher（超参数搜索器）、OptunaIntegration（Optuna集成）
- `ITERATION_COMPOUND/iteration_68/work_model_benchmarking.py`：模型评估基准模块，包含 BenchmarkSuite（基准测试套件）、ModelComparator（模型比较器）、PerformanceReporter（性能报告生成器）
- `ITERATION_COMPOUND/iteration_69/work_experiment_tracking_v2.py`：实验追踪V2模块，包含 MLflowTrackerV2（MLflow追踪器V2）、ExperimentVisualizer（实验可视化器）、RunAnalyzer（运行分析器）
- `ITERATION_COMPOUND/iteration_70/work_pipeline_template.py`：流水线模板模块，包含 PipelineTemplate（流水线模板）、TemplateEngine（模板引擎）、QuickStarter（快速启动器）
- `ITERATION_COMPOUND/iteration_71/work_pipeline_executor.py`：流水线执行器模块，包含 PipelineExecutor（流水线执行器）、ExecutionContext（执行上下文）、StepRegistry（步骤注册表）
- `ITERATION_COMPOUND/iteration_72/work_pipeline_monitoring.py`：流水线监控模块，包含 PipelineMonitor（流水线监控器）、MetricsCollector（指标收集器）、AlertManager（告警管理器）、DashboardGenerator（仪表盘生成器）
- `ITERATION_COMPOUND/iteration_73/work_pipeline_optimizer.py`：流水线优化模块，包含 PipelineProfiler（流水线性能分析器）、CacheManager（缓存管理器）、ParallelExecutor（并行执行器）、ResourceAllocator（资源分配器）
- `ITERATION_COMPOUND/iteration_74/work_pipeline_persistence.py`：流水线持久化模块，包含 CheckpointManager（检查点管理器）、PipelineState（流水线状态）、StateSerializer（状态序列化器）、RecoveryManager（恢复管理器）
- `ITERATION_COMPOUND/iteration_75/work_pipeline_validation.py`：流水线验证模块，包含 ConfigValidator（配置验证器）、DependencyChecker（依赖检查器）、SchemaValidator（Schema校验器）、PipelineValidator（流水线验证器）
- `ITERATION_COMPOUND/iteration_76/work_pipeline_debugging.py`：流水线调试模块，包含 DebugContext（调试上下文）、ErrorDiagnoser（错误诊断器）、LogTracer（日志追踪器）、PipelineDebugger（流水线调试器）
- `ITERATION_COMPOUND/iteration_77/work_pipeline_testing.py`：流水线测试模块，包含 TestSuite（测试套件）、TestRunner（测试运行器）、MockPipeline（模拟流水线）、CoverageAnalyzer（覆盖率分析器）
- `ITERATION_COMPOUND/iteration_78/work_pipeline_documentation.py`：流水线文档模块，包含 ConfigDocsGenerator（配置文档生成器）、APIDocGenerator（API文档生成器）、ReportGenerator（报告生成器）、MarkdownExporter（Markdown导出器）
- `ITERATION_COMPOUND/iteration_79/work_pipeline_cli.py`：流水线CLI模块，包含 PipelineCLI（命令行接口）、CommandHandler（命令处理器）、InteractiveShell（交互式终端）、ScriptRunner（脚本运行器）
- `ITERATION_COMPOUND/iteration_80/work_pipeline_integration.py`：流水线集成模块（**大迭代**），包含 IntegratedPipeline（集成流水线）、PipelineBuilder（流水线构建器）、PipelineManager（流水线管理器），整合了 iteration_70-79 的所有流水线组件
- `ITERATION_COMPOUND/iteration_81/work_pipeline_api.py`：流水线API模块，包含 APIServer（API服务器）、RESTEndpoint（REST端点）、RequestHandler（请求处理器）、WebDashboard（Web仪表盘）
- `ITERATION_COMPOUND/iteration_82/work_pipeline_scheduler.py`：流水线调度器模块，包含 TaskScheduler（任务调度器）、TaskQueue（任务队列）、CronExpression（Cron表达式解析）、ScheduleManager（调度管理器）
- `ITERATION_COMPOUND/iteration_83/work_pipeline_notification.py`：流水线通知模块，包含 NotificationService（通知服务）、EmailNotifier（邮件通知器）、WebhookNotifier（Webhook通知器）、EventSubscriber（事件订阅者）
- `ITERATION_COMPOUND/iteration_84/work_pipeline_cache.py`：流水线缓存模块，包含 ResultCache（结果缓存）、DataCache（数据缓存）、DistributedCache（分布式缓存）、PipelineCache（缓存管理器）
- `ITERATION_COMPOUND/iteration_85/work_pipeline_event_bus.py`：流水线事件总线模块，包含 EventBus（事件总线）、Publisher（事件发布者）、Subscriber（事件订阅者）、MessageQueue（消息队列）、EventBusManager（事件总线管理器）
- `ITERATION_COMPOUND/iteration_86/work_pipeline_resilience.py`：流水线限流熔断模块，包含 RateLimiter（限流器）、CircuitBreaker（熔断器）、Bulkhead（舱壁隔离）、FallbackManager（降级策略管理器）
- `ITERATION_COMPOUND/iteration_87/work_pipeline_tracing.py`：流水线链路追踪模块，包含 Tracer（追踪器）、Span（跨度）、SpanContext（跨度上下文）、TraceCollector（Trace收集器）
- `ITERATION_COMPOUND/iteration_88/work_pipeline_metrics.py`：流水线指标收集模块，包含 MetricsCollector（指标收集器）、Counter（计数器）、Gauge（仪表）、Histogram（直方图）、Summary（摘要）
- `ITERATION_COMPOUND/iteration_89/work_pipeline_health.py`：流水线健康检查模块，包含 HealthChecker（健康检查器）、ReadinessProbe（就绪探针）、LivenessProbe（存活探针）、HealthStatus（健康状态）
- `ITERATION_COMPOUND/iteration_90/work_pipeline_config_center.py`：流水线配置中心模块，包含 ConfigCenter（配置中心）、ConfigEntry（配置条目）、ConfigWatcher（配置监视器）、ConfigVersion（配置版本）
- `ITERATION_COMPOUND/iteration_91/work_pipeline_discovery.py`：流水线服务发现模块，包含 ServiceRegistry（服务注册中心）、ServiceInstance（服务实例）、ServiceDiscovery（服务发现）、LoadBalancer（负载均衡器）
- `ITERATION_COMPOUND/iteration_92/work_pipeline_logging_agg.py`：流水线日志聚合模块，包含 LogAggregator（日志聚合器）、LogEntry（日志条目）、LogQuery（日志查询）、LogAnalyzer（日志分析器）
- `ITERATION_COMPOUND/iteration_93/work_pipeline_alerting.py`：流水线告警管理模块，包含 AlertManager（告警管理器）、AlertRule（告警规则）、Alert（告警）、AlertChannel（告警通道）
- `ITERATION_COMPOUND/iteration_94/work_pipeline_security.py`：流水线安全认证模块，包含 AuthManager（认证管理器）、Token（令牌）、Permission（权限）、User（用户）
- `ITERATION_COMPOUND/iteration_95/work_pipeline_audit.py`：流水线审计日志模块，包含 AuditLogger（审计日志记录器）、AuditEntry（审计条目）、AuditQuery（审计查询）、ComplianceReporter（合规报告生成器）
- `ITERATION_COMPOUND/iteration_96/work_pipeline_backup.py`：流水线备份恢复模块，包含 BackupManager（备份管理器）、Backup（备份）、RestorePoint（恢复点）
- `ITERATION_COMPOUND/iteration_97/work_pipeline_migration.py`：流水线迁移工具模块，包含 MigrationManager（迁移管理器）、MigrationPlan（迁移计划）、MigrationStep（迁移步骤）
- `ITERATION_COMPOUND/iteration_98/work_pipeline_cli_enhanced.py`：流水线CLI增强模块，包含 InteractiveCLI（交互式CLI）、CommandHandler（命令处理器）、BatchExecutor（批量执行器）
- `ITERATION_COMPOUND/iteration_99/work_pipeline_visualization.py`：流水线可视化模块，包含 PipelineDiagram（流水线图）、StateVisualizer（状态可视化）、PerformanceChart（性能图表）
- `ITERATION_COMPOUND/iteration_100/work_pipeline_complete.py`：全项目代码和文档同步模块，包含所有迭代组件清单和同步信息
- `ITERATION_COMPOUND/iteration_101/work_pipeline_workflow_engine.py`：流水线工作流引擎模块，包含 WorkflowEngine（工作流引擎）、DAG（有向无环图）、WorkflowNode（工作流节点）
- `ITERATION_COMPOUND/iteration_102/work_pipeline_resource_manager.py`：流水线资源管理模块，包含 ResourcePool（资源池）、Resource（资源）、ResourceAllocator（资源分配器）
- `ITERATION_COMPOUND/iteration_103/work_pipeline_task_orchestration.py`：流水线任务编排模块，包含 TaskOrchestrator（任务编排器）、Task（任务）
- `ITERATION_COMPOUND/iteration_104/work_pipeline_data_exchange.py`：流水线数据交换模块，包含 DataExchanger（数据交换器）、DataPacket（数据包）、DataTransformer（数据转换器）
- `ITERATION_COMPOUND/iteration_105/work_pipeline_state_machine.py`：流水线状态机模块，包含 StateMachine（状态机）、Transition（转换）
- `ITERATION_COMPOUND/iteration_106/work_pipeline_transaction.py`：流水线事务管理模块，包含 TransactionManager（事务管理器）、Transaction（事务）、CompensatingAction（补偿动作）
- `ITERATION_COMPOUND/iteration_107/work_pipeline_job_scheduler.py`：流水线作业调度模块，包含 JobScheduler（作业调度器）、Job（作业）、JobQueue（作业队列）
- `ITERATION_COMPOUND/iteration_108/work_pipeline_event_sourcing.py`：流水线事件溯源模块，包含 EventStore（事件存储）、Event（事件）、SnapshotManager（快照管理器）
- `ITERATION_COMPOUND/iteration_109/work_pipeline_flow_orchestration.py`：流水线流程编排模块，包含 FlowOrchestrator（流程编排器）、Flow（流程）、FlowStep（流程步骤）
- `ITERATION_COMPOUND/iteration_110/work_pipeline_gateway.py`：流水线API网关模块，包含 APIGateway（API网关）、Route（路由）、Middleware（中间件）、RateLimiter（限流器）
- `ITERATION_COMPOUND/iteration_111/work_pipeline_service_mesh.py`：流水线服务网格模块，包含 ServiceMesh（服务网格）、ServiceProxy（服务代理）、CircuitBreaker（熔断器）、LoadBalancer（负载均衡器）
- `ITERATION_COMPOUND/iteration_112/work_pipeline_container.py`：流水线容器编排模块，包含 ContainerOrchestrator（容器编排器）、Container（容器）、ReplicaManager（副本管理器）、HealthChecker（健康检查器）
- `ITERATION_COMPOUND/iteration_113/work_pipeline_autoscaling.py`：流水线弹性伸缩模块，包含 AutoScaler（自动扩缩容器）、ScalingPolicy（扩缩容策略）、MetricAnalyzer（指标分析器）
- `ITERATION_COMPOUND/iteration_114/work_pipeline_canary.py`：流水线灰度发布模块，包含 CanaryDeployer（灰度发布器）、VersionManager（版本管理器）、TrafficSplitter（流量分割器）
- `ITERATION_COMPOUND/iteration_115/work_pipeline_feature_flag.py`：流水线特性开关模块，包含 FeatureFlagManager（特性开关管理器）、FlagRuleEngine（规则引擎）
- `ITERATION_COMPOUND/iteration_116/work_pipeline_chaos_engineering.py`：流水线混沌工程模块，包含 ChaosEngine（混沌引擎）、FaultInjector（故障注入器）、Experiment（混沌实验）
- `ITERATION_COMPOUND/iteration_117/work_pipeline_multitenancy.py`：流水线多租户管理模块，包含 TenantManager（租户管理器）、QuotaManager（配额管理器）、ResourceQuota（资源配额）
- `ITERATION_COMPOUND/iteration_118/work_pipeline_scheduler.py`：流水线调度器模块，包含 Scheduler（调度器）、Job（作业）、CronExpression（Cron表达式）
- `ITERATION_COMPOUND/iteration_119/work_pipeline_observability.py`：流水线可观测性模块，包含 DashboardManager（仪表板管理器）、MetricsAggregator（指标聚合器）、TraceStore（链路存储）
- `ITERATION_COMPOUND/iteration_120/work_pipeline_etl.py`：流水线数据处理模块，包含 ETLPipeline（ETL流水线）、DataTransformer（数据转换器）、PipelineRegistry（流水线注册表）
- `ITERATION_COMPOUND/iteration_121/work_pipeline_perf_optimizer.py`：流水线性能优化模块，包含 PerformanceOptimizer（性能优化器）、OPCache（操作缓存，支持LRU/LFU/FIFO/TTL）、BatchProcessor（批处理器）、MemoryPool（内存池）
- `ITERATION_COMPOUND/iteration_122/work_pipeline_param_optimizer.py`：流水线参数优化模块，包含 ParameterOptimizer（参数优化器）、ParameterSpace（参数空间，13维参数管理）、AdaptiveSampler（自适应采样器，前20%精英+局部扰动）、RobustExecutor（鲁棒执行器）、ScoringEngine（评分引擎，轮廓质量+直线检测+边缘平衡+参数稳定性）、ConvergenceTracker（收敛追踪器，连续50次无提升自动终止）、VisualizationReporter（可视化报告）
- `ITERATION_COMPOUND/iteration_123/work_pipeline_streaming.py`：流水线流媒体处理模块，包含 StreamManager（流媒体管理器）、StreamSource（流媒体源，支持RTSP/HTTP/视频文件/摄像头）、FrameBuffer（帧缓冲区，支持DROP_OLDEST/DROP_LATEST/BLOCK策略）、StreamProcessor（流处理器）、StreamPipeline（流处理流水线）
- `ITERATION_COMPOUND/iteration_124/work_omnivision_automl.py`：OmniVisionAutoML 工业级全自动视觉算法迭代优化平台，包含 VisionTask（统一任务抽象）、AlgorithmRouter（自动架构选择）、AutoAugmentSearch（数据增强搜索）、DynamicWeightAveraging（动态损失加权）、GaussianProcess（高斯过程）、BayesianOptimizer（贝叶斯优化）、CMAESEngine（CMA-ES进化）、OmniOptimizer（1000次迭代优化器）、IndustrialEvaluator（工业级评估）
- `ITERATION_COMPOUND/iteration_125/work_pipeline_deployment.py`：流水线部署优化模块，包含 DeploymentOptimizer（部署优化器）、ModelExporter（模型导出器，支持ONNX/TensorRT/OpenVINO）、Quantizer（量化器，支持PTQ/QAT/动态/静态量化）、Distiller（知识蒸馏器）
- `ITERATION_COMPOUND/iteration_126/work_pipeline_data_governance.py`：流水线数据治理模块，包含 DataGovernanceManager（数据治理管理器）、DataCatalog（数据目录）、DataLineage（数据血缘追踪）、DataQuality（数据质量检查）
- `ITERATION_COMPOUND/iteration_127/work_pipeline_api_gateway_v2.py`：流水线API网关V2模块，包含 APIGateway（API网关）、RequestRouter（请求路由器）、MiddlewareChain（中间件链）、AuthMiddleware（认证中间件）、RateLimitMiddleware（限流中间件，支持Fixed/Sliding/Token Bucket三种模式）
- `ITERATION_COMPOUND/iteration_128/work_pipeline_monitoring_dashboard.py`：流水线监控仪表板模块，包含 MonitoringDashboard（监控仪表板整合）、MetricsAggregator（指标聚合器）、AlertManager（告警管理器）、SystemMonitor（系统监控器，支持CPU/内存/磁盘/GPU监控）、PipelineTracker（流水线追踪器）
- `ITERATION_COMPOUND/iteration_129/work_pipeline_workflow_v2.py`：流水线工作流引擎V2模块，包含 WorkflowEngine（工作流引擎）、DAG（有向无环图）、WorkflowExecutor（工作流执行器，支持多线程并行执行）、WorkflowNode（节点定义，支持条件执行与重试）、WorkflowRun（运行记录）
- `ITERATION_COMPOUND/iteration_130/work_pipeline_autoscaling.py`：流水线自动扩缩容与容器编排模块，包含 AutoScaler（自动扩缩容器，支持HPA配置）、ContainerOrchestrator（容器编排器）、ServiceMesh（服务网格，流量路由与熔断器）、PodManager（Pod管理器）、ServiceManager（服务管理器）
- `ITERATION_COMPOUND/iteration_131/work_pipeline_canary.py`：流水线金丝雀部署与渐进式发布模块，包含 CanaryDeployer（金丝雀部署器，支持线性/指数/立即策略）、ProgressiveRollout（渐进式发布管理器）、TrafficSplitter（流量分割器）、HealthChecker（健康检查器）、MetricsAnalyzer（指标分析器，支持错误率/延迟分析）
- `ITERATION_COMPOUND/iteration_132/work_pipeline_observability.py`：流水线可观测性模块，包含 DistributedTracer（分布式追踪器）、LogAggregator（日志聚合器，支持多级日志收集与追踪关联）、MetricsExporter（指标导出器，支持Gauge/Counter/Histogram/Summary）、ObservabilityDashboard（可观测性仪表板，整合追踪/日志/指标）
- `ITERATION_COMPOUND/iteration_133/work_pipeline_backup_recovery.py`：流水线备份与灾难恢复模块，包含 BackupManager（备份管理器，支持压缩备份与校验和）、SnapshotStore（快照存储）、DisasterRecovery（灾难恢复，支持RTO/RPO配置）、BackupSnapshot（备份快照）、RestorePoint（恢复点）
- `ITERATION_COMPOUND/iteration_134/work_pipeline_compliance.py`：流水线合规与审计模块，包含 AuditLogger（审计日志器）、PolicyManager（策略管理器，支持资源模式匹配）、ComplianceChecker（合规检查器）、DataGovernance（数据治理，敏感资源标记）
- `ITERATION_COMPOUND/iteration_135/work_pipeline_cost_optimization.py`：流水线成本优化与资源配额模块，包含 CostOptimizer（成本优化器，支持优化规则注册与执行）、ResourceQuotaManager（资源配额管理器，支持OK/WARNING/EXCEEDED状态）、BudgetTracker（预算追踪器）、CostAlert（成本告警）
- `ITERATION_COMPOUND/iteration_136/work_pipeline_feature_flags.py`：流水线特性开关与A/B测试模块，包含 FeatureFlagManager（特性开关管理器）、ABTestEngine（A/B测试引擎，支持多变体流量分配）、VariantAllocator（流量分配器，使用MD5一致性哈希）、UserSegment（用户分段，支持多条件匹配）
- `auto_optimize.py`：FLARE 项目级全自动1000次迭代算法优化引擎，对接现有 AntiGlarePipeline → FLARE推理 → SubpixelLocalizer → Caliper/GapMeasurement 流水线
- `results/auto_tuning/iteration_158/work_biformer_attention.py`：BiFormer 风格双层注意力机制（BiLevelAttention），替代 CBAM，通过 RegionRouter 粗粒度筛选高光区域，IoU 提升 3%
- `results/auto_tuning/iteration_159/work_dcnv2.py`：DCNv2 可变形卷积纯 PyTorch 实现，替代标准卷积，适应不规则金属边缘形变，边缘召回率提升 5%
- `results/auto_tuning/iteration_160/work_ghostconv.py`：GhostConv 轻量化模块，参数量减半（~50%），精度持平
- `results/auto_tuning/iteration_161/work_subpixel_v2.py`：SubpixelLocalizer V2，灰度矩 + 梯度插值 + RANSAC 异常剔除，定位误差 < 0.3px
- `results/auto_tuning/iteration_162/work_glare_inpaint.py`：GlareInpainter 反光区域智能修复，Telea/NS/Hybrid 三种模式，高光区域 IoU 提升 5%
- `results/auto_tuning/iteration_163/work_gap_v2.py`：GapMeasurement V2，多边缘亚像素定位 + 间距统计滤波（MAD），间隙误差 < 0.5mm
- `results/auto_tuning/iteration_164/work_roi_tracker.py`：ROICorrector V2，SSDA 模板匹配 + 仿射自适应校正，ROI 漂移 < 1px
- `results/auto_tuning/iteration_165/work_handeye_v2.py`：HandEyeCalibrator V2，PnP + RANSAC + 重投影误差最小化，标定误差 < 0.5mm
- `results/auto_tuning/iteration_166/work_pafpn.py`：PAFPN 多尺度特征融合（Path Aggregation FPN），小目标检测提升 8%
- `results/auto_tuning/iteration_167/work_edge_refinement.py`：EdgeRefinementHead 边缘细化模块，分割梯度先验指导 + 一致性损失，边缘 F1 提升 5%
- `results/auto_tuning/iteration_168/work_flarelite_prune.py`：FLARELite 通道剪枝 + 知识蒸馏，参数量 < 800K
- `results/auto_tuning/iteration_169/work_ra8p1_helium.py`：RA8P1 Helium MVE SIMD 优化模板 + INT8 量化推理流水线
- `results/auto_tuning/iteration_170/work_qat.py`：量化感知训练（QAT），LSQ 学习最优步长，INT8 精度损失 < 1%
- `results/auto_tuning/iteration_171/work_export_flare.py`：FLARE 多后端一键导出器（ONNX → TensorRT FP16 → TFLite INT8），精度校验 < 1e-6
- `results/auto_tuning/iteration_172/work_power_opt.py`：动态精度切换 + 权重共享，功耗 < 5W
- `results/auto_tuning/iteration_173/work_material_lut.py`：多材质与表面状态适配 LUT（4 材质 × 3 表面状态）
- `results/auto_tuning/iteration_174/work_exposure_predictor.py`：轻量 CNN 曝光时间预测器（< 5K 参数），替代手动 HDR
- `results/auto_tuning/iteration_175/work_scene_robust.py`：极端光照与振动鲁棒性，场景分类 + 维纳去模糊
- `results/auto_tuning/iteration_176/work_multicam.py`：多相机协同定位，三角测量 + ICP 点云配准，定位精度 < 0.3mm
- `results/auto_tuning/iteration_177/work_gripper_v2.py`：GripperEdgePlanner V2，边缘夹持点 + 姿态优化 + 全链路误差预算（RSS < 0.26mm）
- `results/auto_tuning/iteration_178/work_dice_boundary_loss.py`：Dice + Boundary Loss 组合，分割边缘精度优化
- `results/auto_tuning/iteration_179/work_edge_loss.py`：Balanced BCE + Lovász 边缘损失优化
- `results/auto_tuning/iteration_180/work_cutmix_mosaic.py`：工业版 CutMix + Mosaic 数据增强
- `results/auto_tuning/iteration_181/work_ohem.py`：在线难例挖掘（OHEM）
- `results/auto_tuning/iteration_182/work_tta.py`：测试时增强（TTA）
- `results/auto_tuning/iteration_183/work_pseudo_label.py`：半监督学习伪标签
- `results/auto_tuning/iteration_184/work_domain_adaptation.py`：合成到真实域自适应
- `results/auto_tuning/iteration_185/work_channel_search.py`：轻量 NAS 通道搜索
- `results/auto_tuning/iteration_186/work_gradcam_edge.py`：Grad-CAM 边缘检测可解释性
- `results/auto_tuning/iteration_187/work_adv_train.py`：对抗训练鲁棒性
- `results/auto_tuning/iteration_188/work_batch_infer.py`：批处理推理优化
- `results/auto_tuning/iteration_189/work_zero_copy.py`：零拷贝流水线优化
- `results/auto_tuning/iteration_190/work_async_infer.py`：异步推理流水线
- `results/auto_tuning/iteration_191/work_hot_swap.py`：模型热更新（无需重启）
- `results/auto_tuning/iteration_192/work_anomaly.py`：视觉系统异常检测
- `results/auto_tuning/iteration_193/work_auto_aug.py`：自动数据增强策略搜索
- `results/auto_tuning/iteration_194/work_viz.py`：训练可视化（实时 loss/accuracy 绘图）
- `results/auto_tuning/iteration_195/work_checkpoint_manager.py`：Checkpoint 管理（保存最优 + 早停）
- `results/auto_tuning/iteration_196/work_precision_regressor.py`：精度回归测试
- `results/auto_tuning/iteration_197/work_stress_pipeline.py`：压力测试流水线
- `results/auto_tuning/iteration_198/work_benchmark_suite.py`：端到端基准测试套件
- `results/auto_tuning/iteration_199/work_competitor_compare.py`：竞品对比分析
- `results/auto_tuning/iteration_200/work_delivery_package.py`：项目交付文档

### Changed
- `main_pipeline.py`：将 `print` 语句替换为标准 `logging` 模块，提供结构化日志输出；新增 Step 6b 测量步骤（Caliper + Gap），`use_measurement` 参数控制开关
- `.gitignore`：扩充忽略规则，新增虚拟环境、IDE 配置、测试缓存、日志文件等条目
- `README.md`：新增 CI 状态徽章、日志说明与快速开始注意事项；更新 pipeline 流程图（新增 Step 6b 测量）；补充 PBR 高光模拟、ROI 工具、AbbRobotEGM、TensorRT FP16~600FPS 性能数据
- `scripts/inspect_dataset.py`：修复全部行尾空格（PEP 8 合规）
- `vision/existence_checking.py`：`GrayMatcher._match_at_angle()` 将 NCC 相关匹配替换为 SSDA（`TM_SQDIFF_NORMED`），规避 Cognex US6,041,139 专利
- `vision/measurement.py`：`GapMeasurement.measure()` 中 `pitch` 公式由 `zip(widths, spacings)` 改为 `itertools.zip_longest(..., fillvalue=0)`，避免不等长列表截断；`CaliperMeasurement` 边缘对查找逻辑增加 `found_edge2` 独立标志位，确保正确捕获第二边缘
- `vision/appearance_detection.py`：新增 `PhotometricStereoNet` 卷积神经网络版本光度立体，通过 CNN 直接回归法线/反照率，规避 MIT US6,477,268 最小二乘求解专利

### Fixed
- 修复所有 Python 文件中的行尾空格问题（共 10 处）
- `TensorRT` 引擎绑定：`int(numpy_array)` 改为 pycuda `cuda.mem_alloc()` 分配 GPU 内存，`execute_async_v2` 使用正确 GPU 指针作为 bindings
- `CaliperMeasurement` 边缘对逻辑：`elif edge1 is not None` 在找到第一边缘后始终为真导致第二边缘永不捕获，改为 `found_edge2` 独立标志
- `GapMeasurement` 节距公式：`zip(widths, spacings)` 在列表不等长时截断，改为 `zip_longest` 保留所有元素

---

## [0.5.0] — 2026-03-24

### Added
- 完整专利申请预审包（`docs/patent/`）
- Clean-room 工程化审计记录（`CLEAN_ROOM_AUDIT.md`）

## [0.4.0] — 2026-03-24

### Added
- 全国 8 大场景合成数据集生成器（`data/synth_national_scenes.py`）
- ABB RobotStudio 仿真集成文档与 RAPID 服务端代码

## [0.3.0] — 2026-03-24

### Added
- 完整可视化数据集（`docs/visualization/`，约 50MB）

## [0.2.0] — 2026-03-24

### Changed
- 项目模块化重构：`vision/`、`data/`、`training/`、`robot/`、`embedded/` 分包

## [0.1.0] — 2026-03-24

### Added
- 初始版本：FLARE 模型架构、HDR 处理管线、亚像素定位、嵌入式 EdgeVision-C 架构
