# Anti-Interference 2D Vision-Guided Precision Sorting for High-Reflectivity Workpieces

## 抗干扰 2D 视觉引导高反光工件精准分拣系统

基于 **Renesas RZ/V2H + RA8P1** 平台，使用深度学习实现高反光金属工件（汽车门板钢、铝合金等）的精准边缘识别与机器人抓取。

---

## 核心特性

| 特性 | 描述 |
|------|------|
| **AGEANet 模型** | Anti-Glare Edge-Aware U-Net，带 CBAM 注意力机制和边缘感知分支 |
| **反光抑制** | HDR 融合 + 高光检测修复 + 偏振模拟 + 自适应增强 |
| **亚像素定位** | Zernike 矩亚像素边缘精修，达到 0.1 像素级精度 |
| **EdgeVision-C 架构** | **[NEW]** 纯 C AI 视觉架构，支持分层运行时、INT8 量化与 Helium 优化 |
| **Clean-room 工程化** | **[NEW]** 严格遵循独立开发流程，规避专利风险，建立审计追踪 |
| **专利保护** | **[NEW]** 核心算法与架构受专利保护，采用 Apache 2.0 协议授权 |

---

## 落地路线图 (Implementation Roadmap)

本项目遵循严格的 30/60/90 天落地计划，确保从原型到量产的平滑过渡。

### 第一阶段：30 天 - 核心框架与基础验证
- **目标**: 完成 `EdgeVision-C` 核心框架、静态内存管理器及基础算子（Conv2D, ReLU, Pooling）的参考实现。
- **产出**: 
    - `core/include/`: 定义张量结构与算子接口。
    - `core/src/memory_pool.c`: 实现静态内存分配与复用逻辑。
    - **验证**: 在 PC 端通过单元测试验证 INT8 算子的数学正确性。

### 第二阶段：60 天 - 算子全覆盖与 Helium 优化
- **目标**: 完成所有视觉常用算子，并针对 Arm Helium (MVE) 进行深度优化。
- **产出**:
    - `core/kernels/helium/`: 使用 `arm_mve.h` 内联函数重写核心卷积与矩阵乘法算子。
    - **批量回归**: 建立自动化测试脚本，对比参考实现与优化实现的输出一致性。
    - **专利规避文档**: 更新 `CLEAN_ROOM_AUDIT.md`，记录每个算子的设计来源与实现路径。

### 第三阶段：90 天 - 产线联调与量产准备
- **目标**: 完成产线异常追溯系统、版本冻结与完整技术文档。
- **产出**:
    - **异常追溯**: 实现算子级性能 Profiling 与中间层张量 Dump 功能。
    - **版本管理**: 建立语义化版本控制，完成代码冻结。
    - **文档**: 提供《算子规格说明书》、《内存占用分析报告》、《量产集成指南》。

---

## 项目结构

```
├── core/                          # [NEW] EdgeVision-C 纯 C 推理引擎
│   ├── include/                   # 核心头文件 (vision_types.h, operator_interface.h)
│   ├── src/                       # 引擎实现 (memory_pool.c, inference_engine.c)
│   └── kernels/                   # 算子内核 (reference/, helium/)
├── docs/                          # 文档与数学规格说明书
├── tests/                         # 单元测试与回归测试
├── feature_extraction.py          # AGEANet / AGEANet-Lite 模型架构
├── train.py                       # 完整训练流程
├── data_augmentation.py           # 高反光专用数据增强
├── hdr_processing.py              # HDR 融合与反光抑制管线
├── ra8p1_tflm_adapter.py          # 嵌入式部署导出工具
├── LICENSE                        # Apache 2.0 许可证
├── PATENTS                        # 专利声明
└── CLEAN_ROOM_AUDIT.md            # Clean-room 审计追踪
```

---

## 许可证与专利声明

本项目采用 **Apache License 2.0** 协议开源。

**专利声明**：本项目实现的 `AGEANet` 架构、`EdgeVision-C` 静态内存管理算法及 `Helium` 优化算子受专利保护。访问源代码不代表获得专利许可，商业用途请联系 RussellCooper。详情请参阅 [PATENTS](./PATENTS) 文件。

**合规性声明**：本项目遵循 **Clean-room 工程化**标准独立开发，所有核心算子实现均基于数学定义，未参考任何受保护的第三方源代码。详情请参阅 [CLEAN_ROOM_AUDIT.md](./CLEAN_ROOM_AUDIT.md)。
