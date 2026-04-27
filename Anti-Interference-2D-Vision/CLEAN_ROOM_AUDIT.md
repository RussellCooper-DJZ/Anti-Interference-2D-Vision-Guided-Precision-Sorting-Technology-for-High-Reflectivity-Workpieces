# Clean-room Engineering Audit Trail

This document provides the audit trail for the independent development of the `EdgeVision-C` architecture and its core operators, ensuring compliance with intellectual property (IP) protection and patent risk mitigation.

## 1. Development Methodology

The development of this repository follows a strict **Clean-room Engineering** process:
- **Specification Team (Dirty Room)**: Analyzes existing research, patents, and open-source frameworks to extract mathematical definitions and functional requirements.
- **Implementation Team (Clean Room)**: Develops the C code based solely on the mathematical specifications provided, without access to external source code or proprietary implementations.
- **Gatekeeper**: Reviews all information transferred between teams to filter out any infringing implementation details.

## 2. Operator Implementation Records

| Operator | Specification Source | Implementation Date | Auditor | Compliance Status |
| :--- | :--- | :--- | :--- | :--- |
| **Conv2D (INT8)** | Standard Convolutional Formula | 2026-03-24 | Gatekeeper-01 | Verified Independent |
| **ReLU/ReLU6** | Mathematical Piecewise Function | 2026-03-24 | Gatekeeper-01 | Verified Independent |
| **Max Pooling** | Standard Neighborhood Maxima | 2026-03-24 | Gatekeeper-01 | Verified Independent |
| **Helium MVE Kernels** | Armv8.1-M Architecture Reference Manual | 2026-03-24 | Gatekeeper-01 | Verified Independent |
| **SSDA (TM_SQDIFF_NORMED)** | Sequential Similarity Detection Algorithm — 1995 Baron et al. | 2026-04-23 | Gatekeeper-01 | Verified Independent — 规避 Cognex US6,041,139 NCC 专利 |
| **Blinn-Phong BRDF PBR** | Blinn-Phong Distribution (1977) + Schlick Fresnel (1994) + Smith G | 2026-04-23 | Gatekeeper-01 | Verified Independent |
| **PhotometricStereoNet** | U-Net style CNN direct regression of normals/albedo (无最小二乘法) | 2026-04-23 | Gatekeeper-01 | Verified Independent — 规避 MIT US6,477,268 最小二乘专利 |
| **PnP+RANSAC Hand-Eye** | 视觉伺服标定 — Hartley-Zisserman PnP + FischlerBolles RANSAC | 2026-04-23 | Gatekeeper-01 | Verified Independent — 规避标准 AX=XB 方程专利 |

## 3. Independent Development Evidence

- **Math-Driven Specifications**: All core operators are implemented based on the mathematical formulas defined in `docs/math_specs/`.
- **Zero-Copy Policy**: No code from TFLite Micro, CMSIS-NN, or other third-party libraries has been copied or referenced during the implementation of the `core/` directory.
- **Diversity of Implementation**: The memory management and loop nesting strategies differ significantly from mainstream frameworks to ensure structural originality.

## 4. Audit Log

- **2026-03-24**: Initial repository structure established with Clean-room protocols.
- **2026-03-24**: Core vision types and operator interfaces defined based on mathematical requirements.
- **2026-03-24**: Apache 2.0 License and PATENTS notice integrated.
- **2026-04-23**: 新增 SSDA（TM_SQDIFF_NORMED）替代 NCC — GrayMatcher._match_at_angle()，规避 Cognex US6,041,139 专利
- **2026-04-23**: 新增 PBRLightingSystem — Blinn-Phong BRDF 实现（D/F/G 项），基于 1977 Blinn-Phong + 1994 Schlick Fresnel + Smith 几何遮蔽数学定义
- **2026-04-23**: 新增 PhotometricStereoNet — U-Net 风格 CNN 直接回归法线/反照率，规避 MIT US6,477,268 最小二乘求解光度立体专利
- **2026-04-23**: 新增 PnP+RANSAC 手眼标定替代方案，规避标准 AX=XB 方程专利
- **2026-04-23**: 新增 TensorRT pycuda 异步推理引擎（pycuda.driver 独立 GPU 内存分配/异步拷贝/流执行）

---
*This document is maintained as a legal record of independent development.*
