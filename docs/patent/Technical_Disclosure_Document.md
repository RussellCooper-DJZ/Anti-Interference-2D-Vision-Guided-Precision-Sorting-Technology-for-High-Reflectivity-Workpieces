# 专利技术交底书：高反光工件抗干扰 2D 视觉引导精准分拣系统
# Patent Technical Disclosure Document: Anti-Interference 2D Vision-Guided Precision Sorting System for High-Reflectivity Workpieces

:Author: RussellCooper
:Date: 2026-03-27

## 1. 发明名称 (Title of Invention)

**中**: 一种基于边缘感知深度学习与多模态融合的高反光工件抗干扰 2D 视觉引导精准分拣系统及方法
**En**: An Anti-Interference 2D Vision-Guided Precision Sorting System and Method for High-Reflectivity Workpieces Based on Edge-Aware Deep Learning and Multi-Modal Fusion

## 2. 所属技术领域 (Field of the Invention)

本发明属于机器视觉、工业自动化、深度学习及机器人控制领域，尤其涉及在高反光、复杂光照环境下对金属工件进行高精度识别、定位与分拣的技术。

This invention belongs to the fields of machine vision, industrial automation, deep learning, and robot control, particularly concerning high-precision recognition, localization, and sorting of metal workpieces in highly reflective and complex lighting environments.

## 3. 背景技术 (Background Art)

现有工业自动化分拣系统在处理高反光金属工件（如汽车门板、铝合金型材、大型船舶钢板等）时面临严峻挑战。传统机器视觉方法易受高光、阴影、环境光干扰，导致图像质量差、特征提取困难、定位精度低。基于深度学习的方法虽有进步，但仍难以有效处理高反光带来的特征丢失和误识别问题，且在嵌入式端部署时，对算力、内存和实时性要求高，现有通用深度学习框架难以满足。

Existing industrial automated sorting systems face severe challenges when dealing with highly reflective metal workpieces (e.g., automotive door panels, aluminum alloy profiles, large ship steel plates). Traditional machine vision methods are susceptible to glare, shadows, and ambient light interference, leading to poor image quality, difficult feature extraction, and low localization accuracy. Although deep learning-based methods have made progress, they still struggle to effectively handle feature loss and misidentification caused by high reflectivity. Furthermore, deploying these methods on embedded devices demands high computational power, memory, and real-time performance, which existing general-purpose deep learning frameworks struggle to meet.

## 4. 发明内容 (Summary of Invention)

本发明旨在提供一种能够有效克服高反光、复杂光照干扰，实现对金属工件精准识别、亚像素级定位，并支持嵌入式高效部署的 2D 视觉引导分拣系统及方法。

The present invention aims to provide an anti-interference 2D vision-guided sorting system and method for metal workpieces that effectively overcomes glare and complex lighting interference, achieves precise recognition and sub-pixel level localization, and supports efficient embedded deployment.

### 4.1 核心发明点 (Core Inventive Points)

#### 4.1.1 AGEANet：抗眩光边缘感知深度学习网络 (Anti-Glare Edge-Aware Network)

- **创新性**: 提出一种新型的 U-Net 变体，集成 **CBAM (Convolutional Block Attention Module)**，增强网络对关键特征的关注度，同时引入**边缘感知分支 (Edge-Aware Branch)**，在主分割任务之外，显式地学习和预测工件的精确边缘。这使得网络在强高光导致纹理信息丢失时，仍能通过边缘信息进行鲁棒识别。
- **技术效果**: 显著提升在高反光场景下工件边缘的识别精度和鲁棒性，有效避免高光区域的误分割，为后续亚像素定位提供高质量输入。

#### 4.1.2 多模态融合反光抑制与图像增强 (Multi-Modal Fusion Glare Suppression & Image Enhancement)

- **创新性**: 结合**多曝光 HDR 融合**、**偏振图像处理**（模拟）与**自适应高光检测修复**技术。通过融合不同曝光、不同偏振角度的图像信息，并利用深度学习模型对高光区域进行像素级修复，有效消除高光饱和、光晕和阴影，生成纹理细节丰富、亮度均匀的无眩光图像。
- **技术效果**: 从根本上解决高反光图像质量差的问题，为 AGEANet 提供更清晰、更稳定的输入，大幅提高系统对复杂光照环境的适应性。

#### 4.1.3 亚像素级边缘定位与位姿解算 (Sub-Pixel Edge Localization & Pose Estimation)

- **创新性**: 采用**Zernike 矩**或**高斯拟合**等先进算法对 AGEANet 预测的边缘进行亚像素级精修，将边缘定位精度从像素级提升至 **0.1 像素**甚至更高。结合相机标定参数，通过 PnP (Perspective-n-Point) 算法或基于模板匹配的位姿解算，精确计算工件在机器人坐标系下的三维位姿。
- **技术效果**: 确保机器人抓取或加工所需的超高定位精度，满足工业 4.0 对精密操作的要求。

#### 4.1.4 EdgeVision-C：嵌入式纯 C 推理引擎 (Embedded Pure C Inference Engine)

- **创新性**: 针对 Renesas RA8P1 等资源受限的嵌入式平台，设计并实现了一套**纯 C 语言**编写的轻量级深度学习推理引擎。该引擎采用**静态内存管理**、**分层运行时**（参考实现与硬件加速实现分离）和**Arm Helium (MVE) 向量化优化**，支持 INT8 量化推理。
- **技术效果**: 在不依赖复杂操作系统和第三方库的情况下，实现 AGEANet 模型在嵌入式端的超低延迟、高吞吐量推理，满足实时性要求，同时大幅降低内存占用和功耗。

#### 4.1.5 全国复杂场景合成训练数据集生成器 (National Complex Scene Synthetic Training Data Generator)

- **创新性**: 开发了一套高度可配置的合成数据集生成器，能够模拟中国工业场景中 8 种典型大型金属高光面工件（如船厂、钢厂、桥梁、港口起重机、铁路、建筑幕墙、管道/储罐、风电）及其 7 种复杂光照条件（强侧光、顶光、水面反射、阴天、夜间 LED、焊接弧光、混合光源）。
- **技术效果**: 解决了真实高反光数据集难以获取、标注成本高昂的问题，通过合成数据大幅提升模型在各种极端环境下的泛化能力和鲁棒性，加速模型迭代与部署。

## 5. 有益效果 (Beneficial Effects)

1.  **高精度与高鲁棒性**: 在高反光、复杂光照环境下，实现对工件边缘的亚像素级精准识别与定位，分拣成功率和精度远超现有技术。
2.  **实时性与低成本**: EdgeVision-C 引擎确保模型在嵌入式端以极低延迟运行，降低对昂贵 GPU 硬件的依赖，显著降低系统总成本。
3.  **泛化能力强**: 结合合成数据集训练，模型能够适应全国范围内的多样化工业场景和极端光照条件，无需大量真实数据采集。
4.  **易于集成与部署**: 提供模块化的 Python 接口和 ABB RobotStudio 仿真接口，便于与现有机器人系统和工业产线快速集成与调试。
5.  **知识产权保护**: 核心算法与架构具有显著创新性，已申请专利保护，形成技术壁垒。

## 6. 附图简要说明 (Brief Description of Drawings)

本发明将通过以下附图进一步说明：

- **图 1**: 系统总体架构示意图。
- **图 2**: AGEANet 网络结构示意图。
- **图 3**: 多模态融合反光抑制流程图。
- **图 4**: 亚像素定位与位姿解算流程图。
- **图 5**: EdgeVision-C 引擎架构图。
- **图 6**: 全国复杂场景合成数据集生成流程与示例图。

## 7. 具体实施方式 (Detailed Description of Embodiments)

（此处将详细描述每个发明点的具体实现细节、算法步骤、数学模型、代码实现框架等，例如 AGEANet 的损失函数设计、HDR 融合的具体算法、Zernike 矩的计算过程、EdgeVision-C 的内存池实现机制等。此部分内容将结合代码库中的具体实现进行阐述。）

---

*@copyright Licensed under the Apache License, Version 2.0. Commercial use please contact RussellCooper.*
