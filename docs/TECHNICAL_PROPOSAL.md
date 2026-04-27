# 高反光船舶钢板视觉检测技术方案书

> 基于 AGEANet (Anti-Glare Edge-Aware Network) 项目 v1.0
> 借鉴论文：《大型船舶钢板面高反光工况2D视觉检测技术调研报告》

---

## 1. 技术难点与解决方案

### 1.1 技术难点

高反光钢板面视觉检测在船舶制造中面临诸多挑战：

- **镜面反射问题**：钢板表面高反射率导致传统视觉系统产生过度曝光、光晕和伪影，严重影响图像质量。
- **环境光干扰**：工业现场复杂的光照条件容易造成阴影、反光等，进一步增加检测难度。
- **表面特性差异**：不同镀层钢板反光率差异大，光照不均匀情况下检测准确率低。
- **堆积钢板边缘检测困难**：多层钢板堆叠情况下边缘特征不明显。
- **大尺寸检测挑战**：船舶钢板尺寸大，需要高视场和高分辨率成像系统。

### 1.2 解决方案

本项目 AGEANet 针对上述难点，采用多层次技术栈：

| 难点 | 解决方案 | 项目实现模块 |
|------|---------|------------|
| 镜面反射 | 偏振光技术 + HDR 融合 | `vision/hdr_processing.py`, `vision/polarization_advanced.py` |
| 环境光干扰 | 多角度照明模拟 + 自适应增强 | `vision/hdr_processing.py::AntiGlarePipeline` |
| 表面特性差异 | 生成式图像增强（Diffusion/GAN） | `vision/generative_enhancement.py` |
| 边缘检测困难 | 深度学习双分支输出（分割+边缘） | `vision/feature_extraction.py::FLARE` |
| 大尺寸挑战 | 亚像素定位 + 多尺度特征融合 | `vision/localization_and_calibration.py` |

---

## 2. 检测现状与方法论

### 2.1 项目定位

AGEANet 是一款专为**高反光金属工件**（船舶钢板、汽车门板钢、铝合金等）设计的 2D 视觉检测系统。核心目标是在 2D 相机限制下达到接近 3D 相机的测量/分拣精度，部署于 Renesas RZ/V2H + RA8P1 嵌入式平台。

### 2.2 方法论对比

本项目采用**混合范式**（Hybrid Paradigm）：

1. **传统图像处理层**：HDR 融合、偏振光抑制、形态学滤波、亚像素边缘定位
2. **深度学习层**：FLARE/FLARELite 网络（U-Net + CBAM + 可变形卷积）
3. **生成增强层**：条件扩散 / CGAN 修复高光区域纹理（借鉴生成模型范式革命）

---

## 3. 2D 视觉定位抓取算法

### 3.1 传统算法

- **边缘检测**：Canny、Sobel、HED/RCF 多尺度边缘头
- **特征匹配**：SSDA 模板匹配（规避专利风险）
- **形态学操作**：腐蚀、膨胀、开闭运算处理缺陷识别

### 3.2 深度学习算法

- **FLARE 网络**：基于 U-Net 的编码器-解码器结构
  - `CoordConv`：坐标感知卷积
  - `CBAM`：通道+空间注意力（抑制高光虚假激活）
  - `GlareGatedSkip`：高光感知跳跃连接
  - `DeformConv2d`：可变形卷积适配不规则边缘
  - `GhostConv`：轻量化幽灵卷积（参数量减半 ~50%）
- **双分支输出头**：
  - 分割头：语义分割掩膜
  - 边缘头：HED/RCF 多尺度边缘检测
  - `EdgeRefinementHead`：分割梯度指导的边缘细化
- **ResNet 骨干支持**（v1.0 新增）：
  - 可选 ResNet-18/34/50 替代 U-Net 编码器
  - ImageNet 预训练权重加速收敛

### 3.3 生成模型增强（v1.0 新增）

借鉴《视觉的范式革命》论文中"生成即理解"的思想：

- **GlareInpaintGAN**：轻量级 CGAN，单步推理修复高光区域（~10ms）
- **DiffusionGlareRemover**：条件扩散模型，50 步迭代生成真实金属纹理
- **应用场景**：作为 HDR 管线的增强步骤，提升反光区域缺陷检出率

---

## 4. 边缘检测与定位算法实现

### 4.1 边缘检测

```
输入图像 → HDR 融合 → 偏振/高光修复 → FLARE 推理 → 双分支输出
                                                    ├── 分割掩膜 (seg)
                                                    └── 边缘掩膜 (edge)
```

- **阈值分割**：自适应阈值 + Otsu
- **边缘跟踪**：轮廓提取 + 多边形近似
- **亚像素细化**：灰度矩 + 梯度插值

### 4.2 缺陷定位算法

- **SubpixelLocalizer**：
  - 灰度加权矩质心估计
  - PCA 主轴方向分析
  - 高光区域掩膜排除
  - 支持 blob / line / region 三种特征类型
- **CoordinateTransformer**：
  - 像素 → 相机坐标系 → 机器人基坐标系变换
- **HandEyeCalibrator**：
  - PnP + RANSAC 手眼标定（规避 AX=XB 专利）

### 4.3 精度指标

| 指标 | 数值 |
|------|------|
| 定位精度 | < 0.5 mm |
| 分割 IoU | 0.40+ |
| 边缘 F1 | 持续优化中（核心攻坚目标） |
| 推理延迟（TensorRT FP16）| ~10 ms @ 512×512 |

---

## 5. 抓取规划与路径优化

### 5.1 抓取规划

- **抓取点选择**：基于边缘特征确定最佳抓取位置
- **RRT* 路径规划**：快速扩展随机树最优路径
- **碰撞避免**：结合深度信息实时调整路径

### 5.2 机器人接口

| 类 | 模式 | 说明 |
|----|------|------|
| `AbbRobotStub` | 纯 Python 模拟桩 | 无需外部软件，开发测试用 |
| `AbbRobotStudioSim` | TCP Socket | ABB RobotStudio 仿真（端口 10000）|
| `AbbRobotEGM` | UDP EGM | 真实 ABB 机器人（端口 6510）|
| `MultiRobotCoordinator` | 多机协调 | 任务分配、碰撞避免 |

### 5.3 C# WPF 上位机（v1.0 新增）

- **技术栈**：WPF (.NET 8) + MVVM + FastAPI 通信
- **功能模块**：
  - 图像检测：单张上传 → 四宫格可视化
  - 批量处理：文件夹批量推理 → CSV 导出
  - 参数配置：HDR/阈值/模型后端实时调优
  - 模型管理：PyTorch / ONNX / TensorRT 切换
  - 机器人控制：坐标显示 / 标定 / 目标发送

---

## 6. 代码实现与开源方案

### 6.1 项目架构

```
Anti-Interference-2D-Vision/
├── api/                    # FastAPI 后端服务（供 C# GUI 调用）
├── core/                   # 统一配置、日志、指标监控
├── data/                   # 数据加载与增强
│   ├── synth_dataset_generator.py   # PBR 合成数据
│   ├── neu_dataset.py               # NEU 数据集适配（v1.0 新增）
│   └── gc10_dataset.py              # GC10-DET 适配（v1.0 新增）
├── vision/                 # 视觉算法核心
│   ├── feature_extraction.py        # FLARE/ResNet 网络
│   ├── hdr_processing.py            # HDR + 反光抑制
│   ├── polarization_advanced.py     # 物理级偏振模拟（v1.0 新增）
│   ├── generative_enhancement.py    # 生成式修复（v1.0 新增）
│   ├── localization_and_calibration.py
│   ├── appearance_detection.py
│   ├── measurement.py
│   └── inference_engine.py          # PyTorch/ONNX/TensorRT
├── robot/                  # ABB 机器人接口
├── training/               # 训练与评估
├── gui/                    # PyQt6 原生桌面 GUI
└── tests/                  # 测试套件（207 passed）

AntiInterference2D.GUI/     # C# WPF 上位机（v1.0 新增）
├── Views/
├── ViewModels/
├── Services/
└── Models/
```

### 6.2 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习框架 | PyTorch ≥2.0, torchvision ≥0.15 |
| 计算机视觉 | OpenCV ≥4.5, NumPy, SciPy |
| 模型导出 | ONNX ≥1.14, TensorRT, TorchScript |
| 后端服务 | FastAPI, Uvicorn |
| 上位机 | WPF (.NET 8), CommunityToolkit.Mvvm |
| 数据格式 | NEU (BMP), GC10-DET (YOLO txt), 自定义合成 |

---

## 7. 技术参数与系统配置

### 7.1 关键技术参数

| 参数 | 数值 |
|------|------|
| 检测精度 | 0.1–0.5 mm 定位精度 |
| 处理速度 | TensorRT FP16: ~10 ms/帧（512×512）|
| 图像分辨率 | 支持 256–2048 自适应 |
| 检测范围 | 覆盖 1–12 m 宽钢板（多相机协同）|

### 7.2 系统配置要点

- **照明系统**：LED 条形光源或环形光源，色温 5500K–6500K
- **相机选择**：工业面阵相机，12–20MP，根据检测精度要求选择
- **镜头配置**：根据检测距离选择合适焦距，考虑畸变控制
- **处理单元**：
  - 服务器端：NVIDIA GPU + TensorRT
  - 嵌入式端：Renesas RZ/V2H (DRP-AI) + RA8P1 Cortex-M85

### 7.3 部署流程

```bash
# 1. 启动 FastAPI 后端
uvicorn api.server:app --host 0.0.0.0 --port 8000

# 2. 启动 C# WPF 上位机（Visual Studio 或 dotnet run）
cd AntiInterference2D.GUI
dotnet run

# 3. 在 WPF 界面中配置服务器地址（默认 localhost:8000）
```

---

## 8. 未来演进方向

1. **多模态融合**：结合红外热成像、激光扫描数据（论文提及）
2. **生成式数据增强**：利用 Diffusion 合成多样化缺陷样本
3. **边缘 F1 攻坚**：当前核心瓶颈，计划引入 Transformer-based 边缘头
4. **3D 视觉扩展**：从 2D 检测向 3D 点云检测演进
5. **云端协同**：远程模型更新、分布式检测调度
