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
| **伪边缘剔除** | 基于梯度一致性分析，剔除高光反射产生的伪边缘 |
| **嵌入式部署** | 支持 INT8 量化导出到 TFLite，适配 RA8P1 Cortex-M85 |
| **合成数据训练** | 内置高反光金属工件合成数据生成器，支持冷启动训练 |

---

## 系统架构

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  CMOS 传感器  │───▶│  ISP 处理     │───▶│  HDR 融合     │───▶│  反光抑制     │
│  (多重曝光)   │    │  (RZ/V2H)    │    │  (Mertens)   │    │  (高光修复)   │
└─────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                  │
                                                                  ▼
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  机器人抓取   │◀───│  坐标转换     │◀───│  亚像素定位   │◀───│  AGEANet     │
│  (位姿输出)   │    │  (手眼标定)   │    │  (Zernike)   │    │  (分割+边缘)  │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

---

## 项目结构

```
├── feature_extraction.py          # AGEANet / AGEANet-Lite 模型架构
├── train.py                       # 完整训练流程 (损失函数/数据加载/训练循环)
├── data_augmentation.py           # 高反光专用数据增强与合成数据生成
├── hdr_processing.py              # HDR 融合与反光抑制管线
├── simple_isp_simulator.py        # RZ/V2H ISP 模拟器
├── localization_and_calibration.py # 亚像素定位与手眼标定
├── main_system.py                 # 系统主控与推理管线
├── algorithm_verification.py      # 算法验证与评估
├── ra8p1_tflm_adapter.py          # 嵌入式部署导出工具
├── ra8p1_helium_processing.c      # RA8P1 Helium SIMD 加速 (C)
├── ra8p1_main_app.c               # RA8P1 嵌入式主应用 (C)
├── requirements.txt               # Python 依赖
├── high_reflectivity_sorting_solution.md  # 技术方案文档
├── embedded_pure_c_ai_vision_system_zh.md  # 纯C嵌入式视觉系统架构与实践
└── hardware_list_zh.md            # 硬件清单
```

---

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 使用合成数据快速训练

```bash
# 纯合成数据冷启动训练 (无需真实数据)
python train.py --synthetic_only --syn_count 2000 --epochs 50 --model lite

# 使用 AGEANet 标准模型
python train.py --synthetic_only --syn_count 5000 --epochs 100 --model standard
```

### 3. 使用真实数据训练

准备数据集目录：

```
datasets/metal_workpieces/
├── images/     # 原始图像 (.png, .jpg, .bmp)
├── masks/      # 分割掩膜 (同名，单通道，0=背景 255=工件)
└── edges/      # 边缘掩膜 (可选，自动生成)
```

```bash
# 真实数据 + 合成数据混合训练
python train.py --data_dir ./datasets/metal_workpieces --syn_count 1000 --epochs 100

# 仅真实数据
python train.py --data_dir ./datasets/metal_workpieces --epochs 100
```

### 4. 推理

```python
from main_system import HighReflectiveSortingSystem

# 初始化系统
system = HighReflectiveSortingSystem(
    model_path='./checkpoints/best_model.pth',
    model_type='lite',
)

# 处理图像
import cv2
image = cv2.imread('test_image.png')
result = system.process_single_image(image)

print(f"检测到 {result['num_workpieces']} 个工件")
for pose in result['grasp_poses']:
    print(f"  位置: {pose['position']}, 角度: {pose['angle']:.1f}°")
```

### 5. 嵌入式部署

```bash
# 导出 ONNX + TFLite (INT8 量化)
python ra8p1_tflm_adapter.py --export --model-path ./checkpoints/best_model.pth

# 生成 C 头文件
python ra8p1_tflm_adapter.py --generate-header ./export/ageanet_lite.tflite
```

### 6. 算法验证

```bash
# 完整验证 (合成场景 + 精度评估 + 速度测试)
python algorithm_verification.py --output ./verification_results

# 使用训练好的模型验证
python algorithm_verification.py --model ./checkpoints/best_model.pth
```

---

## 数据集准备指南

### 推荐公开数据集

| 数据集 | 类型 | 适用场景 | 来源 |
|--------|------|----------|------|
| **NEU Surface Defect** | 钢表面 | 热轧钢板边缘检测 | [Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) |
| **Aluminum Profile Defects** | 铝型材 | 铝合金表面分割 | [Kaggle](https://www.kaggle.com/datasets/weihaoreal/aluminum-profile-surface-defects-data-set) |
| **MVTec AD (Metal Nut/Screw)** | 金属零件 | 金属工件分割 | [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **DIMO** | 工业金属 | 反光金属物体检测 | [DIMO](https://pderoovere.github.io/dimo) |
| **ROBI** | 反光物体 | 高反光物体位姿估计 | [ROBI](https://www.trailab.utias.utoronto.ca/robi) |
| **OSPD** | 汽车冲压件 | 汽车门板钢检测 | [arXiv:2403.10369](https://arxiv.org/abs/2403.10369) |

### 自采数据建议

1. **拍摄环境**：模拟实际产线光照条件，包含不同角度的光源
2. **多重曝光**：每个场景拍摄 3-5 张不同曝光的图像
3. **标注工具**：推荐使用 [LabelMe](https://github.com/wkentaro/labelme) 进行多边形标注
4. **标注规范**：
   - 沿工件真实边缘精确标注，忽略反光区域的伪边缘
   - 掩膜保存为单通道 PNG (0=背景, 255=工件)
   - 文件名与原图一致

---

## 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `standard` | 模型类型: `standard` (AGEANet, ~4.1M参数) 或 `lite` (AGEANet-Lite, ~1.0M参数) |
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 8 | 批次大小 |
| `--lr` | 1e-3 | 初始学习率 (OneCycleLR 调度) |
| `--img_size` | 256 | 训练图像尺寸 |
| `--dice_w` | 1.0 | Dice Loss 权重 |
| `--focal_w` | 1.0 | Focal Loss 权重 |
| `--boundary_w` | 0.5 | Boundary Loss 权重 |
| `--edge_w` | 0.5 | 边缘 Loss 权重 |
| `--specular_w` | 0.3 | 高光一致性 Loss 权重 |
| `--patience` | 20 | 早停耐心值 |

---

## 模型架构

### AGEANet (标准版)

- **编码器**：4 层，通道数 [64, 128, 256, 512]，每层含 CBAM 注意力
- **解码器**：4 层，跳跃连接 + 上采样
- **高光抑制前端**：检测并抑制高光区域的特征响应
- **边缘感知分支**：独立的边缘检测分支，输出边缘概率图
- **参数量**：约 4.1M

### AGEANet-Lite (轻量版)

- **编码器**：3 层，通道数 [32, 64, 128]，深度可分离卷积
- **解码器**：3 层
- **参数量**：约 1.0M
- **适用**：RA8P1 嵌入式部署

---

## 损失函数

```
L_total = λ_dice * L_dice + λ_focal * L_focal + λ_boundary * L_boundary
        + λ_edge * L_edge + λ_specular * L_specular
```

- **Dice Loss**：处理类别不平衡
- **Focal Loss**：聚焦难分类的边缘像素
- **Boundary Loss**：基于 Sobel 梯度的边缘距离损失
- **Edge Loss**：边缘分支的 BCE 损失
- **Specular Consistency Loss**：高光区域分割一致性损失

---

## 硬件平台

| 组件 | 型号 | 用途 |
|------|------|------|
| 主处理器 | Renesas RZ/V2H | DRP-AI 加速推理 + ISP |
| 协处理器 | Renesas RA8P1 (Cortex-M85) | Helium SIMD 图像预处理 |
| 相机 | CMOS 工业相机 | 多重曝光采集 |
| 机器人 | 6-DOF 工业机器人 | 抓取执行 |

---

## 详细文档

- [技术方案文档 (中文)](./high_reflectivity_sorting_solution.md)
- [硬件清单](./hardware_list_zh.md)

---

## 许可证

MIT License
