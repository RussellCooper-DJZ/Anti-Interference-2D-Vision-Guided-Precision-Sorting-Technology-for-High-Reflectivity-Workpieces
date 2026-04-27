# 数据集准备指南

## 1. 数据集目录结构

训练系统要求数据集按以下结构组织：

```
datasets/
├── metal_workpieces/          # 主数据集
│   ├── images/                # 原始图像 (.png, .jpg, .bmp)
│   ├── masks/                 # 分割掩膜 (同名文件, 单通道, 0/255)
│   └── edges/                 # 边缘掩膜 (可选, 自动从 mask 生成)
├── steel_panels/              # 钢板数据 (可选)
│   ├── images/
│   └── masks/
└── aluminum_profiles/         # 铝型材数据 (可选)
    ├── images/
    └── masks/
```

**关键要求：**
- `images/` 和 `masks/` 中的文件必须同名（扩展名可不同）
- 掩膜为单通道图像：工件区域 = 255（白色），背景 = 0（黑色）
- 如果没有 `edges/` 目录，系统会自动从掩膜生成边缘标注

## 2. 推荐公开数据集

### 2.1 钢板/钢材类

| 数据集 | 描述 | 规模 | 获取方式 |
|--------|------|------|----------|
| **NEU Surface Defect Database** | 东北大学钢表面缺陷数据集，6类缺陷各300张 | 1,800 张 | [Kaggle](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) |
| **Severstal Steel Defect** | Kaggle 竞赛钢板缺陷分割数据集 | 12,568 张 | [Kaggle](https://www.kaggle.com/c/severstal-steel-defect-detection) |
| **OSPD** | 汽车冲压件开放数据集 | 1,500+ 张 | [GitHub](https://github.com/2Obe/OSPD) |

### 2.2 铝合金类

| 数据集 | 描述 | 规模 | 获取方式 |
|--------|------|------|----------|
| **铝型材表面缺陷** | 天池竞赛铝型材缺陷检测数据集 | 10,000+ 张 | [Kaggle](https://www.kaggle.com/datasets/weihaoreal/aluminum-profile-surface-defects-data-set) |

### 2.3 反光金属物体

| 数据集 | 描述 | 规模 | 获取方式 |
|--------|------|------|----------|
| **MVTec AD (Metal Nut/Screw/Grid)** | 工业异常检测基准，含金属类别 | 各类 ~400 张 | [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad) |
| **DIMO** | 工业金属物体数据集，含 6DoF 标注 | 多类别 | [GitHub](https://pderoovere.github.io/dimo) |
| **ROBI** | 反光物体数据集，含分割标注 | 多类别 | [ROBI](https://www.trailab.utias.utoronto.ca/robi) |

## 3. 自采数据建议

### 3.1 拍摄环境

- **光源**：使用工业环形光源或条形光源，模拟实际产线光照
- **相机**：与部署相同型号（如 Renesas RZ/V2H + CMOS 传感器）
- **角度**：固定俯拍角度，与实际部署一致
- **背景**：使用深色哑光传送带或工作台

### 3.2 拍摄策略

1. **多曝光拍摄**：每个场景拍摄 3 张不同曝光（欠曝/正常/过曝）
2. **多角度反光**：旋转工件或调整光源角度，覆盖不同反光模式
3. **多工件组合**：单个工件、多个工件、重叠工件
4. **干扰因素**：油污、灰尘、划痕、不同材质混合

### 3.3 标注工具

推荐使用以下工具进行分割标注：

- **[LabelMe](https://github.com/wkentaro/labelme)**：多边形标注，导出 JSON
- **[CVAT](https://github.com/opencv/cvat)**：在线标注平台，支持团队协作
- **[Labelimg](https://github.com/HumanSignal/labelImg)**：矩形框标注（目标检测）

### 3.4 标注转换脚本

LabelMe JSON → 二值掩膜：

```python
import json
import numpy as np
import cv2

def labelme_to_mask(json_path, output_path):
    with open(json_path) as f:
        data = json.load(f)
    
    h, w = data['imageHeight'], data['imageWidth']
    mask = np.zeros((h, w), dtype=np.uint8)
    
    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    
    cv2.imwrite(output_path, mask)
```

## 4. 训练命令

### 4.1 使用真实数据训练

```bash
python train.py \
    --data_dir ./datasets/metal_workpieces \
    --model standard \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --save_dir ./checkpoints
```

### 4.2 合成数据冷启动 + 真实数据微调

```bash
# 第一阶段：合成数据预训练
python train.py \
    --synthetic_only \
    --syn_count 2000 \
    --model standard \
    --epochs 30 \
    --save_dir ./checkpoints/pretrain

# 第二阶段：真实数据微调
python train.py \
    --data_dir ./datasets/metal_workpieces \
    --resume ./checkpoints/pretrain/best_model.pth \
    --model standard \
    --epochs 100 \
    --lr 1e-4 \
    --save_dir ./checkpoints/finetune
```

### 4.3 混合训练（推荐）

```bash
python train.py \
    --data_dir ./datasets/metal_workpieces \
    --syn_count 1000 \
    --model standard \
    --epochs 100 \
    --batch_size 8 \
    --save_dir ./checkpoints
```

### 4.4 嵌入式轻量模型训练

```bash
python train.py \
    --data_dir ./datasets/metal_workpieces \
    --model lite \
    --base_ch 32 \
    --img_size 256 \
    --epochs 100 \
    --save_dir ./checkpoints/lite
```

## 5. PBR 高光合成数据生成

### 5.1 光照模式

合成数据生成器支持以下光照模式，可通过 `--lighting_mode` 参数指定：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `default` | 基础环境光 | 通用 |
| `industrial` | 工业光源（顶光+侧光） | 产线环境 |
| `spotlight` | 单点聚光 | 高光集中 |
| `water_reflect` | 水面反射 | 潮湿环境 |
| `dual` | 双光源 | 复杂反光 |
| `pbr` | **Blinn-Phong BRDF PBR** | 高反光金属（汽车门板钢/铝合金） |
| `pbr_sun` | **PBR + 太阳光** | 户外/大场面 |
| `pbr_mixed` | **PBR + 水面反射** | 潮湿高反光工件 |

### 5.2 PBR 物理参数

| 参数 | 说明 | 取值范围 |
|------|------|----------|
| `roughness` | 表面粗糙度 | 0.01（镜面）~ 1.0（漫反射） |
| `metallic` | 金属度 | 0.0（非金属）~ 1.0（纯金属） |
| `light_angle_deg` | 光源水平角度 | 0° ~ 360° |
| `light_elevation_deg` | 光源仰角 | 0° ~ 90° |
| `specular_scale` | 高光强度缩放 | 0.0 ~ 2.0 |

PBR 渲染使用 Blinn-Phong BRDF 模型：
- **D 项**（Blinn-Phong 微平面分布）：控制高光锐度
- **F 项**（Schlick Fresnel 近似）：掠射时增强反射
- **G 项**（Smith 几何遮蔽）：自遮挡衰减

### 5.3 PBR 合成数据命令示例

```bash
# 生成高反光金属工件合成数据（500张）
python data/synth_dataset_generator.py \
    --count 500 \
    --output ./dataset_pbr \
    --lighting_mode pbr \
    --specular_scale 1.5

# 生成混合光照（高反光 + 水面反射）
python data/synth_dataset_generator.py \
    --count 300 \
    --output ./dataset_pbr_mixed \
    --lighting_mode pbr_mixed \
    --specular_scale 1.2

# 生成 PBR + 太阳光户外场景
python data/synth_dataset_generator.py \
    --count 200 \
    --output ./dataset_pbr_sun \
    --lighting_mode pbr_sun
```

### 5.4 PBR 数据集用于训练

```bash
# PBR 合成数据预训练
python train.py \
    --synthetic_only \
    --syn_dir ./dataset_pbr \
    --epochs 50 \
    --batch_size 8 \
    --save_dir ./checkpoints/pbr_pretrain

# 真实数据微调
python train.py \
    --data_dir ./datasets/metal_workpieces \
    --resume ./checkpoints/pbr_pretrain/best_model.pth \
    --epochs 100 \
    --lr 1e-4 \
    --save_dir ./checkpoints/finetune
```

## 6. 数据量建议

| 场景 | 最低数据量 | 推荐数据量 | 说明 |
|------|-----------|-----------|------|
| 概念验证 | 50 张 | 200 张 | 配合合成数据 |
| 基本可用 | 200 张 | 500 张 | 单一工件类型 |
| 生产部署 | 500 张 | 2000+ 张 | 多工件、多场景 |
| 高精度要求 | 1000 张 | 5000+ 张 | 覆盖所有干扰因素 |

> **提示**：即使真实数据较少（50-200张），配合合成数据增强也能取得不错的效果。
> 建议先用合成数据预训练，再用少量真实数据微调。
