# 真实工业数据接入与混合训练操作手册
# Real-world Data Integration and Mixed Training Guide

本手册详细说明了如何将您收集的真实工业图像（汽车门板、铝材等）接入本系统的训练流程。

---

## 1. 准备标注数据 (LabelMe)

我们推荐使用 [LabelMe](https://github.com/wkentaro/labelme) 进行标注。

### 标注规范：
- **标签名称**：统一使用 `workpiece`。
- **标注类型**：多边形 (Polygon)。
- **标注要点**：
    - 沿工件真实边缘标注。
    - 即使边缘被高光遮挡，也要根据几何形状补全。
    - 排除阴影区域。

---

## 2. 转换标注为 Mask

使用提供的 `labelme_to_mask.py` 脚本将 JSON 转换为模型可读的二值图像。

### 操作步骤：
1. 将所有原图放入 `images/` 目录。
2. 将所有 LabelMe JSON 放入 `annotations/` 目录。
3. 运行转换脚本：
   ```bash
   python labelme_to_mask.py
   ```
   *注意：请在脚本中修改 `base_dataset_dir` 为您的实际路径。*

---

## 3. 开启“合成+真实”混合训练 (Mixed Training)

为了解决真实数据初期不足（例如只有 50-100 张）导致的过拟合，我们提供了混合训练模式。系统会自动按比例混合合成数据与真实数据。

### 运行命令示例：
```bash
python train.py \
    --real_img_dir /path/to/your/images \
    --real_mask_dir /path/to/your/masks \
    --real_ratio 0.5 \
    --syn_count 1000 \
    --epochs 100 \
    --batch_size 8
```

### 参数说明：
- `--real_img_dir`: 真实照片所在的文件夹。
- `--real_mask_dir`: 转换后的二值 Mask 文件夹。
- `--real_ratio`: 真实数据在训练中的占比（0.5 表示一半真实，一半合成）。
- `--syn_count`: 每轮生成的合成数据量。

---

## 4. 数据校验与可视化

在训练前，强烈建议使用 `inspect_dataset.py` 检查数据对齐情况。

### 校验一致性：
```python
from inspect_dataset import validate_dataset_consistency
validate_dataset_consistency('path/to/images', 'path/to/masks')
```

### 可视化对齐：
```python
from inspect_dataset import inspect_data_pair
inspect_data_pair('img1.jpg', 'mask1.png', 'result.png')
```
*这会生成一张对比图，绿色轮廓表示 Mask 覆盖的区域，方便肉眼确认标注是否偏移。*

---

## 5. 针对高反光的自动增强

系统在加载真实数据时，会自动应用以下针对性处理：
1. **CLAHE (对比度受限自适应直方图均衡化)**：自动平衡高光和阴影，增强边缘对比度。
2. **多尺度缩放**：自动将不同分辨率的工业相机照片调整为模型输入尺寸。
3. **动态边缘生成**：自动从 Mask 计算 Canny 边缘，用于边缘感知分支的监督学习。
4. **PBR 高光数据预训练**：建议使用 `data/synth_dataset_generator.py` 生成 PBR 模式（`pbr`/`pbr_sun`/`pbr_mixed`）合成数据预训练，让模型先学会高反光特征，再用真实数据微调。

---

## 6. 测量工具在真实数据上的应用

拍摄真实数据时，建议同步记录标定板图像，以便后续使用 `CaliperMeasurement` 和 `GapMeasurement` 工具进行几何尺寸测量验证：

| 测量工具 | 功能 | 适用场景 |
|----------|------|----------|
| `CaliperMeasurement` | 双平行边缘卡尺测量，支持任意方向搜索 | 测量工件宽度、边缘间距 |
| `GapMeasurement` | 多边缘间隙/节距/宽度测量 | 测量孔洞间距、槽宽、节距 |

使用方式：将标定图像输入 `main_pipeline.py`（`use_measurement=True`），系统会在定位后自动输出像素级和物理毫米级测量值。

---
**提示**：如果您的真实数据非常少（少于 20 张），建议先将 `--real_ratio` 设为 0.2，随着数据增加逐步提升到 0.8 或 1.0。
