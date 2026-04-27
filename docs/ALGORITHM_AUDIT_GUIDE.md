# 算法引用正确性审计指南（Algorithm Audit Guide）

> **目标**：在每次迭代（Plan→Work→Review→Compound）的 `Work` 阶段结束后、`Review` 阶段开始前，对算法实现的命名、公式、行为进行标准化审查，确保"声称的算法 = 实现的算法"。

---

## 一、审计触发条件

以下任一情况发生时，必须执行本审计：

1. **新算法模块**：work 文件引入了一个新的算法名称（如 Lovász Loss、SSDA、AutoAugment 等）
2. **算法改进版**：声称是某算法的 V2 / Lite / 改进版（如 ROICorrectorV2、SubpixelLocalizerV2）
3. **论文复现**：docstring 或 plan 中引用了具体论文（作者、年份、会议）
4. **性能对比**：声称比 baseline 快/准/省，需验证实现是否支撑该结论

---

## 二、三维审计框架

对每一个声称的算法，从三个维度进行审查：

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   命名维度 (A)   │ ×  │   公式维度 (B)   │ ×  │   行为维度 (C)   │
│  Name Accuracy  │    │ Formula Accuracy │    │Behavior Accuracy│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

三个维度**全部通过**才算审计合格。任一维度失败，进入"问题分级→修复→复验"流程。

---

## 三、维度 A：命名准确性（Name Accuracy）

### 3.1 核心问题
> **代码声称自己是 X，但它真的是 X 吗？**

### 3.2 检查清单（按算法类别）

#### A. 损失函数类
| 算法名 | 必须满足的条件 | 常见陷阱 |
|--------|---------------|---------|
| **Lovász hinge** | 使用 hinge error `max(0, 1-yf)`；按误差降序排序；累积计算 IoU 变化率；最终为 `Σ errors·Δ(1-IoU)` | 用 sigmoid+L1 替代 hinge；不对多维张量正确展平；union 公式写错 |
| **OHEM** | 在整个 batch 的所有像素上取 loss 最高的前 k 个；**不**先设阈值筛选 | 先 `bce > threshold` 再 topk；只在子集上操作 |
| **Focal Loss** | 必须有 `(1-p)^γ` 调制因子和 `α` 权重 | 缺少 γ 或 α；对 logits 而非 probs 计算 |
| **Dice Loss** | 交集/并集计算需包含 smooth 项；通常对 probs 计算 | 忘记 smooth；对 logits 直接计算 |

#### B. 数据增强类
| 算法名 | 必须满足的条件 | 常见陷阱 |
|--------|---------------|---------|
| **CutMix** | 图像：用 λ 确定矩形区域，从另一张图复制；标签：分类任务用 λ 混合 one-hot；**分割任务直接复制 patch 区域标签** | 分割 mask 用 `torch.max` 或 λ 混合（会破坏类别语义） |
| **Mosaic** | 4 张图拼接成 1 张；每张图随机缩放后放在 4 个象限 | 固定缩放比例；不随机选择子图 |
| **AutoAugment** | 必须包含：① 预定义的操作空间 ② **策略学习机制**（RL/PPO/网格搜索）③ 策略 = {操作, 概率, 幅度} | 仅随机采样参数组合，无策略学习过程 |
| **MixUp** | 图像和标签都按 λ 线性插值 | 只混合图像不混合标签 |

#### C. 注意力/卷积类
| 算法名 | 必须满足的条件 | 常见陷阱 |
|--------|---------------|---------|
| **CBAM** | 通道注意力（GAP+GAP²→MLP）+ 空间注意力（GAP+Max→7×7 conv） | 缺少其中一个分支；reduction 比率错误 |
| **SE-Net** | Squeeze（GAP）+ Excitation（FC→ReLU→FC→Sigmoid） | 缺少 reduction；用 conv 替代 FC |
| **DCNv2** | 必须学习 offset 和 mask；用 unfold/grid_sample 实现可变形采样 | 只有 offset 没有 mask；用标准卷积替代采样 |
| **CoordConv** | 输入特征拼接 2 个坐标通道（y, x） | 坐标归一化方式错误 |

#### D. 模板匹配/跟踪类
| 算法名 | 必须满足的条件 | 常见陷阱 |
|--------|---------------|---------|
| **SSDA** | ① 随机采样模板像素子集 ② 逐像素累积误差 ③ **当累积误差 > 当前最优时提前终止** | 只是随机打乱搜索顺序的 SAD；计算完整 diff.sum() |
| **NCC** | 归一化互相关：(template-mean)·(image-mean) / (σ_t·σ_i) | 忘记减均值或除标准差 |

#### E. 可解释性/可视化类
| 算法名 | 必须满足的条件 | 常见陷阱 |
|--------|---------------|---------|
| **Grad-CAM** | 目标必须是**特定类别的分数**（`logits[class]`），对之求梯度 | `logits.sum()` 作为目标（无类别区分性） |

#### F. 推理优化类
| 算法名 | 必须满足的条件 | 常见陷阱 |
|--------|---------------|---------|
| **Async Inference** | `submit()` 启动 kernel 后立即返回，**不在内部 synchronize**；同步推迟到 `get()` | `submit()` 内部调用 `stream.synchronize()` 或 `torch.cuda.synchronize()` |
| **TensorRT/ONNX** | 必须有实际的转换/序列化/推理代码，而非仅封装 PyTorch forward | 只是 PyTorch forward 的 wrapper |

### 3.3 快速决策树

```
work 文件中是否出现了知名算法名称？
    └─ 否 → 命名维度通过
    └─ 是 → 该算法是否有公认的"最小充分条件"？（见上表）
        └─ 否 → 标记为 "未验证的命名"，要求补充文档说明
        └─ 是 → 检查代码是否满足所有充分条件
            └─ 满足 → 通过
            └─ 不满足 → 记录为 "命名不匹配"，进入修复流程
```

---

## 四、维度 B：公式准确性（Formula Accuracy）

### 4.1 核心问题
> **论文里的公式，代码写对了吗？**

### 4.2 审查步骤

**Step 1 — 定位公式来源**
- 在 docstring / comment / plan.md 中找到论文引用
- 如果**没有任何引用**，标记为 "未标注来源"（非致命，但建议补充）

**Step 2 — 建立公式对照表**

| 检查项 | 方法 | 工具/来源 |
|--------|------|----------|
| 论文原文 | 搜索论文 PDF 或 arXiv | Google Scholar / arXiv / OpenReview |
| 官方/高星实现 | 对照 PyTorch 社区实现 | GitHub search `pytorch <algorithm_name>` |
| 公式推导验证 | 用 sympy 或手动推导关键步骤 | `sympy` / 纸笔 |

**Step 3 — 逐项核对**

对公式中的每一项，检查代码是否一致：

```python
# 示例：Lovász hinge loss 核对
# 论文公式:  L = Σ_i errors_i · (IoU_i - IoU_{i-1})
#            errors_i = max(0, 1 - y_i · f_i)
# 代码应体现:
#   1. hinge error 而非 L1/sigmoid
#   2. 按 errors 降序排序
#   3. 累积 intersection / union
#   4. 加权和的形式
```

### 4.3 常见公式错误模式

| 错误模式 | 示例 | 后果 |
|----------|------|------|
| 用近似替代精确 | sigmoid + L1 替代 hinge | 损失函数性质改变，非凸、无理论保证 |
| 维度处理错误 | 一维 sort_idx 直接索引多维张量 | RuntimeError 或 silently wrong |
| 归一化遗漏 | 忘记除标准差 / 减均值 | 数值不稳定，输出分布偏移 |
| 常数错误 | `2π` 写成 `π`；`log` 底数错误 | 频率/尺度计算错误 |

---

## 五、维度 C：行为准确性（Behavior Accuracy）

### 5.1 核心问题
> **代码运行起来，行为是否符合算法定义？**

### 5.2 必做测试清单

#### 测试 1：前向传播通过性（Forward Pass）
```python
# 最小测试模板
model = MyAlgorithmModule(...)
x = torch.randn(B, C, H, W)
try:
    out = model(x)
    print(f"[PASS] output shape: {out.shape}")
except Exception as e:
    print(f"[FAIL] {e}")
```
- **必须测试**：任何 `nn.Module` 子类
- **常见失败**：shape mismatch、类型错误（如 `torch.cos(float)`）、未定义的变量

#### 测试 2：语义行为测试（Semantic Test）

| 算法 | 语义测试方法 | 预期结果 |
|------|-------------|---------|
| **SSDA** | 对比完整 SAD 和 SSDA 的输出位置 | 两者找到的 best_pos 应相同或相近；SSDA 计算量更少 |
| **CutMix** | 检查 mask 的 patch 区域是否完全来自同一张图 | `out_masks[i, :, y1:y2, x1:x2]` 应等于 `masks[idx[i], :, y1:y2, x1:x2]` |
| **OHEM** | 构造一个已知 hard example 的 batch | 损失应主要由 hard pixel 贡献 |
| **Async** | 测量 `submit()` 的耗时 | 应 < GPU 推理时间；如果 ≈ 推理时间，说明有同步点 |
| **Grad-CAM** | 对比 `target_mask=None` 和 `target_mask=全1` | 两者不应产生完全相同的梯度模式 |

#### 测试 3：数值不变性测试（Invariance Test）

```python
# 示例：测试平移不变性
out1 = model(x)
out2 = model(torch.roll(x, shifts=(2, 3), dims=(2, 3)))
# 若算法声称平移不变，out1 和 out2 应在允许误差内匹配
```

#### 测试 4：边界条件测试（Edge Case）

| 边界条件 | 测试方法 |
|----------|----------|
| 空张量 / 全零输入 | `x = torch.zeros(...)` |
| 极大/极小值 | `x = torch.full(..., 1e6)` |
| 单一像素 | `H=W=1` |
| batch_size=1 | 验证没有硬编码 batch dim |

---

## 六、审计执行流程（SOP）

```
迭代 Work 文件完成后
        │
        ▼
┌───────────────────┐
│ Step 1: 快速扫描   │  5 min
│ 检查 docstring 中  │  是否出现知名算法名称？
│ 是否有算法声称     │  是否有论文引用？
└───────────────────┘
        │
    有算法声称?
    ├─ 否 ──→ 审计通过，进入 Review 阶段
    └─ 是
        ▼
┌───────────────────┐
│ Step 2: 维度 A 审查│  10 min
│ 对照"检查清单"     │  命名是否匹配最小充分条件？
└───────────────────┘
        │
    命名通过?
    ├─ 否 ──→ 标记问题，进入修复流程
    └─ 是
        ▼
┌───────────────────┐
│ Step 3: 维度 B 审查│  15 min
│ 查找论文/参考实现   │  公式是否正确实现？
│ 核对关键数学步骤   │
└───────────────────┘
        │
    公式通过?
    ├─ 否 ──→ 标记问题，进入修复流程
    └─ 是
        ▼
┌───────────────────┐
│ Step 4: 维度 C 审查│  10 min
│ 运行前向+语义测试  │  行为是否符合预期？
└───────────────────┘
        │
    行为通过?
    ├─ 否 ──→ 标记问题，进入修复流程
    └─ 是
        ▼
    审计通过，更新 review.md
```

---

## 七、问题分级与修复策略

| 级别 | 定义 | 示例 | 修复策略 |
|------|------|------|---------|
| **P0 - 致命** | 实现完全不是声称的算法，会误导用户和审稿人 | Lovász Loss 公式完全错误；SSDA 没有 early termination | **必须修复**：重写核心逻辑，或改名并更新文档 |
| **P1 - 严重** | 算法核心机制缺失/错误，但框架正确 | OHEM 先阈值再 topk；CutMix mask 用 torch.max | **必须修复**：修正核心机制 |
| **P2 - 中等** | 实现正确但非最优，或文档不准确 | AsyncInferencer 内部 synchronize；AutoAugment 只是随机搜索 | **建议修复**：调整实现或明确标注"简化版" |
| **P3 - 轻微** | 命名/文档不清晰，但实现无大误 | 类名过于宽泛；docstring 缺少论文引用 | **可选修复**：补充文档、加引用 |

---

## 八、自动化审计脚本模板

将以下模板保存为 `scripts/audit_algorithm.py`，每次迭代运行：

```python
#!/usr/bin/env python3
"""
算法审计快速检查脚本。
用法: python scripts/audit_algorithm.py <iteration_number>
"""
import sys, importlib, inspect, ast, pathlib

ITERATION = int(sys.argv[1])
WORK_DIR = pathlib.Path(f"results/auto_tuning/iteration_{ITERATION}")

def find_algorithm_claims(file_path):
    """从 docstring / 类名中提取声称的算法。"""
    text = file_path.read_text(encoding='utf-8')
    tree = ast.parse(text)
    claims = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            claims.append(('class', node.name))
        if isinstance(node, ast.FunctionDef):
            claims.append(('func', node.name))
    return claims

def check_forward_pass(module_path, class_name):
    """测试 nn.Module 是否能前向传播。"""
    try:
        spec = importlib.util.spec_from_file_location("work", module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = getattr(mod, class_name)
        if not inspect.isclass(cls):
            return "SKIP"
        obj = cls()
        if hasattr(obj, 'forward'):
            import torch
            x = torch.randn(1, 3, 64, 64)
            out = obj(x)
            return f"OK shape={out.shape if hasattr(out, 'shape') else type(out)}"
        return "SKIP"
    except Exception as e:
        return f"FAIL: {e}"

# 主流程
for py_file in WORK_DIR.glob("work_*.py"):
    print(f"\n=== {py_file.name} ===")
    claims = find_algorithm_claims(py_file)
    for kind, name in claims:
        print(f"  [{kind}] {name}")
    # 对 nn.Module 子类做前向测试
    for kind, name in claims:
        if kind == 'class':
            result = check_forward_pass(py_file, name)
            print(f"    forward: {result}")
```

---

## 九、常见陷阱速查表（我们踩过的坑）

| # | 陷阱 | 出现位置 | 检测方法 |
|---|------|---------|---------|
| 1 | `np` 未导入就用 `np.pi` | `feature_extraction.py` WaveletScattering | 全局搜索 `np\.`，确认模块顶部有 `import numpy as np` |
| 2 | 类定义在文件末尾，运行时 NameError | `feature_extraction.py` DetectionHead | 在 `__main__` 块中实例化所有类 |
| 3 | `torch.cos(float)` 类型错误 | `feature_extraction.py` _morlet_wavelet | 测试时传入 Python float 而非 Tensor |
| 4 | 多尺度特征图尺寸不一致却 `torch.cat` | `feature_extraction.py` WaveletScattering | 检查 `forward()` 中所有 append 到 list 的张量 shape |
| 5 | PowerShell `&&` 和嵌套引号解析失败 | WSL 执行脚本 | 使用 `bash -c '...'` 单层引号；或写临时 .py 文件执行 |
| 6 | `logits.sum()` 替代 class score | `work_gradcam_edge.py` | Grad-CAM 必须对**特定类别**分数求梯度 |
| 7 | `stream.synchronize()` 在 submit 中 | `work_async_infer.py` | 测量 submit() 耗时，若 ≈ 推理时间则有同步点 |

---

## 十、审计记录模板

每次审计后，在 iteration 的 `review.md` 中追加以下章节：

```markdown
## 算法引用审计

| 维度 | 结果 | 说明 |
|------|------|------|
| A-命名 | ✅/⚠️/❌ | |
| B-公式 | ✅/⚠️/❌ | |
| C-行为 | ✅/⚠️/❌ | |

### 发现的问题
- [级别] 问题描述 → 修复方式

### 参考来源
- 论文：作者, 标题, 会议, 年份
- 代码参考：GitHub URL（如有）
```

---

*本指南由 Iteration 158~200 的算法审计经验沉淀而成，随项目演进持续更新。*
