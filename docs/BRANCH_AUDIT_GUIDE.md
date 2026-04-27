# 全分支迭代审计指南（Multi-Branch Audit Guide）

> **目标**：为项目中六大技术分支建立各自独立的审计维度与检查清单，确保每个分支的迭代产出都经过与其技术领域相匹配的审查。

---

## 一、为什么每个分支需要不同的审计框架？

在 Iteration 158~200 中，我们建立了针对 **Vision 算法** 的三维审计框架（命名/公式/行为）。但当迭代触及 **Robot 通信**、**Data 合成**、**Embedded C 代码** 时，同样的维度会失效：

| 分支 | Vision 审计的问题 | 该分支真正需要审计什么？ |
|------|------------------|------------------------|
| **Robot** | "公式正确吗？" → 机器人控制没有数学公式 | 协议一致性、实时性、安全边界 |
| **Data** | "前向传播通过吗？" → 数据增强不是 nn.Module | 物理合理性、mask-图像同步、可复现性 |
| **Training** | "是标准的 OHEM 吗？" → 只覆盖损失函数 | 学习率调度、梯度流、检查点一致性 |
| **Embedded** | "torch.cos(float) 类型对吗？" → 这是 C 代码 | 内存安全、SIMD 正确性、尾像素处理 |
| **Pipeline** | "算法引用正确吗？" → Pipeline 是胶水代码 | 模块接口兼容性、错误降级、端到端时延 |

因此，**"一刀切"的审计清单是低效的**。每个分支需要独立的审计维度。

---

## 二、六大分支总览与审计核心

```
┌─────────────┬──────────────────────────┬─────────────────────────────┐
│   分支      │     核心技术资产          │        审计核心问题          │
├─────────────┼──────────────────────────┼─────────────────────────────┤
│  Vision     │ CNN, 注意力, 损失函数     │ "声称的算法 = 实现的算法？"   │
│  Data       │ 数据增强, 合成生成        │ "增强后的数据物理上合理吗？"  │
│  Training   │ 训练循环, 优化器, 调度器  │ "梯度流正确吗？能收敛吗？"    │
│  Robot      │ TCP/UDP 通信, 运动学      │ "协议一致吗？安全吗？"        │
│  Embedded   │ C, Helium SIMD, 量化     │ "内存安全吗？SIMD 对吗？"     │
│  Pipeline   │ 模块集成, 主流程          │ "接口兼容吗？错误能降级吗？"  │
└─────────────┴──────────────────────────┴─────────────────────────────┘
```

---

## 三、Vision 分支审计（简要回顾）

详见 `docs/ALGORITHM_AUDIT_GUIDE.md`。

**审计维度**：命名准确性 × 公式准确性 × 行为准确性

**核心检查清单**：
- 损失函数：Lovász / OHEM / Focal / Dice 的公式与实现一致性
- 数据增强：CutMix/Mosaic/AutoAugment 的最小充分条件
- 注意力/卷积：CBAM / SE / DCNv2 / CoordConv 的结构完整性
- 可解释性：Grad-CAM 的目标选择正确性

---

## 四、Data 分支审计（数据与增强）

### 4.1 审计核心问题
> **增强/合成后的数据，在物理上合理吗？几何变换中 mask 和图像还对应吗？**

### 4.2 四维审计框架

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  物理合理性(P)   │ ×  │  一致性(C)      │ ×  │  可复现性(R)    │ ×  │  边界安全(S)    │
│ Physical Sanity │    │ Consistency     │    │ Reproducibility │    │ Safety          │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 4.3 检查清单

#### P — 物理合理性（Physical Sanity）

| 数据操作 | 物理合理性检查 | 常见陷阱 |
|----------|---------------|---------|
| **光照增强** | 亮度变化后像素值是否仍在 [0,255] 或 [0,1] 范围内？Gamma 曲线是否单调？ | Gamma < 0 导致非单调；亮度加减后未 clip |
| **高光注入** | 注入的高光区域是否在物理上合理（如太阳方向一致、水面反射角度合理）？ | 随机椭圆高光方向与光源方向矛盾 |
| **噪声注入** | 噪声幅度是否与传感器特性匹配（如 8-bit 图像噪声 σ 不应 > 50）？ | 噪声幅度过大导致图像完全失真 |
| **模糊增强** | 运动模糊核的方向是否与场景运动一致？散焦模糊的核大小是否符合光学限制？ | 50×50 的散焦模糊核在 512×512 图像上不现实 |
| **合成数据** | PBR 光照模型中入射角=反射角吗？材质粗糙度与高光大小是否一致？ | 镜面反射违反菲涅尔定律；粗糙度与高亮相悖 |

**快速测试**：
```python
# 检查像素范围
assert aug_image.min() >= 0 and aug_image.max() <= 255
# 检查 mask 值域（二值 mask 只能有 0 和 255）
assert set(np.unique(mask_aug)).issubset({0, 255})
```

#### C — 一致性（Consistency）

| 检查项 | 测试方法 | 预期结果 |
|--------|---------|---------|
| **几何变换同步** | 对 image 和 mask 应用同一组随机参数 | `mask_aug` 中目标区域的轮廓应与 `image_aug` 中可见轮廓一致 |
| **CutMix mask 同步** | 检查 patch 区域的 mask 是否来自同一张图 | `out_masks[i, y1:y2, x1:x2] == masks[idx[i], y1:y2, x1:x2]` |
| **边缘与 mask 一致性** | 从 mask 提取的边缘应与提供的 edge 一致 | `generate_edge_from_mask(mask)` ≈ `edge`（允许 1-2px 偏差） |
| **多通道同步** | RGB 三通道应用相同的 color jitter | 不应出现单通道异常导致伪彩色 |

**快速测试**：
```python
# 一致性测试：相同 seed 应产生相同结果
set_seed(42)
img1, msk1 = augment(image, mask)
set_seed(42)
img2, msk2 = augment(image, mask)
np.testing.assert_array_equal(img1, img2)
np.testing.assert_array_equal(msk1, msk2)
```

#### R — 可复现性（Reproducibility）

| 检查项 | 要求 | 常见陷阱 |
|--------|------|---------|
| **全局 seed 控制** | `set_seed()` 必须同时设置 `random`, `np.random`, `torch` 的 seed | 只设置一个库的 seed |
| **多进程安全** | `DataLoader(num_workers>1)` 时各 worker 的 seed 应不同但确定 | worker 间 seed 冲突导致重复样本 |
| **合成数据集** | 相同 seed 应生成完全相同的样本序列 | 使用了未 seed 的随机源（如 `uuid`, `time`） |

#### S — 边界安全（Safety）

| 检查项 | 测试方法 | 预期结果 |
|--------|---------|---------|
| **越界坐标处理** | 对接近图像边界的 bbox 应用旋转/裁剪 | 不应出现负坐标或超出图像范围的访问 |
| **除零保护** | 归一化、计算均值等操作 | 全零区域不应导致 `inf` 或 `nan` |
| **空 mask 处理** | 输入 mask 全为零 | 应返回全零 edge，不崩溃 |

---

## 五、Training 分支审计（训练与优化）

### 5.1 审计核心问题
> **损失曲线下降是因为网络在学习，还是因为 bug？梯度流是通畅的吗？**

### 5.2 四维审计框架

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  梯度健康(G)     │ ×  │  损失正确性(L)   │ ×  │  调度合理性(S)   │ ×  │  检查点完整(C)   │
│ Gradient Health │    │ Loss Correctness│    │ Scheduler Sanity│    │ Checkpoint OK   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 5.3 检查清单

#### G — 梯度健康（Gradient Health）

| 检查项 | 测试方法 | 预期结果 | 常见陷阱 |
|--------|---------|---------|---------|
| **梯度非 NaN/Inf** | 打印首步梯度统计 | 无 `nan`/`inf` | Loss 函数未加 smooth；输入未 normalize |
| **梯度幅值合理** | 检查首层和末层梯度范数 | 应在 1e-6 ~ 1e2 范围内 | 梯度爆炸（>1e6）或梯度消失（<1e-10） |
| **混合精度稳定** | `GradScaler` 是否被正确使用 | `scaler.scale(loss).backward()` + `scaler.step(optimizer)` | 忘记 `scaler.update()`；Loss 在 scale 下溢出 |
| **梯度裁剪生效** | `torch.nn.utils.clip_grad_norm_` | 梯度范数被限制在阈值内 | 裁剪阈值过大（无效）或过小（阻止学习） |

**快速测试**：
```python
# 梯度健康检查
for name, param in model.named_parameters():
    if param.grad is not None:
        assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
        assert not torch.isinf(param.grad).any(), f"Inf grad in {name}"
```

#### L — 损失正确性（Loss Correctness）

| 检查项 | 测试方法 | 预期结果 | 常见陷阱 |
|--------|---------|---------|---------|
| **Loss 值范围** | 全随机输入 vs 完美预测 | 随机输入 Loss > 完美预测 Loss | Loss 符号反了；分母为负 |
| **损失单调性** | 逐步改善预测，观察 Loss 是否单调下降 | Loss 应单调下降 | 用了 IoU 但计算的是 `1-IoU` 的相反数 |
| **多任务平衡** | 单独训练每个头 vs 联合训练 | 联合训练时两个 Loss 都不应被压制到接近 0 | 一个 Loss 比另一个大 1000 倍，导致梯度主导 |
| **权重衰减** | `weight_decay` 是否只对可学习参数生效 | Bias 和 BN 参数不应被 weight_decay | `AdamW` 参数分组错误；BN bias 被衰减 |

#### S — 调度合理性（Scheduler Sanity）

| 检查项 | 测试方法 | 预期结果 | 常见陷阱 |
|--------|---------|---------|---------|
| **Warmup** | 前 N 步学习率应从 0 线性/指数增长到初始值 | `lr[0] ≈ 0`, `lr[warmup] = base_lr` | Warmup 步数太长（占训练 50%）；未从 0 开始 |
| **Cosine Annealing** | 学习率应按余弦曲线下降到 min_lr | 曲线平滑，无跳变 | `T_max` 设置错误；每个 epoch 重置导致锯齿 |
| **学习率数量级** | 打印每个参数组的实际学习率 | 应在 1e-5 ~ 1e-1 范围内 | 学习率过大（Loss 发散）或过小（不收敛） |

**快速测试**：
```python
# 学习率曲线可视化
lrs = []
for epoch in range(epochs):
    for batch in dataloader:
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
plt.plot(lrs)  # 应看到 warmup + cosine 形状
```

#### C — 检查点完整（Checkpoint Completeness）

| 检查项 | 要求 | 常见陷阱 |
|--------|------|---------|
| **保存内容** | `torch.save` 应包含 `model`, `optimizer`, `scheduler`, `epoch`, `best_metric` | 只保存 model.state_dict()，恢复训练时学习率从头开始 |
| **恢复一致性** | 从检查点恢复后，继续训练应无缝衔接 | 恢复后 Loss 突然跳变（optimizer 状态丢失） |
| **格式兼容** | 检查点应在 CPU/GPU 之间可迁移 | 保存时未 `map_location='cpu'` |
| **版本控制** | 检查点应记录代码版本/配置哈希 | 无法追溯检查点对应的代码版本 |

---

## 六、Robot 分支审计（机器人通信与控制）

### 6.1 审计核心问题
> **发送给机器人的指令，协议格式对吗？机器人突然断线时系统会崩溃吗？**

### 6.2 四维审计框架

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  协议一致性(P)   │ ×  │  安全边界(S)    │ ×  │  故障容错(F)    │ ×  │  实时性(R)      │
│ Protocol Match  │    │ Safety Bounds   │    │ Fault Tolerance │    │ Real-time       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 6.3 检查清单

#### P — 协议一致性（Protocol Match）

| 检查项 | 测试方法 | 预期结果 | 常见陷阱 |
|--------|---------|---------|---------|
| **JSON 格式** | 发送的 JSON 能被对端正确解析 | `json.loads(send_data)` 成功 | 浮点数精度丢失；中文编码错误；缺少换行分隔符 |
| **坐标系** | 发送的 (x,y,z) 与机器人实际坐标系一致 | 模拟桩和真实机器人的坐标变换一致 | 单位混淆（mm vs m）；欧拉角顺序错误（ZYX vs XYZ） |
| **指令枚举** | MoveL / MoveJ / MoveC 等指令拼写正确 | 与 RAPID 程序中的指令名完全匹配 | 大小写错误（`movel` vs `MoveL`） |
| **速度/区域** | speed 和 zone 参数在有效范围内 | speed > 0, zone ∈ {"z0", "z1", ..., "z200"} | 负数速度；不存在的 zone 名称 |

**快速测试**：
```python
# 协议一致性测试
robot = AbbRobotStub()
robot.connect()
msg = robot._build_move_message(500.0, 0.0, 400.0)
assert json.loads(msg)  # 必须可解析
assert "cmd" in json.loads(msg)
```

#### S — 安全边界（Safety Bounds）

| 检查项 | 要求 | 常见陷阱 |
|--------|------|---------|
| **工作空间限制** | 目标位置应在机器人可达空间内 | 未检查奇异点；目标在机器人背后 |
| **速度上限** | 发送的速度不应超过安全阈值 | 用户输入 99999 mm/s 直接发送 |
| **加速度约束** | 相邻指令的速度变化应平滑 | 从 10 mm/s 突变到 5000 mm/s |
| **碰撞检测** | 模拟环境中应检测自碰撞和环境碰撞 | 只在真实机器人上才检查碰撞 |

**快速测试**：
```python
# 安全边界测试
assert robot.send_target(99999, 0, 0) == False  # 超界应拒绝
assert robot.send_target(-100, 0, 0) == False   # 负数应拒绝
```

#### F — 故障容错（Fault Tolerance）

| 检查项 | 测试方法 | 预期结果 |
|--------|---------|---------|
| **连接断开** | 运行中关闭机器人服务端 | 应抛出可捕获的异常，不 segfault |
| **超时处理** | 机器人长时间不响应 | 应有超时机制，返回错误状态 |
| **重连机制** | 断线后尝试重新连接 | 应有指数退避重试，不无限循环 |
| **状态查询** | 查询机器人当前状态时断线 | 应返回上一次已知状态或错误码 |

#### R — 实时性（Real-time）

| 检查项 | 要求 | 常见陷阱 |
|--------|------|---------|
| **指令延迟** | 从调用 `send_target` 到数据发出应 < 10ms | 在 `send` 前做大量计算 |
| **EGM 频率** | EGM 模式下的控制周期应稳定在 4ms (250Hz) |  Python GIL 导致抖动；未使用独立线程 |
| **缓冲区管理** | 不应无限堆积未发送的指令 | 队列无上限，导致内存泄漏 |

---

## 七、Embedded 分支审计（嵌入式 C/SIMD）

### 7.1 审计核心问题
> **C 代码有内存越界吗？SIMD 指令处理尾像素了吗？量化后的精度损失在可接受范围内吗？**

### 7.2 四维审计框架

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  内存安全(M)    │ ×  │  SIMD 正确性(V)  │ ×  │  数值精度(N)    │ ×  │  硬件兼容(H)    │
│ Memory Safety   │    │ SIMD Correctness│    │ Numeric Precision│    │ HW Compatibility│
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 6.3 检查清单

#### M — 内存安全（Memory Safety）

| 检查项 | 测试方法 | 预期结果 | 常见陷阱 |
|--------|---------|---------|---------|
| **指针非空** | 所有函数入口处检查 `NULL` | `if (!ptr) return;` | 直接解引用用户传入的指针 |
| **缓冲区大小** | 计算所需的输出缓冲区大小 | 调用者分配的缓冲区 >= 实际需求 | 未考虑通道数（如 RGB 需要 ×3） |
| **循环边界** | `for (i=0; i<num; i++)` 中 `num` 是否正确 | 不访问数组越界 |  off-by-one：`<=` 写成 `<`；忘记处理 remainder |
| **栈溢出防护** | 大数组应分配在堆上而非栈上 | 局部变量 < 1KB | 在函数内声明 `float buffer[1000000]` |

**快速测试（Code Review）**：
```c
// 反例：未检查指针
void bad_process(uint8_t* in, uint8_t* out, int n) {
    for (int i=0; i<n; i++) out[i] = in[i] * 2;  // 如果 in/out 为 NULL？
}

// 正例
void good_process(const uint8_t* in, uint8_t* out, uint32_t n) {
    if (!in || !out || n == 0) return;
    for (uint32_t i=0; i<n; i++) out[i] = in[i] << 1;
}
```

#### V — SIMD 正确性（SIMD Correctness）

| 检查项 | 测试方法 | 预期结果 | 常见陷阱 |
|--------|---------|---------|---------|
| **向量加载对齐** | `vld1q_u8` 的地址是否 16 字节对齐？ | 不对齐时应使用 `vld1q_u8`（ARM 支持非对齐加载）或手动对齐 | 假设指针总是对齐 |
| **尾像素处理** | `num_pixels & 0xF` 的 remainder 循环是否正确？ | remainder 像素与向量化路径结果一致 | remainder 循环与向量化路径逻辑不一致 |
| **饱和 vs 环绕** | 8-bit 加法应使用饱和指令 `vqaddq_u8` 而非普通加法 | 结果不会溢出回绕 | `vaddq_u8` 导致 200+200=144（回绕） |
| ** Helium intrinsic 可用性** | 使用的 intrinsic 是否在目标芯片上支持？ | RA8P1 (Cortex-M85) 支持 Helium MVE | 使用了 NEON intrinsic（A 系列）而非 MVE |

**快速测试**：
```c
// 测试向量化路径和标量路径结果一致
uint8_t in[32], out_vec[32], out_scalar[32];
// 填充随机数据...
helium_process(in, out_vec, 32);      // 向量化实现
scalar_process(in, out_scalar, 32);   // 参考标量实现
assert(memcmp(out_vec, out_scalar, 32) == 0);
```

#### N — 数值精度（Numeric Precision）

| 检查项 | 要求 | 常见陷阱 |
|--------|------|---------|
| **量化精度** | INT8 量化后的 Top-1 / IoU 下降应 < 2% | 未做 per-channel 量化；calibration 数据集不具代表性 |
| **定点数转换** | `float` → `int8` 的 scale/zero_point 计算正确 | scale = (max-min)/255 而非 /256；zero_point 未四舍五入 |
| **累加器位宽** | 8-bit 乘法累加应使用 32-bit 累加器 | 16-bit 累加器在大量累加时溢出 |
| **除法替代** | 嵌入式避免除法，用移位或乘法近似 | 直接用 `/`，在 M85 上极慢 |

#### H — 硬件兼容（HW Compatibility）

| 检查项 | 要求 | 常见陷阱 |
|--------|------|---------|
| **编译器支持** | 代码应能用 GCC/Arm Compiler 6 编译 | 使用了编译器特定扩展 |
| **CMSIS 版本** | 使用的 CMSIS-NN 版本与 SDK 匹配 | CMSIS-NN API 在不同版本间有变化 |
| **内存布局** | 模型权重应放在 Flash，激活值放在 SRAM | 权重太大放不进 Flash；SRAM 溢出 |
| **Cache 一致性** | DMA 传输后是否需要 invalidate cache？ | DMA 写完数据后 CPU 读到的还是旧 cache 内容 |

---

## 八、Pipeline 分支审计（主流程集成）

### 8.1 审计核心问题
> **各个模块拼在一起能跑通吗？某个模块失败时，系统会优雅降级还是直接崩溃？**

### 8.2 四维审计框架

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  接口兼容(I)    │ ×  │  流程完整(F)    │ ×  │  降级策略(G)    │ ×  │  性能瓶颈(B)    │
│ Interface Match│    │ Flow Completion │    │ Graceful Degradation│    │ Bottleneck      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 8.3 检查清单

#### I — 接口兼容（Interface Match）

| 检查项 | 测试方法 | 预期结果 | 常见陷阱 |
|--------|---------|---------|---------|
| **数据类型** | A 模块输出 `uint8`，B 模块输入是否接受 `uint8`？ | 类型匹配或内部有转换 | 期望 `float32 [0,1]` 但收到 `uint8 [0,255]` |
| **坐标系** | Vision 输出像素坐标，Robot 期望世界坐标 mm | 中间有标定转换层 | 忘记乘像素尺寸；未考虑相机畸变 |
| **Batch 维度** | DataLoader 输出 `(B,C,H,W)`，模型是否接受？ | shape 匹配 | 单张图像推理时忘记 `unsqueeze(0)` |
| **设备一致性** | 图像在 CPU，模型期望 GPU | 有 `.to(device)` 调用 | 某一步忘移回 CPU 导致后续 numpy 操作失败 |

#### F — 流程完整（Flow Completion）

| 检查项 | 测试方法 | 预期结果 |
|--------|---------|---------|
| **端到端路径** | 从原始图像输入到机器人指令输出 | 无断点，每个阶段都有输出 |
| **预处理链** | 图像是否经过 resize → normalize → tensor → device？ | 顺序正确，不遗漏 |
| **后处理链** | 模型输出是否经过 threshold → resize → 坐标转换？ | 阈值在 resize 前还是 resize 后（应在 resize 后） |
| **日志完整** | 每个阶段是否有输入/输出 shape 和耗时日志？ | 调试时可追溯问题阶段 |

**快速测试**：
```python
# 端到端流程测试（使用模拟桩）
pipeline = Pipeline(mode='demo')
result = pipeline.run_single(image)
assert 'robot_target' in result
assert result['robot_target'] is not None
```

#### G — 降级策略（Graceful Degradation）

| 故障场景 | 期望行为 | 常见陷阱 |
|----------|---------|---------|
| **模型推理失败** | 使用上一帧结果或发送机器人 HOLD | 直接抛出异常导致整个系统崩溃 |
| **摄像头断开** | 切换到本地测试图像或报警 | 无限重试导致 CPU 占满 |
| **机器人通信超时** | 缓存目标位置，等待重连 | 丢弃目标，机器人停在半空 |
| **检测结果置信度低** | 请求重新拍摄或扩大 ROI 再检测 | 直接发送低置信度坐标 |
| **HDR 融合失败** | 使用单张曝光最好的图像 | 返回全黑图像给下游 |

#### B — 性能瓶颈（Bottleneck）

| 检查项 | 测试方法 | 预期结果 | 常见陷阱 |
|--------|---------|---------|---------|
| **端到端时延** | 从拍照到机器人收到指令的总时间 | 应 < 500ms（实时场景） | 在循环内反复加载模型 |
| **各阶段占比** | 打印每个阶段的耗时 | 找到最慢的阶段优先优化 | 90% 时间花在数据预处理而非推理 |
| **内存占用** | 长时间运行后的内存增长 | 无内存泄漏 | 每帧都 `torch.from_numpy()` 但不释放 |
| **GPU 利用率** | `nvidia-smi` 观察 | 应稳定在 70-100% | CPU 预处理太慢导致 GPU 空闲等待 |

---

## 九、跨分支一致性检查

某些迭代会同时触及多个分支，此时需要额外的**跨分支一致性审计**：

| 跨分支变更 | 一致性检查点 | 示例 |
|-----------|-------------|------|
| **Vision 改输出格式** | Data 的 dataloader、Pipeline 的预处理、Robot 的坐标转换是否同步更新？ | edge head 输出从 `(B,1,H,W)` 改为 `(B,2,H,W)`（多尺度） |
| **Data 改增强策略** | Training 的损失函数是否需要调整？Eval 的 TTA 是否一致？ | 新增 CutMix 后，训练时的 loss 计算应忽略混合区域的虚假标签 |
| **Robot 改协议格式** | Pipeline 的 JSON 构建代码是否同步？ | zone 参数从 string 改为 int |
| **Embedded 改量化方式** | Training 是否导出对应的 QAT 模型？ | per-channel 量化需要在训练时插入 FakeQuantize |
| **Training 改检查点格式** | Eval 和 Pipeline 的加载代码是否兼容？ | 新增了 optimizer 状态但 eval 只加载 model |

**跨分支审计触发条件**：任何修改了 `__all__`、函数签名、数据格式的迭代，必须检查所有 import 该模块的地方。

---

## 十、迭代债务预防机制

在 158~200 的实践中，我们发现了一些反复出现的问题模式。建立以下预防机制：

### 10.1 预提交检查清单（Pre-Commit Checklist）

每个 work 文件在提交前，作者必须自检：

```markdown
- [ ] 本迭代是否修改了跨分支共享的接口？（如模型输入输出格式）
- [ ] 新增代码是否有 docstring / 类型注解？
- [ ] 是否有 `TODO` / `FIXME` / `HACK` 需要解释？
- [ ] 是否在 `__main__` 块中运行过基本测试？
- [ ] 是否更新了 CHANGELOG.md？
```

### 10.2 自动化门控（Automated Gates）

建议在 CI 中加入：

```yaml
# .github/workflows/ci.yml（建议）
jobs:
  vision-audit:
    runs-on: ubuntu-latest
    steps:
      - name: Forward pass test
        run: python -c "from vision.feature_extraction import FLARE; FLARE()(torch.randn(1,3,512,512))"
      - name: Algorithm name scan
        run: python scripts/audit_algorithm.py --scan-all

  data-audit:
    runs-on: ubuntu-latest
    steps:
      - name: Augmentation consistency test
        run: python -m pytest tests/test_data_consistency.py
      - name: Synthetic data physical sanity
        run: python -m pytest tests/test_synth_physics.py

  pipeline-audit:
    runs-on: ubuntu-latest
    steps:
      - name: End-to-end stub test
        run: python main_pipeline.py --mode demo --model_path checkpoints/stub.pth
```

### 10.3 迭代分类标签

给每个 iteration 打标签，明确其所属分支和审计维度：

```
iteration_201/
├── plan.md          # 标签: [Robot] [Protocol] [P0-Safety]
├── work_*.py        # 实现
├── review.md        # 按 Robot 分支的四维框架审计
└── compound.md      # 跨分支影响分析
```

---

## 十一、速查卡（Quick Reference Card）

### 分支 → 审计维度对照表

| 你的迭代涉及... | 首要审计维度 | 次要审计维度 | 必做测试 |
|----------------|-------------|-------------|---------|
| 新的 CNN/注意力/损失 | Vision: A-命名 | Vision: B-公式, C-行为 | `python -c "import model; model(x)"` |
| 数据增强/合成生成 | Data: P-物理 | Data: C-一致性, R-可复现 | 相同 seed → 相同输出 |
| 训练循环/优化器/调度器 | Training: G-梯度 | Training: L-损失, S-调度 | 首步梯度无 NaN |
| ABB/机器人通信 | Robot: P-协议 | Robot: S-安全, F-容错 | JSON 可解析，超界拒绝 |
| C 代码/Helium SIMD | Embedded: M-内存 | Embedded: V-SIMD, N-精度 | 向量化与标量结果一致 |
| 主流程/模块集成 | Pipeline: I-接口 | Pipeline: F-流程, G-降级 | `main_pipeline.py --mode demo` |

### 问题分级 → 修复策略（全分支通用）

| 级别 | 判定标准 | 修复时限 |
|------|---------|---------|
| **P0** | 会导致系统崩溃、安全事件、或完全错误的结果 | 阻塞发布，必须立即修复 |
| **P1** | 功能不正确但系统不崩溃，或性能严重下降 | 当前迭代内修复 |
| **P2** | 实现正确但非最优，或文档/命名不准确 | 下一个迭代修复 |
| **P3** | 代码风格、注释、日志等轻微问题 | 随代码重构时修复 |

---

*本指南覆盖项目六大分支：Vision / Data / Training / Robot / Embedded / Pipeline，与 `ALGORITHM_AUDIT_GUIDE.md` 配合使用。*
