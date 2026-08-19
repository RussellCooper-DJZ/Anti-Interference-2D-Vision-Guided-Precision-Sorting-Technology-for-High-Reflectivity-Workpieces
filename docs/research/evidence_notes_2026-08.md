# 商业计划书与技术白皮书证据笔记

## 经过核验的外部资料

| 编号 | 来源 | 可用于支撑的表述 | 使用边界 |
|---|---|---|---|
| [1] | MarketsandMarkets，2025 年机器视觉市场报告 | 该报告估计全球机器视觉市场从 2025 年 158.3 亿美元增长至 2030 年 236.3 亿美元，预测 CAGR 为 8.3%；报告将自动化、AI/深度学习与质量控制列为需求驱动因素。 | 第三方市场研究估计，不宜表述为审计口径或本项目收入预测。 |
| [2] | Advanced Illumination，偏振滤光在机器视觉中的应用说明 | 偏振片能够抑制曲面镜面反射，但会减少可用光通量；改变光源—相机几何关系通常应与偏振方案共同评估。 | 厂商应用说明，应用于工程设计原则而非性能承诺。 |
| [3] | Yu 等，2025，《Photonics》 | 高反光曲面会形成眩光、过曝、图像失真并遮蔽缺陷；论文提出将偏振成像与边缘/多尺度深度学习结构联合使用。 | 学术场景与本项目工件、镜头、光源不同，不能直接外推其性能指标。 |
| [4] | Arm Helium 技术页面 | Arm 将 Helium/MVE 定位为适用于嵌入式、边缘 AI 的处理能力；具体算子加速倍数须以目标编译器、时钟、内存与模型实测为准。 | 不引用该页推导本项目具体毫秒级结果。 |

## 项目内已存在的工程证据

| 证据 | 位置 | 可支持的表述 |
|---|---|---|
| HDR、光斑模拟与抗高光算法单测 | `tests/test_antiglare_core.py` | 在标准 Python 环境下，五个无硬件核心测试已通过；并不等同于产线准确率验证。 |
| Simple ISP 模拟与 HDR、定位处理代码 | `src/vision/` | 已实现 PC 端算法验证原型与模块化视觉处理链路。 |
| RA8P1/Helium C 算子与 FSP 模板 | `src/embedded/` | 已提供嵌入式部署的代码骨架与部分向量化算子实现；须以实际 RA8P1 工具链、板卡和相机模组完成编译、计时与 HIL 验证。 |
| IMX 寄存器辅助与动态高光策略 | `src/hardware/imx_reg_helper.c`, `dynamic_glare_control.c` | 已提供适配设计起点；寄存器地址、模式时序必须以所选 IMX 模组、Sony NDA 数据手册和模组厂 BSP 为准。 |

## 引用链接

[1]: https://www.marketsandmarkets.com/Market-Reports/industrial-machine-vision-market-234246734.html
[2]: https://advancedillumination.com/application-notes/using-polarizing-filters-in-machine-vision/
[3]: https://www.mdpi.com/2304-6732/12/4/368
[4]: https://www.arm.com/technologies/helium
