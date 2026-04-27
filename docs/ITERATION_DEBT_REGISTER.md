# 迭代技术债务登记册（Iteration Debt Register）

> **维护规则**：每完成一次债务偿还，更新本登记册。新债务发现时，按 P0–P3 分级登记。

---

## P0 — 致命债务（已偿还 / 已归档）

| # | 债务项 | 影响 | 涉及迭代 | 状态 | 偿还方式 |
|---|--------|------|---------|------|---------|
| P0-1 | Phase 3 质量崩塌：18个连续迭代无 Plan，24个无 Compound | 迭代流程完全失控，无法追溯 | 120–137 | ✅ 已偿还 | 补全通用 Plan/Compound |
| P0-2 | 平台化幻象：40+迭代投入通用SaaS平台 | 60+迭代的产出是技术垃圾 | 81–119, 138–157 | ✅ 已偿还 | 标记偏离 + 归档孤立代码 |
| P0-3 | 未沉淀代码：60+迭代的 work 文件孤立无价值 | 占用空间，误导后续开发者 | 51–119, 138–157 | ✅ 已偿还 | 移至 `archive/iteration_graveyard/` |

---

## P1 — 严重债务（已偿还 / 进行中）

| # | 债务项 | 影响 | 涉及迭代 | 状态 | 偿还方式 |
|---|--------|------|---------|------|---------|
| P1-1 | 算法引用不准确 | 误导审稿人，公式错误导致训练失效 | 164, 167, 179–193 | ✅ 已偿还 | 逐一修正公式/命名/文档 |
| P1-2 | 重复实现：配置管理、日志、监控、检查点 | 维护成本倍增 | 38/195, 43/140, 44/194, 46/155, 48/119/194 | ✅ 已偿还 | `core/__init__.py` 统一标注 Legacy deprecated；内部导入已统一为 Unified 实现；Legacy 将在 v0.7.0 移除 |
| P1-3 | 测试覆盖不足：仅3个测试文件 | 无法保证核心模块正确性 | 全局 | ⏳ 进行中 | 新增 `test_data_modules.py`、`test_training_modules.py`；CI 已恢复 6 个被 ignore 的测试文件；覆盖率 29% → 目标 50%+ |
| P1-4 | 主代码库整合：Phase 5算法未全部整合 | 算法改进孤立在 results/ 中 | 158–200 | ✅ 已偿还 | DeformConv2d/GhostConv/PAFPN/EdgeRefinementHead/SubpixelLocalizerV2/ROICorrectorV2/HandEyeCalibratorV2/GlareInpainter 已整合 |
| P1-5 | 后端 API 与算法库严重脱节：measurement.py(1314行)/roi_tools.py(1230行)/appearance_detection.py/existence_checking.py 等 8 个 vision 模块完全没有暴露到 FastAPI | GUI 无法调用卡尺、圆拟合、线拟合、ROI 工具、缺陷检测等核心算法 | 全局 | ⏳ 待偿还 | 后端新增 /measure 系列端点 |
| P1-6 | WPF GUI 功能壳化：7 个 ViewModel 中仅 MainWindowViewModel 被主界面使用；其余 6 个存在但未集成到主界面导航 | 机器人通信、独立仪表盘、专用参数面板等功能不可见 | 全局 | ⏳ 待偿还 | 主界面增加 ContentControl + DataTemplateSelector 切换各功能面板 |
| P1-7 | 坐标定位键名映射错误 | localize() 返回 centroid_px，server.py 错误使用 cx/cy 访问，导致所有坐标为 0 | 全局 | ✅ 已偿还 | 修正 infer_image/infer_base64/_create_visualization 中的键名映射 |
| P1-8 | 模型加载使用 strict=False 且无版本校验 | 兼容旧版 checkpoint 但跳过键名不匹配层，可能引入静默精度损失 | 全局 | ⏳ 待偿还 | 增加 checkpoint 版本标签和兼容性检查层 |

---

## P2 — 中等债务（部分偿还）

| # | 债务项 | 影响 | 涉及迭代 | 状态 |
|---|--------|------|---------|------|
| P2-1 | Phase 5 代码量偏低（部分仅30–75行） | 可能只是概念验证，缺乏深度 | 179–200 | ⏳ 待评估 |
| P2-2 | 179–200 Plan 风格突变（全英文） | 与之前中文风格不一致 | 179–200 | ✅ 已偿还 | compound.md 统一为中文为主风格 |
| P2-3 | CHANGELOG 早期迭代记录不完整 | 无法追溯 1–50 的详细变更 | 1–50 | ⏳ 待补全（1–22 已补全，23–50 仍缺失） |
| P2-4 | CI 测试忽略核心基础设施测试 | Unified 基础设施从未在 CI 中验证 | 全局 | ✅ 已偿还 | `.github/workflows/python-app.yml` 移除全部 `--ignore`；Python 升级至 3.12 |
| P2-5 | Python 版本声明不一致 | CI/Badge/pyproject 版本冲突 | 全局 | ✅ 已偿还 | 统一声明为 >=3.10（CI 3.12） |
| P2-6 | 可视化输出硬编码为四宫格 | `_create_visualization` 固定输出 2×2 拼接，无法按需返回单张分割图/边缘图/原图 | 全局 | ⏳ 待偿还 | 增加 visualization_mode 参数：grid/original/seg/edge/overlay |
| P2-7 | 配置管理是全局替换而非增量更新 | `/api/v1/config` POST 直接替换整个配置对象，并发时可能丢配置 | 全局 | ⏳ 待偿还 | 改为 PATCH 语义或增加配置版本号乐观锁 |
| P2-8 | localization_and_calibration.py 过于庞大（3192行） | 包含定位、标定、焊缝检测、角点检测、手眼标定等多个不相关领域，违反单一职责 | 全局 | ⏳ 待偿还 | 拆分为 localization.py / calibration.py / weld_detection.py / corner_detection.py |
| P2-9 | 后端缺少 graceful shutdown | Uvicorn 直接终止，推理锁 _lock 和模型显存没有清理逻辑 | 全局 | ⏳ 待偿还 | 注册 lifespan 事件，释放 GPU 内存和推理锁 |
| P2-10 | GUI 完全没有单元测试 | 157 个 Python 模块有 pytest，WPF 项目零测试 | 全局 | ⏳ 待偿还 | 补充 xUnit / MSTest 测试项目，覆盖 ViewModel 命令和属性变更 |

---

## P3 — 轻微债务（低优先级）

| # | 债务项 | 影响 | 涉及迭代 | 状态 |
|---|--------|------|---------|------|
| P3-1 | compound.md 命名不统一（大小写混用） | 美观问题 | 114–137 | ✅ 已偿还 |
| P3-2 | work 文件命名风格不一致 | 美观问题 | 全局 | ⏳ 待统一 |
| P3-3 | API 只有英文自动文档（/docs），没有中文 API 使用说明 | 对国内工厂现场工程师不友好 | 全局 | ⏳ 待偿还 | 补充 docs/API_GUIDE_CN.md |
| P3-4 | 日志分散：后端用 Python logging，GUI 用 TextBox 绑定，没有统一日志收集 | 故障排查困难 | 全局 | ⏳ 待偿还 | 后端增加文件日志落盘，GUI 增加日志文件读取视图 |
| P3-5 | 没有容器化/打包：依赖 .venv_win 和本地 Python，没有 Docker 或 PyInstaller 打包 | 部署环境不可复现 | 全局 | ⏳ 待偿还 | 提供 Dockerfile + docker-compose.yml；GUI 提供 Installer 项目 |

---

## 债务统计

| 级别 | 总数 | 已偿还 | 待偿还 | 偿还率 |
|------|------|--------|--------|--------|
| P0 | 3 | 3 | 0 | 100% |
| P1 | 8 | 4 | 4 | 50% |
| P2 | 10 | 3 | 7 | 30% |
| P3 | 5 | 1 | 4 | 20% |
| **合计** | **26** | **11** | **15** | **42%** |

---

## 偿还路线图

### 已完成 ✅
- [x] 标记 60 个偏离迭代
- [x] 归档 60 个迭代的孤立 work 文件
- [x] 补全 18 个缺失 Plan + 25 个缺失 Compound
- [x] 修正 9 处算法引用错误
- [x] 整合 8 个 Phase 5 算法到主代码库
- [x] **统一 core 接口**：Legacy 标注 deprecated，给出迁移指南
- [x] **修复 CI**：Python 3.12，移除 `--ignore`，统一使用 `requirements-dev.txt`
- [x] **修复版本声明**：`pyproject.toml`、`requirements.txt`、CI 统一为 3.10+
- [x] **补充测试**：`test_data_modules.py`、`test_training_modules.py`

### 下一步 ⏳
- [ ] 为核心模块补充 pytest 单元测试（目标覆盖率 > 60%）
- [ ] 评估 Phase 5 低代码量迭代是否需要合并深化
- [ ] 补全 CHANGELOG 迭代 23–50 记录
- [ ] 统一全局 work 文件命名风格
- [ ] 后端暴露 measurement.py / roi_tools.py / appearance_detection.py 能力到 API
- [ ] GUI 集成 ImageInspection/BatchProcessing/ModelManagement/RobotControl/Dashboard 面板
- [ ] 拆分 localization_and_calibration.py（3192行）为单一职责模块
- [ ] 后端增加 graceful shutdown + GPU 显存清理
- [ ] WPF 项目补充单元测试
- [ ] 补充中文 API 使用文档 + Docker 打包

---

*最后更新：2026-04-25*
