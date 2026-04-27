# 贡献指南 (Contributing Guide)

感谢您对本项目的关注！以下是参与贡献的基本规范。

---

## 开发环境搭建

```bash
# 1. 克隆仓库
git clone https://github.com/RussellCooper-DJZ/Anti-Interference-2D-Vision-Guided-Precision-Sorting-Technology-for-High-Reflectivity-Workpieces.git
cd Anti-Interference-2D-Vision-Guided-Precision-Sorting-Technology-for-High-Reflectivity-Workpieces

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安装开发依赖
pip install -r requirements-dev.txt
```

---

## 代码风格规范

本项目遵循以下代码规范：

| 工具 | 用途 | 配置 |
|------|------|------|
| **flake8** | 语法与风格检查 | 最大行长 127 字符 |
| **black** | 代码格式化 | 默认配置 |
| **isort** | import 排序 | 默认配置 |
| **mypy** | 静态类型检查 | 宽松模式 |

提交前请确保代码通过以下检查：

```bash
flake8 . --max-line-length=127 --exclude=.venv,build,dist
black --check .
isort --check-only .
```

---

## 提交规范

本项目采用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <subject>

[optional body]

[optional footer]
```

常用 `type` 类型：

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | 缺陷修复 |
| `docs` | 文档更新 |
| `refactor` | 代码重构（不影响功能） |
| `test` | 测试相关 |
| `chore` | 构建/工具链变更 |
| `perf` | 性能优化 |

示例：

```
feat(vision): add multi-scale CBAM attention for improved glare suppression

Extend the CBAM module to support multi-scale feature aggregation,
improving robustness under mixed lighting conditions.

Closes #42
```

---

## Pull Request 流程

1. Fork 本仓库并基于 `main` 分支创建特性分支：`git checkout -b feat/your-feature`
2. 完成开发并确保代码通过 CI 检查
3. 提交 PR 并填写模板中的所有必填项
4. 等待 Code Review，根据反馈进行修改

---

## 专利与许可声明

在贡献代码前，请确认您的贡献不侵犯任何第三方知识产权。本项目核心算法（FLARE、EdgeVision-C）受专利保护，详见 [PATENTS](./PATENTS) 文件。

所有贡献将遵循项目的 **Apache License 2.0** 协议。

---

## 新增模块参考

贡献以下模块时，请确保同步更新对应文档：

| 模块 | 新增内容 | 需同步的文档 |
|------|----------|--------------|
| `vision/measurement.py` | CaliperMeasurement、GapMeasurement | README.md、DATASET_GUIDE.md、REAL_DATA_GUIDE.md、CHANGELOG.md |
| `data/synth_dataset_generator.py` | PBRLightingSystem（Blinn-Phong BRDF）| README.md、DATASET_GUIDE.md、CHANGELOG.md |
| `robot/abb_robotstudio_interface.py` | AbbRobotEGM（UDP EGM 协议） | README.md、CHANGELOG.md |
| `vision/appearance_detection.py` | PhotometricStereoNet（CNN 光度立体） | CHANGELOG.md、CLEAN_ROOM_AUDIT.md |
| `vision/existence_checking.py` | GrayMatcher SSDA 替代 NCC | CHANGELOG.md、CLEAN_ROOM_AUDIT.md |
| `robot/cells/sorting_cell.py` | 真实视觉引导分拣单元 | README.md、CHANGELOG.md |
