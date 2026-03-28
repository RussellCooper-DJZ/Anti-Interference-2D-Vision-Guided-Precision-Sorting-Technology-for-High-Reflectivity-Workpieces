# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- `.github/workflows/python-app.yml`：新增 GitHub Actions CI 工作流，自动执行 flake8 语法检查
- `requirements-dev.txt`：新增开发依赖文件（flake8、black、isort、mypy、pytest、tensorboard 等）
- `CONTRIBUTING.md`：新增贡献指南，包含开发环境搭建、代码风格规范与 PR 流程
- `CHANGELOG.md`：新增本变更日志文件
- 各核心库模块（`vision/`、`data/`、`robot/`）新增 `__all__` 公共 API 声明

### Changed
- `main_pipeline.py`：将 `print` 语句替换为标准 `logging` 模块，提供结构化日志输出
- `.gitignore`：扩充忽略规则，新增虚拟环境、IDE 配置、测试缓存、日志文件等条目
- `README.md`：新增 CI 状态徽章、日志说明与快速开始注意事项
- `scripts/inspect_dataset.py`：修复全部行尾空格（PEP 8 合规）

### Fixed
- 修复所有 Python 文件中的行尾空格问题（共 10 处）

---

## [0.5.0] — 2026-03-24

### Added
- 完整专利申请预审包（`docs/patent/`）
- Clean-room 工程化审计记录（`CLEAN_ROOM_AUDIT.md`）

## [0.4.0] — 2026-03-24

### Added
- 全国 8 大场景合成数据集生成器（`data/synth_national_scenes.py`）
- ABB RobotStudio 仿真集成文档与 RAPID 服务端代码

## [0.3.0] — 2026-03-24

### Added
- 完整可视化数据集（`docs/visualization/`，约 50MB）

## [0.2.0] — 2026-03-24

### Changed
- 项目模块化重构：`vision/`、`data/`、`training/`、`robot/`、`embedded/` 分包

## [0.1.0] — 2026-03-24

### Added
- 初始版本：AGEANet 模型架构、HDR 处理管线、亚像素定位、嵌入式 EdgeVision-C 架构
