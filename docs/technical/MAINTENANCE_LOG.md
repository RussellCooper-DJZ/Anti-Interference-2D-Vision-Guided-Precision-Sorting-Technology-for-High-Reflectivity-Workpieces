# Anti-Interference-2D 项目维护记录 (Maintenance Log)

**日期**: 2026-04-25
**维护人**: Manus AI Agent
**状态**: 已完成核心架构重构与关键 Bug 修复

---

## 1. 架构重构 (Architecture Refactoring)
为了符合现代开源项目的目录规范，对项目结构进行了深度调整：
- **引入 `src/` 布局**：将所有核心逻辑（`vision`, `data`, `training`, `robot`, `embedded`）移入 `src/` 目录，解决根目录文件杂乱问题。
- **文档中心化**：将各类指南、专利和技术文档整合至 `docs/` 下的 `guides/`, `legal/`, `technical/` 子目录。
- **配置规范化**：更新了 `pyproject.toml`，支持 `src` 布局的包发现，并修正了 `pytest` 的覆盖率统计路径。

## 2. 关键 Bug 修复 (Critical Bug Fixes)
在深度审查代码过程中发现并修复了以下问题：
- **AGEANetLite 维度不匹配**：修复了 `src/vision/feature_extraction.py` 中 `AGEANetLite` 的 `edge_head` 输入通道数错误（从 `c` 修正为 `c*2`），解决了模型初始化时的张量维度报错。
- **重构后的导入路径失效**：修复了 `train.py`, `main_pipeline.py` 和 `real_world_dataloader.py` 中由于目录移动导致的 `from data...` 或 `from vision...` 导入失败问题，统一更新为基于 `src` 的导入路径。
- **在线合成循环依赖**：修正了 `OnlineSynthDataset` 中 `synthesize_one_sample` 的延迟导入路径。

## 3. 性能与逻辑优化 (Performance & Logic Optimization)
- **HDR 管线鲁棒性**：审查了 `hdr_processing.py`，确认其 Mertens 融合逻辑在无曝光时间元数据时具有良好的鲁棒性。
- **数据增强多样性**：确认了 `ShipHullAugPipeline` 针对高反光场景的专项增强（如 `random_sun_glare`）逻辑正确。

## 4. 后续建议 (Next Steps)
- **CI/CD 适配**：建议在 GitHub Actions 中更新测试脚本，以适配新的 `src` 目录结构。
- **模型导出测试**：由于引入了 `src` 布局，建议重新测试 `onnx` 导出脚本，确保路径引用正确。
- **文档同步**：建议更新所有外部文档中的代码片段引用路径。

---
*本记录由 Manus 自动生成，作为项目健康度审计的一部分。*
