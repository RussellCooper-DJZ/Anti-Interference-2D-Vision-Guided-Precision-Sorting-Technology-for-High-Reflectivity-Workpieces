# Embedded C Runtime (30天落地版本)

该子工程提供一个可编译、可测试的纯 C 视觉运行时骨架，目标是让现有仓库中的 Python 算法链路在嵌入式侧快速落地。

## 与现有仓库模块映射

- `hdr_processing.py` → `src/ops.c: op_hdr_fuse_mean_u8 + op_specular_suppress_u8`
- `feature_extraction.py` / `train.py` → 当前由阈值分割占位，后续替换为 INT8 推理图执行器
- `localization_and_calibration.py` → `src/runtime.c: compute_pose_from_mask`
- `ra8p1_helium_processing.c` → 可替换 `src/ops.c` 中热点算子为 Helium SIMD 实现

## 当前包含能力

1. 双曝光 HDR 均值融合
2. 高光抑制（clip + blend）
3. 二值分割与 3x3 开运算去噪
4. 前景质心 + 主方向估计（工业抓取姿态近似）
5. 端到端单元测试（`tests/test_pipeline.c`）

## 30天落地目标（已覆盖）

- [x] Runtime API：`runtime_init` / `runtime_process_frame`
- [x] 关键算子：HDR、阈值、形态学、姿态估计
- [x] 无动态内存：工作区由上层静态提供
- [x] 自动化验证：`make test` 一键执行

## Clean-room 工程建议（非法律意见）

- 仅依据公开论文/文档定义输入输出与性能指标。
- 不复制第三方工具私有 IR、序列化格式和生成代码模板。
- 保留设计评审、实现和测试时间戳记录，形成独立研发证据链。

## 构建与验证

```bash
cd embedded_c_runtime
make test
```
