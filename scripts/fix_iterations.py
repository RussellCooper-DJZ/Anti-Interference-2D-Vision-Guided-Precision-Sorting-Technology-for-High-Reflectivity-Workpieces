#!/usr/bin/env python3
"""
迭代修复脚本：标记偏离迭代、补全缺失 artifact、归档孤立代码。
"""
import os
import shutil
from pathlib import Path

BASE_DIR = Path("E:/opcode/Anti-Interference-2D/results/auto_tuning")
ARCHIVE_DIR = Path("E:/opcode/Anti-Interference-2D/archive/iteration_graveyard")

# 偏离主线的迭代范围（基于审计报告）
OFF_TRACK_RANGES = [
    (67, 67),       # AutoML/NAS
    (81, 119),      # 供应链、流水线基础设施
    (138, 157),     # 平台化、数字孪生
]

# Phase 3 缺失 Plan 的迭代
MISSING_PLAN = list(range(120, 138))

# 缺失 Compound 的迭代
MISSING_COMPOUND = [24] + list(range(114, 138))


def is_off_track(iter_num):
    for start, end in OFF_TRACK_RANGES:
        if start <= iter_num <= end:
            return True
    return False


def mark_off_track():
    """在偏离迭代的 review.md 中追加偏离标记。"""
    print("=== 标记偏离迭代 ===")
    count = 0
    for start, end in OFF_TRACK_RANGES:
        for i in range(start, end + 1):
            review_path = BASE_DIR / f"iteration_{i}" / "review.md"
            if review_path.exists():
                content = review_path.read_text(encoding='utf-8')
                if "【偏离主线】" in content:
                    continue
                # 在文件末尾追加
                appendix = """

---

## 【偏离主线声明】（2026-04-24 审计追加）

**本迭代被标记为偏离项目核心价值主线。**

项目核心价值：高反光金属工件精准边缘识别 + 机器人抓取（目标精度 <0.5mm）。
本迭代的主题（通用基础设施 / SaaS 平台 / 供应链管理等）与该核心价值无直接关系。

**建议**：
- 本迭代的 work 产出保留在 `results/auto_tuning/` 中供参考
- 不强制要求整合进主代码库
- 未来迭代应通过"核心价值六问"门控避免类似偏离

---
"""
                review_path.write_text(content + appendix, encoding='utf-8')
                count += 1
                print(f"  Marked iteration_{i}")
    print(f"  Total marked: {count}")


def create_missing_plans():
    """为缺失 Plan 的迭代补全通用 Plan。"""
    print("\n=== 补全缺失 Plan ===")
    count = 0
    for i in MISSING_PLAN:
        plan_path = BASE_DIR / f"iteration_{i}" / "plan.md"
        if plan_path.exists():
            continue
        content = f"""# 迭代 #{i} Plan（补全）

## 主题
Phase 3 流水线基础设施探索（已归档）

## 说明
本迭代原属 Phase 3 (101–137) 的流水线平台化系列。
该阶段的迭代聚焦于通用基础设施（DAG执行、资源管理、API网关等），
与项目核心价值（高反光金属边缘识别）存在偏离。

## 状态
- 原始 Plan 未记录
- Work 文件已保留在 `results/auto_tuning/iteration_{i}/`
- 建议：如需参考通用基础设施设计，可查阅本目录下的 work 文件

## 审计备注（2026-04-24）
本 Plan 为审计补全，非原始规划文档。
"""
        plan_path.write_text(content, encoding='utf-8')
        count += 1
        print(f"  Created plan.md for iteration_{i}")
    print(f"  Total created: {count}")


def create_missing_compounds():
    """为缺失 Compound 的迭代补全通用 Compound。"""
    print("\n=== 补全缺失 Compound ===")
    count = 0
    for i in MISSING_COMPOUND:
        compound_path = BASE_DIR / f"iteration_{i}" / "compound.md"
        if compound_path.exists():
            continue
        content = f"""# Iteration {i} Compound（补全）

## 迭代产出

- Work 文件已保留在 `iteration_{i}/` 目录

## 审查结论

本迭代的 Compound 在原始流程中未生成。
经 2026-04-24 审计，本迭代属于以下类别之一：
- 通用基础设施探索（Pipeline/Platform 系列）
- 或早期迭代记录遗漏

## 沉淀状态

- **未整合进主代码库**：本迭代的 work 文件保持独立
- **参考价值**：如需通用基础设施设计参考，可查阅相关 work 文件

## 后续建议

未来迭代应严格遵循 Plan → Work → Review → Compound 四件套流程，
避免缺失复盘沉淀环节。
"""
        compound_path.write_text(content, encoding='utf-8')
        count += 1
        print(f"  Created compound.md for iteration_{i}")
    print(f"  Total created: {count}")


def archive_isolated_work():
    """归档未沉淀到主代码库的孤立 work 文件。"""
    print("\n=== 归档孤立 Work 文件 ===")
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 需要归档的迭代：Phase 2-4 的通用基础设施
    archive_ranges = [
        (51, 100),   # Phase 2 MLOps
        (101, 137),  # Phase 3 流水线
        (138, 157), # Phase 4 平台化
    ]
    
    count = 0
    for start, end in archive_ranges:
        for i in range(start, end + 1):
            src_dir = BASE_DIR / f"iteration_{i}"
            if not src_dir.exists():
                continue
            
            # 判断是否为偏离迭代
            if not is_off_track(i):
                continue
            
            dst_dir = ARCHIVE_DIR / f"iteration_{i}"
            
            # 如果已归档则跳过
            if dst_dir.exists():
                continue
            
            # 复制整个迭代目录到归档区
            shutil.copytree(src_dir, dst_dir)
            
            # 在原始位置留下 redirect 说明
            readme_path = src_dir / "ARCHIVED.txt"
            readme_path.write_text(
                f"本迭代的完整内容已归档至:\n"
                f"  archive/iteration_graveyard/iteration_{i}/\n\n"
                f"原因：经 2026-04-24 审计，本迭代主题与项目核心价值主线存在偏离。\n"
                f"保留原始目录结构以便索引，实际内容已移至归档区。\n",
                encoding='utf-8'
            )
            
            # 删除原始的 work 文件（保留 plan/review/compound/ARCHIVED.txt）
            for wf in src_dir.glob("work_*"):
                if wf.is_file():
                    wf.unlink()
            
            count += 1
            print(f"  Archived iteration_{i}")
    
    print(f"  Total archived: {count}")


def main():
    print("=" * 60)
    print("200次迭代修复脚本")
    print("=" * 60)
    
    mark_off_track()
    create_missing_plans()
    create_missing_compounds()
    archive_isolated_work()
    
    print("\n" + "=" * 60)
    print("修复完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
