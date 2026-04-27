#!/usr/bin/env python3
"""
backfill_history.py — 将 iteration 21-200 的元数据回填到 full_history.json

从每个 iteration 目录的 plan.md 和 review.md 提取:
- iteration number
- 主题 (topic)
- 完成度评分
- 质量评估
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RESULTS_DIR = PROJECT_ROOT / "results" / "auto_tuning"
HISTORY_FILE = RESULTS_DIR / "full_history.json"


def extract_topic(plan_path: Path) -> str:
    """从 plan.md 提取主题"""
    if not plan_path.exists():
        return "未知主题"
    content = plan_path.read_text(encoding="utf-8")
    # 匹配 "## 主题" 或 "## 目标"
    match = re.search(r"## 主题\s*\n\s*(.+?)(?:\n|$)", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    # 尝试从第一行提取
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    for line in lines:
        if line.startswith("#") and "迭代" in line:
            return line.replace("#", "").strip()
    return "未知主题"


def extract_score(review_path: Path) -> float:
    """从 review.md 提取综合评分"""
    if not review_path.exists():
        return 0.0
    content = review_path.read_text(encoding="utf-8")
    # 匹配质量评估表格中的评分
    matches = re.findall(r"评分[:：]\s*(\d+)/10", content)
    if matches:
        return sum(int(s) for s in matches) / len(matches)
    return 0.0


def extract_checklist_status(review_path: Path) -> Dict[str, int]:
    """从 review.md 提取完成度"""
    if not review_path.exists():
        return {"completed": 0, "total": 0}
    content = review_path.read_text(encoding="utf-8")
    completed = len(re.findall(r"### ✅ 已完成|### ✅", content))
    total = completed + len(re.findall(r"### ⚠️|### ❌", content))
    return {"completed": completed, "total": max(total, completed)}


def backfill_history():
    """回填 iteration 21-200 的元数据"""
    # 读取现有的 history
    if HISTORY_FILE.exists():
        history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    else:
        history = []

    existing_iters = {entry.get("iteration") for entry in history if "iteration" in entry}
    print(f"现有记录: {len(history)} 条, iterations: {sorted(existing_iters)[:5]}...")

    new_entries = []
    for i in range(21, 201):
        iter_dir = RESULTS_DIR / f"iteration_{i}"
        if not iter_dir.exists():
            continue

        plan_path = iter_dir / "plan.md"
        review_path = iter_dir / "review.md"

        topic = extract_topic(plan_path)
        score = extract_score(review_path)
        checklist = extract_checklist_status(review_path)

        entry = {
            "iteration": i,
            "topic": topic,
            "score": score,
            "checklist": checklist,
            "phase": "learning"  # 区别于前20的优化迭代
        }
        new_entries.append(entry)

        if i % 50 == 0:
            print(f"  处理 iteration_{i}...")

    # 追加新条目
    history.extend(new_entries)

    # 保存
    HISTORY_FILE.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n✅ 完成: 原有 {len(history) - len(new_entries)} 条 + 新增 {len(new_entries)} 条 = {len(history)} 条")


if __name__ == "__main__":
    backfill_history()
