#!/usr/bin/env python3
"""
200次迭代自动化审计脚本
扫描 results/auto_tuning/iteration_001~200，提取元数据并生成分段统计。
"""
import os
import re
import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path("E:/opcode/Anti-Interference-2D/results/auto_tuning")

# 分支关键词映射
BRANCH_KEYWORDS = {
    "Vision": ["FLARE", "edge", "segment", "attention", "CBAM", "conv", "hdr", "highlight", "glare", "localiz", "subpixel", "caliper", "gap", "measurement", "defect", "scratch", "inpaint", "repair", "biformer", "dcn", "ghost", "pafpn", "rcf", "refinement", "wavelet", "fourier", "gabor"],
    "Data": ["data", "augment", "synth", "dataset", "cutmix", "mixup", "mosaic", "pipeline_data", "loader", "quality", "validation", "integrity", "anomaly"],
    "Training": ["train", "loss", "optim", "scheduler", "lr_", "early_stop", "checkpoint", "ema", "distill", "prun", "quant", "focal", "dice", "lovasz", "ohem", "cutmix", "mosaic", "mixup", "augment", "knowledge", "hyperparameter", "hpo", "ensemble", "benchmark"],
    "Robot": ["robot", "ABB", "EGM", "RAPID", "gripper", "sorting", "cell", "coord", "hand_eye", "calib", "motion", "trajectory", "collision"],
    "Embedded": ["embedded", "RA8", "Helium", "quant", "INT8", "TFLite", "ONNX", "deploy", "edge_comput", "inference_engine"],
    "Pipeline": ["pipeline", "workflow", "orchestr", "gateway", "service_mesh", "container", "autoscaling", "canary", "feature_flag", "chaos", "multitenancy", "scheduler", "observability", "etl", "streaming", "transaction", "state_machine", "event_sourcing", "flow_"],
    "Infra": ["config", "logging", "log_", "experiment", "version", "registry", "monitor", "alert", "profil", "debug", "drift", "feature_store", "automl", "nas", "searcher", "serving", "api", "fastapi", "torchserve"],
    "Docs": ["README", "doc", "guide", "report", "culture", "compound", "summary", "delivery", "model_card"],
}

# 偏离主线的关键词（通用基础设施）
OFF_TRACK_KEYWORDS = [
    "chaos_engineering", "multitenancy", "feature_flag", "service_mesh",
    "container", "gateway", "observability", "event_sourcing", "state_machine",
    "transaction", "autoscaling", "canary", "perf_optimizer", "param_optimizer",
    "streaming", "etl", "scheduler", "workflow_engine", "resource_manager",
    "task_orchestration", "data_exchange", "job_scheduler", "flow_orchestration",
    "pipeline_platform", "pipeline_", "omnivision", "autom", "nas", "searcher",
    "supply_chain", "quality_control", "digital_twin", "simulation",
    "smart_factory", "industrial_iot", "mes", "erp", "inventory"
]


def classify_branch(text: str) -> str:
    """根据文本内容分类到分支。"""
    text_lower = text.lower()
    scores = {}
    for branch, keywords in BRANCH_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in text_lower)
        if score > 0:
            scores[branch] = score
    if not scores:
        return "Other"
    return max(scores, key=scores.get)


def is_off_track(text: str) -> bool:
    """判断是否偏离视觉主线。"""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in OFF_TRACK_KEYWORDS)


def extract_plan_topic(plan_path: Path) -> str:
    """从 plan.md 提取主题。"""
    try:
        content = plan_path.read_text(encoding='utf-8')
        # 查找 "## 主题" 或 "## 目标" 后的第一行
        for pattern in [r"##\s*主题\s*\n+(.+)", r"##\s*目标\s*\n+(.+)", r"#\s*Iteration\s+\d+\s*Plan:\s*(.+)"]:
            m = re.search(pattern, content, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        # 取第一行非标题内容
        lines = [l.strip() for l in content.split('\n') if l.strip() and not l.strip().startswith('#')]
        if lines:
            return lines[0][:80]
        return "(empty)"
    except Exception as e:
        return f"(error: {e})"


def audit_iteration(iter_dir: Path) -> dict:
    """审计单个迭代目录。"""
    iter_num = int(iter_dir.name.split('_')[1])
    plan_path = iter_dir / "plan.md"
    review_path = iter_dir / "review.md"
    compound_path = iter_dir / "compound.md"
    work_files = list(iter_dir.glob("work_*"))

    has_plan = plan_path.exists()
    has_review = review_path.exists()
    has_compound = compound_path.exists()

    plan_topic = extract_plan_topic(plan_path) if has_plan else "(missing)"
    branch = classify_branch(plan_topic) if has_plan else "Unknown"
    off_track = is_off_track(plan_topic) if has_plan else False

    total_lines = 0
    for wf in work_files:
        try:
            total_lines += len(wf.read_text(encoding='utf-8').split('\n'))
        except:
            pass

    return {
        "iteration": iter_num,
        "topic": plan_topic,
        "branch": branch,
        "off_track": off_track,
        "has_plan": has_plan,
        "has_review": has_review,
        "has_compound": has_compound,
        "work_files": len(work_files),
        "work_lines": total_lines,
    }


def main():
    print("=" * 70)
    print("200次迭代自动化审计")
    print("=" * 70)

    results = []
    for i in range(1, 201):
        iter_dir = BASE_DIR / f"iteration_{i}"
        if not iter_dir.exists():
            results.append({
                "iteration": i,
                "topic": "(missing directory)",
                "branch": "Missing",
                "off_track": False,
                "has_plan": False,
                "has_review": False,
                "has_compound": False,
                "work_files": 0,
                "work_lines": 0,
            })
            continue
        result = audit_iteration(iter_dir)
        results.append(result)

    # === 统计输出 ===
    print("\n【一、迭代完整度统计】")
    missing_dirs = sum(1 for r in results if r["branch"] == "Missing")
    missing_plan = sum(1 for r in results if not r["has_plan"])
    missing_review = sum(1 for r in results if not r["has_review"])
    missing_compound = sum(1 for r in results if not r["has_compound"])
    no_work = sum(1 for r in results if r["work_files"] == 0)

    print(f"  缺失目录: {missing_dirs}/200")
    print(f"  缺失 plan.md: {missing_plan}/200")
    print(f"  缺失 review.md: {missing_review}/200")
    print(f"  缺失 compound.md: {missing_compound}/200")
    print(f"  无 work 文件: {no_work}/200")

    print("\n【二、分支分布统计】")
    branch_counts = Counter(r["branch"] for r in results if r["branch"] != "Missing")
    for branch, count in branch_counts.most_common():
        pct = count / 200 * 100
        print(f"  {branch:12s}: {count:3d} ({pct:5.1f}%)")

    print("\n【三、偏离主线分析】")
    off_track_count = sum(1 for r in results if r["off_track"])
    print(f"  偏离主线迭代: {off_track_count}/200 ({off_track_count/200*100:.1f}%)")
    print("  偏离主题列表:")
    for r in results:
        if r["off_track"]:
            print(f"    Iter {r['iteration']:3d}: {r['topic'][:60]}")

    print("\n【四、分阶段统计】")
    phases = [
        ("Phase 1", 1, 50),
        ("Phase 2", 51, 100),
        ("Phase 3", 101, 137),
        ("Phase 4", 138, 157),
        ("Phase 5", 158, 200),
    ]
    for name, start, end in phases:
        phase_results = [r for r in results if start <= r["iteration"] <= end]
        total = len(phase_results)
        off = sum(1 for r in phase_results if r["off_track"])
        avg_work = sum(r["work_files"] for r in phase_results) / total if total else 0
        avg_lines = sum(r["work_lines"] for r in phase_results) / total if total else 0
        missing_rv = sum(1 for r in phase_results if not r["has_review"])
        missing_cp = sum(1 for r in phase_results if not r["has_compound"])
        print(f"  {name} ({start}-{end}):")
        print(f"    总数: {total}, 偏离: {off}, 偏离率: {off/total*100:.1f}%")
        print(f"    平均 work 文件: {avg_work:.1f}, 平均代码行: {avg_lines:.0f}")
        print(f"    缺失 review: {missing_rv}, 缺失 compound: {missing_cp}")

    print("\n【五、代码量统计】")
    total_lines = sum(r["work_lines"] for r in results)
    print(f"  全部 work 文件总代码行: {total_lines:,}")

    # 保存 JSON
    output_path = Path("E:/opcode/Anti-Interference-2D/results/iteration_audit_raw.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  原始数据已保存: {output_path}")

    print("\n" + "=" * 70)
    print("审计完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
