import json
from collections import Counter, defaultdict

with open('E:/opcode/Anti-Interference-2D/results/iteration_audit_raw.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# === 1. 重复主题识别 ===
print("=== Duplicate / Similar Topics ===")

def normalize_topic(t):
    t = t.lower()
    for kw in ['pipeline', 'workflow', 'orchestr', 'config', 'logging', 'log_',
               'experiment', 'version', 'monitor', 'alert', 'profil', 'debug',
               'drift', 'feature_store', 'automl', 'nas', 'searcher', 'serving',
               'deploy', 'canary', 'chaos', 'multitenancy', 'observability',
               'scheduler', 'gateway', 'service_mesh', 'container', 'autoscaling',
               'event_sourcing', 'state_machine', 'transaction', 'etl', 'streaming',
               'resource_manager', 'task_orchestration', 'data_exchange',
               'job_scheduler', 'flow_orchestration', 'perf_optimizer',
               'param_optimizer']:
        if kw in t:
            return kw
    return t[:40]

topic_groups = defaultdict(list)
for r in data:
    if r['has_plan']:
        nt = normalize_topic(r['topic'])
        topic_groups[nt].append(r['iteration'])

for topic, iters in sorted(topic_groups.items(), key=lambda x: -len(x[1])):
    if len(iters) > 1:
        print(f"  {topic}: {len(iters)} iterations -> {iters}")

# === 2. 未沉淀到主代码库的迭代 ===
print("\n=== Not Integrated into Main Codebase ===")
# 检查哪些 work 文件的内容与主代码库无交集
main_code_files = [
    'feature_extraction.py', 'hdr_processing.py', 'localization_and_calibration.py',
    'measurement.py', 'appearance_detection.py', 'inference_engine.py',
    'data_augmentation.py', 'synth_dataset_generator.py', 'real_world_dataloader.py',
    'train.py', 'evaluate.py', 'abb_robotstudio_interface.py',
    'ra8p1_helium_processing.c', 'main_pipeline.py'
]

for r in data:
    if not r['has_plan']:
        continue
    # 简单判断：如果 topic 不包含主代码库中的模块名，可能未沉淀
    t = r['topic'].lower()
    # 这些主题通常是纯基础设施，未进入主代码库
    infra_keywords = ['pipeline', 'orchestr', 'gateway', 'service_mesh', 'container',
                      'autoscaling', 'canary', 'chaos', 'multitenancy', 'observability',
                      'scheduler', 'event_sourcing', 'state_machine', 'transaction',
                      'etl', 'streaming', 'resource_manager', 'task_orchestration',
                      'data_exchange', 'job_scheduler', 'flow_orchestration',
                      'perf_optimizer', 'param_optimizer', 'omnivision', 'automl',
                      'nas', 'searcher', 'supply_chain', 'digital_twin', 'simulation',
                      'smart_factory', 'industrial_iot', 'mes', 'erp', 'inventory',
                      'cost_optimization', 'capacity_planning', 'sla_management',
                      'root_cause', 'knowledge_graph', 'predictive_maintenance',
                      'devops', 'sre', 'platform', 'api_gateway', 'graphql']
    if any(kw in t for kw in infra_keywords):
        print(f"  Iter {r['iteration']:3d}: {r['topic'][:70]}")

# === 3. Phase 3 & 4 详细分析 ===
print("\n=== Phase 3 (101-137) Topic Analysis ===")
for r in data:
    if 101 <= r['iteration'] <= 137 and r['has_plan']:
        print(f"  {r['iteration']:3d}: {r['topic'][:70]}")

# === 4. Phase 5 (158-200) 精简度分析 ===
print("\n=== Phase 5 (158-200) Code Volume Trend ===")
for r in data:
    if 158 <= r['iteration'] <= 200:
        print(f"  {r['iteration']:3d}: {r['work_lines']:4d} lines - {r['topic'][:60]}")

# === 5. Missing artifacts summary ===
print("\n=== Missing Artifacts Summary ===")
missing_plan = [r['iteration'] for r in data if not r['has_plan']]
missing_review = [r['iteration'] for r in data if not r['has_review']]
missing_compound = [r['iteration'] for r in data if not r['has_compound']]
print(f"Missing plan: {missing_plan}")
print(f"Missing review: {missing_review}")
print(f"Missing compound: {missing_compound}")

# === 6. Debt scoring ===
print("\n=== Iteration Debt Score ===")
for r in data:
    score = 0
    if not r['has_plan']: score += 3
    if not r['has_review']: score += 2
    if not r['has_compound']: score += 1
    if r['work_lines'] < 50: score += 1
    if score >= 3:
        print(f"  Iter {r['iteration']:3d}: debt_score={score}, lines={r['work_lines']}")
