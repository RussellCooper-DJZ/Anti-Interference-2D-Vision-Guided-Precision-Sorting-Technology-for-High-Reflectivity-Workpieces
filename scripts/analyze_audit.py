import json
from collections import Counter

with open('E:/opcode/Anti-Interference-2D/results/iteration_audit_raw.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 1. Basic stats
print('=== Iteration Completeness ===')
print(f'Missing plan: {sum(1 for r in data if not r["has_plan"])}/200')
print(f'Missing review: {sum(1 for r in data if not r["has_review"])}/200')
print(f'Missing compound: {sum(1 for r in data if not r["has_compound"])}/200')
print(f'No work files: {sum(1 for r in data if r["work_files"] == 0)}/200')

# 2. Branch distribution
branches = Counter(r['branch'] for r in data)
print('\n=== Branch Distribution ===')
for b, c in branches.most_common():
    print(f'  {b}: {c} ({c/200*100:.1f}%)')

# 3. Off-track iterations
off = [r for r in data if r['off_track']]
print(f'\n=== Off-Track Iterations ({len(off)}) ===')
for r in off:
    print(f'  Iter {r["iteration"]}: {r["topic"]}')

# 4. Phase analysis
phases = [(1,50,'Phase 1'),(51,100,'Phase 2'),(101,137,'Phase 3'),(138,157,'Phase 4'),(158,200,'Phase 5')]
print('\n=== Phase Analysis ===')
for start, end, name in phases:
    pr = [r for r in data if start <= r['iteration'] <= end]
    off_c = sum(1 for r in pr if r['off_track'])
    avg_lines = sum(r['work_lines'] for r in pr) / len(pr)
    missing_rv = sum(1 for r in pr if not r['has_review'])
    missing_cp = sum(1 for r in pr if not r['has_compound'])
    print(f'{name} ({start}-{end}):')
    print(f'  Total: {len(pr)}, Off-track: {off_c}, Avg lines: {avg_lines:.0f}')
    print(f'  Missing review: {missing_rv}, Missing compound: {missing_cp}')

# 5. Missing plan
no_plan = [r for r in data if not r['has_plan']]
print(f'\n=== Missing Plan ({len(no_plan)}) ===')
for r in no_plan:
    print(f'  Iter {r["iteration"]}')

# 6. Missing compound
no_cp = [r for r in data if not r['has_compound']]
print(f'\n=== Missing Compound ({len(no_cp)}) ===')
for r in no_cp:
    print(f'  Iter {r["iteration"]}')

# 7. Vision iterations
vision = [r for r in data if r['branch'] == 'Vision']
print(f'\n=== Vision Iterations ({len(vision)}) ===')
for r in vision:
    print(f'  Iter {r["iteration"]}: {r["topic"]}')

# 8. Training iterations
training = [r for r in data if r['branch'] == 'Training']
print(f'\n=== Training Iterations ({len(training)}) ===')
for r in training:
    print(f'  Iter {r["iteration"]}: {r["topic"]}')

# 9. Phase 4 detail
p4 = [r for r in data if 138 <= r['iteration'] <= 157]
print('\n=== Phase 4 (138-157) Topics ===')
for r in p4:
    print(f'  Iter {r["iteration"]}: {r["topic"]}')

# 10. Phase 3 detail
p3 = [r for r in data if 101 <= r['iteration'] <= 137]
print('\n=== Phase 3 (101-137) Topics ===')
for r in p3:
    print(f'  Iter {r["iteration"]}: {r["topic"]}')

# 11. Total lines
print(f'\n=== Total Work Code Lines: {sum(r["work_lines"] for r in data):,} ===')
