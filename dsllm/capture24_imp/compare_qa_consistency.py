#!/usr/bin/env python3
"""
Compare baseline and upsampled QA files for consistency.
"""
import json

# Paths to baseline and upsampled QA files (edit as needed)
baseline_path = "/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_compare_buffer/300seconds_100DS/test/capture24_test_qa_stage2_cls.json"
upsampled_path = "/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_upsampled/300seconds_100DS_upsampled/test/capture24_test_qa_stage2_cls.json"

with open(baseline_path) as f:
    baseline = {int(entry['index']): entry['qa_pair'] for entry in json.load(f)['dataset']}
with open(upsampled_path) as f:
    upsampled = {int(entry['index']): entry['qa_pair'] for entry in json.load(f)['dataset']}

indices = sorted(set(baseline.keys()) & set(upsampled.keys()))

# Check a few indices in detail
for idx in [0, 1, 10, 100, 1000]:
    if idx not in indices:
        continue
    b = baseline[idx]
    u = upsampled[idx]
    print(f"Index {idx}:")
    for key in ['Q', 'trend_text', 'A']:
        print(f"  {key} identical? {b[key] == u[key]}")
    print(f"  smry baseline: {b['smry'][:80]} ...")
    print(f"  smry upsampled: {u['smry'][:80]} ...")
    print(f"  corr_text baseline: {b['corr_text'][:80]} ...")
    print(f"  corr_text upsampled: {u['corr_text'][:80]} ...")
    print()

# Summary statistics
identical_Q = 0
identical_trend = 0
identical_A = 0
total = 0
for idx in indices:
    b = baseline[idx]
    u = upsampled[idx]
    if b['Q'] == u['Q']:
        identical_Q += 1
    if b['trend_text'] == u['trend_text']:
        identical_trend += 1
    if b['A'] == u['A']:
        identical_A += 1
    total += 1
print(f"\nSummary:")
print(f"  Q identical: {identical_Q}/{total}")
print(f"  trend_text identical: {identical_trend}/{total}")
print(f"  A identical: {identical_A}/{total}")
print(f"  (smry and corr_text are expected to differ)") 