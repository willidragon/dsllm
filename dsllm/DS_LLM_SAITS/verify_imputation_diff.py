#!/usr/bin/env python3
"""
verify_imputation_diff.py
========================
Compare imputed and original (ground truth) .pkl files for each split, and print summary statistics of their differences.
"""
import pickle
import numpy as np
from pathlib import Path

# Settings
DATA_TAG = "300seconds"
TARGET_DS = "100"
SPLITS = ["train", "val", "test"]

# Paths
base_pred = Path("/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_upsampled_saits/imputations")
base_gt = Path("/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_compare_buffer")

for split in SPLITS:
    # Predicted (imputed) data
    pred_pkl = base_pred / split / f"capture24_{split}_data_stage2_{DATA_TAG}_{TARGET_DS}DS_saits.pkl"
    # Ground truth data
    gt_pkl = base_gt / f"{DATA_TAG}_{TARGET_DS}DS" / split / f"capture24_{split}_data_stage2_{DATA_TAG}_{TARGET_DS}DS.pkl"

    if not pred_pkl.exists() or not gt_pkl.exists():
        print(f"[WARNING] Missing file for split {split}:\n  pred: {pred_pkl}\n  gt:  {gt_pkl}")
        continue

    with open(pred_pkl, "rb") as f:
        pred = np.array(pickle.load(f))
    with open(gt_pkl, "rb") as f:
        gt = np.array(pickle.load(f))

    if pred.shape != gt.shape:
        print(f"[ERROR] Shape mismatch for split {split}: pred {pred.shape}, gt {gt.shape}")
        continue

    diff = pred - gt
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff**2)
    max_abs = np.max(np.abs(diff))
    print(f"=== {split.upper()} ===")
    print(f"  Shape: {pred.shape}")
    print(f"  MAE: {mae:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Max abs diff: {max_abs:.6f}")
    print(f"  Identical: {np.allclose(pred, gt)}\n") 