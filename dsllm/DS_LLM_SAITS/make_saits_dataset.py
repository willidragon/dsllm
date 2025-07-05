#!/usr/bin/env python3
"""
make_saits_dataset.py
=====================
Convert Capture24 stage-2 .pkl datasets into an HDF5 file that SAITS
(`SAITS-main/run_models.py`) can read.

The HDF5 structure complies with the expectation of
`modeling/unified_dataloader.py`:
  /<split>/X               – ground-truth complete series (target_ds)
           /X_hat          – observed series with NaNs (input_ds on grid)
           /missing_mask   – 1 where value present in X_hat, 0 where NaN
           /indicating_mask – 1 where value is missing in X_hat (i.e., to be imputed)

Example
-------
python make_saits_dataset.py \
  --base_output_dir /project/.../dsllm/dsllm/data \
  --data_tag 300seconds \
  --input_ds 1000 --target_ds 100 \
  --splits train val test

This will write an HDF5 file named:
  capture24_100DS_impute_from_1000DS.h5
inside   {base_output_dir}/saits_ds/
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import pickle
import h5py
import numpy as np


def convert_split(base_dir: Path, data_tag: str, input_ds: int, target_ds: int, split: str, seq_len: int, feature_dim: int):
    """Load .pkl files for a split and convert to arrays.

    Returns X, X_hat, missing_mask, indicating_mask (all np.ndarray)
    with shape (N, seq_len, feature_dim).
    """
    # Ground-truth high-res path (target_ds)
    hi_file = base_dir / "stage_2_compare_buffer" / f"{data_tag}_{target_ds}DS" / split / f"capture24_{split}_data_stage2_{data_tag}_{target_ds}DS.pkl"
    # Low-res path (input_ds)
    lo_file = base_dir / "stage_2_compare_buffer" / f"{data_tag}_{input_ds}DS" / split / f"capture24_{split}_data_stage2_{data_tag}_{input_ds}DS.pkl"

    if not (hi_file.exists() and lo_file.exists()):
        raise FileNotFoundError(f"Required pkl files not found for split '{split}'.\nHigh-res: {hi_file}\nLow-res : {lo_file}")

    with open(hi_file, "rb") as f:
        hi_segments = pickle.load(f)
    with open(lo_file, "rb") as f:
        lo_segments = pickle.load(f)

    assert len(hi_segments) == len(lo_segments), "hi and lo segment counts mismatch"

    up = input_ds // target_ds  # e.g. 10
    if seq_len % up != 0:
        raise ValueError("seq_len must be divisible by upsampling factor")

    X = np.stack(hi_segments).astype(np.float32)  # (N, seq_len, feature_dim)
    N = X.shape[0]

    X_hat = np.full_like(X, np.nan, dtype=np.float32)
    idx = np.arange(0, seq_len, up)  # indices where low-res samples exist
    # Ensure low-res seg length matches idx count
    assert lo_segments[0].shape[0] == len(idx), "Low-res timesteps inconsistent with factor"
    X_hat[:, idx, :] = np.stack(lo_segments).astype(np.float32)

    missing_mask = (~np.isnan(X_hat)).astype(np.float32)
    indicating_mask = 1.0 - missing_mask
    return X, X_hat, missing_mask, indicating_mask


def main():
    parser = argparse.ArgumentParser(description="Convert Capture24 pkl to SAITS HDF5 dataset")
    parser.add_argument("--base_output_dir", type=str, default="/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data", help="Base data directory (same as paths.yaml base_output_dir)")
    parser.add_argument("--data_tag", type=str, default="300seconds", help="Data tag prefix, e.g. 300seconds")
    parser.add_argument("--input_ds", type=int, default=1000, help="Input down-sample rate (e.g. 1000)")
    parser.add_argument("--target_ds", type=int, default=100, help="Target down-sample rate (e.g. 100)")
    parser.add_argument("--seq_len", type=int, default=300, help="Number of timesteps for target_ds sequences")
    parser.add_argument("--feature_dim", type=int, default=3, help="Number of sensor channels")
    parser.add_argument("--splits", nargs="*", default=["train", "val", "test"], help="Dataset splits to process")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to save the HDF5 file (default: {base_output_dir}/saits_ds)")
    args = parser.parse_args()

    base_dir = Path(args.base_output_dir)
    if args.output_dir is None:
        # Save to /.../data/SAITS/capture24_{target_ds}DS_impute_from_{input_ds}DS/datasets.h5
        out_dir = base_dir / "SAITS" / f"capture24_{args.target_ds}DS_impute_from_{args.input_ds}DS"
        out_dir.mkdir(parents=True, exist_ok=True)
        h5_path = out_dir / "datasets.h5"
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        h5_path = out_dir / "datasets.h5"

    print("Creating SAITS dataset at", h5_path)
    with h5py.File(h5_path, "w") as hf:
        for split in args.splits:
            print(f"  Processing split: {split}")
            X, X_hat, miss, indi = convert_split(base_dir, args.data_tag, args.input_ds, args.target_ds, split, args.seq_len, args.feature_dim)
            grp = hf.create_group(split)
            grp.create_dataset("X", data=X, compression="gzip")
            grp.create_dataset("X_hat", data=X_hat, compression="gzip")
            grp.create_dataset("missing_mask", data=miss, compression="gzip")
            grp.create_dataset("indicating_mask", data=indi, compression="gzip")
            print(f"    Saved {X.shape[0]} samples.")
    print("✅ Done. HDF5 saved at", h5_path)


if __name__ == "__main__":
    main() 