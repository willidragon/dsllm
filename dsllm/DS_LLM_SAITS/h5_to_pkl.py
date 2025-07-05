#!/usr/bin/env python3
"""
h5_to_pkl.py
============
Convert SAITS imputed HDF5 file (imputations.h5) to a .pkl file (list of arrays) for SensorLLM pipeline.

Usage:
  python h5_to_pkl.py \
    --h5_path /path/to/imputations.h5 \
    --pkl_path /path/to/output.pkl

If --pkl_path is not specified, will save as 'imputed_data.pkl' in the same directory as the HDF5 file.
"""
import argparse
import h5py
import pickle
from pathlib import Path

parser = argparse.ArgumentParser(description="Convert SAITS imputed HDF5 to .pkl list of arrays.")
parser.add_argument('--h5_path', type=str, required=True, help='Path to imputations.h5')
parser.add_argument('--pkl_path', type=str, default=None, help='Output .pkl path (default: /project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_upsampled_saits/)')
parser.add_argument('--split', type=str, default='test', help='Data split name for output file (default: test)')
parser.add_argument('--label_src', type=str, default=None, help='Path to source label .pkl file to copy alongside imputed data')
args = parser.parse_args()

h5_path = Path(args.h5_path)
if args.pkl_path is None:
    # Compose output path in /stage_2_upsampled_saits/<h5_stem>/<split>/
    base_out_dir = Path('/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_upsampled_saits')
    h5_stem = h5_path.stem.replace('.h5', '')
    out_dir = base_out_dir / h5_stem / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_name = f'capture24_{args.split}_data_stage2_300seconds_100DS_saits.pkl'
    pkl_path = out_dir / pkl_name
else:
    pkl_path = Path(args.pkl_path)

with h5py.File(h5_path, 'r') as f:
    keys = list(f.keys())
    print(f"Available keys in HDF5: {keys}")

    # Prefer the split-specific dataset if present, e.g. "imputed_test_set"
    split_key = f"imputed_{args.split}_set"
    if split_key in keys:
        key = split_key
    else:
        # Fall back to generic/common names
        for candidate in ['imputed_data', 'imputation', 'X_imputed', 'imputations', 'data']:
            if candidate in keys:
                key = candidate
                break
        else:
            key = keys[0]  # last resort
            print(f"Warning: Using first key '{key}' in HDF5.")

    arr = f[key][:]
    print(f"Loaded array of shape {arr.shape} from key '{key}'")

# Convert to list of arrays for compatibility
arr_list = [x for x in arr]
with open(pkl_path, 'wb') as f:
    pickle.dump(arr_list, f)
print(f"Saved imputed data as {pkl_path}")

# Optionally copy label file
if args.label_src is not None:
    import shutil
    label_dst = pkl_path.parent / pkl_path.name.replace('data_', 'labels_')
    shutil.copy(args.label_src, label_dst)
    print(f"Copied label file to {label_dst}") 