#!/usr/bin/env python3
"""
Generate QA JSON for upsampled data (100DS_upsampled)
"""

import os
import numpy as np
import pickle
import json
from datetime import datetime
import random
import time
from tqdm import tqdm
import warnings
from collections import defaultdict
import shutil
import yaml
import sys

# Import your QA generation utilities
sys.path.append(os.path.dirname(__file__))
from process_capture24_stage2_2_custom_mins_multiple import select_random_pair, QA_gen

# Load base paths from paths.yaml in project root
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paths_config = os.path.join(root_dir, 'paths.yaml')
with open(paths_config, 'r') as f:
    paths = yaml.safe_load(f)
base_output_dir = paths['base_output_dir']

# Settings for upsampled data
window_size_seconds = 300
sampling_rate = 1  # 100DS_upsampled is 1Hz
save_interval = 100

# Set up warning tracking
warning_counts = defaultdict(int)
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    warning_key = f"{category.__name__}: {str(message)}"
    warning_counts[warning_key] += 1
    if warning_counts[warning_key] <= 5:
        print(f"\nWarning #{warning_counts[warning_key]} - {warning_key}")
        print(f"In file: {filename}, line {lineno}")

warnings.showwarning = custom_warning_handler

def save_qa_data(qa_dict, path, filename):
    full_path = os.path.join(path, filename)
    temp_path = full_path + ".tmp"
    try:
        with open(temp_path, 'w') as f:
            json.dump(qa_dict, f, indent=2)
        shutil.move(temp_path, full_path)
        print(f"Successfully saved to: {full_path}")
        print(f"File size: {os.path.getsize(full_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Error saving to {full_path}: {str(e)}")
        backup_path = os.path.join(os.getcwd(), f"backup_{filename}")
        print(f"Attempting to save backup to: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(qa_dict, f, indent=2)
        print(f"Backup saved successfully to: {backup_path}")

def process_upsampled_split(split):
    print(f"\n{'='*80}")
    print(f"Processing upsampled {split} split")
    print(f"{'='*80}")

    output_tag = "300seconds_100DS_upsampled"
    base_output_path = os.path.join(base_output_dir, "stage_2_upsampled", output_tag, split)
    data_file = os.path.join(base_output_path, f"capture24_{split}_data_stage2_{output_tag}.pkl")
    labels_file = os.path.join(base_output_path, f"capture24_{split}_labels_stage2_{output_tag}.pkl")
    if not os.path.exists(data_file) or not os.path.exists(labels_file):
        print(f"Skipping {split} split: files not found.")
        return

    print(f"Loading data from: {data_file}")
    print(f"Loading labels from: {labels_file}")

    with open(data_file, 'rb') as f:
        all_segments = pickle.load(f)
    with open(labels_file, 'rb') as f:
        all_labels = pickle.load(f)

    qa_dict = {
        "author": "",
        "version": "",
        "date": str(datetime.now().date()),
        "dataset": []
    }

    print(f"\nStarting processing for {len(all_labels)} segments...")
    print("First 10 examples will be printed for verification.")

    start_time = time.time()
    last_save = 0
    error_count = 0
    processed_count = 0
    total_segments = len(all_labels)
    processed_indices = set()

    try:
        for idx, (l, d) in enumerate(tqdm(zip(all_labels, all_segments), total=total_segments)):
            if idx in processed_indices:
                continue
            try:
                assert d.shape[0] == window_size_seconds * sampling_rate
                assert d.shape[1] == 3

                if np.any(np.isnan(d)) or np.all(d == 0, axis=0).any():
                    print(f"\nWarning: Segment {idx} contains NaN or all-zero channel")
                    error_count += 1
                    continue

                window_id = l.get('window_id', idx)
                random.seed(hash(str(window_id)))
                trend_pair_list = select_random_pair(window_size_seconds)
                data_dict = {
                    "index": idx,
                    "qa_pair": QA_gen(l, d, trend_pair_list, window_size_seconds, sampling_rate)
                }
                qa_dict["dataset"].append(data_dict)
                processed_count += 1

                if idx < 10:
                    print(f"\nExample {idx + 1}:")
                    print(data_dict["qa_pair"])

                current_time = time.time()
                elapsed_time = current_time - start_time
                estimated_total = elapsed_time / (idx + 1) * total_segments
                remaining_time = estimated_total - elapsed_time

                if (idx + 1) % 100 == 0:
                    print(f"\nProgress: {(idx+1)/total_segments*100:.2f}% ({idx+1}/{total_segments})")
                    print(f"Processed successfully: {processed_count}")
                    print(f"Errors encountered: {error_count}")
                    print(f"Elapsed: {elapsed_time/60:.1f}min | Remaining: {remaining_time/60:.1f}min")
                    print("-" * 50)

                if idx - last_save >= save_interval:
                    print(f"\nAuto-saving at segment {idx}...")
                    save_qa_data(qa_dict, base_output_path, f"capture24_{split}_qa_stage2_cls.json")
                    last_save = idx

            except Exception as e:
                error_count += 1
                print(f"\nError processing segment {idx}: {str(e)}")
                continue

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Saving current progress...")
    finally:
        print("\nSaving final results...")
        save_qa_data(qa_dict, base_output_path, f"capture24_{split}_qa_stage2_cls.json")

        total_time = time.time() - start_time
        print(f"\nProcessing complete for {split} split!")
        print(f"Total segments processed successfully: {processed_count}/{total_segments}")
        print(f"Total errors encountered: {error_count}")
        print(f"Total time taken: {total_time:.2f}s ({total_time/60:.2f}min)")
        print(f"Average time per segment: {total_time/total_segments:.2f}s")

        print("\nWarning Summary:")
        for warning, count in warning_counts.items():
            print(f"{warning}: {count} times")

def main():
    for split in ["train", "test"]:
        process_upsampled_split(split)
    print("\nAll upsampled splits processed successfully!")

if __name__ == "__main__":
    main() 