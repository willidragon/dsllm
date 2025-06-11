import os
import numpy as np
import pickle
import json
from datetime import datetime
import random
from process_capture24_stage2_2_custom_mins import select_random_pair, QA_gen
import time
from tqdm import tqdm
import warnings
from collections import defaultdict
import shutil
from process_capture24_stage2_1_custom_mins import window_size, sampling_rate, output_tag

# Constants
save_interval = 100  # Save every 2500 segments

# Set up warning tracking
warning_counts = defaultdict(int)
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    warning_key = f"{category.__name__}: {str(message)}"
    warning_counts[warning_key] += 1
    if warning_counts[warning_key] <= 5:  # Only print first 5 occurrences
        print(f"\nWarning #{warning_counts[warning_key]} - {warning_key}")
        print(f"In file: {filename}, line {lineno}")

warnings.showwarning = custom_warning_handler

# Set up output paths
base_output_path = f"/home/willidragon/william-research/sensorllm/sensorllm/data/stage_2/{output_tag}"
split = "train"
output_path = os.path.join(base_output_path, split)

# Create directories if they don't exist
os.makedirs(base_output_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)
print(f"Saving data to: {output_path}")

def save_qa_data(qa_dict, path, filename):
    """Helper function to save QA data with error handling and atomic write"""
    full_path = os.path.join(path, filename)
    temp_path = full_path + ".tmp"
    try:
        with open(temp_path, 'w') as f:
            json.dump(qa_dict, f, indent=2)
        shutil.move(temp_path, full_path)  # Atomic move
        print(f"Successfully saved to: {full_path}")
        print(f"File size: {os.path.getsize(full_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Error saving to {full_path}: {str(e)}")
        # Try saving to current directory as backup
        backup_path = os.path.join(os.getcwd(), f"backup_{filename}")
        print(f"Attempting to save backup to: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(qa_dict, f, indent=2)
        print(f"Backup saved successfully to: {backup_path}")

qa_file = os.path.join(
    f"/home/willidragon/william-research/sensorllm/sensorllm/data/stage_2/",
    __import__('process_capture24_stage2_1_custom_mins').output_tag,
    'train',
    'capture24_train_qa_stage2_cls.json'
)

qa_dict = {
    "author": "",
    "version": "",
    "date": str(datetime.now().date()),
    "dataset": []
}

print("\nChecking data paths...")
data_file = os.path.join(output_path, f"capture24_{split}_data_stage2_{output_tag}.pkl")
labels_file = os.path.join(output_path, f"capture24_{split}_labels_stage2_{output_tag}.pkl")
print(f"Loading data from: {data_file}")
print(f"Loading labels from: {labels_file}")

# Load the data
with open(data_file, 'rb') as f:
    all_train_segments = pickle.load(f)
with open(labels_file, 'rb') as f:
    all_train_labels = pickle.load(f)

print(f"\nStarting processing for {len(all_train_labels)} segments...")
print("First 10 examples will be printed for verification.")

start_time = time.time()
last_save = 0
error_count = 0
processed_count = 0
total_segments = len(all_train_labels)

processed_indices = set()

try:
    # Use tqdm for progress bar
    for idx, (l, d) in enumerate(tqdm(zip(all_train_labels, all_train_segments), total=total_segments)):
        if idx in processed_indices:
            continue  # Skip already processed segments
        try:
            assert d.shape[0] == window_size  # Use window_size variable for assertion
            assert d.shape[1] == 3  # Capture24 has 3 channels (x, y, z)
            
            # Check for NaN or zero values in the data
            if np.any(np.isnan(d)) or np.all(d == 0, axis=0).any():
                print(f"\nWarning: Segment {idx} contains NaN or all-zero channel")
                error_count += 1
                continue
                
            trend_pair_list = select_random_pair()
            data_dict = {
                "index": idx,
                "qa_pair": QA_gen(l, d, trend_pair_list)
            }
            qa_dict["dataset"].append(data_dict)
            processed_count += 1
            
            # Print first 10 examples
            if idx < 10:
                print(f"\nExample {idx + 1}:")
                print(data_dict["qa_pair"])
            
            # Calculate and print progress
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
            
            # Auto-save every save_interval segments
            if idx - last_save >= save_interval:
                print(f"\nAuto-saving at segment {idx}...")
                save_qa_data(qa_dict, output_path, f"capture24_{split}_qa_stage2_cls.json")
                last_save = idx
                
        except Exception as e:
            error_count += 1
            print(f"\nError processing segment {idx}: {str(e)}")
            continue

except KeyboardInterrupt:
    print("\nProcess interrupted by user. Saving current progress...")
finally:
    # Save final results
    print("\nSaving final results...")
    save_qa_data(qa_dict, output_path, f"capture24_{split}_qa_stage2_cls.json")

    total_time = time.time() - start_time
    print(f"\nProcessing complete!")
    print(f"Total segments processed successfully: {processed_count}/{total_segments}")
    print(f"Total errors encountered: {error_count}")
    print(f"Total time taken: {total_time:.2f}s ({total_time/60:.2f}min)")
    print(f"Average time per segment: {total_time/total_segments:.2f}s")

    print("\nWarning Summary:")
    for warning, count in warning_counts.items():
        print(f"{warning}: {count} times")
