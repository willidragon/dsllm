import os
import numpy as np
import pickle
import json
from datetime import datetime
import random
from process_capture24_stage2_2_custom_mins_multiple import select_random_pair, QA_gen
import time
from tqdm import tqdm
import warnings
from collections import defaultdict
import shutil
import yaml

# Load base paths from paths.yaml in project root
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paths_config = os.path.join(root_dir, 'paths.yaml')
with open(paths_config, 'r') as f:
    paths = yaml.safe_load(f)
base_data_dir = paths['base_data_dir']
base_output_dir = paths['base_output_dir']

# Load config from YAML file
config_path = os.path.join(os.path.dirname(__file__), 'process_capture24_stage2_compare.yaml')
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Get settings from config
window_size_seconds = config["window_size_seconds"]
original_sampling_rate = config["original_sampling_rate"]
downsample_factors = config["downsample_factors"]

# Constants
save_interval = 100  # Save every 100 segments

# Set up warning tracking
warning_counts = defaultdict(int)
def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
    warning_key = f"{category.__name__}: {str(message)}"
    warning_counts[warning_key] += 1
    if warning_counts[warning_key] <= 5:  # Only print first 5 occurrences
        print(f"\nWarning #{warning_counts[warning_key]} - {warning_key}")
        print(f"In file: {filename}, line {lineno}")

warnings.showwarning = custom_warning_handler

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

def process_rate(rate):
    """
    Process one downsampling rate and generate QA pairs for both train and test splits
    
    Note: Fixed data quality check to handle sleep data where one channel becomes zero due to 
    downsampling (e.g., x-axis during minimal movement periods). Only skips data with 2+ 
    all-zero channels or completely zero data to prevent data/QA count mismatches.
    """
    print(f"\n{'='*80}")
    print(f"Processing downsampling rate: {rate}")
    print(f"{'='*80}")
    
    # Set up paths for this rate
    output_tag = f"{window_size_seconds}seconds_{rate}DS"
    base_output_path = os.path.join(base_output_dir, "stage_2_compare_buffer", output_tag)
    
    # Calculate window size for this rate
    sampling_rate = original_sampling_rate / rate
    window_size = int(window_size_seconds * sampling_rate)
    
    for split in ["train", "val", "test"]:
        output_path = os.path.join(base_output_path, split)
        data_file = os.path.join(output_path, f"capture24_{split}_data_stage2_{output_tag}.pkl")
        labels_file = os.path.join(output_path, f"capture24_{split}_labels_stage2_{output_tag}.pkl")
        if not os.path.exists(data_file) or not os.path.exists(labels_file):
            print(f"Skipping {split} split for rate {rate}: files not found.")
            continue
        print(f"\nChecking data paths...")
        print(f"Loading data from: {data_file}")
        print(f"Loading labels from: {labels_file}")

        # Load the data
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
            # Use tqdm for progress bar
            for idx, (l, d) in enumerate(tqdm(zip(all_labels, all_segments), total=total_segments)):
                if idx in processed_indices:
                    continue  # Skip already processed segments
                try:
                    assert d.shape[0] == window_size  # Use window_size variable for assertion
                    assert d.shape[1] == 3  # Capture24 has 3 channels (x, y, z)
                    
                    # Check for NaN or invalid data
                    if np.any(np.isnan(d)):
                        print(f"\nWarning: Segment {idx} contains NaN values")
                        error_count += 1
                        continue
                    
                    # More sophisticated check for problematic data:
                    # Only skip if ALL data is zero or if 2+ channels are all zero
                    # This allows valid sleep data where one channel (e.g., x-axis) becomes zero due to downsampling
                    num_all_zero_channels = np.sum(np.all(d == 0, axis=0))
                    if np.all(d == 0) or num_all_zero_channels >= 2:
                        print(f"\nWarning: Segment {idx} has {num_all_zero_channels} all-zero channels or all data is zero")
                        error_count += 1
                        continue
                    
                    # Log single all-zero channels for monitoring (but don't skip)
                    if num_all_zero_channels == 1:
                        zero_channels = [i for i, is_zero in enumerate(np.all(d == 0, axis=0)) if is_zero]
                        channel_names = ['x-axis', 'y-axis', 'z-axis']
                        if idx < 20 or idx % 1000 == 0:  # Only log occasionally to avoid spam
                            print(f"\nInfo: Segment {idx} has all-zero {channel_names[zero_channels[0]]} (acceptable for sleep data)")
                    
                    # --- Deterministic randomization per window ---
                    window_id = l.get('window_id', idx)  # fallback to idx if not present
                    random.seed(hash(str(window_id)))
                    trend_pair_list = select_random_pair(window_size_seconds)
                    data_dict = {
                        "index": idx,
                        "qa_pair": QA_gen(l, d, trend_pair_list, window_size_seconds, sampling_rate)
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
            print(f"\nProcessing complete for {split} split, rate {rate}!")
            print(f"Total segments processed successfully: {processed_count}/{total_segments}")
            print(f"Total errors encountered: {error_count}")
            print(f"Total time taken: {total_time:.2f}s ({total_time/60:.2f}min)")
            print(f"Average time per segment: {total_time/total_segments:.2f}s")

            print("\nWarning Summary:")
            for warning, count in warning_counts.items():
                print(f"{warning}: {count} times")

def main():
    """Main function to process all downsampling rates"""
    print(f"Processing {len(downsample_factors)} downsampling rates: {downsample_factors}")
    
    for rate in downsample_factors:
        process_rate(rate)
    
    print("\nAll rates processed successfully!")

if __name__ == "__main__":
    main()
