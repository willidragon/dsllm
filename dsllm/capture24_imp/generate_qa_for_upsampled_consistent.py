#!/usr/bin/env python3
"""
Generate QA JSON for upsampled data (100DS_upsampled) with consistent structure to baseline 100DS QA.
"""
import os
import numpy as np
import pickle
import json
from datetime import datetime
from collections import defaultdict
import shutil
import yaml
import sys
from tqdm import tqdm
import random
import re

# Load config for this script
config_path = os.path.join(os.path.dirname(__file__), 'generate_qa_for_upsampled_consistent_config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

base_output_dir = config['base_output_dir']

upsampled_tag = config['upsampled_tag']
upsampled_dir = config['upsampled_dir'].format(base_output_dir=base_output_dir, upsampled_tag=upsampled_tag)
splits = config['splits']

# Settings for upsampled data
window_size_seconds = 300
sampling_rate = 1  # 100DS_upsampled is 1Hz
save_interval = 100

# Import QA_gen and select_random_pair for dynamic trend_text generation
sys.path.append(os.path.dirname(__file__))
from process_capture24_stage2_2_custom_mins_multiple import select_random_pair, QA_gen

def round_to_8_decimals(number):
    return f'{number:.8f}'.rstrip('0').rstrip('.')

def get_correlation_description(corr_val):
    if corr_val >= 0.7:
        return "strongly positively correlated"
    elif corr_val >= 0.3:
        return "moderately positively correlated"
    elif corr_val >= 0.1:
        return "weakly positively correlated"
    elif corr_val <= -0.7:
        return "strongly negatively correlated"
    elif corr_val <= -0.3:
        return "moderately negatively correlated"
    elif corr_val <= -0.1:
        return "weakly negatively correlated"
    else:
        return "not significantly correlated"

def recalculate_qa_statistics(template_qa, data_array, label, idx):
    # Extract channels
    acc_x = data_array[:, 0]
    acc_y = data_array[:, 1]
    acc_z = data_array[:, 2]
    # Recalculate summary statistics
    new_smry_lines = ["Statistics for each channel:\n"]
    for r, n in zip([acc_x, acc_y, acc_z], ["x-axis accelerometer", "y-axis accelerometer", "z-axis accelerometer"]):
        mean = np.mean(r)
        std_dev = np.std(r)
        new_smry_lines.append(f"{n}: Mean={round_to_8_decimals(mean)}, StdDev={round_to_8_decimals(std_dev)}\n")
    new_smry = ' '.join(new_smry_lines)
    # Recalculate correlation matrix
    correlation_matrix = np.corrcoef([acc_x, acc_y, acc_z])
    new_corr_text = f"""Pearson Correlation Matrix for each channel:\nThe correlation between x-axis accelerometer and y-axis accelerometer is {get_correlation_description(correlation_matrix[0,1])}.\nThe correlation between x-axis accelerometer and z-axis accelerometer is {get_correlation_description(correlation_matrix[0,2])}.\nThe correlation between y-axis accelerometer and z-axis accelerometer is {get_correlation_description(correlation_matrix[1,2])}.\n"""
    # Set random seed for consistency
    window_id = label.get('window_id', idx) if isinstance(label, dict) else idx
    random.seed(hash(str(window_id)))
    trend_pair_list = select_random_pair(window_size_seconds)
    qa_gen_result = QA_gen(label, data_array, trend_pair_list, window_size_seconds, sampling_rate)
    new_trend_text = qa_gen_result.get('trend_text', template_qa.get('trend_text', ''))
    # Create new QA pair with Q/A from baseline, all else dynamic
    new_qa = {
        "Q": template_qa["Q"],
        "smry": new_smry,
        "trend_text": new_trend_text,
        "corr_text": new_corr_text,
        "A": template_qa["A"]
    }
    return new_qa

def process_upsampled_split_consistent(split):
    print(f"\n{'='*80}")
    print(f"Processing upsampled {split} split (consistent QA)")
    print(f"{'='*80}")
    base_output_path = os.path.join(upsampled_dir, split)
    data_file = os.path.join(base_output_path, f"capture24_{split}_data_stage2_{upsampled_tag}.pkl")
    labels_file = os.path.join(base_output_path, f"capture24_{split}_labels_stage2_{upsampled_tag}.pkl")
    # If upsampled_tag contains '_upsampled_from_', use a dynamic suffix ending with _upsampled.pkl
    if '_upsampled_from_' in upsampled_tag:
        base_tag = re.sub(r'_upsampled_from_.*DS$', '_upsampled', upsampled_tag)
        data_file = os.path.join(base_output_path, f"capture24_{split}_data_stage2_{base_tag}.pkl")
        labels_file = os.path.join(base_output_path, f"capture24_{split}_labels_stage2_{base_tag}.pkl")
    # Load the correct baseline QA file for this split
    baseline_qa_path = config['baseline_qa_paths'][split].format(base_output_dir=base_output_dir)
    with open(baseline_qa_path, 'r') as f:
        baseline_qa_data = json.load(f)
    baseline_qa_dict = {int(entry['index']): entry['qa_pair'] for entry in baseline_qa_data['dataset']}
    print(f"Looking for data file: {data_file}")
    print(f"Looking for labels file: {labels_file}")
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
    for idx, (l, d) in tqdm(list(enumerate(zip(all_labels, all_segments))), total=len(all_labels), desc=f"Processing {split}"):
        if idx not in baseline_qa_dict:
            print(f"Warning: No baseline QA for index {idx}, skipping.")
            continue
        template_qa = baseline_qa_dict[idx]
        new_qa = recalculate_qa_statistics(template_qa, np.array(d), l, idx)
        qa_dict["dataset"].append({
            "index": idx,
            "qa_pair": new_qa
        })
        if idx < 10:
            print(f"\nExample {idx + 1}:")
            print(new_qa)
    # Save QA file
    qa_path = os.path.join(base_output_path, f"capture24_{split}_qa_stage2_cls.json")
    with open(qa_path, 'w') as f:
        json.dump(qa_dict, f, indent=2)
    print(f"Saved consistent QA file to: {qa_path}")

def main():
    for split in splits:
        process_upsampled_split_consistent(split)
    print("\nAll upsampled splits processed with consistent QA!")

if __name__ == "__main__":
    main() 