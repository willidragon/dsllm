#!/usr/bin/env python
# coding: utf-8

# TEMPORARY IMPLEMENTATION - TESTING ONLY
# This is a modified version that only processes 2500 data points
# for quick testing of the pipeline. To be removed after testing.
# Original implementation processes the full dataset.

# ## Imports

# In[3]:


import pandas as pd  # For data manipulation and CSV handling
import numpy as np   # For numerical operations and array handling
import random       # For generating random window sizes
import os          # For file and directory operations
import pickle      # For saving/loading processed data
import gzip        # For handling gzipped CSV files
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
import json
from collections import Counter
import sys
import yaml
import time

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

original_sampling_rate = config["original_sampling_rate"]
# Accept a list of downsampling rates
if "downsample_factors" in config:
    downsample_factors = config["downsample_factors"]
else:
    downsample_factors = [config["downsample_factor"]]

balance_ratio = config["balance_ratio"]
min_label_fraction = config["min_label_fraction"]
window_size_seconds = config["window_size_seconds"]
data_dir = config.get("data_dir", os.path.join(base_data_dir, "capture24"))
num_participants = config["num_participants"]
train_split = config.get("train_split", 70)
val_split = config.get("val_split", 15)
test_split = config.get("test_split", 15)

# Automatically find all participant CSV files (assume .csv.gz extension)
all_participant_files = [f for f in os.listdir(data_dir) if f.endswith('.csv.gz')]
all_participant_ids = [f.split('.')[0] for f in all_participant_files]

# Randomly select a number of participants
random.seed(42)
num_total = len(all_participant_ids)
num_to_use = min(num_participants, num_total)
selected_participants = random.sample(all_participant_ids, num_to_use)

# Split into train, val, test
num_train = int(num_to_use * train_split / 100)
num_val = int(num_to_use * val_split / 100)
num_test = num_to_use - num_train - num_val
train_participants = selected_participants[:num_train]
val_participants = selected_participants[num_train:num_train+num_val]
test_participants = selected_participants[num_train+num_val:]

print(f"Training participants: {train_participants}")
print(f"Validation participants: {val_participants}")
print(f"Testing participants: {test_participants}")

# In[1]:


def extract_windows_with_min_label_coverage(df, window_size, stride, min_label_fraction=0.5):
    """
    Slide a window across the dataframe and keep windows where the most common label
    covers at least min_label_fraction of the window.
    Returns:
        segments: list of np.arrays of shape (window_size, 3)
        labels: list of dicts with 'activity_category', 'subject', and window indices
    """
    segments = []
    labels = []
    n = len(df)
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        window = df.iloc[start:end]
        if len(window) < window_size:
            continue
        # Count label occurrences
        label_counts = window['activity_category'].value_counts(dropna=True)
        if label_counts.empty:
            continue
        top_label = label_counts.idxmax()
        top_count = label_counts.max()
        if top_count / window_size >= min_label_fraction:
            segment = window[['x', 'y', 'z']].values
            segments.append(np.array(segment))
            labels.append({
                'activity_category': int(top_label),
                'subject': window['subject'].iloc[0],
                'window_indices': [start, end-1]
            })
    return segments, labels


# In[5]:


def split_sequences(sequences, window_size, stride=1):
    # We expect x, y, z columns for Capture24 data
    assert len(sequences[0]) == 3, f"Expected 3 columns (x, y, z) but got {len(sequences[0])}"
    
    # Check for null values
    has_null = any(any(pd.isnull(item) or item == '' for item in sublist) for sublist in sequences)
    if has_null:
        raise ValueError("Has null values")

    # If sequence length is less than window size, return empty lists
    if len(sequences) < window_size:
        print(f"Warning: sequence length {len(sequences)} is less than window size {window_size}")
        return [], []

    segments = []
    labels = []

    # Calculate number of complete segments
    num_complete_segments = (len(sequences) - window_size) // stride + 1

    # Create segments with stride
    for i in range(num_complete_segments):
        start = i * stride
        end = start + window_size
        segment = sequences[start:end]
        assert len(segment) == window_size
        segments.append(np.array(segment))
        labels.append([start, end-1])

    # Add final segment if needed and if we have enough data
    if num_complete_segments > 0 and labels[-1][1] < len(sequences) - 1:
        start = len(sequences) - window_size
        end = len(sequences)
        segment = sequences[start:end]
        assert len(segment) == window_size
        segments.append(np.array(segment))
        labels.append([start, end-1])
    
    assert len(labels) == len(segments)
    print(f"sequence length: {len(sequences)}\nsegments: {len(segments)}")
    pd.set_option('display.max_columns', None)
    print(f"First 5 labels: {labels[:5]}")
    print(f"Last 5 labels: {labels[-5:] if len(labels) >= 5 else labels}")
    
    return segments, labels


# In[ ]:


# Define activity mapping based on Capture24 annotation dictionary
activity_map = {
    'sleep': 1,
    'sitting': 2,
    'standing': 3,
    'walking': 4,
    'bicycling': 5,
    'vehicle': 6,
    'household-chores': 7,
    'manual-work': 8,
    'sports': 9,
    'mixed-activity': 10
}

# Reverse mapping for easy lookup
activity_map_reverse = {v: k for k, v in activity_map.items()}

# Load annotation dictionary
annotation_dict = pd.read_csv(os.path.join(data_dir, 'annotation-label-dictionary.csv'))

# Create mapping from specific activities to main categories
activity_to_category = {}
for _, row in annotation_dict.iterrows():
    main_category = row['label:WillettsSpecific2018']  # Using WillettsSpecific2018 as main category
    if main_category in activity_map:
        activity_to_category[row['annotation']] = activity_map[main_category]



# First pass: count windows per activity per participant
activity_participant_windows = {}  # {activity: {participant_id: [segment indices]}}
activity_total_windows = {}        # {activity: total_windows}

for participant_id in train_participants + val_participants + test_participants:
    try:
        df = pd.read_csv(f'{data_dir}/{participant_id}.csv.gz')
        df = df.iloc[::downsample_factors[0]].reset_index(drop=True)
        df['activity_category'] = df['annotation'].map(activity_to_category)
        df['subject'] = participant_id
        df = df.dropna(subset=['activity_category'])
        df['activity_category'] = df['activity_category'].astype(int)
        # segments, label_dicts = extract_windows_with_min_label_coverage(df, window_size, stride, min_label_fraction=min_label_fraction)
    except FileNotFoundError:
        continue

# Determine the most common activity and set the upper limit
max_windows = max(activity_total_windows.values()) if activity_total_windows else 0
upper_limit_activity = int(max_windows * balance_ratio)
activity_caps = {a: upper_limit_activity if n > upper_limit_activity else n for a, n in activity_total_windows.items()}

# Second pass: build balanced datasets
all_train_segments = []
all_val_segments = []
all_test_segments = []
all_train_labels = []
all_val_labels = []
all_test_labels = []

for split_name, split_participants, all_segments, all_labels in [
    ("train", train_participants, all_train_segments, all_train_labels),
    ("val", val_participants, all_val_segments, all_val_labels),
    ("test", test_participants, all_test_segments, all_test_labels)
]:
    for activity, participant_dict in activity_participant_windows.items():
        # Only consider participants in this split
        split_participant_dict = {pid: segs for pid, segs in participant_dict.items() if pid in split_participants}
        total_participants = len(split_participant_dict)
        if total_participants == 0:
            continue
        cap = activity_caps[activity]
        per_participant_cap = cap // total_participants
        remainder = cap % total_participants
        for i, (pid, segs) in enumerate(split_participant_dict.items()):
            n_take = per_participant_cap + (1 if i < remainder else 0)
            n_take = min(n_take, len(segs))
            selected_segs = segs[:n_take]  # preserve temporal order
            for seg in selected_segs:
                all_segments.append(seg)
                label_dict = {
                    "subject": pid,
                    "activity_name": activity_map_reverse[activity],
                    "activity": activity - 1,
                    "segments": [0, len(seg)-1]  # placeholder, not used downstream
                }
                all_labels.append(label_dict)

# Print results
print(f"\nResults:")
print(f"all_train_segments: {len(all_train_segments)}")
print(f"all_train_labels: {len(all_train_labels)}")
print(f"all_val_segments: {len(all_val_segments)}")
print(f"all_val_labels: {len(all_val_labels)}")
print(f"all_test_segments: {len(all_test_segments)}")
print(f"all_test_labels: {len(all_test_labels)}")

# Print activity distribution safely
if all_train_labels:
    activity_counts = pd.DataFrame(all_train_labels)['activity_name'].value_counts()
    print("\nActivity distribution in training set:")
    print(activity_counts)
else:
    print("\nNo training labels found. Check your config or data.")

# Calculate pre-processed (original) activity spread
pre_activity_spread = {str(activity_map_reverse[a]): int(n) for a, n in activity_total_windows.items()}

# Calculate post-processed (final) activity spread for train and test separately
train_post_activity_counter = Counter()
for label in all_train_labels:
    train_post_activity_counter[label['activity_name']] += 1
train_post_activity_spread = {str(k): int(v) for k, v in train_post_activity_counter.items()}

val_post_activity_counter = Counter()
for label in all_val_labels:
    val_post_activity_counter[label['activity_name']] += 1
val_post_activity_spread = {str(k): int(v) for k, v in val_post_activity_counter.items()}

test_post_activity_counter = Counter()
for label in all_test_labels:
    test_post_activity_counter[label['activity_name']] += 1
test_post_activity_spread = {str(k): int(v) for k, v in test_post_activity_counter.items()}

# Convert activity_caps keys to str for JSON
activity_caps_str = {str(activity_map_reverse[a]): int(n) for a, n in activity_caps.items()}

# Calculate activity pruning summary for train
activity_pruning_summary_train = {}
for activity in pre_activity_spread:
    pre = pre_activity_spread[activity]
    post = train_post_activity_spread.get(activity, 0)
    cap = activity_caps_str.get(activity, None)
    diff = pre - post
    if diff == 0:
        reason = "No pruning"
    elif cap is not None and str(post) == str(cap):
        reason = f"Capped at balance_ratio ({cap})"
    elif post < pre:
        reason = "Removed due to label fraction, windowing, or insufficient data"
    else:
        reason = "Unknown"
    activity_pruning_summary_train[activity] = {
        "pre": pre,
        "post": post,
        "cut": diff,
        "reason": reason
    }
# Calculate activity pruning summary for val
activity_pruning_summary_val = {}
for activity in pre_activity_spread:
    pre = pre_activity_spread[activity]
    post = val_post_activity_spread.get(activity, 0)
    cap = activity_caps_str.get(activity, None)
    diff = pre - post
    if diff == 0:
        reason = "No pruning"
    elif cap is not None and str(post) == str(cap):
        reason = f"Capped at balance_ratio ({cap})"
    elif post < pre:
        reason = "Removed due to label fraction, windowing, or insufficient data"
    else:
        reason = "Unknown"
    activity_pruning_summary_val[activity] = {
        "pre": pre,
        "post": post,
        "cut": diff,
        "reason": reason
    }
# Calculate activity pruning summary for test
activity_pruning_summary_test = {}
for activity in pre_activity_spread:
    pre = pre_activity_spread[activity]
    post = test_post_activity_spread.get(activity, 0)
    cap = activity_caps_str.get(activity, None)
    diff = pre - post
    if diff == 0:
        reason = "No pruning"
    elif cap is not None and str(post) == str(cap):
        reason = f"Capped at balance_ratio ({cap})"
    elif post < pre:
        reason = "Removed due to label fraction, windowing, or insufficient data"
    else:
        reason = "Unknown"
    activity_pruning_summary_test[activity] = {
        "pre": pre,
        "post": post,
        "cut": diff,
        "reason": reason
    }

# --- New: Window first, then downsample for each rate ---

all_rate_windows = {rate: {} for rate in downsample_factors}

for participant_id in train_participants + val_participants + test_participants:
    try:
        df_full = pd.read_csv(f'{data_dir}/{participant_id}.csv.gz')
    except FileNotFoundError:
        continue
    # Map annotation to activity category
    df_full['activity_category'] = df_full['annotation'].map(activity_to_category)
    df_full['subject'] = participant_id
    df_full = df_full.dropna(subset=['activity_category'])
    df_full['activity_category'] = df_full['activity_category'].astype(int)
    # Window the full-rate data
    sampling_rate = original_sampling_rate
    window_size = int(window_size_seconds * sampling_rate)
    stride = max(1, int(window_size * 0.5))
    segments, label_dicts = extract_windows_with_min_label_coverage(df_full, window_size, stride, min_label_fraction=min_label_fraction)
    print(f"[DEBUG] Participant {participant_id}: found {len(segments)} windows at full rate")
    for i, (seg_full, lab) in enumerate(zip(segments, label_dicts)):
        start_idx, end_idx = lab['window_indices']
        start_timestamp = df_full.iloc[start_idx]['timestamp'] if 'timestamp' in df_full.columns else start_idx
        end_timestamp = df_full.iloc[end_idx]['timestamp'] if 'timestamp' in df_full.columns else end_idx
        window_id = (participant_id, start_timestamp, end_timestamp)
        # For each rate, downsample this window
        for rate in downsample_factors:
            ds_seg = seg_full[::rate]
            # If the downsampled window is too short, skip
            if len(ds_seg) < 2:
                continue
            # Copy label dict and update window_id
            lab_ds = lab.copy()
            lab_ds['window_id'] = window_id
            lab_ds['activity_name'] = activity_map_reverse[lab['activity_category']]
            if participant_id not in all_rate_windows[rate]:
                all_rate_windows[rate][participant_id] = {}
            all_rate_windows[rate][participant_id][window_id] = (ds_seg, lab_ds)
        # Print debug info for first 2 windows
        if i < 2:
            print(f"[DEBUG] Window {i} ({window_id}): lengths {[len(seg_full[::r]) for r in downsample_factors]}")

# --- Output: For each rate, collect all segments/labels and save ---
for rate in downsample_factors:
    # Collect train, val, and test segments/labels for this rate
    train_segments = []
    train_labels = []
    val_segments = []
    val_labels = []
    test_segments = []
    test_labels = []
    for participant_id in all_rate_windows[rate]:
        for wid, (seg, lab) in all_rate_windows[rate][participant_id].items():
            # Determine if this participant is in train, val, or test
            if participant_id in train_participants:
                train_segments.append(seg)
                train_labels.append(lab)
            elif participant_id in val_participants:
                val_segments.append(seg)
                val_labels.append(lab)
            elif participant_id in test_participants:
                test_segments.append(seg)
                test_labels.append(lab)
    output_tag = f"{window_size_seconds}seconds_{rate}DS"
    output_path = os.path.join(base_output_dir, "stage_2_compare_buffer", output_tag)
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)
    # Save train segments/labels
    with open(os.path.join(output_path, 'train', f'capture24_train_data_stage2_{output_tag}.pkl'), 'wb') as f:
        pickle.dump(train_segments, f)
    with open(os.path.join(output_path, 'train', f'capture24_train_labels_stage2_{output_tag}.pkl'), 'wb') as f:
        pickle.dump(train_labels, f)
    # Save val segments/labels
    with open(os.path.join(output_path, 'val', f'capture24_val_data_stage2_{output_tag}.pkl'), 'wb') as f:
        pickle.dump(val_segments, f)
    with open(os.path.join(output_path, 'val', f'capture24_val_labels_stage2_{output_tag}.pkl'), 'wb') as f:
        pickle.dump(val_labels, f)
    # Save test segments/labels
    with open(os.path.join(output_path, 'test', f'capture24_test_data_stage2_{output_tag}.pkl'), 'wb') as f:
        pickle.dump(test_segments, f)
    with open(os.path.join(output_path, 'test', f'capture24_test_labels_stage2_{output_tag}.pkl'), 'wb') as f:
        pickle.dump(test_labels, f)
    print(f"[INFO] Saved {len(train_segments)} train, {len(val_segments)} val, and {len(test_segments)} test segments for rate {rate} at {output_path}")

    # Train settings for this rate
    train_settings = {
        "window_size_seconds": window_size_seconds,
        "window_size": window_size,
        "stride": stride,
        "downsample_factor": rate,
        "original_sampling_rate": original_sampling_rate,
        "sampling_rate": sampling_rate,
        "num_participants": num_participants,
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "balance_ratio": balance_ratio,
        "activity_caps": activity_caps_str,
        "pre_activity_spread": pre_activity_spread,
        "post_activity_spread": train_post_activity_spread,
        "all_participants": all_participant_ids,
        "selected_participants": selected_participants,
        "train_participants": train_participants,
        "val_participants": val_participants,
        "test_participants": test_participants,
        "activity_pruning_summary": activity_pruning_summary_train
    }

    # Val settings for this rate
    val_settings = {
        "window_size_seconds": window_size_seconds,
        "window_size": window_size,
        "stride": stride,
        "downsample_factor": rate,
        "original_sampling_rate": original_sampling_rate,
        "sampling_rate": sampling_rate,
        "num_participants": num_participants,
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "balance_ratio": balance_ratio,
        "activity_caps": activity_caps_str,
        "pre_activity_spread": pre_activity_spread,
        "post_activity_spread": val_post_activity_spread,
        "all_participants": all_participant_ids,
        "selected_participants": selected_participants,
        "train_participants": train_participants,
        "val_participants": val_participants,
        "test_participants": test_participants,
        "activity_pruning_summary": activity_pruning_summary_val
    }

    # Test settings for this rate
    test_settings = {
        "window_size_seconds": window_size_seconds,
        "window_size": window_size,
        "stride": stride,
        "downsample_factor": rate,
        "original_sampling_rate": original_sampling_rate,
        "sampling_rate": sampling_rate,
        "num_participants": num_participants,
        "train_split": train_split,
        "val_split": val_split,
        "test_split": test_split,
        "balance_ratio": balance_ratio,
        "activity_caps": activity_caps_str,
        "pre_activity_spread": pre_activity_spread,
        "post_activity_spread": test_post_activity_spread,
        "all_participants": all_participant_ids,
        "selected_participants": selected_participants,
        "train_participants": train_participants,
        "val_participants": val_participants,
        "test_participants": test_participants,
        "activity_pruning_summary": activity_pruning_summary_test
    }

    # Save settings for this rate
    with open(os.path.join(output_path, 'train', 'settings.json'), 'w') as f:
        json.dump(train_settings, f, indent=2)
    with open(os.path.join(output_path, 'val', 'settings.json'), 'w') as f:
        json.dump(val_settings, f, indent=2)
    with open(os.path.join(output_path, 'test', 'settings.json'), 'w') as f:
        json.dump(test_settings, f, indent=2)

print("\nData saved successfully for all downsampling rates!")

if __name__ == "__main__":
    if "--print-output-dir" in sys.argv:
        output_tag = f"{window_size_seconds}seconds_{rate}DS"
        output_path = os.path.join(base_output_dir, "stage_2_compare_buffer", output_tag)
        print(output_path)
