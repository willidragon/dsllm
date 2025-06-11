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

# Load config from YAML file
config_path = os.path.join(os.path.dirname(__file__), 'config_stage2.yaml')
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

original_sampling_rate = config["original_sampling_rate"]
downsample_factor = config["downsample_factor"]
sampling_rate = original_sampling_rate / downsample_factor  # Use float division for sub-Hz rates
balance_ratio = config["balance_ratio"]
min_label_fraction = config["min_label_fraction"]
window_size_seconds = config["window_size_seconds"]
# Ensure stride is at least 1
window_size = int(window_size_seconds * sampling_rate)
stride = max(1, int(window_size * 0.5))
data_dir = config.get("data_dir", '/home/willidragon/william-research/datasets/capture24')
num_participants = config["num_participants"]
train_test_split = config["train_test_split"]

# Automatically find all participant CSV files (assume .csv.gz extension)
all_participant_files = [f for f in os.listdir(data_dir) if f.endswith('.csv.gz')]
all_participant_ids = [f.split('.')[0] for f in all_participant_files]

# Randomly select a number of participants
random.seed(42)
num_total = len(all_participant_ids)
num_to_use = min(num_participants, num_total)
selected_participants = random.sample(all_participant_ids, num_to_use)

# Split into train and test
num_train = max(1, int(len(selected_participants) * train_test_split / 100))
train_participants = selected_participants[:num_train]
test_participants = selected_participants[num_train:]

print(f"Training participants: {train_participants}")
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
annotation_dict = pd.read_csv('/home/willidragon/william-research/datasets/capture24/annotation-label-dictionary.csv')

# Create mapping from specific activities to main categories
activity_to_category = {}
for _, row in annotation_dict.iterrows():
    main_category = row['label:WillettsSpecific2018']  # Using WillettsSpecific2018 as main category
    if main_category in activity_map:
        activity_to_category[row['annotation']] = activity_map[main_category]



# First pass: count windows per activity per participant
activity_participant_windows = {}  # {activity: {participant_id: [segment indices]}}
activity_total_windows = {}        # {activity: total_windows}

for participant_id in train_participants + test_participants:
    try:
        df = pd.read_csv(f'/home/willidragon/william-research/datasets/capture24/{participant_id}.csv.gz')
        df = df.iloc[::downsample_factor].reset_index(drop=True)
        df['activity_category'] = df['annotation'].map(activity_to_category)
        df['subject'] = participant_id
        df = df.dropna(subset=['activity_category'])
        df['activity_category'] = df['activity_category'].astype(int)
        # Use the new function here
        segments, label_dicts = extract_windows_with_min_label_coverage(df, window_size, stride, min_label_fraction=min_label_fraction)
        for seg, label_dict in zip(segments, label_dicts):
            activity = label_dict['activity_category']
            if activity not in activity_participant_windows:
                activity_participant_windows[activity] = {}
            if participant_id not in activity_participant_windows[activity]:
                activity_participant_windows[activity][participant_id] = []
            activity_participant_windows[activity][participant_id].append(seg)
            activity_total_windows[activity] = activity_total_windows.get(activity, 0) + 1
    except FileNotFoundError:
        continue

# Determine the most common activity and set the upper limit
max_windows = max(activity_total_windows.values()) if activity_total_windows else 0
upper_limit_activity = int(max_windows * balance_ratio)
activity_caps = {a: upper_limit_activity if n > upper_limit_activity else n for a, n in activity_total_windows.items()}

# Second pass: build balanced datasets
all_train_segments = []
all_test_segments = []
all_train_labels = []
all_test_labels = []

for split_name, split_participants, all_segments, all_labels in [
    ("train", train_participants, all_train_segments, all_train_labels),
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

# Output directory and file naming
output_tag = f"{window_size_seconds}seconds_{downsample_factor}DS"
output_path = f"/home/willidragon/william-research/sensorllm/sensorllm/data/stage_2/{output_tag}"

# Create output directories if they don't exist
os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'test'), exist_ok=True)

with open(os.path.join(output_path, 'train', f'capture24_train_data_stage2_{output_tag}.pkl'), 'wb') as f:
    pickle.dump(all_train_segments, f)

with open(os.path.join(output_path, 'test', f'capture24_test_data_stage2_{output_tag}.pkl'), 'wb') as f:
    pickle.dump(all_test_segments, f)

with open(os.path.join(output_path, 'train', f'capture24_train_labels_stage2_{output_tag}.pkl'), 'wb') as f:
    pickle.dump(all_train_labels, f)

with open(os.path.join(output_path, 'test', f'capture24_test_labels_stage2_{output_tag}.pkl'), 'wb') as f:
    pickle.dump(all_test_labels, f)

# Train settings
train_settings = {
    "window_size_seconds": window_size_seconds,
    "window_size": window_size,
    "stride": stride,
    "downsample_factor": downsample_factor,
    "original_sampling_rate": original_sampling_rate,
    "sampling_rate": sampling_rate,
    "num_participants": num_participants,
    "train_test_split": train_test_split,
    "balance_ratio": balance_ratio,
    "activity_caps": activity_caps_str,
    "pre_activity_spread": pre_activity_spread,
    "post_activity_spread": train_post_activity_spread,
    "all_participants": all_participant_ids,
    "selected_participants": selected_participants,
    "train_participants": train_participants,
    "test_participants": test_participants,
    "activity_pruning_summary": activity_pruning_summary_train
}

# Test settings
test_settings = {
    "window_size_seconds": window_size_seconds,
    "window_size": window_size,
    "stride": stride,
    "downsample_factor": downsample_factor,
    "original_sampling_rate": original_sampling_rate,
    "sampling_rate": sampling_rate,
    "num_participants": num_participants,
    "train_test_split": train_test_split,
    "balance_ratio": balance_ratio,
    "activity_caps": activity_caps_str,
    "pre_activity_spread": pre_activity_spread,
    "post_activity_spread": test_post_activity_spread,
    "all_participants": all_participant_ids,
    "selected_participants": selected_participants,
    "train_participants": train_participants,
    "test_participants": test_participants,
    "activity_pruning_summary": activity_pruning_summary_test
}

with open(os.path.join(output_path, 'train', 'settings.json'), 'w') as f:
    json.dump(train_settings, f, indent=2)
with open(os.path.join(output_path, 'test', 'settings.json'), 'w') as f:
    json.dump(test_settings, f, indent=2)

print("\nData saved successfully!")

if __name__ == "__main__":
    if "--print-output-dir" in sys.argv:
        output_tag = f"{window_size_seconds}seconds_{downsample_factor}DS"
        output_path = f"/home/willidragon/william-research/sensorllm/sensorllm/data/stage_2/{output_tag}"
        print(output_path)
