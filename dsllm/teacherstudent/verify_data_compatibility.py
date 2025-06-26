#!/usr/bin/env python3
"""
Verify Data Compatibility - Compare Original 100DS vs Upsampled Data

This script checks that the upsampled data has EXACTLY the same format,
shapes, and specifications as the original 100DS data for SensorLLM compatibility.
"""

import pickle
import numpy as np
import json
from pathlib import Path

def compare_data_formats():
    print('🔍 COMPARING ORIGINAL 100DS vs UPSAMPLED DATA')
    print('='*60)
    
    # Paths to original 100DS data
    orig_base_path = Path("/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_compare/300seconds_100DS")
    ups_base_path = Path("/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_upsampled/300seconds_100DS_upsampled")
    
    for split in ["train", "test"]:
        print(f"\n📋 COMPARING {split.upper()} SPLIT:")
        print("-" * 40)
        
        # Load original 100DS data
        orig_data_path = orig_base_path / split / f"capture24_{split}_data_stage2_300seconds_100DS.pkl"
        orig_labels_path = orig_base_path / split / f"capture24_{split}_labels_stage2_300seconds_100DS.pkl"
        orig_settings_path = orig_base_path / split / "settings.json"
        
        # Load upsampled data
        ups_data_path = ups_base_path / split / f"capture24_{split}_data_stage2_300seconds_100DS_upsampled.pkl"
        ups_labels_path = ups_base_path / split / f"capture24_{split}_labels_stage2_300seconds_100DS_upsampled.pkl"
        ups_settings_path = ups_base_path / split / "settings.json"
        
        # Check if files exist
        if not orig_data_path.exists():
            print(f"❌ Original {split} data not found: {orig_data_path}")
            continue
        if not ups_data_path.exists():
            print(f"❌ Upsampled {split} data not found: {ups_data_path}")
            continue
            
        # Load data
        with open(orig_data_path, 'rb') as f:
            orig_data = pickle.load(f)
        with open(orig_labels_path, 'rb') as f:
            orig_labels = pickle.load(f)
        with open(orig_settings_path, 'r') as f:
            orig_settings = json.load(f)
            
        with open(ups_data_path, 'rb') as f:
            ups_data = pickle.load(f)
        with open(ups_labels_path, 'rb') as f:
            ups_labels = pickle.load(f)
        with open(ups_settings_path, 'r') as f:
            ups_settings = json.load(f)
        
        # Compare data formats
        print(f"📊 DATA COMPARISON:")
        print(f"   Original samples: {len(orig_data)}")
        print(f"   Upsampled samples: {len(ups_data)}")
        print(f"   ✅ Sample count match: {len(orig_data) == len(ups_data)}")
        
        if orig_data and ups_data:
            print(f"   Original shape: {orig_data[0].shape}")
            print(f"   Upsampled shape: {ups_data[0].shape}")
            print(f"   ✅ Shape match: {orig_data[0].shape == ups_data[0].shape}")
            
            print(f"   Original dtype: {orig_data[0].dtype}")
            print(f"   Upsampled dtype: {ups_data[0].dtype}")
            print(f"   ✅ Dtype match: {orig_data[0].dtype == ups_data[0].dtype}")
            
            print(f"   Original type: {type(orig_data[0])}")
            print(f"   Upsampled type: {type(ups_data[0])}")
            print(f"   ✅ Type match: {type(orig_data[0]) == type(ups_data[0])}")
        
        # Compare labels format
        print(f"\n🏷️  LABELS COMPARISON:")
        print(f"   Original labels: {len(orig_labels)}")
        print(f"   Upsampled labels: {len(ups_labels)}")
        print(f"   ✅ Label count match: {len(orig_labels) == len(ups_labels)}")
        
        if orig_labels and ups_labels:
            orig_keys = set(orig_labels[0].keys())
            ups_keys = set(ups_labels[0].keys())
            
            print(f"   Original keys: {sorted(orig_keys)}")
            print(f"   Upsampled keys: {sorted(ups_keys)}")
            
            # Core keys that must match
            core_keys = {'activity_category', 'subject', 'window_indices', 'activity_name'}
            orig_core = orig_keys.intersection(core_keys)
            ups_core = ups_keys.intersection(core_keys)
            
            print(f"   ✅ Core keys match: {orig_core == ups_core}")
            
            # Check if upsampled has extra metadata (which is expected)
            extra_keys = ups_keys - orig_keys
            if extra_keys:
                print(f"   📝 Extra metadata in upsampled: {sorted(extra_keys)}")
        
        # Data value comparison (sample a few)
        print(f"\n📈 DATA VALUES COMPARISON:")
        if orig_data and ups_data and len(orig_data) > 0 and len(ups_data) > 0:
            # Compare data ranges
            orig_min = np.min([np.min(seg) for seg in orig_data[:10]])
            orig_max = np.max([np.max(seg) for seg in orig_data[:10]])
            ups_min = np.min([np.min(seg) for seg in ups_data[:10]])
            ups_max = np.max([np.max(seg) for seg in ups_data[:10]])
            
            print(f"   Original range: [{orig_min:.4f}, {orig_max:.4f}]")
            print(f"   Upsampled range: [{ups_min:.4f}, {ups_max:.4f}]")
            
            # Check if ranges are reasonable (upsampled should be in similar range)
            range_reasonable = (abs(orig_min - ups_min) < 2.0) and (abs(orig_max - ups_max) < 2.0)
            print(f"   ✅ Value ranges reasonable: {range_reasonable}")
        
        print()
    
    # Overall compatibility check
    print("🎯 SENSORLLM COMPATIBILITY CHECK:")
    print("-" * 40)
    
    # Load one sample from each for final verification
    try:
        with open(orig_base_path / "train" / "capture24_train_data_stage2_300seconds_100DS.pkl", 'rb') as f:
            orig_sample = pickle.load(f)[0]
        with open(ups_base_path / "train" / "capture24_train_data_stage2_300seconds_100DS_upsampled.pkl", 'rb') as f:
            ups_sample = pickle.load(f)[0]
        
        shape_match = orig_sample.shape == ups_sample.shape
        dtype_match = orig_sample.dtype == ups_sample.dtype
        type_match = type(orig_sample) == type(ups_sample)
        
        print(f"✅ Tensor shapes compatible: {shape_match}")
        print(f"✅ Data types compatible: {dtype_match}")
        print(f"✅ Object types compatible: {type_match}")
        
        if shape_match and dtype_match and type_match:
            print(f"\n🎉 SUCCESS: Upsampled data is 100% compatible with SensorLLM!")
            print(f"   Shape: {ups_sample.shape}")
            print(f"   Dtype: {ups_sample.dtype}")
            print(f"   Type: {type(ups_sample)}")
        else:
            print(f"\n❌ WARNING: Compatibility issues detected!")
            
    except Exception as e:
        print(f"❌ Error during compatibility check: {e}")

if __name__ == "__main__":
    compare_data_formats() 