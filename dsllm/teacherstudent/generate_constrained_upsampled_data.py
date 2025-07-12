#!/usr/bin/env python
"""
Generate Upsampled Data Using Constrained Enhancement Model

This script:
1. Loads the trained constrained enhancement model (with constraint logic)
2. Loads low-res data (e.g., 1000DS)
3. Uses the model to upsample to high-res equivalent quality (e.g., 100DS)
4. Saves the upsampled data to stage_2_upsampled_constraint folder
"""

import sys
import os
from pathlib import Path
import pickle
import json
import yaml
import numpy as np
import torch
import argparse

# Add the current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from run_data_enhancement_constrained import ConstrainedEnhancementModel
from teacher_student_model import DataEnhancementConfig

def load_paths():
    """Load base paths from paths.yaml"""
    paths_file = current_dir.parent.parent / "paths.yaml"
    if not paths_file.exists():
        print(f"⚠️  Warning: paths.yaml not found at {paths_file}")
        return None
    with open(paths_file) as f:
        paths = yaml.safe_load(f)
    return paths

def load_constrained_enhancement_model(model_path: str, config: DataEnhancementConfig):
    """Load the trained constrained enhancement model (with constraint logic)"""
    print(f"🏗️  Loading constrained enhancement model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=config.device)
    # Detect label conditioning
    num_classes = checkpoint.get('num_classes', 10) if 'num_classes' in checkpoint else 10
    label_embed_dim = checkpoint.get('label_embed_dim', 16) if 'label_embed_dim' in checkpoint else 16
    model = ConstrainedEnhancementModel(config, num_classes=num_classes, label_embed_dim=label_embed_dim).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Constrained enhancement model loaded successfully!")
    return model

def enhance_data_batch(model, low_res_data, batch_size=32, device='cpu'):
    """Enhance data in batches to avoid memory issues"""
    enhanced_segments = []
    with torch.no_grad():
        for i in range(0, len(low_res_data), batch_size):
            batch = low_res_data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(np.array(batch)).to(device)
            enhanced_batch = model(batch_tensor)
            enhanced_batch_np = enhanced_batch.cpu().numpy().astype(np.float64)
            enhanced_segments.extend(enhanced_batch_np)
    return enhanced_segments

def generate_constrained_upsampled_data():
    parser = argparse.ArgumentParser(description="Generate upsampled data using constrained enhancement model.")
    parser.add_argument('--input_ds', type=int, default=1000, help='Input downsampling rate (e.g., 1000 for 1000DS)')
    parser.add_argument('--output_ds', type=int, default=100, help='Output (target) downsampling rate (e.g., 100 for 100DS)')
    parser.add_argument('--sequence_length', type=int, default=300, help='Number of timesteps for output (default: 300)')
    parser.add_argument('--data_tag', type=str, default='300seconds', help='Data tag prefix (default: 300seconds)')
    parser.add_argument('--model_path', type=str, 
                       default='/project/cc-20250120231604/ssd/users/kwsu/data/trained_model/enhancement_model/constrained_enhancement_10x_20250709_042111/constrained_enhancement_model_10x.pth',
                       help='Path to trained constrained enhancement model (.pth)')
    args = parser.parse_args()

    input_ds = args.input_ds
    output_ds = args.output_ds
    sequence_length = args.sequence_length
    data_tag = args.data_tag
    upsampling_factor = input_ds // output_ds

    print("🚀 Starting Constrained Upsampled Data Generation")
    print("=" * 60)
    print(f"PURPOSE: Use constrained enhancement model to upsample {input_ds}DS → {output_ds}DS equivalent")
    print(f"OUTPUT: Save to stage_2_upsampled_constraint folder")
    print("=" * 60)
    print(f"Input DS: {input_ds}")
    print(f"Output DS: {output_ds}")
    print(f"Upsampling factor: {upsampling_factor}x")
    print(f"Sequence length (output): {sequence_length}")
    print(f"Data tag: {data_tag}")
    print(f"Model path: {args.model_path}")

    # Load paths
    paths = load_paths()
    if paths is None:
        print("❌ Error: Could not load paths.yaml")
        return
    base_output_dir = Path(paths['base_output_dir'])
    input_base_path = base_output_dir / "stage_2_compare_buffer" / f"{data_tag}_{input_ds}DS"
    output_base_path = base_output_dir / "stage_2_upsampled_constraint" / f"{data_tag}_{output_ds}DS_upsampled_from_{input_ds}DS"
    output_base_path.mkdir(parents=True, exist_ok=True)
    (output_base_path / "train").mkdir(exist_ok=True)
    (output_base_path / "test").mkdir(exist_ok=True)
    (output_base_path / "val").mkdir(exist_ok=True)
    print(f"📂 Paths:")
    print(f"   Input ({input_ds}DS): {input_base_path}")
    print(f"   Output (constrained upsampled): {output_base_path}")
    print()
    high_res_timesteps = sequence_length
    low_res_timesteps = sequence_length // upsampling_factor
    print(f"Timesteps: {low_res_timesteps} (input) → {high_res_timesteps} (output)")
    config = DataEnhancementConfig(
        high_res_timesteps=high_res_timesteps,
        low_res_timesteps=low_res_timesteps,
        high_res_path=str(input_base_path / "train" / f"capture24_train_data_stage2_{data_tag}_{input_ds}DS.pkl"),
        low_res_path=str(input_base_path / "train" / f"capture24_train_data_stage2_{data_tag}_{input_ds}DS.pkl"),
        feature_dim=3,
        teacher_hidden_dim=512,
        student_hidden_dim=256,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    if not Path(args.model_path).exists():
        print(f"❌ Error: Constrained enhancement model not found at {args.model_path}")
        print("   Please ensure the model path is correct")
        return
    model = load_constrained_enhancement_model(args.model_path, config)
    device = config.device
    for split in ["train", "test", "val"]:
        print(f"\n🔄 Processing {split} split...")
        input_data_path = input_base_path / split / f"capture24_{split}_data_stage2_{data_tag}_{input_ds}DS.pkl"
        input_labels_path = input_base_path / split / f"capture24_{split}_labels_stage2_{data_tag}_{input_ds}DS.pkl"
        if not input_data_path.exists() or not input_labels_path.exists():
            print(f"⚠️  Warning: Input files not found for {split} split")
            print(f"   Data: {input_data_path}")
            print(f"   Labels: {input_labels_path}")
            continue
        with open(input_data_path, 'rb') as f:
            low_res_segments = pickle.load(f)
        with open(input_labels_path, 'rb') as f:
            labels = pickle.load(f)
        print(f"   📥 Loaded {len(low_res_segments)} segments")
        print(f"   📏 Input shape: {low_res_segments[0].shape if low_res_segments else 'No data'}")
        print(f"   🚀 Enhancing segments with constrained model...")
        enhanced_segments = enhance_data_batch(model, low_res_segments, batch_size=32, device=device)
        print(f"   📤 Enhanced to shape: {enhanced_segments[0].shape if enhanced_segments else 'No data'}")
        print(f"   📊 Upsampling: {low_res_segments[0].shape[0]} → {enhanced_segments[0].shape[0]} timesteps")
        updated_labels = []
        for label in labels:
            updated_label = label.copy()
            updated_label['upsampled'] = True
            updated_label['constrained_upsampled'] = True
            updated_label['original_downsample_factor'] = input_ds
            updated_label['target_downsample_factor'] = output_ds
            updated_label['enhancement_model'] = 'constrained_enhancement_10x'
            updated_label['model_path'] = args.model_path
            updated_labels.append(updated_label)
        output_data_path = output_base_path / split / f"capture24_{split}_data_stage2_{data_tag}_{output_ds}DS_constrained_upsampled.pkl"
        output_labels_path = output_base_path / split / f"capture24_{split}_labels_stage2_{data_tag}_{output_ds}DS_constrained_upsampled.pkl"
        with open(output_data_path, 'wb') as f:
            pickle.dump(enhanced_segments, f)
        with open(output_labels_path, 'wb') as f:
            pickle.dump(updated_labels, f)
        print(f"   💾 Saved to:")
        print(f"      Data: {output_data_path}")
        print(f"      Labels: {output_labels_path}")
        input_settings_path = input_base_path / split / "settings.json"
        if input_settings_path.exists():
            with open(input_settings_path, 'r') as f:
                settings = json.load(f)
            settings['constrained_upsampled'] = True
            settings['enhancement_model'] = 'constrained_enhancement_10x'
            settings['original_downsample_factor'] = input_ds
            settings['target_downsample_factor'] = output_ds
            settings['enhancement_timestamp'] = str(Path(args.model_path).stat().st_mtime)
            settings['output_tag'] = f'{data_tag}_{output_ds}DS_constrained_upsampled'
            settings['model_path'] = args.model_path
            output_settings_path = output_base_path / split / "settings.json"
            with open(output_settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"      Settings: {output_settings_path}")
    print(f"\n✅ Constrained upsampled data generation completed!")
    print(f"📁 Output directory: {output_base_path}")
    print(f"📊 Ready for use with SensorLLM pipeline!")
    print(f"\n📈 Summary:")
    train_data_path = output_base_path / "train" / f"capture24_train_data_stage2_{data_tag}_{output_ds}DS_constrained_upsampled.pkl"
    test_data_path = output_base_path / "test" / f"capture24_test_data_stage2_{data_tag}_{output_ds}DS_constrained_upsampled.pkl"
    if train_data_path.exists():
        with open(train_data_path, 'rb') as f:
            train_data = pickle.load(f)
        print(f"   Train samples: {len(train_data)}")
        print(f"   Train shape: {train_data[0].shape if train_data else 'No data'}")
    if test_data_path.exists():
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        print(f"   Test samples: {len(test_data)}")
        print(f"   Test shape: {test_data[0].shape if test_data else 'No data'}")

if __name__ == "__main__":
    generate_constrained_upsampled_data() 