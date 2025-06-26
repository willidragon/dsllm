#!/usr/bin/env python3
"""
Generate Upsampled Data Using Trained Enhancement Model

This script:
1. Loads the trained enhancement model
2. Loads 1000DS (low-res) data 
3. Uses the model to upsample to 100DS equivalent quality
4. Saves the upsampled data in the same format as the original processing pipeline
"""

import sys
import os
from pathlib import Path
import pickle
import json
import yaml
import numpy as np
import torch

# Add the current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from teacher_student_model import DataEnhancementTeacherStudentModel, DataEnhancementConfig

def load_paths():
    """Load base paths from paths.yaml"""
    paths_file = current_dir.parent.parent / "paths.yaml"
    if not paths_file.exists():
        print(f"⚠️  Warning: paths.yaml not found at {paths_file}")
        return None
    
    with open(paths_file) as f:
        paths = yaml.safe_load(f)
    return paths

def load_enhancement_model(model_path: str, config: DataEnhancementConfig):
    """Load the trained enhancement model"""
    print(f"🏗️  Loading trained model from {model_path}...")
    
    # Initialize model
    model = DataEnhancementTeacherStudentModel(config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config.device, weights_only=False)
    
    # Load state dicts
    model.student_encoder.load_state_dict(checkpoint['student_encoder_state_dict'])
    model.student_decoder.load_state_dict(checkpoint['student_decoder_state_dict'])
    model.teacher_encoder.load_state_dict(checkpoint['teacher_encoder_state_dict'])
    model.teacher_decoder.load_state_dict(checkpoint['teacher_decoder_state_dict'])
    model.feature_projector.load_state_dict(checkpoint['feature_projector_state_dict'])
    
    # Set to evaluation mode
    model.student_encoder.eval()
    model.student_decoder.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Upsampling factor: {checkpoint.get('high_to_low_ratio', 'Unknown')}x")
    
    return model

def enhance_data_batch(model, low_res_data, batch_size=32):
    """Enhance data in batches to avoid memory issues"""
    enhanced_segments = []
    
    with torch.no_grad():
        for i in range(0, len(low_res_data), batch_size):
            batch = low_res_data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(np.array(batch)).to(model.config.device)
            
            # Run enhancement
            student_features, _ = model.student_encoder(batch_tensor)
            enhanced_batch = model.student_decoder(student_features)
            
            # Convert back to numpy with float64 dtype to match original 100DS data
            enhanced_batch_np = enhanced_batch.cpu().numpy().astype(np.float64)
            enhanced_segments.extend(enhanced_batch_np)
    
    return enhanced_segments

def generate_upsampled_data():
    """Main function to generate upsampled data"""
    print("🚀 Starting Upsampled Data Generation")
    print("=" * 60)
    print("PURPOSE: Use trained enhancement model to upsample 1000DS → 100DS equivalent")
    print("OUTPUT: Save in same format as original processing pipeline")
    print("=" * 60)
    
    # Load paths
    paths = load_paths()
    if paths is None:
        print("❌ Error: Could not load paths.yaml")
        return
    
    base_output_dir = Path(paths['base_output_dir'])
    
    # Define input and output paths
    input_base_path = base_output_dir / "stage_2_compare" / "300seconds_1000DS"
    output_base_path = base_output_dir / "stage_2_upsampled" / "300seconds_100DS_upsampled"
    
    # Create output directories
    output_base_path.mkdir(parents=True, exist_ok=True)
    (output_base_path / "train").mkdir(exist_ok=True)
    (output_base_path / "test").mkdir(exist_ok=True)
    
    print(f"📂 Paths:")
    print(f"   Input (1000DS): {input_base_path}")
    print(f"   Output (upsampled): {output_base_path}")
    print()
    
    # Set up configuration for model loading
    config = DataEnhancementConfig(
        high_res_timesteps=300,  # Target: 300 timesteps (1Hz equivalent)
        low_res_timesteps=30,    # Input: 30 timesteps (0.1Hz)
        high_res_path=str(input_base_path / "train" / "capture24_train_data_stage2_300seconds_1000DS.pkl"),
        low_res_path=str(input_base_path / "train" / "capture24_train_data_stage2_300seconds_1000DS.pkl"),
        feature_dim=3,
        teacher_hidden_dim=512,
        student_hidden_dim=256,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Load the trained model
    model_path = "trained/enhancement_model_10x.pth"
    if not Path(model_path).exists():
        print(f"❌ Error: Trained model not found at {model_path}")
        print("   Please train the model first using run_data_enhancement.py")
        return
    
    model = load_enhancement_model(model_path, config)
    
    # Process train and test splits
    for split in ["train", "test"]:
        print(f"\n🔄 Processing {split} split...")
        
        # Load input data and labels
        input_data_path = input_base_path / split / f"capture24_{split}_data_stage2_300seconds_1000DS.pkl"
        input_labels_path = input_base_path / split / f"capture24_{split}_labels_stage2_300seconds_1000DS.pkl"
        
        if not input_data_path.exists() or not input_labels_path.exists():
            print(f"⚠️  Warning: Input files not found for {split} split")
            print(f"   Data: {input_data_path}")
            print(f"   Labels: {input_labels_path}")
            continue
        
        # Load data
        with open(input_data_path, 'rb') as f:
            low_res_segments = pickle.load(f)
        with open(input_labels_path, 'rb') as f:
            labels = pickle.load(f)
        
        print(f"   📥 Loaded {len(low_res_segments)} segments")
        print(f"   📏 Input shape: {low_res_segments[0].shape if low_res_segments else 'No data'}")
        
        # Enhance all segments
        print(f"   🚀 Enhancing segments...")
        enhanced_segments = enhance_data_batch(model, low_res_segments, batch_size=32)
        
        print(f"   📤 Enhanced to shape: {enhanced_segments[0].shape if enhanced_segments else 'No data'}")
        print(f"   📊 Upsampling: {low_res_segments[0].shape[0]} → {enhanced_segments[0].shape[0]} timesteps")
        
        # Update labels to reflect the upsampled nature
        updated_labels = []
        for label in labels:
            # Copy original label
            updated_label = label.copy()
            # Add metadata about upsampling
            updated_label['upsampled'] = True
            updated_label['original_downsample_factor'] = 1000
            updated_label['target_downsample_factor'] = 100
            updated_label['enhancement_model'] = 'teacher_student_10x'
            updated_labels.append(updated_label)
        
        # Save enhanced data and labels
        output_data_path = output_base_path / split / f"capture24_{split}_data_stage2_300seconds_100DS_upsampled.pkl"
        output_labels_path = output_base_path / split / f"capture24_{split}_labels_stage2_300seconds_100DS_upsampled.pkl"
        
        with open(output_data_path, 'wb') as f:
            pickle.dump(enhanced_segments, f)
        with open(output_labels_path, 'wb') as f:
            pickle.dump(updated_labels, f)
        
        print(f"   💾 Saved to:")
        print(f"      Data: {output_data_path}")
        print(f"      Labels: {output_labels_path}")
        
        # Load and update settings
        input_settings_path = input_base_path / split / "settings.json"
        if input_settings_path.exists():
            with open(input_settings_path, 'r') as f:
                settings = json.load(f)
            
            # Update settings for upsampled data
            settings['upsampled'] = True
            settings['enhancement_model'] = 'teacher_student_10x'
            settings['original_downsample_factor'] = 1000
            settings['target_downsample_factor'] = 100
            settings['enhancement_timestamp'] = str(Path(model_path).stat().st_mtime)
            settings['output_tag'] = '300seconds_100DS_upsampled'
            
            # Save updated settings
            output_settings_path = output_base_path / split / "settings.json"
            with open(output_settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            
            print(f"      Settings: {output_settings_path}")
    
    print(f"\n✅ Upsampled data generation completed!")
    print(f"📁 Output directory: {output_base_path}")
    print(f"📊 Ready for use with SensorLLM pipeline!")
    
    # Print summary statistics
    print(f"\n📈 Summary:")
    train_data_path = output_base_path / "train" / "capture24_train_data_stage2_300seconds_100DS_upsampled.pkl"
    test_data_path = output_base_path / "test" / "capture24_test_data_stage2_300seconds_100DS_upsampled.pkl"
    
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
    generate_upsampled_data() 