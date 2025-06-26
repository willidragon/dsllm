#!/usr/bin/env python3
"""
Teacher-Student Model Training Script for Sensor Data

This script implements knowledge distillation between:
- Teacher: Trained on high-resolution sensor data (100DS)
- Student: Trained on low-resolution sensor data (1000DS) with teacher guidance

The goal is to achieve good performance on low-resolution data by learning 
from a teacher that has access to high-resolution patterns.
"""

import sys
import os
from pathlib import Path
import pickle
import json
import yaml
import numpy as np
import torch

# Add the current directory to path since we moved the files
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from teacher_student_model import TeacherStudentModel, TeacherStudentConfig

def load_paths():
    """Load base paths from paths.yaml"""
    paths_file = current_dir.parent.parent / "paths.yaml"
    if not paths_file.exists():
        print(f"⚠️  Warning: paths.yaml not found at {paths_file}")
        return None
    
    with open(paths_file) as f:
        paths = yaml.safe_load(f)
    return paths

def get_num_classes(data_path: str) -> int:
    """Automatically determine number of classes from the data"""
    try:
        labels_file = Path(data_path) / "train" / f"capture24_train_labels_stage2_300seconds_{data_path.split('_')[-1]}.pkl"
        with open(labels_file, 'rb') as f:
            labels = pickle.load(f)
        
        if isinstance(labels, list):
            labels = np.array(labels)
        
        num_classes = len(np.unique(labels))
        print(f"Detected {num_classes} classes in the dataset")
        return num_classes
    except Exception as e:
        print(f"Warning: Could not determine number of classes automatically: {e}")
        print("Using default value of 8 classes")
        return 8

def main():
    print("🚀 Starting Teacher-Student Model Training for Sensor Data")
    print("=" * 60)
    
    # Load paths from paths.yaml
    paths = load_paths()
    if paths is None:
        print("❌ Error: Could not load paths.yaml")
        return
    
    # Set up paths using base_output_dir from paths.yaml
    base_path = Path(paths['base_output_dir']) / "stage_2_compare"
    high_res_path = str(base_path / "300seconds_100DS")
    med_res_path = str(base_path / "300seconds_200DS")
    low_res_path = str(base_path / "300seconds_1000DS")
    
    print("📂 Data paths:")
    print(f"   Base path: {base_path}")
    print(f"   High-res: {high_res_path}")
    print(f"   Med-res:  {med_res_path}")
    print(f"   Low-res:  {low_res_path}")
    print()
    
    # Check if paths exist
    for path in [high_res_path, low_res_path]:
        if not Path(path).exists():
            print(f"❌ Error: Path does not exist: {path}")
            return
    
    # Automatically determine number of classes
    num_classes = get_num_classes(high_res_path)
    
    # Configuration
    config = TeacherStudentConfig(
        # Data paths
        high_res_path=high_res_path,
        med_res_path=med_res_path,
        low_res_path=low_res_path,
        
        # Model architecture
        teacher_hidden_dim=512,
        student_hidden_dim=256,
        num_classes=10,  # We need 10 classes to handle indices 0-9 after conversion from 1-based [1,2,3,4,6,7,9,10]
        
        # Training parameters
        teacher_epochs=15,  # Reduced for faster experimentation
        student_epochs=25,
        teacher_lr=1e-3,
        student_lr=1e-3,
        batch_size=8,  # Smaller batch size for limited data
        
        # Knowledge distillation
        temperature=4.0,
        alpha=0.7,  # Higher weight for distillation
        beta=0.3,   # Lower weight for hard targets
        
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"📊 Configuration:")
    print(f"   Teacher data: {config.high_res_path}")
    print(f"   Student data: {config.low_res_path}")
    print(f"   Device: {config.device}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Classes: {config.num_classes}")
    print()
    
    try:
        # Create model
        print("🏗️  Initializing models...")
        model = TeacherStudentModel(config)
        
        # Train the complete pipeline
        print("🎓 Starting training pipeline...")
        model.train()
        
        # Compare results
        print("\n📈 Final comparison:")
        model.compare_models()
        
        print("\n✅ Training completed successfully!")
        print(f"💾 Models saved as 'best_teacher.pth' and 'best_student.pth'")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 