#!/usr/bin/env python3
"""
Data Enhancement Teacher-Student Model Training Script

This script implements sensor data super-resolution:
- Teacher: Learns features from high-resolution sensor data (100DS)  
- Student: Reconstructs high-resolution data from low-resolution input (1000DS)

Goal: Upsample low-res sensor data to high-res quality for SensorLLM pipeline.
"""

import sys
import os
from pathlib import Path
import pickle
import json
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import pandas as pd

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

def visualize_enhancement_results(model, num_samples=3, output_dir=None):
    """Visualize enhancement results"""
    model.student_encoder.eval()
    model.student_decoder.eval()
    
    # Get some test samples
    test_loader_iter = iter(model.test_loader)
    batch_data = next(test_loader_iter)
    
    # Handle both cases: with and without labels
    if len(batch_data) == 3:
        low_res_batch, high_res_batch, _ = batch_data  # Ignore labels for visualization
    else:
        low_res_batch, high_res_batch = batch_data
    
    # Select first few samples
    low_res_samples = low_res_batch[:num_samples].to(model.config.device)
    high_res_samples = high_res_batch[:num_samples].to(model.config.device)
    
    # Generate enhanced data
    with torch.no_grad():
        student_features, _ = model.student_encoder(low_res_samples)
        enhanced_samples = model.student_decoder(student_features)
    
    # Move to CPU for plotting
    low_res_cpu = low_res_samples.cpu().numpy()
    high_res_cpu = high_res_samples.cpu().numpy()
    enhanced_cpu = enhanced_samples.cpu().numpy()
    
    # Plot comparison
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    feature_names = ['X-axis', 'Y-axis', 'Z-axis']
    
    for sample_idx in range(num_samples):
        for feature_idx in range(3):
            # Low-res data (upsampled for visualization)
            low_res_upsampled = np.repeat(low_res_cpu[sample_idx, :, feature_idx], 10)
            time_low = np.linspace(0, 300, len(low_res_upsampled))
            
            # High-res and enhanced data
            time_high = np.linspace(0, 300, len(high_res_cpu[sample_idx, :, feature_idx]))
            time_enhanced = np.linspace(0, 300, len(enhanced_cpu[sample_idx, :, feature_idx]))
            
            ax = axes[sample_idx, feature_idx] if num_samples > 1 else axes[feature_idx]
            
            ax.plot(time_low, low_res_upsampled, 'b-', alpha=0.7, linewidth=2, 
                   label='Low-res (1Hz)')
            ax.plot(time_high, high_res_cpu[sample_idx, :, feature_idx], 'g-', alpha=0.8, 
                   linewidth=1, label='True High-res (100Hz)')
            ax.plot(time_enhanced, enhanced_cpu[sample_idx, :, feature_idx], 'r--', alpha=0.8, 
                   linewidth=1, label='Enhanced (reconstructed)')
            
            ax.set_title(f'Sample {sample_idx+1} - {feature_names[feature_idx]}')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Acceleration (g)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to results_dir if provided, else evaluation folder
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / 'enhancement_comparison.png'
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = Path(f"evaluation/training_demo_{timestamp}")
        eval_dir.mkdir(parents=True, exist_ok=True)
        plot_path = eval_dir / 'enhancement_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training demo visualization saved to: {plot_path}")
    plt.show()
    
    # Calculate and print MSE for each sample
    mse_per_sample = np.mean((high_res_cpu - enhanced_cpu)**2, axis=(1, 2))
    print(f"\n📊 Enhancement Quality (MSE per sample):")
    for i, mse in enumerate(mse_per_sample):
        print(f"   Sample {i+1}: {mse:.6f}")
    print(f"   Average MSE: {np.mean(mse_per_sample):.6f}")

def demo_enhancement_pipeline(model, num_samples=5, output_dir=None):
    """Demonstrate the complete enhancement pipeline"""
    print("\n🚀 Data Enhancement Pipeline Demo")
    print("=" * 50)
    
    # Load some low-res test data
    test_loader_iter = iter(model.test_loader)
    batch_data = next(test_loader_iter)
    
    # Handle both cases: with and without labels
    if len(batch_data) == 3:
        low_res_batch, high_res_batch, _ = batch_data  # Ignore labels for demo
    else:
        low_res_batch, high_res_batch = batch_data
    
    # Select samples
    low_res_demo = low_res_batch[:num_samples]
    high_res_demo = high_res_batch[:num_samples]
    
    print(f"📥 Input: Low-resolution data")
    print(f"   Shape: {low_res_demo.shape}")
    print(f"   Timesteps: {low_res_demo.shape[1]}")
    print(f"   Features: {low_res_demo.shape[2]}")
    
    # Enhance data
    enhanced_data = model.enhance_data(low_res_demo)
    
    print(f"\n📤 Output: Enhanced high-resolution data")
    print(f"   Shape: {enhanced_data.shape}")
    print(f"   Timesteps: {enhanced_data.shape[1]} ({enhanced_data.shape[1]/low_res_demo.shape[1]:.1f}x upsampling)")
    print(f"   Features: {enhanced_data.shape[2]}")
    
    print(f"\n🎯 Target: True high-resolution data")
    print(f"   Shape: {high_res_demo.shape}")
    print(f"   Timesteps: {high_res_demo.shape[1]}")
    
    # Calculate reconstruction quality
    mse = np.mean((enhanced_data - high_res_demo.numpy())**2)
    print(f"\n📊 Reconstruction Quality:")
    print(f"   Mean Squared Error: {mse:.6f}")
    print(f"   RMSE: {np.sqrt(mse):.6f}")
    
    # Calculate temporal correlation
    correlations = []
    for i in range(num_samples):
        for j in range(3):  # For each axis
            corr = np.corrcoef(enhanced_data[i, :, j], high_res_demo[i, :, j].numpy())[0, 1]
            correlations.append(corr)
    
    avg_correlation = np.mean(correlations)
    print(f"   Temporal Correlation: {avg_correlation:.4f}")
    
    print(f"\n💡 Usage: This enhanced data can now be fed into your SensorLLM pipeline!")
    print(f"   Original SensorLLM input: {high_res_demo.shape}")
    print(f"   Enhanced data output: {enhanced_data.shape}")
    print(f"   ✅ Shapes match - ready for SensorLLM inference!")
    
    # Optionally save demo results to output_dir
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save a summary text file
        with open(output_dir / 'demo_summary.txt', 'w') as f:
            f.write(f"Input shape: {low_res_demo.shape}\n")
            f.write(f"Output shape: {enhanced_data.shape}\n")
            f.write(f"Target shape: {high_res_demo.shape}\n")
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"RMSE: {np.sqrt(mse):.6f}\n")
            f.write(f"Temporal Correlation: {avg_correlation:.4f}\n")
    return enhanced_data

def plot_loss_curves(results_dir):
    """Plot and save teacher and student loss curves from CSV logs."""
    import matplotlib.pyplot as plt
    import pandas as pd
    results_dir = Path(results_dir)
    # Teacher loss curve
    teacher_csv = results_dir / 'training_log_teacher.csv'
    if teacher_csv.exists():
        df_teacher = pd.read_csv(teacher_csv)
        plt.figure(figsize=(8,4))
        plt.plot(df_teacher['epoch'], df_teacher['total_loss'], label='Teacher Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Teacher Loss Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / 'teacher_loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    # Student loss curve
    student_csv = results_dir / 'training_log_student.csv'
    if student_csv.exists():
        df_student = pd.read_csv(student_csv)
        plt.figure(figsize=(8,4))
        plt.plot(df_student['epoch'], df_student['total_loss'], label='Student Total Loss')
        plt.plot(df_student['epoch'], df_student['recon_loss'], label='Recon Loss', linestyle='--')
        plt.plot(df_student['epoch'], df_student['feature_loss'], label='Feature Loss', linestyle='--')
        plt.plot(df_student['epoch'], df_student['smooth_loss'], label='Smooth Loss', linestyle='--')
        plt.plot(df_student['epoch'], df_student['freq_loss'], label='Freq Loss', linestyle='--')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Student Loss Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / 'student_loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Data Enhancement Teacher-Student Training")
    parser.add_argument('--high_ds', type=int, default=100, help='High-res (teacher) DS rate, e.g., 100')
    parser.add_argument('--low_ds', type=int, default=1000, help='Low-res (student) DS rate, e.g., 1000 or 500')
    parser.add_argument('--data_subdir', type=str, default='stage_2_compare_buffer', help='Subdirectory under base_output_dir for input data (e.g., stage_2_compare_buffer)')
    args = parser.parse_args()

    print("🚀 Starting Data Enhancement Teacher-Student Training")
    print("=" * 60)
    print("PURPOSE: Learn to upsample low-res sensor data to high-res quality")
    print("GOAL: Create enhanced data for SensorLLM pipeline")
    print("=" * 60)
    print(f"Using high-res DS: {args.high_ds}")
    print(f"Using low-res DS: {args.low_ds}")
    
    # Load paths from paths.yaml
    paths = load_paths()
    if paths is None:
        print("❌ Error: Could not load paths.yaml")
        return
    
    # Set up paths
    high_res_path = f"/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_compare_buffer/300seconds_{args.high_ds}DS/train/capture24_train_data_stage2_300seconds_{args.high_ds}DS.pkl"
    low_res_path = f"/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_compare_buffer/300seconds_{args.low_ds}DS/train/capture24_train_data_stage2_300seconds_{args.low_ds}DS.pkl"
    
    print("📂 Data paths:")
    print(f"   High-res (teacher): {high_res_path}")
    print(f"   Low-res (student input): {low_res_path}")
    print()
    
    # Check if paths exist
    for path in [high_res_path, low_res_path]:
        if not Path(path).exists():
            print(f"❌ Error: Path does not exist: {path}")
            return
    
    # Get data dimensions from sample files
    with open(high_res_path, 'rb') as f:
        sample_high_res = pickle.load(f)[0]  # Get first sample
    with open(low_res_path, 'rb') as f:
        sample_low_res = pickle.load(f)[0]  # Get first sample
    
    # Calculate downsampling ratio for filename
    high_to_low_ratio = sample_high_res.shape[0] // sample_low_res.shape[0]
    
    # Set output directory as requested
    trained_dir = Path("/project/cc-20250120231604/ssd/users/kwsu/data/trained_model/enhancement_model")
    trained_dir.mkdir(parents=True, exist_ok=True)
    # Compose a descriptive model name
    enhancement_tags = []
    enhancement_tags.append(f"{high_to_low_ratio}x")
    enhancement_tags.append("stftloss")
    enhancement_tags.append("activityaware")
    model_name = trained_dir / ("enhancement_model_" + "_".join(enhancement_tags) + ".pth")
    
    # Configuration for data enhancement
    config = DataEnhancementConfig(
        high_res_path=high_res_path,
        low_res_path=low_res_path,
        high_res_timesteps=sample_high_res.shape[0],
        low_res_timesteps=sample_low_res.shape[0],
        feature_dim=sample_high_res.shape[1],  # x, y, z accelerometer
        teacher_hidden_dim=512,
        student_hidden_dim=256,
        teacher_epochs=9999,
        student_epochs=9999,
        batch_size=64,           # Smaller batch for memory
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"🔧 Configuration:")
    print(f"   Upsampling: {config.low_res_timesteps} -> {config.high_res_timesteps} timesteps")
    print(f"   Factor: {config.high_res_timesteps // config.low_res_timesteps}x")
    print(f"   Device: {config.device}")
    print(f"   Batch size: {config.batch_size}")
    print()
    
    try:
        print("🏗️  Initializing data enhancement models...")
        model = DataEnhancementTeacherStudentModel(config)
        
        print("🎓 Starting enhancement training...")
        results_dir = model.train()
        
        if results_dir is not None:
            print("\n📊 Evaluating enhancement quality...")
            test_mse = model.evaluate_enhancement_quality()
            # Save the trained model in results_dir
            model_save_path = results_dir / ("enhancement_model_" + "_".join(enhancement_tags) + ".pth")
            print(f"\n💾 Saving model as {model_save_path}")
            torch.save({
                'config': config,
                'teacher_encoder_state_dict': model.teacher_encoder.state_dict(),
                'teacher_decoder_state_dict': model.teacher_decoder.state_dict(),
                'student_encoder_state_dict': model.student_encoder.state_dict(),
                'student_decoder_state_dict': model.student_decoder.state_dict(),
                'feature_projector_state_dict': model.feature_projector.state_dict(),
                'high_to_low_ratio': high_to_low_ratio,
                'student_optimizer_state_dict': model.student_optimizer.state_dict(),
                'final_loss': test_mse,
                'enhancement_tags': enhancement_tags,
            }, model_save_path)
            # Plot and save loss curves
            plot_loss_curves(results_dir)
            print("\n📊 Demonstrating enhancement on sample data...")
            visualize_enhancement_results(model, output_dir=results_dir)
            demo_enhancement_pipeline(model, output_dir=results_dir)
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 