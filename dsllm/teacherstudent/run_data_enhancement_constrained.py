#!/usr/bin/env python3
"""
Constrained Data Enhancement - Forces enhanced data through low-res points
"""

import sys
import os
from pathlib import Path
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import yaml

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from teacher_student_model import DataEnhancementTeacherStudentModel, DataEnhancementConfig

class ConstrainedEnhancementModel(nn.Module):
    """Model that enforces enhanced data passes through low-res points"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.low_res_timesteps = config.low_res_timesteps
        self.high_res_timesteps = config.high_res_timesteps
        self.feature_dim = config.feature_dim
        self.upsampling_factor = config.high_res_timesteps // config.low_res_timesteps
        
        # Enhanced encoder-decoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(config.low_res_timesteps * config.feature_dim, config.student_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.student_hidden_dim * 2, config.student_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.student_hidden_dim, config.student_hidden_dim // 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(config.student_hidden_dim // 2, config.student_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.student_hidden_dim, config.student_hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.student_hidden_dim * 2, config.high_res_timesteps * config.feature_dim)
        )
        
        # Calculate where low-res points go in high-res sequence
        self.low_res_indices = torch.arange(0, config.high_res_timesteps, self.upsampling_factor)
        
    def forward(self, low_res_data):
        batch_size = low_res_data.shape[0]
        
        # Encode and decode
        x = low_res_data.view(batch_size, -1)
        features = self.encoder(x)
        decoded = self.decoder(features)
        decoded = decoded.view(batch_size, self.high_res_timesteps, self.feature_dim)
        
        # HARD CONSTRAINT: Force enhanced data through low-res points
        enhanced_data = self.enforce_constraint(decoded, low_res_data)
        
        return enhanced_data
    
    def enforce_constraint(self, decoded_data, low_res_data):
        """Force enhanced data to pass exactly through low-res points"""
        enhanced = decoded_data.clone()
        
        # Set low-res points exactly
        for i, idx in enumerate(self.low_res_indices):
            if i < low_res_data.shape[1]:
                enhanced[:, idx, :] = low_res_data[:, i, :]
        
        # Smooth interpolation between fixed points
        enhanced = self.smooth_between_points(enhanced, low_res_data)
        
        return enhanced
    
    def smooth_between_points(self, enhanced_data, low_res_data):
        """Apply smooth interpolation between the fixed low-res points"""
        result = enhanced_data.clone()
        
        # Interpolate between consecutive low-res points
        for i in range(len(self.low_res_indices) - 1):
            start_idx = self.low_res_indices[i]
            end_idx = self.low_res_indices[i + 1]
            
            start_val = low_res_data[:, i, :]
            end_val = low_res_data[:, i + 1, :]
            
            # Smooth interpolation with learned features
            for t in range(start_idx + 1, end_idx):
                alpha = (t - start_idx) / (end_idx - start_idx)
                linear_interp = (1 - alpha) * start_val + alpha * end_val
                learned_offset = enhanced_data[:, t, :] - linear_interp
                result[:, t, :] = linear_interp + 0.2 * learned_offset
        
        return result

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.min_val_loss = float('inf')
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"  ⚠️ Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("  🛑 Early stopping triggered!")
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

class ConstrainedTrainer:
    """Trainer with consistency loss to enforce constraint"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Load data using existing infrastructure
        self.base_model = DataEnhancementTeacherStudentModel(config)
        self.train_loader = self.base_model.train_loader
        self.test_loader = self.base_model.test_loader
        
        # Initialize constrained model
        self.model = ConstrainedEnhancementModel(config).to(self.device)
        
        # Optimizer with higher initial learning rate and cosine annealing
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.002,  # Higher initial learning rate
            weight_decay=1e-4,  # Slightly stronger regularization
            betas=(0.9, 0.999)  # Default Adam betas
        )
        
        # Cosine annealing scheduler with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,  # Double the restart interval after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        
        # Loss weights - adjusted for better balance
        self.reconstruction_weight = 1.0
        self.consistency_weight = 10.0  # Reduced from 50.0 to allow more flexibility
        self.smoothness_weight = 0.05   # Reduced to prioritize reconstruction
        
    def consistency_loss(self, enhanced_data, low_res_data):
        """Ensure enhanced data exactly matches low-res data at sampled points"""
        total_loss = 0.0
        count = 0
        
        for i, idx in enumerate(self.model.low_res_indices):
            if i < low_res_data.shape[1]:
                loss = F.mse_loss(enhanced_data[:, idx, :], low_res_data[:, i, :])
                total_loss += loss
                count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def smoothness_loss(self, enhanced_data):
        """Encourage smooth transitions"""
        diff1 = enhanced_data[:, 1:, :] - enhanced_data[:, :-1, :]
        diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]
        return torch.mean(diff2 ** 2)
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_consistency = 0.0
        
        for batch_idx, (low_res_batch, high_res_batch) in enumerate(self.train_loader):
            low_res_batch = low_res_batch.to(self.device)
            high_res_batch = high_res_batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass and loss calculation
            enhanced_batch = self.model(low_res_batch)
            reconstruction_loss = F.mse_loss(enhanced_batch, high_res_batch)
            consistency_loss = self.consistency_loss(enhanced_batch, low_res_batch)
            smoothness_loss = self.smoothness_loss(enhanced_batch)
            
            # Combined loss with high consistency weight
            total_batch_loss = (
                self.reconstruction_weight * reconstruction_loss +
                self.consistency_weight * consistency_loss +
                self.smoothness_weight * smoothness_loss
            )
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_consistency += consistency_loss.item()
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                avg_consistency = total_consistency / (batch_idx + 1)
                print(f"Batch [{batch_idx:4d}/{len(self.train_loader)}] - "
                      f"Loss: {avg_loss:.6f}, Consistency: {avg_consistency:.8f}")
        
        return total_loss / len(self.train_loader), total_consistency / len(self.train_loader)
    
    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_consistency = 0.0
        
        with torch.no_grad():
            for low_res_batch, high_res_batch in self.test_loader:
                low_res_batch = low_res_batch.to(self.device)
                high_res_batch = high_res_batch.to(self.device)
                
                enhanced_batch = self.model(low_res_batch)
                
                loss = F.mse_loss(enhanced_batch, high_res_batch)
                consistency = self.consistency_loss(enhanced_batch, low_res_batch)
                
                total_loss += loss.item()
                total_consistency += consistency.item()
        
        return total_loss / len(self.test_loader), total_consistency / len(self.test_loader)
    
    def train(self, epochs):
        print("🎓 Starting CONSTRAINED enhancement training...")
        print(f"🎯 CONSTRAINT: Enhanced data MUST pass through low-res points")
        print(f"📊 Training configuration:")
        print(f"   - Initial learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"   - Reconstruction weight: {self.reconstruction_weight}")
        print(f"   - Consistency weight: {self.consistency_weight}")
        print(f"   - Smoothness weight: {self.smoothness_weight}")
        
        best_loss = float('inf')
        early_stopping = EarlyStopping(patience=10, min_delta=0.0001)  # Stop if no improvement for 10 epochs
        
        for epoch in range(epochs):
            # Train epoch
            train_loss, train_consistency = self.train_epoch()
            
            # Step the scheduler every epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            # Evaluate
            val_loss, val_consistency = self.evaluate()
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train - Loss: {train_loss:.6f}, Consistency: {train_consistency:.8f}")
            print(f"  Val   - Loss: {val_loss:.6f}, Consistency: {val_consistency:.8f}")
            print(f"  LR: {current_lr:.6f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model("best_constrained_model.pth")
                print(f"  ✅ Best model saved!")
            
            if val_consistency < 1e-10:
                print(f"  🎯 Perfect constraint satisfaction achieved!")
            
            # Early stopping check
            if early_stopping(val_loss):
                print(f"Training stopped early at epoch {epoch+1}")
                break
        
        print(f"✅ Training completed! Best loss: {best_loss:.6f}")
    
    def save_model(self, filename):
        save_path = Path(f"/project/cc-20250120231604/ssd/users/kwsu/data/trained_model/enhancement_model/{filename}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        print(f"💾 Model saved to {save_path}")

def visualize_constrained_results(trainer, num_samples=3):
    """Visualize results and verify constraints are satisfied"""
    trainer.model.eval()
    
    test_iter = iter(trainer.test_loader)
    low_res_batch, high_res_batch = next(test_iter)
    
    low_res_samples = low_res_batch[:num_samples].to(trainer.device)
    high_res_samples = high_res_batch[:num_samples].to(trainer.device)
    
    with torch.no_grad():
        enhanced_samples = trainer.model(low_res_samples)
    
    # Convert to numpy
    low_res_cpu = low_res_samples.cpu().numpy()
    high_res_cpu = high_res_samples.cpu().numpy()
    enhanced_cpu = enhanced_samples.cpu().numpy()
    
    # Plot results
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    feature_names = ['X-axis', 'Y-axis', 'Z-axis']
    upsampling_factor = trainer.model.upsampling_factor
    
    for sample_idx in range(num_samples):
        for feature_idx in range(3):
            ax = axes[sample_idx, feature_idx] if num_samples > 1 else axes[feature_idx]
            
            # Time arrays
            time_high = np.linspace(0, 300, len(high_res_cpu[sample_idx, :, feature_idx]))
            time_low = np.linspace(0, 300, len(low_res_cpu[sample_idx, :, feature_idx]))
            
            # Plot signals
            ax.plot(time_high, high_res_cpu[sample_idx, :, feature_idx], 'g-', 
                   linewidth=1, label='True High-res', alpha=0.8)
            ax.plot(time_high, enhanced_cpu[sample_idx, :, feature_idx], 'r-', 
                   linewidth=2, label='Enhanced (CONSTRAINED)', alpha=0.9)
            ax.scatter(time_low, low_res_cpu[sample_idx, :, feature_idx], 
                      c='blue', s=50, label='Low-res Input', zorder=5)
            
            # Verify constraint: check if enhanced passes through blue dots
            constraint_satisfied = True
            for i, low_val in enumerate(low_res_cpu[sample_idx, :, feature_idx]):
                enhanced_val = enhanced_cpu[sample_idx, i * upsampling_factor, feature_idx]
                error = abs(enhanced_val - low_val)
                if error > 1e-6:
                    constraint_satisfied = False
                    ax.plot(time_low[i], enhanced_val, 'rx', markersize=10, 
                           markeredgewidth=3, label='VIOLATION!')
            
            constraint_text = "✅ CONSTRAINT SATISFIED" if constraint_satisfied else "❌ CONSTRAINT VIOLATED"
            ax.set_title(f'Sample {sample_idx+1} - {feature_names[feature_idx]}\n{constraint_text}')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Acceleration (g)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('constrained_enhancement_verification.png', dpi=300, bbox_inches='tight')
    print("📊 Constraint verification plot saved to: constrained_enhancement_verification.png")
    plt.show()

def load_paths():
    paths_file = current_dir.parent.parent / "paths.yaml"
    if not paths_file.exists():
        return None
    with open(paths_file) as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Constrained Data Enhancement Training")
    parser.add_argument('--high_ds', type=int, default=100, help='High-res DS rate')
    parser.add_argument('--low_ds', type=int, default=1000, help='Low-res DS rate')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--data_subdir', type=str, default='stage_2_compare_buffer')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with reduced dataset and epochs')
    args = parser.parse_args()

    # Override settings if in test mode
    if args.test_mode:
        args.epochs = 3
        args.batch_size = 8
        print("\n🧪 Running in TEST MODE:")
        print("   - Reduced dataset (10%)")
        print("   - Epochs:", args.epochs)
        print("   - Batch size:", args.batch_size)
        print("   - Progress logging: every 50 batches")
        print("=" * 60)

    print("🚀 CONSTRAINED Data Enhancement Training")
    print("=" * 60)
    print("🎯 INNOVATION: Enhanced data MUST pass through low-res points!")
    print("🔒 HARD CONSTRAINT: Zero tolerance for deviation!")
    print("=" * 60)
    
    # Load paths and data
    paths = load_paths()
    if paths is None:
        print("❌ Error: Could not load paths.yaml")
        return
    
    base_path = Path(paths['base_output_dir']) / args.data_subdir
    high_res_path = str(base_path / f"300seconds_{args.high_ds}DS/train/capture24_train_data_stage2_300seconds_{args.high_ds}DS.pkl")
    low_res_path = str(base_path / f"300seconds_{args.low_ds}DS/train/capture24_train_data_stage2_300seconds_{args.low_ds}DS.pkl")
    
    # Check paths exist
    for path in [high_res_path, low_res_path]:
        if not Path(path).exists():
            print(f"❌ Error: Path does not exist: {path}")
            return
    
    # Get data dimensions
    with open(high_res_path, 'rb') as f:
        sample_high_res = pickle.load(f)[0]
    with open(low_res_path, 'rb') as f:
        sample_low_res = pickle.load(f)[0]
    
    upsampling_factor = sample_high_res.shape[0] // sample_low_res.shape[0]
    
    # Configuration
    config = DataEnhancementConfig(
        high_res_path=high_res_path,
        low_res_path=low_res_path,
        high_res_timesteps=sample_high_res.shape[0],
        low_res_timesteps=sample_low_res.shape[0],
        feature_dim=sample_high_res.shape[1],
        teacher_hidden_dim=512,
        student_hidden_dim=256,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"📊 Configuration:")
    print(f"   Upsampling: {config.low_res_timesteps} -> {config.high_res_timesteps} timesteps")
    print(f"   Factor: {upsampling_factor}x")
    print(f"   🎯 CONSTRAINT: Enhanced MUST pass through {config.low_res_timesteps} low-res points")
    
    try:
        # Initialize and train
        trainer = ConstrainedTrainer(config)
        
        # Reduce dataset size in test mode
        if args.test_mode:
            trainer.train_loader.dataset.data = trainer.train_loader.dataset.data[:5000]
            trainer.test_loader.dataset.data = trainer.test_loader.dataset.data[:500]
            print(f"📊 Reduced dataset sizes:")
            print(f"   Train: {len(trainer.train_loader.dataset)} samples")
            print(f"   Test: {len(trainer.test_loader.dataset)} samples")
        
        trainer.train(args.epochs)
        
        # Visualize and verify constraints
        print("\n📊 Generating constraint verification plots...")
        visualize_constrained_results(trainer)
        
        # Save final model
        model_name = f"constrained_enhancement_model_{upsampling_factor}x.pth"
        trainer.save_model(model_name)
        
        print(f"\n✅ CONSTRAINED training completed!")
        print(f"🎯 Enhanced data now GUARANTEED to pass through low-res points!")
        print(f"💾 Model saved as: {model_name}")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 