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
import csv
import datetime
import shutil
import torch.optim as optim

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from teacher_student_model import DataEnhancementTeacherStudentModel, DataEnhancementConfig

class TeacherModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.high_res_timesteps * config.feature_dim, config.teacher_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.teacher_hidden_dim * 2, config.teacher_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.teacher_hidden_dim, config.teacher_hidden_dim // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.teacher_hidden_dim // 2, config.teacher_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.teacher_hidden_dim, config.teacher_hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.teacher_hidden_dim * 2, config.high_res_timesteps * config.feature_dim)
        )
        self.high_res_timesteps = config.high_res_timesteps
        self.feature_dim = config.feature_dim
    def forward(self, high_res_data):
        batch_size = high_res_data.shape[0]
        x = high_res_data.view(batch_size, -1)
        features = self.encoder(x)
        decoded = self.decoder(features)
        decoded = decoded.view(batch_size, self.high_res_timesteps, self.feature_dim)
        return decoded, features

class ConstrainedEnhancementModel(nn.Module):
    """Model that enforces enhanced data passes through low-res points, with optional label conditioning"""
    
    def __init__(self, config, num_classes=None, label_embed_dim=16):
        super().__init__()
        self.config = config
        self.low_res_timesteps = config.low_res_timesteps
        self.high_res_timesteps = config.high_res_timesteps
        self.feature_dim = config.feature_dim
        self.upsampling_factor = config.high_res_timesteps // config.low_res_timesteps
        self.label_embed_dim = label_embed_dim
        self.num_classes = num_classes
        # Enhanced encoder-decoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(config.low_res_timesteps * config.feature_dim, config.student_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.student_hidden_dim * 2, config.student_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.student_hidden_dim, config.student_hidden_dim // 2)
        )
        # Decoder input size depends on label conditioning
        decoder_input_dim = config.student_hidden_dim // 2 + (label_embed_dim if num_classes else 0)
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, config.student_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.student_hidden_dim, config.student_hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.student_hidden_dim * 2, config.high_res_timesteps * config.feature_dim)
        )
        # Label embedding if num_classes is provided
        if num_classes:
            self.label_embedding = nn.Embedding(num_classes, label_embed_dim)
        else:
            self.label_embedding = None
        # Calculate where low-res points go in high-res sequence
        self.low_res_indices = torch.arange(0, config.high_res_timesteps, self.upsampling_factor)
    
    def forward(self, low_res_data, labels=None):
        batch_size = low_res_data.shape[0]
        x = low_res_data.view(batch_size, -1)
        features = self.encoder(x)
        # Label conditioning
        if self.label_embedding is not None and labels is not None:
            label_embeds = self.label_embedding(labels)
            features = torch.cat([features, label_embeds], dim=-1)
        elif self.label_embedding is not None:
            # Use zeros if no label is provided (test time)
            device = features.device
            label_embeds = torch.zeros(features.shape[0], self.label_embed_dim, device=device)
            features = torch.cat([features, label_embeds], dim=-1)
        decoded = self.decoder(features)
        decoded = decoded.view(batch_size, self.high_res_timesteps, self.feature_dim)
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

def detect_spike(current_loss, previous_loss, threshold=0.05):
    """Detects a spike in loss if the increase exceeds the threshold."""
    return (current_loss - previous_loss) / previous_loss > threshold

class ConstrainedTrainer:
    """Trainer with consistency loss to enforce constraint and optional label conditioning"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        # Load data using existing infrastructure
        self.base_model = DataEnhancementTeacherStudentModel(config)
        self.train_loader = self.base_model.train_loader
        self.test_loader = self.base_model.test_loader
        # Always use num_classes=10 (labels 1-10, shifted to 0-9)
        num_classes = 10
        print(f"[DEBUG] num_classes set to: {num_classes}")
        # Initialize constrained model with label conditioning if available
        self.model = ConstrainedEnhancementModel(config, num_classes=num_classes).to(self.device)
        # Optimizer with more conservative learning rate
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,  # Lower initial learning rate
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
        # Use a more stable learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,  # Reduce LR by half when plateauing
            patience=5,   # Wait 5 epochs before reducing LR
            verbose=True,
            min_lr=1e-6
        )
        # Loss weights - adjusted for better stability
        self.reconstruction_weight = 1.0
        self.consistency_weight = 5.0  # Further reduced from 10.0 to improve stability
        self.smoothness_weight = 0.01  # Reduced to focus on main objectives
        # Internal list to store per-epoch metrics for optional logging / plotting
        self._epoch_history = []  # (epoch, train_loss, val_loss, train_cons, val_cons, lr)
        self.feature_projector = nn.Linear(config.student_hidden_dim // 2, config.teacher_hidden_dim // 2).to(self.device)
    
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
    
    def train_teacher(self, epochs):
        print("\n===== TEACHER TRAINING =====")
        self.teacher_model.train()
        optimizer = optim.AdamW(self.teacher_model.parameters(), lr=0.001, weight_decay=1e-4)
        best_loss = float('inf')
        early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
        log_file_teacher = open('training_log_teacher.csv', 'w', newline='')
        log_writer_teacher = csv.writer(log_file_teacher)
        log_writer_teacher.writerow(['epoch', 'train_loss', 'val_loss'])
        previous_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in self.train_loader:
                if len(batch) == 3:
                    _, high_res_batch, _ = batch
                else:
                    _, high_res_batch = batch
                high_res_batch = high_res_batch.to(self.device)
                optimizer.zero_grad()
                recon, _ = self.teacher_model(high_res_batch)
                loss = F.mse_loss(recon, high_res_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            print(f"[Teacher] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
            print(f"[Teacher] Early stopping counter: {early_stopping.counter}/{early_stopping.patience}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.teacher_model.state_dict(), "best_teacher_model.pth")
            if early_stopping(avg_loss):
                print(f"[Teacher] Early stopping at epoch {epoch+1}")
                break
            if detect_spike(avg_loss, previous_loss):
                torch.save(self.teacher_model.state_dict(), f"spike_teacher_model_epoch_{epoch+1}.pth")
                print(f"Spike detected! Model saved at epoch {epoch+1}")
            previous_loss = avg_loss
        print(f"[Teacher] Training complete. Best loss: {best_loss:.6f}")
        self.teacher_model.load_state_dict(torch.load("best_teacher_model.pth"))
        self.teacher_model.eval()
        log_writer_teacher.writerow([epoch+1, avg_loss, best_loss])
        log_file_teacher.close()

    def train_student(self, epochs, feature_loss_weight=0.5):
        print("\n===== STUDENT (CONSTRAINED) TRAINING =====")
        self.model.train()
        best_loss = float('inf')
        early_stopping = EarlyStopping(patience=10, min_delta=0.0001)
        log_file_student = open('training_log_student.csv', 'w', newline='')
        log_writer_student = csv.writer(log_file_student)
        log_writer_student.writerow(['epoch', 'train_loss', 'val_loss', 'train_consistency', 'val_consistency', 'lr'])
        previous_loss = float('inf')
        for epoch in range(epochs):
            total_loss = 0.0
            total_consistency = 0.0
            total_feature = 0.0
            for batch_idx, batch in enumerate(self.train_loader):
                if len(batch) == 3:
                    low_res_batch, high_res_batch, labels = batch
                    labels = labels.to(self.device) - 1
                else:
                    low_res_batch, high_res_batch = batch
                    labels = None
                low_res_batch = low_res_batch.to(self.device)
                high_res_batch = high_res_batch.to(self.device)
                self.optimizer.zero_grad()
                # Student forward
                enhanced_batch = self.model(low_res_batch, labels)
                # Teacher features
                with torch.no_grad():
                    _, teacher_features = self.teacher_model(high_res_batch)
                # Student features (from encoder, before label concat)
                batch_size = low_res_batch.shape[0]
                x = low_res_batch.view(batch_size, -1)
                student_features = self.model.encoder(x)
                projected_student_features = self.feature_projector(student_features)
                # Losses
                reconstruction_loss = F.mse_loss(enhanced_batch, high_res_batch)
                consistency_loss = self.consistency_loss(enhanced_batch, low_res_batch)
                smoothness_loss = self.smoothness_loss(enhanced_batch)
                feature_loss = F.mse_loss(projected_student_features, teacher_features)
                total_batch_loss = (
                    self.reconstruction_weight * reconstruction_loss +
                    self.consistency_weight * consistency_loss +
                    self.smoothness_weight * smoothness_loss +
                    feature_loss_weight * feature_loss
                )
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += total_batch_loss.item()
                total_consistency += consistency_loss.item()
                total_feature += feature_loss.item()
                if batch_idx % 50 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    avg_consistency = total_consistency / (batch_idx + 1)
                    avg_feature = total_feature / (batch_idx + 1)
                    print(f"[Student] Batch [{batch_idx:4d}/{len(self.train_loader)}] - Loss: {avg_loss:.6f}, Consistency: {avg_consistency:.8f}, Feature: {avg_feature:.6f}")
                    print(f"[Student] Early stopping counter: {early_stopping.counter}/{early_stopping.patience}")
            avg_loss = total_loss / len(self.train_loader)
            avg_consistency = total_consistency / len(self.train_loader)
            avg_feature = total_feature / len(self.train_loader)
            print(f"[Student] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}, Consistency: {avg_consistency:.8f}, Feature: {avg_feature:.6f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                # self.save_model("best_constrained_model.pth") # REMOVE repeated saving
            if early_stopping(avg_loss):
                print(f"[Student] Early stopping at epoch {epoch+1}")
                break
            if detect_spike(avg_loss, previous_loss):
                torch.save(self.model.state_dict(), f"spike_student_model_epoch_{epoch+1}.pth")
                print(f"Spike detected! Model saved at epoch {epoch+1}")
            previous_loss = avg_loss
        print(f"[Student] Training complete. Best loss: {best_loss:.6f}")
        # self.model.load_state_dict(torch.load("/project/cc-20250120231604/ssd/users/kwsu/data/trained_model/enhancement_model/best_constrained_model.pth")) # REMOVE repeated loading
        self.model.eval()
        log_writer_student.writerow([epoch+1, avg_loss, best_loss, avg_consistency, avg_consistency, self.optimizer.param_groups[0]['lr']])
        log_file_student.close()

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_consistency = 0.0
        for batch_idx, batch in enumerate(self.train_loader):
            if len(batch) == 3:
                low_res_batch, high_res_batch, labels = batch
                labels = labels.to(self.device) - 1  # shift to zero-based
            else:
                low_res_batch, high_res_batch = batch
                labels = None
            low_res_batch = low_res_batch.to(self.device)
            high_res_batch = high_res_batch.to(self.device)
            self.optimizer.zero_grad()
            enhanced_batch = self.model(low_res_batch, labels)
            reconstruction_loss = F.mse_loss(enhanced_batch, high_res_batch)
            consistency_loss = self.consistency_loss(enhanced_batch, low_res_batch)
            smoothness_loss = self.smoothness_loss(enhanced_batch)
            total_batch_loss = (
                self.reconstruction_weight * reconstruction_loss +
                self.consistency_weight * consistency_loss +
                self.smoothness_weight * smoothness_loss
            )
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += total_batch_loss.item()
            total_consistency += consistency_loss.item()
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
            for batch in self.test_loader:
                if len(batch) == 3:
                    low_res_batch, high_res_batch, labels = batch
                    labels = labels.to(self.device) - 1  # shift to zero-based
                else:
                    low_res_batch, high_res_batch = batch
                    labels = None
                low_res_batch = low_res_batch.to(self.device)
                high_res_batch = high_res_batch.to(self.device)
                enhanced_batch = self.model(low_res_batch, labels)
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
        early_stopping = EarlyStopping(patience=15, min_delta=0.0001)  # Increased patience
        
        # --- CSV logging setup (mimics original pipeline) ---
        log_file = open('training_log_constrained.csv', 'w', newline='')
        log_writer = csv.writer(log_file)
        log_writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_consistency', 'val_consistency', 'lr'])

        for epoch in range(epochs):
            # Train epoch
            train_loss, train_consistency = self.train_epoch()
            
            # Evaluate
            val_loss, val_consistency = self.evaluate()
            
            # Step the scheduler using validation loss
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train - Loss: {train_loss:.6f}, Consistency: {train_consistency:.8f}")
            print(f"  Val   - Loss: {val_loss:.6f}, Consistency: {val_consistency:.8f}")
            print(f"  LR: {current_lr:.6f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"  ✅ Best model so far!")
            
            if val_consistency < 1e-10:
                print(f"  🎯 Perfect constraint satisfaction achieved!")
            
            # Early stopping check
            if early_stopping(val_loss):
                print(f"Training stopped early at epoch {epoch+1}")
                break
            
            # Log metrics
            log_writer.writerow([epoch+1, train_loss, val_loss, train_consistency, val_consistency, current_lr])
            log_file.flush()
        
        # --- Close log file ---
        log_file.close()

        print(f"✅ Training completed! Best loss: {best_loss:.6f}")

        # --- Collate results directory like the original script ---
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        ratio_tag = f"{self.config.high_res_timesteps//self.config.low_res_timesteps}x"
        results_dir = Path(f"/project/cc-20250120231604/ssd/users/kwsu/data/trained_model/enhancement_model/constrained_enhancement_{ratio_tag}_{now}")
        results_dir.mkdir(parents=True, exist_ok=True)
        # Move log file inside results dir
        shutil.move('training_log_constrained.csv', results_dir / 'training_log_constrained.csv')
        print(f"📁 All logs saved to {results_dir}")

        # --- Save final model in results_dir ---
        model_save_path = results_dir / f"constrained_enhancement_model_{ratio_tag}.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Optionally add teacher model, feature projector, etc. if needed
        }, model_save_path)
        print(f"💾 Model saved to {model_save_path}")
        return results_dir
    
    def save_model(self, filename):
        save_path = Path(f"/project/cc-20250120231604/ssd/users/kwsu/data/trained_model/enhancement_model/{filename}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        print(f"💾 Model saved to {save_path}")

    def enhance_data(self, low_res_data):
        """Public helper that mirrors DataEnhancementTeacherStudentModel.enhance_data."""
        self.model.eval()
        with torch.no_grad():
            if isinstance(low_res_data, np.ndarray):
                low_res_data = torch.FloatTensor(low_res_data)
            low_res_data = low_res_data.to(self.device)
            enhanced = self.model(low_res_data)
            return enhanced.cpu().numpy()

    def evaluate_enhancement_quality(self):
        """Compute average MSE of reconstructed signal on the held-out test set."""
        self.model.eval()
        total_mse = 0.0
        total_samples = 0
        with torch.no_grad():
            for batch in self.test_loader:
                low_res_batch, high_res_batch = batch[:2]
                low_res_batch = low_res_batch.to(self.device)
                high_res_batch = high_res_batch.to(self.device)
                recon = self.model(low_res_batch)
                mse = F.mse_loss(recon, high_res_batch)
                total_mse += mse.item() * low_res_batch.size(0)
                total_samples += low_res_batch.size(0)
        return total_mse / max(total_samples, 1)

def visualize_teacher_results(trainer, num_samples=3, output_dir=None):
    """Visualize teacher model results and compare to ground truth."""
    trainer.teacher_model.eval()
    test_iter = iter(trainer.test_loader)
    batch = next(test_iter)
    low_res_batch, high_res_batch = batch[:2]
    high_res_samples = high_res_batch[:num_samples].to(trainer.device)
    with torch.no_grad():
        teacher_recon, _ = trainer.teacher_model(high_res_samples)
    # Convert to numpy
    high_res_cpu = high_res_samples.cpu().numpy()
    teacher_cpu = teacher_recon.cpu().numpy()
    # Plot results
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    feature_names = ['X-axis', 'Y-axis', 'Z-axis']
    for sample_idx in range(num_samples):
        for feature_idx in range(3):
            ax = axes[sample_idx, feature_idx] if num_samples > 1 else axes[feature_idx]
            time_high = np.linspace(0, 300, len(high_res_cpu[sample_idx, :, feature_idx]))
            ax.plot(time_high, high_res_cpu[sample_idx, :, feature_idx], 'g-', linewidth=1, label='True High-res', alpha=0.8)
            ax.plot(time_high, teacher_cpu[sample_idx, :, feature_idx], 'b-', linewidth=2, label='Teacher Output', alpha=0.9)
            ax.set_title(f'Sample {sample_idx+1} - {feature_names[feature_idx]}')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Acceleration (g)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_dir is not None:
        import os
        plt.savefig(os.path.join(output_dir, 'teacher_verification.png'), dpi=300, bbox_inches='tight')
        print(f"📊 Teacher verification plot saved to: {os.path.join(output_dir, 'teacher_verification.png')}")
    else:
        plt.savefig('teacher_verification.png', dpi=300, bbox_inches='tight')
        print("📊 Teacher verification plot saved to: teacher_verification.png")
    plt.show()

def visualize_constrained_results(trainer, num_samples=3, output_dir=None):
    """Visualize results and verify constraints are satisfied"""
    trainer.model.eval()
    test_iter = iter(trainer.test_loader)
    batch = next(test_iter)
    low_res_batch, high_res_batch = batch[:2]
    low_res_samples = low_res_batch[:num_samples].to(trainer.device)
    high_res_samples = high_res_batch[:num_samples].to(trainer.device)
    with torch.no_grad():
        enhanced_samples = trainer.model(low_res_samples)
    # Convert to numpy
    low_res_cpu = low_res_samples.cpu().numpy()
    high_res_cpu = high_res_samples.cpu().numpy()
    enhanced_cpu = enhanced_samples.cpu().numpy()
    # Plot results
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    feature_names = ['X-axis', 'Y-axis', 'Z-axis']
    upsampling_factor = trainer.model.upsampling_factor
    for sample_idx in range(num_samples):
        for feature_idx in range(3):
            ax = axes[sample_idx, feature_idx] if num_samples > 1 else axes[feature_idx]
            time_high = np.linspace(0, 300, len(high_res_cpu[sample_idx, :, feature_idx]))
            time_low = np.linspace(0, 300, len(low_res_cpu[sample_idx, :, feature_idx]))
            ax.plot(time_high, high_res_cpu[sample_idx, :, feature_idx], 'g-', 
                   linewidth=1, label='True High-res', alpha=0.8)
            ax.plot(time_high, enhanced_cpu[sample_idx, :, feature_idx], 'r-', 
                   linewidth=2, label='Enhanced (CONSTRAINED)', alpha=0.9)
            ax.scatter(time_low, low_res_cpu[sample_idx, :, feature_idx], 
                      c='blue', s=50, label='Low-res Input', zorder=5)
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
    if output_dir is not None:
        import os
        plt.savefig(os.path.join(output_dir, 'constrained_enhancement_verification.png'), dpi=300, bbox_inches='tight')
        print(f"📊 Constraint verification plot saved to: {os.path.join(output_dir, 'constrained_enhancement_verification.png')}")
    else:
        plt.savefig('constrained_enhancement_verification.png', dpi=300, bbox_inches='tight')
        print("📊 Constraint verification plot saved to: constrained_enhancement_verification.png")
    plt.show()

def load_paths():
    paths_file = current_dir.parent.parent / "paths.yaml"
    if not paths_file.exists():
        return None
    with open(paths_file) as f:
        return yaml.safe_load(f)

def plot_loss_curves(results_dir):
    """Plot train/val loss curves for the constrained run."""
    import pandas as pd
    results_dir = Path(results_dir)
    csv_path = results_dir / 'training_log_constrained.csv'
    if not csv_path.exists():
        print(f"⚠️  No training log found at {csv_path}, skipping loss plot.")
        return

    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8,4))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Constrained Model Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / 'constrained_loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

### DEMO / PIPELINE FUNCTION ###

def demo_enhancement_pipeline(trainer, num_samples: int = 5, output_dir=None):
    """Mirror of original demo pipeline but using constrained trainer."""
    print("\n🚀 Data Enhancement Pipeline Demo (CONSTRAINED)")
    print("=" * 50)
    test_iter = iter(trainer.test_loader)
    batch = next(test_iter)
    low_res_batch, high_res_batch = batch[:2]
    
    low_res_demo = low_res_batch[:num_samples]
    high_res_demo = high_res_batch[:num_samples]

    print(f"📥 Input: Low-resolution data  Shape: {low_res_demo.shape}")
    enhanced_data = trainer.enhance_data(low_res_demo)

    print(f"\n📤 Output: Enhanced high-resolution data  Shape: {enhanced_data.shape}")
    print(f"   Upsampling factor: {enhanced_data.shape[1] // low_res_demo.shape[1]}x")
    mse = np.mean((enhanced_data - high_res_demo.numpy())**2)
    print(f"\n📊 Reconstruction Quality - MSE: {mse:.6f}  RMSE: {np.sqrt(mse):.6f}")
    # Optionally save
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'demo_summary.txt', 'w') as f:
            f.write(f"Input shape: {low_res_demo.shape}\n")
            f.write(f"Output shape: {enhanced_data.shape}\n")
            f.write(f"Target shape: {high_res_demo.shape}\n")
            f.write(f"MSE: {mse:.6f}\n")
            f.write(f"RMSE: {np.sqrt(mse):.6f}\n")
    return enhanced_data

def main():
    parser = argparse.ArgumentParser(description="Constrained Data Enhancement Training")
    parser.add_argument('--high_ds', type=int, default=100, help='High-res DS rate')
    parser.add_argument('--low_ds', type=int, default=1000, help='Low-res DS rate')
    parser.add_argument('--epochs', type=int, default=300, help='Training epochs')
    parser.add_argument('--teacher_epochs', type=int, default=300, help='Teacher training epochs')
    parser.add_argument('--data_subdir', type=str, default='stage_2_compare_buffer')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
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
        trainer.teacher_model = TeacherModel(config).to(trainer.device)
        trainer.train_teacher(epochs=args.teacher_epochs)
        results_dir = trainer.train(epochs=args.epochs)
        
        # Reduce dataset size in test mode
        if args.test_mode:
            trainer.train_loader.dataset.data = trainer.train_loader.dataset.data[:5000]
            trainer.test_loader.dataset.data = trainer.test_loader.dataset.data[:500]
            print(f"📊 Reduced dataset sizes:")
            print(f"   Train: {len(trainer.train_loader.dataset)} samples")
            print(f"   Test: {len(trainer.test_loader.dataset)} samples")
        
        # Visualize and verify constraints
        print("\n📊 Generating constraint verification plots...")
        visualize_teacher_results(trainer, num_samples=3, output_dir=results_dir)
        visualize_constrained_results(trainer, num_samples=3, output_dir=results_dir)
        
        # Evaluate enhancement quality on test set
        print("\n📊 Evaluating enhancement quality on test set...")
        test_mse = trainer.evaluate_enhancement_quality()
        print(f"   Test MSE: {test_mse:.6f}")

        print(f"\n✅ CONSTRAINED training completed!")
        print(f"🎯 Enhanced data now GUARANTEED to pass through low-res points!")
        # print(f"💾 Model saved as: {model_name}") # REMOVE repeated saving

        # Plot loss curves
        plot_loss_curves(results_dir)
        # plot_teacher_loss_curve() # REMOVE repeated saving
        # plot_student_loss_curve() # REMOVE repeated saving

        # Demo pipeline
        demo_enhancement_pipeline(trainer, output_dir=results_dir)
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 