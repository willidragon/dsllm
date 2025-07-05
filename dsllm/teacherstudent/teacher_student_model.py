import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging
from dataclasses import dataclass
import os
import csv
import shutil
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TeacherStudentConfig:
    """Configuration for teacher-student training"""
    # Data paths
    high_res_path: str = "data/stage_2_compare/300seconds_100DS"
    med_res_path: str = "data/stage_2_compare/300seconds_200DS" 
    low_res_path: str = "data/stage_2_compare/300seconds_1000DS"
    
    # Model parameters
    teacher_hidden_dim: int = 512
    student_hidden_dim: int = 256
    num_classes: int = 10  # Capture24 has 10 activity classes (0-9)
    
    # Training parameters
    teacher_epochs: int = 50
    student_epochs: int = 100
    teacher_lr: float = 1e-3
    student_lr: float = 1e-3
    batch_size: int = 32
    
    # Knowledge distillation parameters
    temperature: float = 4.0
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for hard target loss
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DataEnhancementConfig:
    """Configuration for data enhancement teacher-student training"""
    # Required parameters (no defaults)
    high_res_timesteps: int  # Number of timesteps in high resolution data
    low_res_timesteps: int   # Number of timesteps in low resolution data
    
    # Optional parameters (with defaults)
    # Data paths
    high_res_path: str = "/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_compare/300seconds_100DS/train/capture24_train_data_stage2_300seconds_100DS.pkl"
    low_res_path: str = "/project/cc-20250120231604/ssd/users/kwsu/research/dsllm/dsllm/data/stage_2_compare/300seconds_1000DS/train/capture24_train_data_stage2_300seconds_1000DS.pkl"
    
    # Model parameters
    teacher_hidden_dim: int = 512
    student_hidden_dim: int = 256
    feature_dim: int = 3     # x, y, z accelerometer
    
    # Training parameters
    teacher_epochs: int = 30
    student_epochs: int = 50
    teacher_lr: float = 1e-3
    student_lr: float = 1e-3
    batch_size: int = 16
    
    # Enhancement loss weights
    reconstruction_weight: float = 1.0    # MSE loss weight
    feature_matching_weight: float = 0.5  # Teacher feature matching weight
    adversarial_weight: float = 0.1       # GAN-style loss weight (unused)
    smoothness_weight: float = 1.0        # Temporal smoothness weight
    frequency_weight: float = 1.0         # Frequency domain loss weight
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class SensorDataset(Dataset):
    """Dataset for sensor data with varying granularity"""
    
    def __init__(self, data_path: str, split: str = "train"):
        self.data_path = Path(data_path) / split
        
        # Extract downsample factor from path
        ds_factor = data_path.split('_')[-1]
        
        # Load data
        data_file = self.data_path / f"capture24_{split}_data_stage2_300seconds_{ds_factor}.pkl"
        labels_file = self.data_path / f"capture24_{split}_labels_stage2_300seconds_{ds_factor}.pkl"
        
        # Load raw data
        with open(data_file, 'rb') as f:
            raw_data = pickle.load(f)
        
        with open(labels_file, 'rb') as f:
            raw_labels = pickle.load(f)
        
        # Convert data using robust loading
        self.data = self._safe_convert_to_tensor(raw_data, is_label=False)
        self.labels = self._safe_convert_to_tensor(raw_labels, is_label=True)
        
        logger.info(f"Loaded {split} data from {data_path}: {self.data.shape}, labels: {self.labels.shape}")
    
    def _safe_convert_to_tensor(self, data, is_label=False):
        """Safely convert data to PyTorch tensor"""
        if isinstance(data, torch.Tensor):
            return data.long() if is_label else data.float()
        
        # Special handling for label dictionaries
        if is_label and isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Extract activity_category from label dictionaries
            label_values = []
            for item in data:
                if isinstance(item, dict) and 'activity_category' in item:
                    label_values.append(item['activity_category'])
                else:
                    logger.warning(f"Unexpected label format: {item}")
                    label_values.append(0)  # Default to category 0
            
            # For capture24 dataset, activity_category should already be 0-based indices (0-9)
            # according to the schema. If they're 1-based, we need to adjust.
            unique_categories = sorted(set(label_values))
            logger.info(f"Found activity categories: {unique_categories}")
            
            # Check if categories are 1-based (minimum > 0) and adjust if needed
            if min(unique_categories) > 0:
                # Convert from 1-based to 0-based
                adjusted_labels = [cat - 1 for cat in label_values]
                logger.info(f"Converted 1-based categories to 0-based: {unique_categories} -> {[cat-1 for cat in unique_categories]}")
                return torch.LongTensor(adjusted_labels)
            else:
                # Already 0-based
                logger.info(f"Categories are already 0-based: {unique_categories}")
                return torch.LongTensor(label_values)
        
        # Handle different data structures
        if isinstance(data, list):
            # Convert list to numpy array, handling object arrays
            try:
                # First try simple conversion
                np_data = np.array(data, dtype=np.int64 if is_label else np.float32)
            except (ValueError, TypeError):
                # Handle object arrays or ragged arrays
                if len(data) > 0:
                    # For labels, just flatten and convert
                    if is_label:
                        flat_data = []
                        for item in data:
                            if isinstance(item, (list, np.ndarray)):
                                flat_data.extend(np.array(item).flatten())
                            else:
                                flat_data.append(item)
                        np_data = np.array(flat_data, dtype=np.int64)
                    else:
                        # For data, handle more complex structures
                        # Check if all elements have same shape
                        shapes = []
                        for item in data[:min(10, len(data))]:  # Check first 10 items
                            if isinstance(item, (list, np.ndarray)):
                                item_array = np.array(item) if isinstance(item, list) else item
                                shapes.append(item_array.shape)
                        
                        if len(set(shapes)) == 1 and len(shapes) > 0:
                            # All have same shape
                            np_data = np.array([np.array(item, dtype=np.float32) for item in data])
                        else:
                            # Ragged arrays - find max dimensions and pad
                            max_dims = []
                            for item in data:
                                if isinstance(item, (list, np.ndarray)):
                                    item_array = np.array(item, dtype=np.float32)
                                    for i, dim in enumerate(item_array.shape):
                                        if i >= len(max_dims):
                                            max_dims.append(dim)
                                        else:
                                            max_dims[i] = max(max_dims[i], dim)
                            
                            # Pad all arrays to max dimensions
                            padded_data = []
                            for item in data:
                                if isinstance(item, (list, np.ndarray)):
                                    item_array = np.array(item, dtype=np.float32)
                                    
                                    # Pad to max dimensions
                                    pad_widths = []
                                    for i, max_dim in enumerate(max_dims):
                                        if i < len(item_array.shape):
                                            pad_widths.append((0, max_dim - item_array.shape[i]))
                                        else:
                                            pad_widths.append((0, max_dim))
                                    
                                    if len(pad_widths) > 0:
                                        # Reshape if needed
                                        while len(item_array.shape) < len(max_dims):
                                            item_array = np.expand_dims(item_array, axis=-1)
                                        
                                        padded_item = np.pad(item_array, pad_widths[:len(item_array.shape)], mode='constant')
                                        padded_data.append(padded_item)
                                    else:
                                        padded_data.append(item_array)
                            
                            np_data = np.array(padded_data, dtype=np.float32)
                else:
                    np_data = np.array(data, dtype=np.int64 if is_label else np.float32)
        
        elif isinstance(data, np.ndarray):
            if data.dtype == object:
                # Recursively handle object arrays
                return self._safe_convert_to_tensor(data.tolist(), is_label)
            else:
                np_data = data.astype(np.int64 if is_label else np.float32)
        else:
            np_data = np.array(data, dtype=np.int64 if is_label else np.float32)
        
        # Convert to tensor
        if is_label:
            return torch.LongTensor(np_data)
        else:
            return torch.FloatTensor(np_data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SensorEncoder(nn.Module):
    """Base encoder for sensor data"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Adaptive layer to handle varying input dimensions
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Encode
        features = self.encoder(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits, features

# --- Add temporal smoothness loss ---
def temporal_smoothness_loss(pred, weight=0.1):
    diff = pred[:, 1:, :] - pred[:, :-1, :]
    return weight * torch.mean(diff ** 2)

def activity_aware_smoothness_loss(pred, activity_labels=None, sleep_weight=0.5, active_weight=0.1):
    """
    Activity-aware smoothness loss that applies stronger constraints to sleep periods
    
    Args:
        pred: Predicted data (batch, timesteps, features)
        activity_labels: Activity labels (batch,) where 0 = sleep
        sleep_weight: Weight for smoothness loss during sleep
        active_weight: Weight for smoothness loss during active periods
    """
    # Base temporal smoothness
    diff = pred[:, 1:, :] - pred[:, :-1, :]
    base_smooth_loss = torch.mean(diff ** 2)
    
    if activity_labels is None:
        # No activity labels, use moderate smoothness
        return 0.2 * base_smooth_loss
    
    # Apply different weights based on activity
    total_loss = 0.0
    sleep_mask = (activity_labels == 0)  # Assuming 0 = sleep
    active_mask = ~sleep_mask
    
    if sleep_mask.sum() > 0:
        # Strong smoothness for sleep periods
        sleep_loss = torch.mean(diff[sleep_mask] ** 2)
        total_loss += sleep_weight * sleep_loss
    
    if active_mask.sum() > 0:
        # Moderate smoothness for active periods
        active_loss = torch.mean(diff[active_mask] ** 2)
        total_loss += active_weight * active_loss
    
    return total_loss

# ============================================================
#  STFT-based Spectral Loss                                    
#  Replaces the previous global FFT loss with a short-time     
#  Fourier transform loss that compares magnitude spectra in   
#  local windows, helping the student recover high-frequency   
#  details more relevant for activity recognition.             
# ============================================================

def frequency_domain_loss(pred: torch.Tensor, target: torch.Tensor,
                          n_fft: int = 64, hop_length: Optional[int] = None,
                          weight: float = 0.05) -> torch.Tensor:
    """Short-time Fourier transform (STFT) magnitude loss."""
    pred = pred.float()
    target = target.float()
    if pred.dim() != 3 or target.dim() != 3:
        raise ValueError("Inputs must be 3-D tensors (batch, time/channel, feature)")
    if pred.size(1) < pred.size(2):
        pred = pred.permute(0, 2, 1)
    if target.size(1) < target.size(2):
        target = target.permute(0, 2, 1)
    B, T, C = pred.shape
    pred_flat = pred.reshape(-1, T)
    target_flat = target.reshape(-1, T)
    hop_length = hop_length or n_fft // 4
    device = pred.device
    window = torch.hann_window(n_fft, device=device)
    pred_stft = torch.stft(pred_flat, n_fft=n_fft, hop_length=hop_length,
                           return_complex=True, center=True, pad_mode='reflect',
                           window=window)
    target_stft = torch.stft(target_flat, n_fft=n_fft, hop_length=hop_length,
                             return_complex=True, center=True, pad_mode='reflect',
                             window=window)
    pred_mag = torch.abs(pred_stft)
    target_mag = torch.abs(target_stft)
    freq_loss = F.mse_loss(pred_mag, target_mag)
    return weight * freq_loss

class SensorDataEnhancementDataset(Dataset):
    """Dataset for sensor data enhancement with normalization"""
    _norm_stats = None  # Class variable to store mean/std for normalization

    def __init__(self, high_res_path: str, low_res_path: str, split: str = "train", 
                 load_labels: bool = True):
        # Load high-res data
        with open(high_res_path, 'rb') as f:
            raw_high_data = pickle.load(f)
        self.high_res_data = self._safe_convert_to_tensor(raw_high_data)
        # Load low-res data  
        with open(low_res_path, 'rb') as f:
            raw_low_data = pickle.load(f)
        self.low_res_data = self._safe_convert_to_tensor(raw_low_data)
        
        # Load activity labels if requested (for training)
        self.labels = None
        if load_labels:
            try:
                # Try to load labels from the same directory structure
                high_labels_path = high_res_path.replace('_data_', '_labels_')
                low_labels_path = low_res_path.replace('_data_', '_labels_')
                
                with open(high_labels_path, 'rb') as f:
                    raw_high_labels = pickle.load(f)
                self.labels = self._safe_convert_to_labels(raw_high_labels)
                
                logger.info(f"Loaded activity labels: {self.labels.shape}")
            except FileNotFoundError:
                logger.warning(f"Could not load activity labels from {high_labels_path}")
                logger.warning("Enhancement will proceed without activity-aware loss")
                
        # Ensure same number of samples
        min_samples = min(len(self.high_res_data), len(self.low_res_data))
        self.high_res_data = self.high_res_data[:min_samples]
        self.low_res_data = self.low_res_data[:min_samples]
        if self.labels is not None:
            self.labels = self.labels[:min_samples]
            
        # --- Normalization ---
        if split == "train":
            # Compute mean/std from training set (across all samples and timesteps, per feature)
            mean = self.low_res_data.mean(dim=(0, 1), keepdim=True)
            std = self.low_res_data.std(dim=(0, 1), keepdim=True) + 1e-8
            SensorDataEnhancementDataset._norm_stats = (mean, std)
            logger.info(f"[Normalization] Computed mean: {mean.flatten().tolist()} std: {std.flatten().tolist()}")
        else:
            mean, std = SensorDataEnhancementDataset._norm_stats
        # Apply normalization
        self.low_res_data = (self.low_res_data - mean) / std
        self.high_res_data = (self.high_res_data - mean) / std
        logger.info(f"Loaded {split} enhancement data:")
        logger.info(f"  High-res: {self.high_res_data.shape}")
        logger.info(f"  Low-res: {self.low_res_data.shape}")
        if self.labels is not None:
            logger.info(f"  Labels: {self.labels.shape}")
    
    def _safe_convert_to_tensor(self, data):
        """Convert data to tensor safely"""
        if isinstance(data, torch.Tensor):
            return data.float()
        
        if isinstance(data, list):
            # Convert to numpy first
            data_arrays = [np.array(item, dtype=np.float32) for item in data]
            return torch.FloatTensor(np.array(data_arrays))
        
        return torch.FloatTensor(np.array(data, dtype=np.float32))
    
    def _safe_convert_to_labels(self, labels):
        """Convert labels to tensor safely"""
        if isinstance(labels, torch.Tensor):
            return labels.long()
        
        if isinstance(labels, list):
            # Handle label dictionaries (common in capture24)
            if len(labels) > 0 and isinstance(labels[0], dict):
                label_values = []
                for item in labels:
                    if isinstance(item, dict) and 'activity_category' in item:
                        label_values.append(item['activity_category'])
                    else:
                        label_values.append(0)  # Default
                return torch.LongTensor(label_values)
            else:
                # Already numeric labels
                return torch.LongTensor(labels)
        
        return torch.LongTensor(np.array(labels, dtype=np.int64))
    
    def __len__(self):
        return len(self.high_res_data)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.low_res_data[idx], self.high_res_data[idx], self.labels[idx]
        else:
            return self.low_res_data[idx], self.high_res_data[idx]

class SensorEnhancementEncoder(nn.Module):
    """Encoder for extracting features from sensor data with temporal awareness"""
    
    def __init__(self, input_timesteps: int, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.input_timesteps = input_timesteps
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Convolutional feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(feature_dim, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # LSTM for temporal modeling (replacing global average pooling)
        self.lstm = nn.LSTM(256, hidden_dim // 2, num_layers=2, 
                           batch_first=True, bidirectional=True, dropout=0.1)
        
        # Final projection
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x: (batch, timesteps, features) -> (batch, features, timesteps)
        x = x.transpose(1, 2)
        
        # Convolutional processing
        features = self.conv_layers(x)  # (batch, 256, timesteps)
        
        # Transpose for LSTM: (batch, timesteps, features)
        features = features.transpose(1, 2)  # (batch, timesteps, 256)
        
        # LSTM processing to preserve temporal information
        lstm_out, (hidden, cell) = self.lstm(features)  # (batch, timesteps, hidden_dim)
        
        # Use final hidden state as encoded features
        # Take last timestep output as the encoded representation
        encoded = lstm_out[:, -1, :]  # (batch, hidden_dim)
        encoded = self.fc(encoded)  # (batch, hidden_dim)
        
        return encoded, features

class SensorEnhancementDecoder(nn.Module):
    """Decoder for reconstructing high-res sensor data with progressive upsampling"""
    
    def __init__(self, low_res_timesteps: int, high_res_timesteps: int, 
                 feature_dim: int, hidden_dim: int):
        super().__init__()
        self.low_res_timesteps = low_res_timesteps
        self.high_res_timesteps = high_res_timesteps
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.upsample_factor = high_res_timesteps // low_res_timesteps
        
        # Project encoded features back to sequence with residual connection
        self.fc_expand = nn.Linear(hidden_dim, 128 * low_res_timesteps)
        
        # Progressive upsampling with smaller steps and residual connections
        self.upsample_layers = nn.ModuleList()
        current_timesteps = low_res_timesteps
        current_channels = 128
        
        # Progressive upsampling in smaller steps
        while current_timesteps < high_res_timesteps:
            # Smaller upsampling steps (max 2x at a time)
            next_timesteps = min(current_timesteps * 2, high_res_timesteps)
            next_channels = max(current_channels // 2, 32)
            
            # Use smaller kernels and proper padding
            self.upsample_layers.append(nn.Sequential(
                nn.ConvTranspose1d(current_channels, next_channels, 
                                 kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm1d(next_channels),
                nn.ReLU(),
                # Add a smoothing convolution
                nn.Conv1d(next_channels, next_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(next_channels),
                nn.ReLU()
            ))
            
            current_timesteps = next_timesteps
            current_channels = next_channels
        
        # Final output layer - no activation to avoid clipping
        self.output_layer = nn.Sequential(
            nn.Conv1d(current_channels, feature_dim, kernel_size=3, padding=1),
            # No Tanh activation - let the data be in its natural range
        )
        
    def forward(self, encoded_features):
        batch_size = encoded_features.size(0)
        
        # Expand to sequence
        x = self.fc_expand(encoded_features)  # (batch, 128 * low_res_timesteps)
        x = x.view(batch_size, 128, self.low_res_timesteps)  # (batch, 128, low_res_timesteps)
        
        # Progressive upsampling with residual connections
        for i, upsample_layer in enumerate(self.upsample_layers):
            x_new = upsample_layer(x)
            
            # Handle exact target length
            if x_new.size(-1) > self.high_res_timesteps:
                x_new = x_new[:, :, :self.high_res_timesteps]
            
            # Skip connection if dimensions match
            if x.size(-1) * 2 == x_new.size(-1) and x.size(1) == x_new.size(1):
                # Upsample x to match x_new and add residual
                x_upsampled = F.interpolate(x, size=x_new.size(-1), mode='linear', align_corners=False)
                x = x_new + x_upsampled
            else:
                x = x_new
        
        # Final output - no activation to preserve natural data range
        x = self.output_layer(x)  # (batch, feature_dim, high_res_timesteps)
        
        # Transpose back to (batch, timesteps, features)
        x = x.transpose(1, 2)
        
        return x

class TeacherStudentModel:
    """Teacher-Student model for sensor data with knowledge distillation"""
    
    def __init__(self, config: TeacherStudentConfig):
        self.config = config
        
        # Clear GPU memory before initialization if using CUDA
        if config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Load datasets
        self.high_res_train = SensorDataset(config.high_res_path, "train")
        self.high_res_test = SensorDataset(config.high_res_path, "test")
        self.low_res_train = SensorDataset(config.low_res_path, "train")
        self.low_res_test = SensorDataset(config.low_res_path, "test")
        
        # Create data loaders
        self.high_res_train_loader = DataLoader(
            self.high_res_train, batch_size=config.batch_size, shuffle=True
        )
        self.high_res_test_loader = DataLoader(
            self.high_res_test, batch_size=config.batch_size, shuffle=False
        )
        self.low_res_train_loader = DataLoader(
            self.low_res_train, batch_size=config.batch_size, shuffle=True
        )
        self.low_res_test_loader = DataLoader(
            self.low_res_test, batch_size=config.batch_size, shuffle=False
        )
        
        # Get input dimensions
        high_res_sample = self.high_res_train.data[0]
        low_res_sample = self.low_res_train.data[0]
        
        high_res_input_dim = np.prod(high_res_sample.shape)
        low_res_input_dim = np.prod(low_res_sample.shape)
        
        # Initialize models on CPU first, then move to device
        try:
            self.teacher = SensorEncoder(
                input_dim=high_res_input_dim,
                hidden_dim=config.teacher_hidden_dim,
                num_classes=config.num_classes
            )
            
            self.student = SensorEncoder(
                input_dim=low_res_input_dim,
                hidden_dim=config.student_hidden_dim,
                num_classes=config.num_classes
            )
            
            # Move models to device with error handling
            if config.device == "cuda" and torch.cuda.is_available():
                try:
                    self.teacher = self.teacher.to(config.device)
                    self.student = self.student.to(config.device)
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        logger.warning(f"CUDA error when moving models to GPU: {e}")
                        logger.warning("Falling back to CPU")
                        config.device = "cpu"
                        self.config.device = "cpu"
                    else:
                        raise e
            
        except Exception as e:
            logger.error(f"Error during model initialization: {e}")
            raise e
        
        # Optimizers
        self.teacher_optimizer = optim.Adam(self.teacher.parameters(), lr=config.teacher_lr)
        self.student_optimizer = optim.Adam(self.student.parameters(), lr=config.student_lr)
        
        # Loss functions
        self.hard_loss = nn.CrossEntropyLoss()
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')
        
        logger.info(f"Teacher input dim: {high_res_input_dim}, Student input dim: {low_res_input_dim}")
        logger.info(f"Models initialized successfully on device: {config.device}")
    
    def train_teacher(self):
        """Train teacher encoder to extract good features from high-res data"""
        logger.info("Training teacher encoder on high-resolution data...")
        
        self.teacher.train()
        best_acc = 0.0
        
        for epoch in range(self.config.teacher_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(self.high_res_train_loader):
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                self.teacher_optimizer.zero_grad()
                
                logits, _ = self.teacher(data)
                loss = self.hard_loss(logits, target)
                
                loss.backward()
                self.teacher_optimizer.step()
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            acc = 100. * correct / total
            avg_loss = total_loss / len(self.high_res_train_loader)
            if not self.is_distributed or self.local_rank == 0:
                logger.info(f"Teacher Epoch {epoch+1}/{self.config.teacher_epochs}: "
                           f"Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
            
            # Evaluate teacher
            test_acc = self.evaluate_teacher()
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.teacher.state_dict(), 'best_teacher.pth')
        
            if not self.is_distributed or self.local_rank == 0:
                logger.info(f"Teacher training completed. Best test accuracy: {best_acc:.2f}%")
    
    def evaluate_teacher(self):
        """Evaluate teacher model"""
        self.teacher.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.high_res_test_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                logits, _ = self.teacher(data)
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        acc = 100. * correct / total
        logger.info(f"Teacher Test Accuracy: {acc:.2f}%")
        return acc
    
    def distillation_loss(self, student_logits, teacher_logits, target, temperature, alpha, beta):
        """Compute knowledge distillation loss"""
        # Soft targets from teacher
        teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
        
        # KL divergence loss (knowledge distillation)
        distill_loss = self.soft_loss(student_log_probs, teacher_probs) * (temperature ** 2)
        
        # Hard target loss
        hard_loss = self.hard_loss(student_logits, target)
        
        # Combined loss
        total_loss = alpha * distill_loss + beta * hard_loss
        
        return total_loss, distill_loss, hard_loss
    
    def train_student(self):
        """Train student model with knowledge distillation"""
        logger.info("Training student model with knowledge distillation...")
        
        # Load best teacher
        self.teacher.load_state_dict(torch.load('best_teacher.pth'))
        self.teacher.eval()
        
        self.student.train()
        best_acc = 0.0
        
        for epoch in range(self.config.student_epochs):
            total_loss = 0.0
            total_distill_loss = 0.0
            total_hard_loss = 0.0
            correct = 0
            total = 0
            
            # Align data loaders
            high_res_iter = iter(self.high_res_train_loader)
            
            for batch_idx, batch_data in enumerate(self.low_res_train_loader):
                # Handle both cases: dataset may return (low_res, high_res, labels) or (low_res, high_res)
                if len(batch_data) == 3:
                    low_res_data, high_res_data, activity_labels = batch_data
                    activity_labels = activity_labels.to(self.config.device)
                else:
                    low_res_data, high_res_data = batch_data
                    activity_labels = None
                low_res_data = low_res_data.to(self.config.device)
                high_res_data = high_res_data.to(self.config.device)
                
                # Get corresponding high-res data for teacher
                try:
                    high_res_data, _ = next(high_res_iter)
                    high_res_data = high_res_data.to(self.config.device)
                except StopIteration:
                    high_res_iter = iter(self.high_res_train_loader)
                    high_res_data, _ = next(high_res_iter)
                    high_res_data = high_res_data.to(self.config.device)
                
                # Match batch sizes
                min_batch = min(low_res_data.size(0), high_res_data.size(0))
                low_res_data = low_res_data[:min_batch]
                high_res_data = high_res_data[:min_batch]
                activity_labels = activity_labels[:min_batch] if activity_labels is not None else None
                
                self.student_optimizer.zero_grad()
                
                # Student forward pass
                student_logits, _ = self.student(low_res_data)
                
                # Teacher forward pass (no gradients)
                with torch.no_grad():
                    teacher_logits, _ = self.teacher(high_res_data)
                
                # Compute distillation loss
                loss, distill_loss, hard_loss = self.distillation_loss(
                    student_logits, teacher_logits, activity_labels,
                    self.config.temperature, self.config.alpha, self.config.beta
                )
                
                loss.backward()
                self.student_optimizer.step()
                
                total_loss += loss.item()
                total_distill_loss += distill_loss.item()
                total_hard_loss += hard_loss.item()
                
                pred = student_logits.argmax(dim=1)
                correct += pred.eq(activity_labels).sum().item() if activity_labels is not None else 0
                total += len(low_res_data)
            
            acc = 100. * correct / total
            avg_loss = total_loss / len(self.low_res_train_loader)
            avg_distill_loss = total_distill_loss / len(self.low_res_train_loader)
            avg_hard_loss = total_hard_loss / len(self.low_res_train_loader)
            if not self.is_distributed or self.local_rank == 0:
                logger.info(f"Student Epoch {epoch+1}/{self.config.student_epochs}: "
                           f"Total Loss: {avg_loss:.4f}, Distill Loss: {avg_distill_loss:.4f}, Hard Loss: {avg_hard_loss:.4f}, Acc: {acc:.2f}%")
            
            # Evaluate student
            test_acc = self.evaluate_student()
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.student.state_dict(), 'best_student.pth')
        
            if not self.is_distributed or self.local_rank == 0:
                logger.info(f"Student training completed. Best test accuracy: {best_acc:.2f}%")
    
    def evaluate_student(self):
        """Evaluate student model"""
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.low_res_test_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                logits, _ = self.student(data)
                pred = logits.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        acc = 100. * correct / total
        logger.info(f"Student Test Accuracy: {acc:.2f}%")
        return acc
    
    def train(self):
        """Complete training pipeline"""
        logger.info("Starting teacher-student training pipeline...")
        
        # Step 1: Train teacher on high-resolution data
        self.train_teacher()
        
        # Step 2: Train student with knowledge distillation
        results_dir = self.train_student()
        
        logger.info("Training pipeline completed!")
        return results_dir
    
    def compare_models(self):
        """Compare teacher and student performance"""
        logger.info("\n=== Model Comparison ===")
        
        # Evaluate teacher
        teacher_acc = self.evaluate_teacher()
        
        # Evaluate student
        student_acc = self.evaluate_student()
        
        # Model sizes
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_params = sum(p.numel() for p in self.student.parameters())
        
        logger.info(f"Teacher: {teacher_acc:.2f}% accuracy, {teacher_params:,} parameters")
        logger.info(f"Student: {student_acc:.2f}% accuracy, {student_params:,} parameters")
        logger.info(f"Compression ratio: {teacher_params/student_params:.2f}x")

class DataEnhancementTeacherStudentModel:
    """Teacher-Student model for sensor data enhancement/super-resolution"""
    
    def __init__(self, config: DataEnhancementConfig):
        self.config = config
        # --- DDP distributed setup ---
        self.is_distributed = False
        self.local_rank = 0
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self.is_distributed = True
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl")
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1
        # Clear GPU memory before initialization if using CUDA
        if config.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Load datasets
        self.train_dataset = SensorDataEnhancementDataset(
            config.high_res_path, config.low_res_path, "train", load_labels=True
        )
        self.test_dataset = SensorDataEnhancementDataset(
            config.high_res_path, config.low_res_path, "test", load_labels=True
        )
        # Create distributed sampler if using DDP
        train_sampler = None
        test_sampler = None
        if self.is_distributed:
            train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, 
            shuffle=(train_sampler is None), sampler=train_sampler,
            num_workers=0, pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=config.batch_size, 
            shuffle=False, sampler=test_sampler,
            num_workers=0, pin_memory=True
        )
        # Initialize models
        try:
            self.teacher_encoder = SensorEnhancementEncoder(
                input_timesteps=30,  # Downsampled high-res (300 -> 30)
                feature_dim=config.feature_dim,
                hidden_dim=config.teacher_hidden_dim
            )
            self.student_encoder = SensorEnhancementEncoder(
                input_timesteps=config.low_res_timesteps,
                feature_dim=config.feature_dim,
                hidden_dim=config.student_hidden_dim
            )
            self.student_decoder = SensorEnhancementDecoder(
                low_res_timesteps=config.low_res_timesteps,
                high_res_timesteps=config.high_res_timesteps,
                feature_dim=config.feature_dim,
                hidden_dim=config.student_hidden_dim
            )
            self.feature_projector = nn.Linear(config.student_hidden_dim, config.teacher_hidden_dim)
            self.teacher_decoder = SensorEnhancementDecoder(
                low_res_timesteps=30,
                high_res_timesteps=config.high_res_timesteps,
                feature_dim=config.feature_dim,
                hidden_dim=config.teacher_hidden_dim
            )
            # Move to device
            self.device = torch.device(config.device)
            self.teacher_encoder = self.teacher_encoder.to(self.device)
            self.teacher_decoder = self.teacher_decoder.to(self.device)
            self.student_encoder = self.student_encoder.to(self.device)
            self.student_decoder = self.student_decoder.to(self.device)
            self.feature_projector = self.feature_projector.to(self.device)
            # Wrap in DDP if distributed
            if self.is_distributed:
                self.teacher_encoder = DDP(self.teacher_encoder, device_ids=[self.local_rank])
                self.teacher_decoder = DDP(self.teacher_decoder, device_ids=[self.local_rank])
                self.student_encoder = DDP(self.student_encoder, device_ids=[self.local_rank])
                self.student_decoder = DDP(self.student_decoder, device_ids=[self.local_rank])
                self.feature_projector = DDP(self.feature_projector, device_ids=[self.local_rank])
        except Exception as e:
            logger.error(f"Error during model initialization: {e}")
            raise e
        # Optimizers
        self.teacher_optimizer = optim.Adam(
            list(self.teacher_encoder.parameters()) + list(self.teacher_decoder.parameters()),
            lr=config.teacher_lr
        )
        self.student_optimizer = optim.Adam(
            list(self.student_encoder.parameters()) + 
            list(self.student_decoder.parameters()) + 
            list(self.feature_projector.parameters()),
            lr=config.student_lr
        )
        # Loss functions
        self.reconstruction_loss = nn.MSELoss()
        self.feature_matching_loss = nn.MSELoss()
        if not self.is_distributed or self.local_rank == 0:
            logger.info(f"Data enhancement models initialized successfully on device: {config.device}")
            logger.info(f"Teacher processes: {config.high_res_timesteps} timesteps")
            logger.info(f"Student reconstructs: {config.low_res_timesteps} -> {config.high_res_timesteps} timesteps")
            logger.info(f"Feature projector: {config.student_hidden_dim} -> {config.teacher_hidden_dim} dimensions")
    
    def train_teacher(self, patience: int = 10):
        if not self.is_distributed or self.local_rank == 0:
            logger.info("Training teacher encoder on high-resolution data...")
        best_loss = float('inf')
        epochs_since_improvement = 0
        stop_early = False
        log_teacher = (not self.is_distributed or self.local_rank == 0)
        tb_writer = None
        if log_teacher:
            teacher_logfile = open('training_log_teacher.csv', 'w', newline='')
            teacher_logwriter = csv.writer(teacher_logfile)
            teacher_logwriter.writerow(['epoch', 'total_loss', 'best_loss_so_far', 'epochs_since_improvement', 'early_stopped'])
            tb_writer = SummaryWriter(log_dir="runs/teacher")
        for epoch in range(self.config.teacher_epochs):
            if stop_early:
                if log_teacher:
                    teacher_logwriter.writerow([epoch+1, avg_loss, best_loss, epochs_since_improvement, 1])
                    teacher_logfile.flush()
                if tb_writer is not None:
                    tb_writer.add_scalar('EarlyStopping/triggered', 1, epoch+1)
                    tb_writer.close()
                break
            total_loss = 0.0
            for batch_idx, batch_data in enumerate(self.train_loader):
                # Handle both cases: dataset may return (low_res, high_res, labels) or (low_res, high_res)
                if len(batch_data) == 3:
                    low_res_data, high_res_data, _ = batch_data  # Teacher does not use activity labels
                else:
                    low_res_data, high_res_data = batch_data
                # Move tensors to device (low_res_data is unused but moved for completeness)
                low_res_data = low_res_data.to(self.device)
                high_res_data = high_res_data.to(self.device)
                self.teacher_optimizer.zero_grad()
                downsampled_data = high_res_data[:, ::10, :]
                teacher_features, _ = self.teacher_encoder(downsampled_data)
                reconstructed_data = self.teacher_decoder(teacher_features)
                loss = self.reconstruction_loss(reconstructed_data, high_res_data)
                loss.backward()
                self.teacher_optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            if not self.is_distributed or self.local_rank == 0:
                logger.info(f"[Teacher][Epoch {epoch+1}/{self.config.teacher_epochs}] Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | Patience: {epochs_since_improvement}/{patience}")
            if log_teacher and not stop_early:
                teacher_logwriter.writerow([epoch+1, avg_loss, best_loss, epochs_since_improvement, 0])
                teacher_logfile.flush()
            if tb_writer is not None:
                tb_writer.add_scalar('Loss/total', avg_loss, epoch+1)
                tb_writer.add_scalar('EarlyStopping/best_loss_so_far', best_loss, epoch+1)
                tb_writer.add_scalar('EarlyStopping/epochs_since_improvement', epochs_since_improvement, epoch+1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    stop_early = True
                    if self.is_distributed:
                        dist.barrier()
        if log_teacher:
            teacher_logfile.close()
        if tb_writer is not None:
            tb_writer.close()
        if not self.is_distributed or self.local_rank == 0:
            logger.info("Teacher training completed!")
    
    def train_student(self, patience: int = 10):
        if not self.is_distributed or self.local_rank == 0:
            logger.info("Training student for data enhancement...")
        best_loss = float('inf')
        epochs_since_improvement = 0
        stop_early = False
        has_labels = hasattr(self.train_dataset, 'labels') and self.train_dataset.labels is not None
        log_student = (not self.is_distributed or self.local_rank == 0)
        tb_writer = None
        if log_student:
            student_logfile = open('training_log_student.csv', 'w', newline='')
            student_logwriter = csv.writer(student_logfile)
            student_logwriter.writerow(['epoch', 'total_loss', 'recon_loss', 'feature_loss', 'smooth_loss', 'freq_loss', 'best_loss_so_far', 'epochs_since_improvement', 'early_stopped'])
            tb_writer = SummaryWriter(log_dir="runs/student")
        for epoch in range(self.config.student_epochs):
            if stop_early:
                if log_student:
                    student_logwriter.writerow([epoch+1, avg_loss, avg_recon, avg_feature, avg_smooth, avg_freq, best_loss, epochs_since_improvement, 1])
                    student_logfile.flush()
                if tb_writer is not None:
                    tb_writer.add_scalar('EarlyStopping/triggered', 1, epoch+1)
                    tb_writer.close()
                break
            total_loss = 0.0
            total_recon_loss = 0.0
            total_feature_loss = 0.0
            total_smooth_loss = 0.0
            total_freq_loss = 0.0
            for batch_idx, batch_data in enumerate(self.train_loader):
                # Handle both cases: dataset may return (low_res, high_res, labels) or (low_res, high_res)
                if len(batch_data) == 3:
                    low_res_data, high_res_data, activity_labels = batch_data
                    activity_labels = activity_labels.to(self.device)
                else:
                    low_res_data, high_res_data = batch_data
                    activity_labels = None
                low_res_data = low_res_data.to(self.device)
                high_res_data = high_res_data.to(self.device)
                self.student_optimizer.zero_grad()
                student_features, _ = self.student_encoder(low_res_data)
                reconstructed_data = self.student_decoder(student_features)
                downsampled_high_res = high_res_data[:, ::10, :]
                with torch.no_grad():
                    teacher_features, _ = self.teacher_encoder(downsampled_high_res)
                projected_student_features = self.feature_projector(student_features)
                recon_loss = self.reconstruction_loss(reconstructed_data, high_res_data)
                feature_loss = self.feature_matching_loss(projected_student_features, teacher_features)
                if has_labels:
                    smooth_loss = activity_aware_smoothness_loss(
                        reconstructed_data, 
                        activity_labels=activity_labels,
                        sleep_weight=1.5,
                        active_weight=0.3
                    )
                else:
                    smooth_loss = activity_aware_smoothness_loss(reconstructed_data)
                freq_loss = frequency_domain_loss(reconstructed_data, high_res_data)
                total_loss_batch = (
                    self.config.reconstruction_weight * recon_loss +
                    self.config.feature_matching_weight * feature_loss +
                    self.config.smoothness_weight * smooth_loss +
                    self.config.frequency_weight * freq_loss
                )
                total_loss_batch.backward()
                self.student_optimizer.step()
                total_loss += total_loss_batch.item()
                total_recon_loss += recon_loss.item()
                total_feature_loss += feature_loss.item()
                total_smooth_loss += smooth_loss.item()
                total_freq_loss += freq_loss.item()
            avg_loss = total_loss / len(self.train_loader)
            avg_recon = total_recon_loss / len(self.train_loader)
            avg_feature = total_feature_loss / len(self.train_loader)
            avg_smooth = total_smooth_loss / len(self.train_loader)
            avg_freq = total_freq_loss / len(self.train_loader)
            if not self.is_distributed or self.local_rank == 0:
                logger.info(f"[Student][Epoch {epoch+1}/{self.config.student_epochs}] Total: {avg_loss:.6f}, Recon: {avg_recon:.6f}, Feature: {avg_feature:.6f}, Smooth: {avg_smooth:.6f}, Freq: {avg_freq:.6f} | Best: {best_loss:.6f} | Patience: {epochs_since_improvement}/{patience}")
            if log_student and not stop_early:
                student_logwriter.writerow([epoch+1, avg_loss, avg_recon, avg_feature, avg_smooth, avg_freq, best_loss, epochs_since_improvement, 0])
                student_logfile.flush()
            if tb_writer is not None:
                tb_writer.add_scalar('Loss/total', avg_loss, epoch+1)
                tb_writer.add_scalar('Loss/recon', avg_recon, epoch+1)
                tb_writer.add_scalar('Loss/feature', avg_feature, epoch+1)
                tb_writer.add_scalar('Loss/smooth', avg_smooth, epoch+1)
                tb_writer.add_scalar('Loss/freq', avg_freq, epoch+1)
                tb_writer.add_scalar('EarlyStopping/best_loss_so_far', best_loss, epoch+1)
                tb_writer.add_scalar('EarlyStopping/epochs_since_improvement', epochs_since_improvement, epoch+1)
            if avg_loss < best_loss:
                best_loss = avg_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= patience:
                    stop_early = True
                    if self.is_distributed:
                        dist.barrier()
        if log_student:
            student_logfile.close()
        if tb_writer is not None:
            tb_writer.close()
        if not self.is_distributed or self.local_rank == 0:
            logger.info(f"Student training completed! Best loss: {best_loss:.6f}")
            # --- Move all outputs to a results folder ---
            now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_tags = []
            model_tags.append(f"{self.config.high_res_timesteps//self.config.low_res_timesteps}x")
            model_tags.append("stftloss")
            model_tags.append("activityaware")
            results_dir = Path(f"/project/cc-20250120231604/ssd/users/kwsu/data/trained_model/enhancement_model/enhancement_model_{'_'.join(model_tags)}_{now}")
            results_dir.mkdir(parents=True, exist_ok=True)
            for f in Path('trained').glob('best_enhancement_model.pth'):
                shutil.move(str(f), results_dir / f.name)
            for logf in ["training_log_teacher.csv", "training_log_student.csv"]:
                if Path(logf).exists():
                    shutil.move(logf, results_dir / logf)
            for tb_dir in ["runs/teacher", "runs/student"]:
                if Path(tb_dir).exists():
                    shutil.move(tb_dir, results_dir / Path(tb_dir).name)
            logger.info(f"All outputs moved to {results_dir}")
            return results_dir
        return None
    
    def enhance_data(self, low_res_data):
        """Use trained student to enhance low-resolution data"""
        self.student_encoder.eval()
        self.student_decoder.eval()
        
        with torch.no_grad():
            if isinstance(low_res_data, np.ndarray):
                low_res_data = torch.FloatTensor(low_res_data)
            
            low_res_data = low_res_data.to(self.device)
            
            # Encode and decode
            student_features, _ = self.student_encoder(low_res_data)
            enhanced_data = self.student_decoder(student_features)
            
            return enhanced_data.cpu().numpy()
    
    def train(self):
        """Train complete enhancement pipeline"""
        logger.info("Starting data enhancement training pipeline...")
        
        # Train teacher
        self.train_teacher()
        
        # Train student
        results_dir = self.train_student()
        
        logger.info("Enhancement training pipeline completed!")
        return results_dir
    
    def evaluate_enhancement_quality(self):
        """Evaluate reconstruction quality on test set"""
        self.student_encoder.eval()
        self.student_decoder.eval()
        
        total_mse = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                # Handle both cases: with and without labels
                if len(batch_data) == 3:
                    low_res_data, high_res_data, _ = batch_data  # Ignore labels during evaluation
                else:
                    low_res_data, high_res_data = batch_data
                    
                low_res_data = low_res_data.to(self.device)
                high_res_data = high_res_data.to(self.device)
                
                # Enhance data
                student_features, _ = self.student_encoder(low_res_data)
                reconstructed_data = self.student_decoder(student_features)
                
                # Compute MSE
                mse = F.mse_loss(reconstructed_data, high_res_data)
                total_mse += mse.item() * low_res_data.size(0)
                total_samples += low_res_data.size(0)
        
        avg_mse = total_mse / total_samples
        logger.info(f"Enhancement Quality - Test MSE: {avg_mse:.6f}")
        return avg_mse

    @staticmethod
    def load_model(model_path: str) -> 'DataEnhancementModel':
        """Load a trained enhancement model from disk."""
        checkpoint = torch.load(model_path)
        config = checkpoint['config']
        
        model = DataEnhancementModel(config)
        model.teacher.load_state_dict(checkpoint['teacher_state_dict'])
        model.student.load_state_dict(checkpoint['student_state_dict'])
        
        # Print model info
        ratio = checkpoint.get('high_to_low_ratio', 
                             config.high_res_timesteps // config.low_res_timesteps)
        print(f"📈 Loaded enhancement model ({ratio}x upsampling)")
        print(f"   High-res timesteps: {config.high_res_timesteps}")
        print(f"   Low-res timesteps: {config.low_res_timesteps}")
        print(f"   Feature dimensions: {config.feature_dim}")
        
        return model

# Example usage
if __name__ == "__main__":
    # Configuration
    config = TeacherStudentConfig(
        high_res_path="research/dsllm/dsllm/data/stage_2_compare/300seconds_100DS",
        low_res_path="research/dsllm/dsllm/data/stage_2_compare/300seconds_1000DS",
        teacher_epochs=20,
        student_epochs=30,
        batch_size=16,
        num_classes=8  # Adjust based on your actual number of activity classes
    )
    
    # Create and train model
    model = TeacherStudentModel(config)
    model.train()
    model.compare_models() 