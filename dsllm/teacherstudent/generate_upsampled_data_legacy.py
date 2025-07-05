#!/usr/bin/env python3
"""
Legacy: Generate Upsampled Data Using Trained Enhancement Model (Old Architecture)

This script:
1. Loads the trained enhancement model (old architecture)
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
import torch.nn as nn
import torch.nn.functional as F

# Add the current directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from teacher_student_model import DataEnhancementConfig

def load_paths():
    paths_file = current_dir.parent.parent / "paths.yaml"
    if not paths_file.exists():
        print(f"⚠️  Warning: paths.yaml not found at {paths_file}")
        return None
    with open(paths_file) as f:
        paths = yaml.safe_load(f)
    return paths

# --- Legacy Encoder/Decoder ---
class LegacySensorEnhancementEncoder(nn.Module):
    """Legacy encoder: conv layers + global average pooling + FC (no LSTM)"""
    def __init__(self, input_timesteps, feature_dim, hidden_dim):
        super().__init__()
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
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, hidden_dim)
    def forward(self, x):
        x = x.transpose(1, 2)
        features = self.conv_layers(x)
        pooled = self.global_pool(features).squeeze(-1)
        encoded = self.fc(pooled)
        return encoded, features

class LegacySensorEnhancementDecoder(nn.Module):
    """Legacy decoder: progressive upsampling, Tanh output, outputs exactly 300 timesteps from 60 input"""
    def __init__(self, low_res_timesteps, high_res_timesteps, feature_dim, hidden_dim):
        super().__init__()
        self.upsample_factor = high_res_timesteps // low_res_timesteps
        self.fc_expand = nn.Linear(hidden_dim, 256 * low_res_timesteps)
        self.upsample_layers = nn.ModuleList()
        current_timesteps = low_res_timesteps
        current_channels = 256
        # Upsample from 60 -> 120 -> 240 -> 300 (last step is stride 1, kernel 61)
        # 60 -> 120 (stride=2), 120 -> 240 (stride=2), 240 -> 300 (stride=1, kernel=61)
        # This ensures exact output length
        # 1st: 60 -> 120
        self.upsample_layers.append(nn.Sequential(
            nn.ConvTranspose1d(current_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        ))
        current_channels = 128
        current_timesteps = 120
        # 2nd: 120 -> 240
        self.upsample_layers.append(nn.Sequential(
            nn.ConvTranspose1d(current_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        ))
        current_channels = 64
        current_timesteps = 240
        # 3rd: 240 -> 300 (stride=1, kernel=61)
        self.upsample_layers.append(nn.Sequential(
            nn.ConvTranspose1d(current_channels, 32, kernel_size=61, stride=1, padding=30),
            nn.BatchNorm1d(32),
            nn.ReLU()
        ))
        current_channels = 32
        current_timesteps = 300
        self.output_layer = nn.Conv1d(current_channels, feature_dim, kernel_size=3, padding=1)
        self.output_activation = nn.Tanh()
    def forward(self, encoded_features):
        batch_size = encoded_features.size(0)
        x = self.fc_expand(encoded_features)
        x = x.view(batch_size, 256, -1)
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)
        # Ensure output is exactly 300 timesteps
        if x.size(-1) > 300:
            x = x[:, :, :300]
        x = self.output_layer(x)
        x = self.output_activation(x)
        x = x.transpose(1, 2)
        return x

# --- Legacy Model Loader ---
def load_enhancement_model_legacy(model_path: str, config: DataEnhancementConfig):
    print(f"🏗️  Loading trained model from {model_path}...")
    encoder = LegacySensorEnhancementEncoder(config.low_res_timesteps, config.feature_dim, config.student_hidden_dim).to(config.device)
    decoder = LegacySensorEnhancementDecoder(config.low_res_timesteps, config.high_res_timesteps, config.feature_dim, config.student_hidden_dim).to(config.device)
    checkpoint = torch.load(model_path, map_location=config.device)
    encoder.load_state_dict(checkpoint['student_encoder_state_dict'])
    decoder.load_state_dict(checkpoint['student_decoder_state_dict'])
    encoder.eval()
    decoder.eval()
    print(f"✅ Model loaded successfully!")
    print(f"📊 Upsampling factor: {checkpoint.get('high_to_low_ratio', 'Unknown')}x")
    return encoder, decoder

def enhance_data_batch_legacy(encoder, decoder, low_res_data, config, batch_size=32):
    enhanced_segments = []
    with torch.no_grad():
        for i in range(0, len(low_res_data), batch_size):
            batch = low_res_data[i:i+batch_size]
            batch_tensor = torch.FloatTensor(np.array(batch)).to(config.device)
            features, _ = encoder(batch_tensor)
            enhanced_batch = decoder(features)
            enhanced_batch_np = enhanced_batch.cpu().numpy().astype(np.float64)
            enhanced_segments.extend(enhanced_batch_np)
    return enhanced_segments

def generate_upsampled_data_legacy():
    print("🚀 Starting Legacy Upsampled Data Generation")
    print("=" * 60)
    print("PURPOSE: Use trained enhancement model (legacy) to upsample 1000DS → 100DS equivalent")
    print("OUTPUT: Save in same format as original processing pipeline")
    print("=" * 60)
    paths = load_paths()
    if paths is None:
        print("❌ Error: Could not load paths.yaml")
        return
    base_output_dir = Path(paths['base_output_dir'])
    input_base_path = base_output_dir / "stage_2_compare_buffer" / "300seconds_500DS"
    output_base_path = base_output_dir / "stage_2_upsampled" / "300seconds_100DS_upsampled"
    output_base_path.mkdir(parents=True, exist_ok=True)
    (output_base_path / "train").mkdir(exist_ok=True)
    (output_base_path / "test").mkdir(exist_ok=True)
    print(f"📂 Paths:")
    print(f"   Input (500DS): {input_base_path}")
    print(f"   Output (upsampled): {output_base_path}")
    print()
    config = DataEnhancementConfig(
        high_res_timesteps=300,  # 100DS = 300 timesteps
        low_res_timesteps=60,    # 500DS = 60 timesteps
        high_res_path=str(input_base_path / "train" / "capture24_train_data_stage2_300seconds_500DS.pkl"),
        low_res_path=str(input_base_path / "train" / "capture24_train_data_stage2_300seconds_500DS.pkl"),
        feature_dim=3,
        teacher_hidden_dim=512,
        student_hidden_dim=256,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model_path = "/project/cc-20250120231604/ssd/users/kwsu/data/trained_model/enhancement_model/enhancement_model_5x_v0.pth"
    if not Path(model_path).exists():
        print(f"❌ Error: Trained model not found at {model_path}")
        return
    encoder, decoder = load_enhancement_model_legacy(model_path, config)
    for split in ["train", "test"]:
        print(f"\n🔄 Processing {split} split...")
        input_data_path = input_base_path / split / f"capture24_{split}_data_stage2_300seconds_500DS.pkl"
        input_labels_path = input_base_path / split / f"capture24_{split}_labels_stage2_300seconds_500DS.pkl"
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
        print(f"   🚀 Enhancing segments...")
        enhanced_segments = enhance_data_batch_legacy(encoder, decoder, low_res_segments, config, batch_size=32)
        print(f"   📤 Enhanced to shape: {enhanced_segments[0].shape if enhanced_segments else 'No data'}")
        print(f"   📊 Upsampling: {low_res_segments[0].shape[0]} → {enhanced_segments[0].shape[0]} timesteps")
        updated_labels = []
        for label in labels:
            updated_label = label.copy()
            updated_label['upsampled'] = True
            updated_label['original_downsample_factor'] = 500
            updated_label['target_downsample_factor'] = 100
            updated_label['enhancement_model'] = 'teacher_student_5x_legacy'
            updated_labels.append(updated_label)
        output_data_path = output_base_path / split / f"capture24_{split}_data_stage2_300seconds_100DS_upsampled.pkl"
        output_labels_path = output_base_path / split / f"capture24_{split}_labels_stage2_300seconds_100DS_upsampled.pkl"
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
            settings['upsampled'] = True
            settings['enhancement_model'] = 'teacher_student_5x_legacy'
            settings['original_downsample_factor'] = 500
            settings['target_downsample_factor'] = 100
            settings['output_tag'] = '300seconds_100DS_upsampled'
            output_settings_path = output_base_path / split / "settings.json"
            with open(output_settings_path, 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"      Settings: {output_settings_path}")
    print(f"\n✅ Legacy upsampled data generation completed!")
    print(f"📁 Output directory: {output_base_path}")
    print(f"📊 Ready for use with SensorLLM pipeline!")
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
    generate_upsampled_data_legacy() 