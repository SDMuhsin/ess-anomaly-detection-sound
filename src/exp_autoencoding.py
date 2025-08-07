#!/usr/bin/env python
"""
Research Experiments Runner for MIMII Dataset
Supports 10 different baseline models for anomaly detection research
"""
import argparse
import csv
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("research_experiments.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

########################################################################
# Model Definitions
########################################################################

class SimpleAutoencoder(nn.Module):
    """Simple 3-layer autoencoder (baseline)"""
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class DeepAutoencoder(nn.Module):
    """Deep autoencoder with multiple hidden layers"""
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for anomaly detection"""
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for spectral features"""
    def __init__(self, input_dim, n_mels=64, frames=5):
        super().__init__()
        self.n_mels = n_mels
        self.frames = frames
        self.input_dim = input_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened size after convolutions
        self.encoded_h = self.n_mels // 4  # Two MaxPool2d with kernel_size=2
        self.encoded_w = self.frames // 4
        self.flat_size = 64 * self.encoded_h * self.encoded_w
        
        self.fc_encode = nn.Linear(self.flat_size, 32)
        self.fc_decode = nn.Linear(32, self.flat_size)
        
        # Decoder - use ConvTranspose2d with proper output_padding to restore exact dimensions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2, padding=0, output_padding=(self.n_mels % 4, self.frames % 4)),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.n_mels, self.frames)
        
        # Encode
        encoded = self.encoder(x)
        flat = encoded.view(batch_size, -1)
        bottleneck = self.fc_encode(flat)
        
        # Decode
        decoded_flat = self.fc_decode(bottleneck)
        decoded_conv = decoded_flat.view(batch_size, 64, self.encoded_h, self.encoded_w)
        output = self.decoder(decoded_conv)
        
        # Ensure output matches input dimensions exactly
        output = output[:, :, :self.n_mels, :self.frames]
        
        return output.view(batch_size, self.input_dim)

class LSTMAutoencoder(nn.Module):
    """LSTM-based Autoencoder for temporal patterns"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True)
        
        # Decoder LSTM
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(-1)  # Add feature dimension
        
        # Encode
        encoded, (hidden, cell) = self.encoder_lstm(x)
        
        # Use last hidden state as context
        context = encoded[:, -1:, :].repeat(1, self.input_dim, 1)
        
        # Decode
        decoded, _ = self.decoder_lstm(context, (hidden, cell))
        output = self.output_layer(decoded)
        
        return output.squeeze(-1)

class AttentionAutoencoder(nn.Module):
    """Autoencoder with attention mechanism"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(64, num_heads=8, batch_first=True)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode
        encoded = self.encoder(x)
        
        # Reshape for attention (treat features as sequence)
        encoded = encoded.unsqueeze(1)  # Add sequence dimension
        
        # Apply attention
        attended, _ = self.attention(encoded, encoded, encoded)
        attended = attended.squeeze(1)
        
        # Decode
        decoded = self.decoder(attended)
        return decoded

class ResidualAutoencoder(nn.Module):
    """Autoencoder with residual connections"""
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 128)
        
        # Encoder blocks
        self.enc_block1 = self._make_residual_block(128, 128)
        self.enc_block2 = self._make_residual_block(128, 64)
        self.bottleneck = nn.Linear(64, 16)
        
        # Decoder blocks
        self.dec_proj = nn.Linear(16, 64)
        self.dec_block1 = self._make_residual_block(64, 64)
        self.dec_block2 = self._make_residual_block(64, 128)
        self.output_proj = nn.Linear(128, input_dim)

    def _make_residual_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        # Encoder
        x = F.relu(self.input_proj(x))
        
        # Residual blocks
        residual = x
        x = self.enc_block1(x)
        if x.size(-1) == residual.size(-1):
            x = x + residual
        x = F.relu(x)
        
        x = self.enc_block2(x)
        x = F.relu(x)
        
        # Bottleneck
        x = F.relu(self.bottleneck(x))
        
        # Decoder
        x = F.relu(self.dec_proj(x))
        
        residual = x
        x = self.dec_block1(x)
        x = x + residual
        x = F.relu(x)
        
        x = self.dec_block2(x)
        x = F.relu(x)
        
        x = self.output_proj(x)
        return x

class DenoisingAutoencoder(nn.Module):
    """Denoising Autoencoder"""
    def __init__(self, input_dim, noise_factor=0.1):
        super().__init__()
        self.noise_factor = noise_factor
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim),
        )

    def add_noise(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_factor
            return x + noise
        return x

    def forward(self, x):
        noisy_x = self.add_noise(x)
        encoded = self.encoder(noisy_x)
        decoded = self.decoder(encoded)
        return decoded

########################################################################
# Classical ML Models (Sklearn-based)
########################################################################

class SklearnAnomalyDetector:
    """Wrapper for sklearn-based anomaly detection models"""
    def __init__(self, model_type='isolation_forest', **kwargs):
        self.model_type = model_type
        if model_type == 'isolation_forest':
            self.model = IsolationForest(random_state=42, **kwargs)
        elif model_type == 'one_class_svm':
            self.model = OneClassSVM(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X):
        self.model.fit(X)
    
    def predict_anomaly_scores(self, X):
        if self.model_type == 'isolation_forest':
            # Isolation Forest returns anomaly scores (lower = more anomalous)
            scores = self.model.decision_function(X)
            return -scores  # Invert so higher = more anomalous
        elif self.model_type == 'one_class_svm':
            # One-Class SVM returns distance to separating hyperplane
            scores = self.model.decision_function(X)
            return -scores  # Invert so higher = more anomalous

class SpectralShapeAutoencoder(nn.Module):
    """
    Final Rework: This version processes each spectral slice within the time-frequency
    window using a shared-weight projection. This efficiently captures spatial features
    before the results are concatenated and passed to the core 'normalized basis' 
    bottleneck. This hierarchical, parameter-efficient design preserves context
    while remaining true to the novel paradigm.
    """
    def __init__(self, input_dim, n_mels=64, frames=5, latent_dim=32, slice_latent_dim=64):
        super().__init__()
        self.n_mels = n_mels
        self.frames = frames
        self.latent_dim = latent_dim
        self.slice_latent_dim = slice_latent_dim
        self.input_dim = input_dim
        
        # Combined dimension after processing each slice
        self.combined_dim = frames * slice_latent_dim

        # === ENCODER ===
        # A shared linear layer to process each of the 5 spectral slices
        self.slice_encoder = nn.Sequential(
            nn.LayerNorm(n_mels),
            nn.Linear(n_mels, slice_latent_dim),
            nn.GELU()
        )
        
        # The core learnable basis, operating on the combined representation
        self.global_basis = nn.Parameter(torch.randn(latent_dim, self.combined_dim))
        nn.init.xavier_uniform_(self.global_basis)
        
        # A final non-linear step in the bottleneck for added capacity
        self.bottleneck_mlp = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU()
        )

        # === DECODER ===
        # Reconstructs the combined vector from the latent code
        self.global_decoder = nn.Linear(latent_dim, self.combined_dim, bias=False)

        # A shared linear layer to reconstruct each spectral slice
        self.slice_decoder = nn.Sequential(
            nn.Linear(slice_latent_dim, n_mels)
        )

    def forward(self, x):
        # x shape: (B, n_mels * frames)
        # Reshape to process each slice: (B, frames, n_mels)
        x_slices = x.view(-1, self.frames, self.n_mels)
        
        # 1. Encode each slice using the shared encoder
        encoded_slices = self.slice_encoder(x_slices)
        
        # 2. Concatenate slice representations into a single vector
        combined_vec = encoded_slices.view(-1, self.combined_dim)
        
        # 3. Normalize the global basis for stability
        normalized_basis = F.normalize(self.global_basis, p=2, dim=1)
        
        # 4. Project onto the global basis to get the final latent code
        latent_code = torch.matmul(combined_vec, normalized_basis.T)
        
        # 5. Pass through bottleneck MLP for more capacity
        latent_code = self.bottleneck_mlp(latent_code)

        # 6. Reconstruct the combined vector using the same basis (or a separate linear layer)
        # Using the basis maintains the tied-weight paradigm
        reconstructed_combined = torch.matmul(latent_code, normalized_basis)
        
        # 7. Reshape back into slice-wise representations
        reconstructed_slices_latent = reconstructed_combined.view(-1, self.frames, self.slice_latent_dim)
        
        # 8. Decode each slice using the shared decoder
        reconstructed_slices = self.slice_decoder(reconstructed_slices_latent)
        
        # 9. Flatten to original output shape
        reconstructed_x = reconstructed_slices.view(-1, self.input_dim)
        
        return reconstructed_x


########################################################################
# Data Handling and Feature Extraction
########################################################################

def file_to_vector_array(file_path: Path, **kwargs) -> np.ndarray:
    """Converts a single audio file to a feature vector array."""
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
    except Exception as e:
        logging.error(f"Could not load {file_path}: {e}")
        return np.empty((0, kwargs['n_mels'] * kwargs['frames']), float)

    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, **{k: v for k, v in kwargs.items() if k not in ['frames', 'power']}
    )
    log_mel = 20.0 / kwargs['power'] * np.log10(mel_spectrogram + sys.float_info.epsilon)

    frames = kwargs['frames']
    n_mels = kwargs['n_mels']
    dims = n_mels * frames
    
    if log_mel.shape[1] < frames:
        return np.empty((0, dims), float)

    # Frame the spectrogram
    vector_array = np.lib.stride_tricks.as_strided(
        log_mel,
        shape=(log_mel.shape[1] - frames + 1, n_mels, frames),
        strides=(log_mel.strides[1], log_mel.strides[0], log_mel.strides[1])
    ).reshape(-1, dims)
    
    return vector_array

def list_to_vector_array(file_list: list, msg: str, **kwargs) -> np.ndarray:
    """Converts a list of audio files to a concatenated feature vector array."""
    feature_vectors = [
        file_to_vector_array(Path(f), **kwargs) for f in tqdm(file_list, desc=msg)
    ]
    return np.concatenate([v for v in feature_vectors if v.size > 0], axis=0)

def get_file_lists(target_dir: Path) -> tuple:
    """Generates training and evaluation file lists from the target directory."""
    # This function now just gets files for one directory, to be aggregated later
    normal_files = sorted(target_dir.glob("normal/*.wav"))
    abnormal_files = sorted(target_dir.glob("abnormal/*.wav"))

    if not normal_files or not abnormal_files:
        raise IOError(f"No WAV data found in normal/ and/or abnormal/ subdirs of {target_dir}")
        
    # Use some normal files for evaluation, the rest for training
    # This split is consistent with the original MIMII baseline
    train_files = normal_files[len(abnormal_files):]
    eval_normal_files = normal_files[:len(abnormal_files)]
    
    eval_files = eval_normal_files + abnormal_files
    eval_labels = [0] * len(eval_normal_files) + [1] * len(abnormal_files)

    return train_files, eval_files, eval_labels

########################################################################
# Training and Evaluation Functions
########################################################################

def train_pytorch_model(model, train_loader, val_loader, args, device):
    """Trains PyTorch models"""
    if isinstance(model, VariationalAutoencoder):
        return train_vae(model, train_loader, val_loader, args, device)
    else:
        return train_standard_ae(model, train_loader, val_loader, args, device)

def train_standard_ae(model, train_loader, val_loader, args, device):
    """Train standard autoencoder models"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    history = {"loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for data, in train_loader:
            inputs = data.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, in val_loader:
                inputs = data.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        
        history["loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        if (epoch + 1) % 10 == 0:
            logging.info(
                f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_train_loss:.6f} - Val Loss: {epoch_val_loss:.6f}"
            )
    return history

def train_vae(model, train_loader, val_loader, args, device):
    """Train Variational Autoencoder"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    history = {"loss": [], "val_loss": []}

    def vae_loss(recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for data, in train_loader:
            inputs = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(inputs)
            loss = vae_loss(recon, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, in val_loader:
                inputs = data.to(device)
                recon, mu, logvar = model(inputs)
                loss = vae_loss(recon, inputs, mu, logvar)
                val_loss += loss.item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        
        history["loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)

        if (epoch + 1) % 10 == 0:
            logging.info(
                f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_train_loss:.6f} - Val Loss: {epoch_val_loss:.6f}"
            )
    return history

def evaluate_pytorch_model(model, eval_files, eval_labels, feat_params, device):
    """Evaluates PyTorch models and calculates metrics"""
    model.eval()
    y_true = eval_labels
    y_pred = []

    with torch.no_grad():
        for file_path in tqdm(eval_files, desc="Evaluating"):
            vectors = file_to_vector_array(file_path, **feat_params)
            if vectors.shape[0] == 0:
                y_pred.append(0.0)
                continue

            data = torch.from_numpy(vectors).float().to(device)
            
            if isinstance(model, VariationalAutoencoder):
                reconstruction, _, _ = model(data)
            else:
                reconstruction = model(data)
            
            errors = torch.mean(torch.square(data - reconstruction), axis=1)
            y_pred.append(torch.mean(errors).item())

    # Calculate metrics
    auc_score = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    return {
        'auc': auc_score,
        'pr_auc': pr_auc,
        'y_pred': y_pred
    }

def evaluate_sklearn_model(model, eval_files, eval_labels, feat_params):
    """Evaluates sklearn models"""
    y_true = eval_labels
    y_pred = []

    for file_path in tqdm(eval_files, desc="Evaluating"):
        vectors = file_to_vector_array(file_path, **feat_params)
        if vectors.shape[0] == 0:
            y_pred.append(0.0)
            continue
        
        # Use mean anomaly score across all frames
        scores = model.predict_anomaly_scores(vectors)
        y_pred.append(np.mean(scores))

    # Calculate metrics
    auc_score = roc_auc_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    return {
        'auc': auc_score,
        'pr_auc': pr_auc,
        'y_pred': y_pred
    }

def measure_model_efficiency(model, input_dim, device, model_path=None):
    """Measure model efficiency metrics"""
    if isinstance(model, SklearnAnomalyDetector):
        # For sklearn models, we can't easily measure these metrics this way
        return {
            'model_size_mb': 0.0,
            'num_parameters': 0,
            'inference_time_ms': 0.0,
            'throughput_samples_per_sec': 0.0,
            'memory_usage_mb': 0.0
        }
    
    # Model size
    model_size_mb = 0.0
    if model_path and Path(model_path).exists():
        model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    
    # Parameter count
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Inference time measurement
    model.eval()
    test_data = torch.randn(1000, input_dim).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            if isinstance(model, VariationalAutoencoder):
                _ = model(test_data[:10])
            else:
                _ = model(test_data[:10])
    
    # Measure inference time
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        if isinstance(model, VariationalAutoencoder):
            outputs, _, _ = model(test_data)
        else:
            outputs = model(test_data)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    inference_time_ms = (total_time / 1000) * 1000
    throughput = 1000 / total_time
    
    # Memory usage
    memory_usage_mb = 0.0
    if device.type == 'cuda':
        memory_usage_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return {
        'model_size_mb': model_size_mb,
        'num_parameters': num_parameters,
        'inference_time_ms': inference_time_ms,
        'throughput_samples_per_sec': throughput,
        'memory_usage_mb': memory_usage_mb
    }

########################################################################
# Model Factory
########################################################################

def create_model(model_name: str, input_dim: int, **kwargs):
    """Factory function to create models"""
    models = {
        'simple_ae': SimpleAutoencoder,
        'deep_ae': DeepAutoencoder,
        'vae': VariationalAutoencoder,
        'conv_ae': ConvAutoencoder,
        'lstm_ae': LSTMAutoencoder,
        'attention_ae': AttentionAutoencoder,
        'residual_ae': ResidualAutoencoder,
        'denoising_ae': DenoisingAutoencoder,
        'isolation_forest': lambda input_dim, **kw: SklearnAnomalyDetector('isolation_forest'),
        'one_class_svm': lambda input_dim, **kw: SklearnAnomalyDetector('one_class_svm'),
        'ss_ae': SpectralShapeAutoencoder,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    if model_name in ['conv_ae','ss_ae']:
        n_mels = kwargs.get('n_mels', 64)
        frames = kwargs.get('frames', 5)
        return models[model_name](input_dim, n_mels, frames)
    elif model_name in ['isolation_forest', 'one_class_svm']:
        return models[model_name](input_dim)
    else:
        return models[model_name](input_dim)

########################################################################
# Main Experiment Runner
########################################################################

def run_experiment(args):
    """
    Run a single experiment with specified parameters.
    This version aggregates all data, trains one model, and evaluates once.
    """
    
    # Setup paths and device
    pickle_path = Path(args.pickle_dir)
    model_path = Path(args.model_dir)
    result_path = Path(args.result_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Feature extraction parameters
    feat_params = {
        "n_mels": args.n_mels, "frames": args.frames, "n_fft": args.n_fft,
        "hop_length": args.hop_length, "power": args.power,
    }
    input_dim = args.n_mels * args.frames

    # --- 1. AGGREGATE DATA from all sub-datasets ---
    logging.info("Aggregating data from all sub-datasets...")
    target_dirs = [p for p in Path(args.base_dir).glob("*/*/*")
                   if p.is_dir() and (p / "normal").exists() and (p / "abnormal").exists()]

    if not target_dirs:
        logging.error(f"No valid sub-directories with normal/ and abnormal/ folders found in {args.base_dir}")
        return []

    all_train_files, all_eval_files, all_eval_labels = [], [], []
    for target_dir in tqdm(target_dirs, desc="Scanning directories"):
        try:
            train_files, eval_files, eval_labels = get_file_lists(target_dir)
            all_train_files.extend(train_files)
            all_eval_files.extend(eval_files)
            all_eval_labels.extend(eval_labels)
        except IOError as e:
            logging.warning(f"Skipping directory {target_dir}: {e}")
            continue

    if not all_train_files or not all_eval_files:
        logging.error("No training or evaluation data found across all directories. Exiting.")
        return []

    logging.info(f"Total training files found: {len(all_train_files)}")
    logging.info(f"Total evaluation files found: {len(all_eval_files)}")

    # --- 2. PREPARE UNIFIED TRAINING DATA ---
    pickle_filename = f"train_data_all_{args.n_mels}mels_{args.frames}frames.pkl"
    train_pickle_file = pickle_path / pickle_filename
    
    if train_pickle_file.exists() and not args.retrain:
        logging.info(f"Loading cached training data from {train_pickle_file}")
        with open(train_pickle_file, "rb") as f:
            train_data = pickle.load(f)
    else:
        logging.info("Generating aggregated training data from all audio files...")
        train_data = list_to_vector_array(all_train_files, "Generating combined train vectors", **feat_params)
        with open(train_pickle_file, "wb") as f:
            pickle.dump(train_data, f)

    # --- 3. TRAIN A SINGLE MODEL ---
    logging.info(f"\n{'='*50}")
    logging.info(f"Training single model '{args.model}' on the entire dataset")
    logging.info(f"{'='*50}")

    model = create_model(args.model, input_dim, **feat_params)
    model_file = model_path / f"model_{args.model}_all_machines.pth"
    
    start_time = time.time()
    
    if isinstance(model, SklearnAnomalyDetector):
        logging.info("Training sklearn model...")
        model.fit(train_data)
        training_time = time.time() - start_time
    else:
        model = model.to(device)
        if model_file.exists() and not args.retrain:
            logging.info(f"Loading saved model from {model_file}")
            model.load_state_dict(torch.load(model_file, map_location=device))
            training_time = 0.0
        else:
            logging.info("Training PyTorch model...")
            dataset = TensorDataset(torch.from_numpy(train_data).float())
            val_size = int(len(dataset) * args.val_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

            history = train_pytorch_model(model, train_loader, val_loader, args, device)
            training_time = time.time() - start_time
            torch.save(model.state_dict(), model_file)
    
    # --- 4. EVALUATE THE MODEL ONCE ---
    logging.info("Evaluating model on the entire aggregated dataset...")
    if isinstance(model, SklearnAnomalyDetector):
        eval_results = evaluate_sklearn_model(model, all_eval_files, all_eval_labels, feat_params)
    else:
        eval_results = evaluate_pytorch_model(model, all_eval_files, all_eval_labels, feat_params, device)
    
    efficiency_metrics = measure_model_efficiency(model, input_dim, device, model_file if not isinstance(model, SklearnAnomalyDetector) else None)
    
    # --- 5. COLLECT AND REPORT ONE SET OF RESULTS ---
    result_entry = {
        'model': args.model,
        'dataset_scope': 'all_machines',
        'auc': eval_results['auc'],
        'pr_auc': eval_results['pr_auc'],
        'training_time_sec': training_time,
        **efficiency_metrics,
        'n_mels': args.n_mels, 'frames': args.frames, 'n_fft': args.n_fft, 'hop_length': args.hop_length,
        'power': args.power, 'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
        'val_split': args.val_split,
        'train_files': len(all_train_files), 'eval_files': len(all_eval_files), 'train_samples': len(train_data),
    }
    
    logging.info(f"\n--- Overall Results for {args.model} ---")
    logging.info(f"  AUC: {eval_results['auc']:.4f}")
    logging.info(f"  PR-AUC: {eval_results['pr_auc']:.4f}")
    logging.info(f"  Training time: {training_time:.2f}s")
    logging.info(f"  Model size: {efficiency_metrics['model_size_mb']:.2f}MB")
    logging.info(f"  Parameters: {efficiency_metrics['num_parameters']:,}")
    
    return [result_entry] # Return as a list with one item

def save_results_to_csv(results: List[Dict], output_file: Path):
    """Save experiment results to CSV file, appending if the file exists."""
    if not results:
        logging.warning("No results to save")
        return
    
    fieldnames = list(results[0].keys())
    
    # Check if file exists to determine if we need to write a header
    write_header = not output_file.exists() or output_file.stat().st_size == 0
    
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)
    
    logging.info(f"Results appended to {output_file}")

def run_all_models_experiment(args):
    """Run experiments for all baseline models."""
    all_models = [
        'simple_ae', 'deep_ae', 'vae', 'conv_ae', 'lstm_ae',
        'attention_ae', 'residual_ae', 'denoising_ae', 
        'isolation_forest', 'ss_ae'
    ]
    
    all_experiment_results = []
    
    for model_name in all_models:
        logging.info(f"\n{'='*60}")
        logging.info(f"RUNNING EXPERIMENTS FOR MODEL: {model_name.upper()}")
        logging.info(f"{'='*60}")
        
        args.model = model_name
        
        try:
            model_results = run_experiment(args)
            all_experiment_results.extend(model_results)
        except Exception as e:
            logging.error(f"Error running experiment for {model_name}: {e}", exc_info=True)
            continue
    
    return all_experiment_results

########################################################################
# Main Function and Argument Parser
########################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Research Experiments Runner for MIMII Dataset - 10 Baseline Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and paths
    parser.add_argument("--base_dir", type=str, required=True, 
                        help="Base directory of the MIMII dataset")
    parser.add_argument("--pickle_dir", type=str, default="./cache/research_features", 
                        help="Directory for cached feature data")
    parser.add_argument("--model_dir", type=str, default="./cache/research_models", 
                        help="Directory to save trained models")
    parser.add_argument("--result_dir", type=str, default="./results", 
                        help="Directory to save results")
    parser.add_argument("--output_csv", type=str, default="research_results.csv",
                        help="Output CSV file for results")
    
    # Model selection
    parser.add_argument("--model", type=str, 
                        choices=['simple_ae', 'deep_ae', 'vae', 'conv_ae', 'lstm_ae',
                                 'attention_ae', 'residual_ae', 'denoising_ae', 
                                 'isolation_forest', 'one_class_svm', 'ss_ae','all'],
                        default='all',
                        help="Model to train and evaluate")
    
    # Feature extraction parameters
    parser.add_argument("--n_mels", type=int, default=64, 
                        help="Number of Mel-frequency bands")
    parser.add_argument("--frames", type=int, default=5, 
                        help="Number of frames to concatenate")
    parser.add_argument("--n_fft", type=int, default=1024, 
                        help="FFT size for spectrogram")
    parser.add_argument("--hop_length", type=int, default=512, 
                        help="Hop length for STFT")
    parser.add_argument("--power", type=float, default=2.0, 
                        help="Exponent for magnitude spectrogram")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, 
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Learning rate for optimizer")
    parser.add_argument("--val_split", type=float, default=0.1, 
                        help="Fraction of training data for validation")
    
    # Experiment control
    parser.add_argument("--retrain", action="store_true", 
                        help="Force feature regeneration and model retraining")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directories
    Path(args.pickle_dir).mkdir(exist_ok=True, parents=True)
    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    Path(args.result_dir).mkdir(exist_ok=True, parents=True)
    
    logging.info("="*60)
    logging.info("MIMII DATASET RESEARCH EXPERIMENTS (UNIFIED TRAINING)")
    logging.info("="*60)
    logging.info(f"Base directory: {args.base_dir}")
    logging.info(f"Model(s): {args.model}")
    logging.info(f"Feature params: n_mels={args.n_mels}, frames={args.frames}")
    logging.info(f"Training params: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    logging.info(f"Output CSV: {Path(args.result_dir) / args.output_csv}")
    logging.info("="*60)
    
    # Run experiments
    start_time = time.time()
    if args.model == 'all':
        results = run_all_models_experiment(args)
    else:
        results = run_experiment(args)
    
    # Save results to CSV
    if results:
        output_path = Path(args.result_dir) / args.output_csv
        save_results_to_csv(results, output_path)
    
    total_time = time.time() - start_time
    
    # Print summary
    if results:
        logging.info("\n" + "="*60)
        logging.info("EXPERIMENT SUMMARY")
        logging.info("="*60)
        
        # Simple printout since each model now has one aggregated score
        for result in results:
            model_name = result['model']
            auc_score = result['auc']
            pr_auc_score = result['pr_auc']
            logging.info(f"{model_name:17s}: AUC = {auc_score:.4f} | PR-AUC = {pr_auc_score:.4f}")
        
        logging.info(f"\nTotal experiments completed: {len(results)}")
        logging.info(f"Total run time: {total_time:.2f} seconds")
        logging.info(f"Results saved to: {output_path}")
    
    logging.info("Experiments completed successfully!")

if __name__ == "__main__":
    main()
