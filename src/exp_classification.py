#!/usr/bin/env python
"""
Research Classification Experiments Runner for MIMII Dataset
Supports 10 different baseline models for supervised classification research
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("research_classification.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

########################################################################
# Classification Model Definitions
########################################################################

class SimpleClassifier(nn.Module):
    """Simple 3-layer classifier (baseline)"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)

class DeepClassifier(nn.Module):
    """Deep classifier with multiple hidden layers"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)

class VariationalClassifier(nn.Module):
    """Variational classifier with probabilistic latent space"""
    def __init__(self, input_dim, num_classes, latent_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
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

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.classifier(z)
        return output, mu, logvar

class ConvClassifier(nn.Module):
    """Convolutional classifier for spectral features"""
    def __init__(self, input_dim, num_classes, n_mels=64, frames=5):
        super().__init__()
        self.n_mels = n_mels
        self.frames = frames
        self.input_dim = input_dim
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        # Calculate flattened size
        self.flat_size = 64 * 4 * 4
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 1, self.n_mels, self.frames)
        
        # Convolutional feature extraction
        features = self.conv_layers(x)
        flat = features.view(batch_size, -1)
        
        # Classification
        output = self.classifier(flat)
        return output

class LSTMClassifier(nn.Module):
    """LSTM-based classifier for temporal patterns"""
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(1, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(-1)  # Add feature dimension
        
        # LSTM processing
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use last hidden state for classification
        last_hidden = hidden[-1]  # Take last layer's hidden state
        
        # Classification
        output = self.classifier(last_hidden)
        return output

class AttentionClassifier(nn.Module):
    """Classifier with attention mechanism"""
    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(64, num_heads=8, batch_first=True, dropout=0.1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode features
        encoded = self.encoder(x)
        
        # Reshape for attention (treat features as sequence)
        encoded = encoded.unsqueeze(1)  # Add sequence dimension
        
        # Apply attention
        attended, _ = self.attention(encoded, encoded, encoded)
        attended = attended.squeeze(1)
        
        # Classification
        output = self.classifier(attended)
        return output

class ResidualClassifier(nn.Module):
    """Classifier with residual connections"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, 128)
        
        # Residual blocks
        self.res_block1 = self._make_residual_block(128, 128)
        self.res_block2 = self._make_residual_block(128, 64)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def _make_residual_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        # Initial projection
        x = F.relu(self.input_proj(x))
        
        # Residual blocks
        residual = x
        x = self.res_block1(x)
        if x.size(-1) == residual.size(-1):
            x = x + residual
        x = F.relu(x)
        
        x = self.res_block2(x)
        x = F.relu(x)
        
        # Classification
        output = self.classifier(x)
        return output

class DenoisingClassifier(nn.Module):
    """Denoising classifier with noise injection during training"""
    def __init__(self, input_dim, num_classes, noise_factor=0.1):
        super().__init__()
        self.noise_factor = noise_factor
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes),
        )

    def add_noise(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_factor
            return x + noise
        return x

    def forward(self, x):
        noisy_x = self.add_noise(x)
        features = self.feature_extractor(noisy_x)
        output = self.classifier(features)
        return output

########################################################################
# Classical ML Models (Sklearn-based)
########################################################################

class SklearnClassifier:
    """Wrapper for sklearn-based classification models"""
    def __init__(self, model_type='random_forest', num_classes=4, **kwargs):
        self.model_type = model_type
        self.num_classes = num_classes
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42, n_estimators=100, **kwargs)
        elif model_type == 'svm':
            self.model = SVC(random_state=42, probability=True, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class SpectralShapeClassifier(nn.Module):
    """
    Classification equivalent of SpectralShapeAutoencoder (ss_ae).
    This version processes each spectral slice within the time-frequency
    window using a shared-weight projection, then uses a normalized basis
    for efficient feature extraction before classification.
    """
    def __init__(self, input_dim, num_classes, n_mels=64, frames=5, latent_dim=32, slice_latent_dim=64):
        super().__init__()
        self.n_mels = n_mels
        self.frames = frames
        self.latent_dim = latent_dim
        self.slice_latent_dim = slice_latent_dim
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Combined dimension after processing each slice
        self.combined_dim = frames * slice_latent_dim

        # === FEATURE EXTRACTOR ===
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
            nn.GELU(),
            nn.Dropout(0.2)
        )

        # === CLASSIFIER ===
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, num_classes),
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
        features = self.bottleneck_mlp(latent_code)

        # 6. Classification
        output = self.classifier(features)
        
        return output

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

def get_classification_file_lists(base_dir: Path) -> tuple:
    """
    Generates training and evaluation file lists with labels for normal vs abnormal classification.
    
    Args:
        base_dir: Base directory containing the dataset
    
    Returns:
        train_files, train_labels, eval_files, eval_labels, label_to_name
    """
    # Find all valid directories
    target_dirs = [p for p in base_dir.glob("*/*/*")
                   if p.is_dir() and (p / "normal").exists() and (p / "abnormal").exists()]

    if not target_dirs:
        raise IOError(f"No valid sub-directories with normal/ and abnormal/ folders found in {base_dir}")

    all_files = []
    all_labels = []
    
    # Binary classification: 0=normal, 1=abnormal
    label_to_name = {0: 'normal', 1: 'abnormal'}
    
    for target_dir in target_dirs:
        # Get normal files (label = 0)
        normal_files = sorted(target_dir.glob("normal/*.wav"))
        for file_path in normal_files:
            all_files.append(file_path)
            all_labels.append(0)  # normal = 0
        
        # Get abnormal files (label = 1)
        abnormal_files = sorted(target_dir.glob("abnormal/*.wav"))
        for file_path in abnormal_files:
            all_files.append(file_path)
            all_labels.append(1)  # abnormal = 1

    # Split into train and eval (80/20 split)
    from sklearn.model_selection import train_test_split
    train_files, eval_files, train_labels, eval_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    return train_files, train_labels, eval_files, eval_labels, label_to_name

########################################################################
# Training and Evaluation Functions
########################################################################

def train_pytorch_classifier(model, train_loader, val_loader, args, device, num_classes):
    """Trains PyTorch classification models"""
    if isinstance(model, VariationalClassifier):
        return train_variational_classifier(model, train_loader, val_loader, args, device, num_classes)
    else:
        return train_standard_classifier(model, train_loader, val_loader, args, device, num_classes)

def train_standard_classifier(model, train_loader, val_loader, args, device, num_classes):
    """Train standard classification models"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, targets in train_loader:
            inputs = data.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                inputs = data.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_train_acc = 100 * train_correct / train_total
        epoch_val_acc = 100 * val_correct / val_total
        
        history["loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["accuracy"].append(epoch_train_acc)
        history["val_accuracy"].append(epoch_val_acc)
        
        scheduler.step(epoch_val_loss)

        if (epoch + 1) % 10 == 0:
            logging.info(
                f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_train_loss:.6f} - Acc: {epoch_train_acc:.2f}% - "
                f"Val Loss: {epoch_val_loss:.6f} - Val Acc: {epoch_val_acc:.2f}%"
            )
    
    return history

def train_variational_classifier(model, train_loader, val_loader, args, device, num_classes):
    """Train Variational Classifier"""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    def vae_classification_loss(outputs, targets, mu, logvar, beta=0.1):
        ce_loss = criterion(outputs, targets)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = kld_loss / targets.size(0)  # Normalize by batch size
        return ce_loss + beta * kld_loss

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for data, targets in train_loader:
            inputs = data.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs, mu, logvar = model(inputs)
            loss = vae_classification_loss(outputs, targets, mu, logvar)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                inputs = data.to(device)
                targets = targets.to(device)
                
                outputs, mu, logvar = model(inputs)
                loss = vae_classification_loss(outputs, targets, mu, logvar)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_train_acc = 100 * train_correct / train_total
        epoch_val_acc = 100 * val_correct / val_total
        
        history["loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["accuracy"].append(epoch_train_acc)
        history["val_accuracy"].append(epoch_val_acc)
        
        scheduler.step(epoch_val_loss)

        if (epoch + 1) % 10 == 0:
            logging.info(
                f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_train_loss:.6f} - Acc: {epoch_train_acc:.2f}% - "
                f"Val Loss: {epoch_val_loss:.6f} - Val Acc: {epoch_val_acc:.2f}%"
            )
    
    return history

def evaluate_pytorch_classifier(model, eval_files, eval_labels, feat_params, device, label_to_name):
    """Evaluates PyTorch classification models and calculates metrics"""
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for file_path, true_label in tqdm(zip(eval_files, eval_labels), desc="Evaluating", total=len(eval_files)):
            vectors = file_to_vector_array(file_path, **feat_params)
            if vectors.shape[0] == 0:
                y_pred.append(0)  # Default prediction
                y_true.append(true_label)
                continue

            data = torch.from_numpy(vectors).float().to(device)
            
            if isinstance(model, VariationalClassifier):
                outputs, _, _ = model(data)
            else:
                outputs = model(data)
            
            # Average predictions across all frames
            probs = F.softmax(outputs, dim=1)
            avg_probs = torch.mean(probs, dim=0)
            predicted_class = torch.argmax(avg_probs).item()
            
            y_pred.append(predicted_class)
            y_true.append(true_label)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'y_pred': y_pred,
        'y_true': y_true
    }

def evaluate_sklearn_classifier(model, eval_files, eval_labels, feat_params, label_to_name):
    """Evaluates sklearn classification models"""
    y_true = []
    y_pred = []

    for file_path, true_label in tqdm(zip(eval_files, eval_labels), desc="Evaluating", total=len(eval_files)):
        vectors = file_to_vector_array(file_path, **feat_params)
        if vectors.shape[0] == 0:
            y_pred.append(0)  # Default prediction
            y_true.append(true_label)
            continue
        
        # Use majority vote across all frames
        frame_predictions = model.predict(vectors)
        predicted_class = np.bincount(frame_predictions).argmax()
        
        y_pred.append(predicted_class)
        y_true.append(true_label)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'y_pred': y_pred,
        'y_true': y_true
    }

def measure_model_efficiency(model, input_dim, device, model_path=None):
    """Measure model efficiency metrics"""
    if isinstance(model, SklearnClassifier):
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
            if isinstance(model, VariationalClassifier):
                _ = model(test_data[:10])
            else:
                _ = model(test_data[:10])
    
    # Measure inference time
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    with torch.no_grad():
        if isinstance(model, VariationalClassifier):
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

def create_model(model_name: str, input_dim: int, num_classes: int, **kwargs):
    """Factory function to create classification models"""
    models = {
        'simple_classifier': SimpleClassifier,
        'deep_classifier': DeepClassifier,
        'vae_classifier': VariationalClassifier,
        'conv_classifier': ConvClassifier,
        'lstm_classifier': LSTMClassifier,
        'attention_classifier': AttentionClassifier,
        'residual_classifier': ResidualClassifier,
        'denoising_classifier': DenoisingClassifier,
        'random_forest': lambda input_dim, num_classes, **kw: SklearnClassifier('random_forest', num_classes),
        'ss_classifier': SpectralShapeClassifier,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    if model_name in ['conv_classifier', 'ss_classifier']:
        n_mels = kwargs.get('n_mels', 64)
        frames = kwargs.get('frames', 5)
        return models[model_name](input_dim, num_classes, n_mels, frames)
    elif model_name in ['random_forest', 'svm']:
        return models[model_name](input_dim, num_classes)
    else:
        return models[model_name](input_dim, num_classes)

########################################################################
# Main Experiment Runner
########################################################################

def run_experiment(args):
    """
    Run a single classification experiment with specified parameters.
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

    # --- 1. GET CLASSIFICATION DATA ---
    logging.info("Preparing classification data for normal vs abnormal...")
    train_files, train_labels, eval_files, eval_labels, label_to_name = get_classification_file_lists(
        Path(args.base_dir)
    )
    
    num_classes = len(label_to_name)
    logging.info(f"Number of classes: {num_classes}")
    logging.info(f"Classes: {label_to_name}")
    logging.info(f"Training files: {len(train_files)}")
    logging.info(f"Evaluation files: {len(eval_files)}")

    # --- 2. PREPARE TRAINING DATA ---
    pickle_filename = f"train_data_normal_vs_abnormal_{args.model}_{args.n_mels}mels_{args.frames}frames.pkl"
    train_pickle_file = pickle_path / pickle_filename
    
    if train_pickle_file.exists() and not args.retrain:
        logging.info(f"Loading cached training data from {train_pickle_file}")
        with open(train_pickle_file, "rb") as f:
            train_data, train_labels_array = pickle.load(f)
    else:
        logging.info("Generating training data from audio files...")
        train_data = list_to_vector_array(train_files, "Generating train vectors", **feat_params)
        
        # Create labels array matching the training data
        train_labels_array = []
        for file_path, label in zip(train_files, train_labels):
            vectors = file_to_vector_array(file_path, **feat_params)
            train_labels_array.extend([label] * len(vectors))
        
        train_labels_array = np.array(train_labels_array)
        
        with open(train_pickle_file, "wb") as f:
            pickle.dump((train_data, train_labels_array), f)

    # --- 3. TRAIN THE MODEL ---
    logging.info(f"\n{'='*50}")
    logging.info(f"Training classification model '{args.model}' for normal vs abnormal")
    logging.info(f"{'='*50}")

    model = create_model(args.model, input_dim, num_classes, **feat_params)
    model_file = model_path / f"model_{args.model}_normal_vs_abnormal.pth"
    
    start_time = time.time()
    
    if isinstance(model, SklearnClassifier):
        logging.info("Training sklearn model...")
        model.fit(train_data, train_labels_array)
        training_time = time.time() - start_time
    else:
        model = model.to(device)
        if model_file.exists() and not args.retrain:
            logging.info(f"Loading saved model from {model_file}")
            model.load_state_dict(torch.load(model_file, map_location=device))
            training_time = 0.0
        else:
            logging.info("Training PyTorch model...")
            dataset = TensorDataset(
                torch.from_numpy(train_data).float(),
                torch.from_numpy(train_labels_array).long()
            )
            val_size = int(len(dataset) * args.val_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

            history = train_pytorch_classifier(model, train_loader, val_loader, args, device, num_classes)
            training_time = time.time() - start_time
            torch.save(model.state_dict(), model_file)
    
    # --- 4. EVALUATE THE MODEL ---
    logging.info("Evaluating model...")
    if isinstance(model, SklearnClassifier):
        eval_results = evaluate_sklearn_classifier(model, eval_files, eval_labels, feat_params, label_to_name)
    else:
        eval_results = evaluate_pytorch_classifier(model, eval_files, eval_labels, feat_params, device, label_to_name)
    
    efficiency_metrics = measure_model_efficiency(model, input_dim, device, model_file if not isinstance(model, SklearnClassifier) else None)
    
    # --- 5. COLLECT RESULTS ---
    result_entry = {
        'model': args.model,
        'classification_type': args.classification_type,
        'num_classes': num_classes,
        'accuracy': eval_results['accuracy'],
        'precision': eval_results['precision'],
        'recall': eval_results['recall'],
        'f1': eval_results['f1'],
        'training_time_sec': training_time,
        **efficiency_metrics,
        'n_mels': args.n_mels, 'frames': args.frames, 'n_fft': args.n_fft, 'hop_length': args.hop_length,
        'power': args.power, 'epochs': args.epochs, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
        'val_split': args.val_split,
        'train_files': len(train_files), 'eval_files': len(eval_files), 'train_samples': len(train_data),
    }
    
    logging.info(f"\n--- Results for {args.model} on {args.classification_type} ---")
    logging.info(f"  Accuracy: {eval_results['accuracy']:.4f}")
    logging.info(f"  Precision: {eval_results['precision']:.4f}")
    logging.info(f"  Recall: {eval_results['recall']:.4f}")
    logging.info(f"  F1-Score: {eval_results['f1']:.4f}")
    logging.info(f"  Training time: {training_time:.2f}s")
    logging.info(f"  Model size: {efficiency_metrics['model_size_mb']:.2f}MB")
    logging.info(f"  Parameters: {efficiency_metrics['num_parameters']:,}")
    
    return [result_entry]

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
    """Run experiments for all baseline classification models."""
    all_models = [
        'simple_classifier', 'deep_classifier', 'vae_classifier', 'conv_classifier', 'lstm_classifier',
        'attention_classifier', 'residual_classifier', 'denoising_classifier', 
         'ss_classifier'
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
        description="Research Classification Experiments Runner for MIMII Dataset - 11 Baseline Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and paths
    parser.add_argument("--base_dir", type=str, required=True, 
                        help="Base directory of the MIMII dataset")
    parser.add_argument("--pickle_dir", type=str, default="./cache/classification_features", 
                        help="Directory for cached feature data")
    parser.add_argument("--model_dir", type=str, default="./cache/classification_models", 
                        help="Directory to save trained models")
    parser.add_argument("--result_dir", type=str, default="./results", 
                        help="Directory to save results")
    parser.add_argument("--output_csv", type=str, default="research_classification_results.csv",
                        help="Output CSV file for results")
    
    # Classification type
    parser.add_argument("--classification_type", type=str, 
                        choices=['machine_type', 'machine_id'],
                        default='machine_type',
                        help="Type of classification: machine_type (fan/pump/valve/slider) or machine_id (individual machines)")
    
    # Model selection
    parser.add_argument("--model", type=str, 
                        choices=['simple_classifier', 'deep_classifier', 'vae_classifier', 'conv_classifier', 'lstm_classifier',
                                 'attention_classifier', 'residual_classifier', 'denoising_classifier', 
                                 'random_forest', 'svm', 'ss_classifier', 'all'],
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
    logging.info("MIMII DATASET CLASSIFICATION EXPERIMENTS")
    logging.info("="*60)
    logging.info(f"Base directory: {args.base_dir}")
    logging.info(f"Classification type: {args.classification_type}")
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
        
        for result in results:
            model_name = result['model']
            accuracy = result['accuracy']
            f1_score = result['f1']
            logging.info(f"{model_name:20s}: Accuracy = {accuracy:.4f} | F1 = {f1_score:.4f}")
        
        logging.info(f"\nTotal experiments completed: {len(results)}")
        logging.info(f"Total run time: {total_time:.2f} seconds")
        logging.info(f"Results saved to: {output_path}")
    
    logging.info("Classification experiments completed successfully!")

if __name__ == "__main__":
    main()
