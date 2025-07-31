#!/usr/bin/env python
"""
 @file   train_pytorch.py
 @brief  PyTorch version of the baseline AE-based anomaly detection for the MIMII Dataset.
 @author Gemini
"""
import argparse
import logging
import pickle
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("baseline_pytorch.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

########################################################################
# Model Definition
########################################################################
class Autoencoder(nn.Module):
    """A simple autoencoder model for anomaly detection."""
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
    logging.info(f"Target directory: {target_dir}")
    
    normal_files = sorted(target_dir.glob("normal/*.wav"))
    abnormal_files = sorted(target_dir.glob("abnormal/*.wav"))

    if not normal_files or not abnormal_files:
        raise IOError(f"No WAV data found in {target_dir}")
        
    # Use some normal files for evaluation, the rest for training
    train_files = normal_files[len(abnormal_files):]
    eval_normal_files = normal_files[:len(abnormal_files)]
    
    eval_files = eval_normal_files + abnormal_files
    eval_labels = [0] * len(eval_normal_files) + [1] * len(abnormal_files)

    logging.info(f"Training files: {len(train_files)}")
    logging.info(f"Evaluation files: {len(eval_files)}")
    return train_files, eval_files, eval_labels

########################################################################
# Training and Evaluation
########################################################################
def train_model(model, train_loader, val_loader, args, device):
    """Trains the autoencoder model."""
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

        logging.info(
            f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_train_loss:.6f} - Val Loss: {epoch_val_loss:.6f}"
        )
    return history

def evaluate_model(model, eval_files, eval_labels, feat_params, device):
    """Evaluates the model and calculates the AUC score."""
    model.eval()
    y_true = eval_labels
    y_pred = [0.0] * len(eval_files)

    with torch.no_grad():
        for i, file_path in enumerate(tqdm(eval_files, desc="Evaluating")):
            vectors = file_to_vector_array(file_path, **feat_params)
            if vectors.shape[0] == 0:
                logging.warning(f"Could not process {file_path}, skipping.")
                # Decide how to handle this case; here we predict a low anomaly score
                y_pred[i] = 0.0 
                continue

            data = torch.from_numpy(vectors).float().to(device)
            reconstruction = model(data)
            errors = torch.mean(torch.square(data - reconstruction), axis=1)
            y_pred[i] = torch.mean(errors).item()

    auc = roc_auc_score(y_true, y_pred)
    logging.info(f"AUC Score: {auc:.4f}")
    return auc

########################################################################
# Main
########################################################################
def main():
    parser = argparse.ArgumentParser(description="PyTorch Autoencoder for MIMII Dataset")
    # Paths
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory of the dataset")
    parser.add_argument("--pickle_dir", type=str, default="./pickle_data", help="Directory for cached data")
    parser.add_argument("--model_dir", type=str, default="./models", help="Directory to save models")
    parser.add_argument("--result_dir", type=str, default="./results", help="Directory to save results")
    # Feature params
    parser.add_argument("--n_mels", type=int, default=64, help="Number of Mel bands")
    parser.add_argument("--frames", type=int, default=5, help="Number of frames to concatenate")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT size")
    parser.add_argument("--hop_length", type=int, default=512, help="Hop length for STFT")
    parser.add_argument("--power", type=float, default=2.0, help="Exponent for the magnitude spectrogram")
    # Training params
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for Adam optimizer")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of training data for validation")
    
    args = parser.parse_args()

    # Create directories
    pickle_path = Path(args.pickle_dir)
    model_path = Path(args.model_dir)
    result_path = Path(args.result_dir)
    pickle_path.mkdir(exist_ok=True, parents=True)
    model_path.mkdir(exist_ok=True, parents=True)
    result_path.mkdir(exist_ok=True, parents=True)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    feat_params = {
        "n_mels": args.n_mels,
        "frames": args.frames,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "power": args.power,
    }
    input_dim = args.n_mels * args.frames

    all_results = {}
    # Find all machine type/ID directories
    target_dirs = [p for p in Path(args.base_dir).glob("*/*") if p.is_dir() and (p / "normal").exists() and (p / "abnormal").exists()]

    for target_dir in target_dirs:
        db = target_dir.parents[1].name
        machine_type = target_dir.parents[0].name
        machine_id = target_dir.name
        logging.info(f"\n{'='*25}\nProcessing: {machine_type} - {machine_id} - {db}\n{'='*25}")
        
        result_key = f"{machine_type}_{machine_id}_{db}"
        train_pickle_file = pickle_path / f"train_{result_key}.pkl"
        
        # === Dataset Generation ===
        train_files, eval_files, eval_labels = get_file_lists(target_dir)

        if train_pickle_file.exists():
            logging.info(f"Loading cached training data from {train_pickle_file}")
            with open(train_pickle_file, "rb") as f:
                train_data = pickle.load(f)
        else:
            logging.info("Generating training data from audio files...")
            train_data = list_to_vector_array(train_files, "Generating train vectors", **feat_params)
            with open(train_pickle_file, "wb") as f:
                pickle.dump(train_data, f)
        
        # === Model Training ===
        model = Autoencoder(input_dim).to(device)
        model_file = model_path / f"model_{result_key}.pth"
        
        if model_file.exists():
            logging.info(f"Loading saved model from {model_file}")
            model.load_state_dict(torch.load(model_file, map_location=device))
        else:
            logging.info("Training new model...")
            # Create dataset and dataloaders
            dataset = TensorDataset(torch.from_numpy(train_data).float())
            val_size = int(len(dataset) * args.val_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

            history = train_model(model, train_loader, val_loader, args, device)
            
            # Save model and plot history
            torch.save(model.state_dict(), model_file)
            plt.figure(figsize=(10, 5))
            plt.plot(history["loss"], label="Train Loss")
            plt.plot(history["val_loss"], label="Validation Loss")
            plt.title(f"Model Loss for {result_key}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig(model_path / f"history_{result_key}.png")
            plt.close()

        # === Evaluation ===
        logging.info("Starting evaluation...")
        auc = evaluate_model(model, eval_files, eval_labels, feat_params, device)
        all_results[result_key] = {"AUC": float(auc)}

    # === Save Final Results ===
    result_file = result_path / "summary_results.yaml"
    logging.info(f"\nSaving final results to {result_file}")
    with open(result_file, "w") as f:
        yaml.dump(all_results, f, default_flow_style=False)
    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
