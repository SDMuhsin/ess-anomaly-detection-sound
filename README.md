# MIMII Dataset Anomaly Detection & Classification Framework

A comprehensive research framework for anomaly detection and classification on the MIMII dataset, featuring 10+ baseline models with both unsupervised (anomaly detection) and supervised (classification) approaches.

## Overview

This framework provides camera-ready baseline implementations for research on industrial machine sound analysis using the MIMII dataset. It includes:

- **10 anomaly detection models** (unsupervised learning)
- **11 classification models** (supervised learning)
- **Comprehensive evaluation metrics** (AUC, PR-AUC, Accuracy, F1-Score, efficiency metrics)
- **Single command execution** with argparse for easy experimentation
- **CSV output** with all metrics for research paper tables
- **Reproducible experiments** with seed control
- **Consolidated directory structure** for clean organization

## Dataset Information

The MIMII Dataset is a sound dataset for malfunctioning industrial machine investigation and inspection. It contains sounds from four types of industrial machines: valves, pumps, fans, and slide rails. Each machine type includes multiple individual models with normal and anomalous sounds.

**Download**: https://zenodo.org/record/3384388

**Citation**: If you use the MIMII Dataset, please cite:
> Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," arXiv preprint arXiv:1909.09347, 2019.

## Quick Start

### 1. Setup Dataset
```bash
cd dataset/
sh 7z.sh  # Extract downloaded MIMII dataset files
cd ..
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Experiments

**Anomaly Detection (Unsupervised):**
```bash
python src/exp_autoencoding.py --base_dir ./dataset --model all
```

**Classification (Supervised):**
```bash
python src/exp_classification.py --base_dir ./dataset --model all --classification_type machine_type
```

## Supported Models

### Anomaly Detection Models (Unsupervised)

#### Deep Learning Models (PyTorch)
1. **Simple Autoencoder** (`simple_ae`) - Basic 3-layer autoencoder baseline
2. **Deep Autoencoder** (`deep_ae`) - Multi-layer deep autoencoder
3. **Variational Autoencoder** (`vae`) - VAE with KL divergence loss
4. **Convolutional Autoencoder** (`conv_ae`) - CNN-based for spectral features
5. **LSTM Autoencoder** (`lstm_ae`) - RNN-based for temporal patterns
6. **Attention Autoencoder** (`attention_ae`) - Transformer-style attention mechanism
7. **Residual Autoencoder** (`residual_ae`) - ResNet-style skip connections
8. **Denoising Autoencoder** (`denoising_ae`) - Noise-robust training
9. **Spectral Shape Autoencoder** (`ss_ae`) - **Novel method** - parameter-efficient spectral processing

#### Classical ML Models (Scikit-learn)
10. **Isolation Forest** (`isolation_forest`) - Tree-based anomaly detection
11. **One-Class SVM** (`one_class_svm`) - Support vector-based approach

### Classification Models (Supervised)

#### Deep Learning Models (PyTorch)
1. **Simple Classifier** (`simple_classifier`) - Basic 3-layer classifier baseline
2. **Deep Classifier** (`deep_classifier`) - Multi-layer deep classifier
3. **Variational Classifier** (`vae_classifier`) - VAE-style classifier with KL regularization
4. **Convolutional Classifier** (`conv_classifier`) - CNN-based for spectral features
5. **LSTM Classifier** (`lstm_classifier`) - RNN-based for temporal patterns
6. **Attention Classifier** (`attention_classifier`) - Transformer-style attention mechanism
7. **Residual Classifier** (`residual_classifier`) - ResNet-style skip connections
8. **Denoising Classifier** (`denoising_classifier`) - Noise-robust training with dropout
9. **Spectral Shape Classifier** (`ss_classifier`) - **Novel method** - classification equivalent of ss_ae

#### Classical ML Models (Scikit-learn)
10. **Random Forest** (`random_forest`) - Tree-based ensemble classifier
11. **SVM** (`svm`) - Support vector machine classifier

## Usage Examples

### Anomaly Detection Experiments

**Run all models:**
```bash
python src/exp_autoencoding.py --base_dir ./dataset --model all
```

**Run single model:**
```bash
python src/exp_autoencoding.py --base_dir ./dataset --model simple_ae
```

**Custom hyperparameters:**
```bash
python src/exp_autoencoding.py \
    --base_dir ./dataset \
    --model deep_ae \
    --epochs 100 \
    --batch_size 512 \
    --learning_rate 0.0001 \
    --n_mels 128 \
    --frames 10
```

### Classification Experiments

**Machine type classification (normal vs abnormal):**
```bash
python src/exp_classification.py --base_dir ./dataset --model all --classification_type machine_type
```

**Run single classifier:**
```bash
python src/exp_classification.py --base_dir ./dataset --model ss_classifier --classification_type machine_type
```

**Custom hyperparameters:**
```bash
python src/exp_classification.py \
    --base_dir ./dataset \
    --model ss_classifier \
    --classification_type machine_type \
    --epochs 100 \
    --batch_size 512 \
    --learning_rate 0.0001
```

## Command Line Arguments

### Common Arguments (Both Scripts)

**Required:**
- `--base_dir`: Path to MIMII dataset directory

**Model Selection:**
- `--model`: Choose from available models or 'all' (default: 'all')

**Feature Extraction:**
- `--n_mels`: Number of Mel-frequency bands (default: 64)
- `--frames`: Number of frames to concatenate (default: 5)
- `--n_fft`: FFT size for spectrogram (default: 1024)
- `--hop_length`: Hop length for STFT (default: 512)
- `--power`: Exponent for magnitude spectrogram (default: 2.0)

**Training:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 256)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--val_split`: Fraction of training data for validation (default: 0.1)

**Control:**
- `--retrain`: Force retraining even if model exists
- `--seed`: Random seed for reproducibility (default: 42)

### Classification-Specific Arguments
- `--classification_type`: Type of classification task
  - `machine_type`: Normal vs abnormal classification
  - `machine_id`: Individual machine identification

## Directory Structure

The framework uses a consolidated directory structure:

```
mimii_baseline/
├── cache/                              # All models and intermediate data
│   ├── research_models/                # Anomaly detection models
│   ├── research_features/              # Anomaly detection cached features
│   ├── classification_models/          # Classification models
│   ├── classification_features/        # Classification cached features
│   └── legacy_*/                       # Legacy data (if migrated)
├── results/                            # All experiment results
│   ├── research_results.csv           # Anomaly detection results
│   ├── research_classification_results.csv # Classification results
│   └── ... (other result files)
├── src/                               # Source code
│   ├── exp_autoencoding.py            # Anomaly detection experiments
│   ├── exp_classification.py          # Classification experiments
│   ├── baseline.py                    # Original baseline implementation
│   └── ... (other source files)
├── dataset/                           # MIMII dataset (download separately)
└── ... (other project files)
```

## Output Metrics

### Anomaly Detection Metrics
- **AUC**: Area Under ROC Curve
- **PR-AUC**: Area Under Precision-Recall Curve

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1-Score**: Weighted average F1-score

### Efficiency Metrics (Both)
- **Training Time**: Time to train model (seconds)
- **Model Size**: Saved model file size (MB)
- **Parameters**: Number of trainable parameters
- **Inference Time**: Average inference time per sample (ms)
- **Throughput**: Samples processed per second
- **Memory Usage**: Peak GPU memory usage (MB)

## Research Paper Usage

This framework is designed for research paper baselines:

### Generate Results Table
```python
import pandas as pd

# Anomaly detection results
df_ad = pd.read_csv('results/research_results.csv')
summary_ad = df_ad.groupby('model').agg({
    'auc': ['mean', 'std'],
    'pr_auc': ['mean', 'std'],
    'training_time_sec': 'mean',
    'num_parameters': 'mean'
}).round(4)

# Classification results
df_cls = pd.read_csv('results/research_classification_results.csv')
summary_cls = df_cls.groupby('model').agg({
    'accuracy': ['mean', 'std'],
    'f1': ['mean', 'std'],
    'training_time_sec': 'mean',
    'num_parameters': 'mean'
}).round(4)

print("Anomaly Detection Results:")
print(summary_ad.to_latex())
print("\nClassification Results:")
print(summary_cls.to_latex())
```

## Expected Performance

### Anomaly Detection (AUC ranges)
- **Simple/Deep AE**: 0.55-0.65
- **VAE**: 0.60-0.70
- **Conv AE**: 0.65-0.75 
- **LSTM AE**: 0.60-0.70
- **Attention AE**: 0.65-0.75
- **Residual AE**: 0.60-0.70
- **Denoising AE**: 0.65-0.75
- **SS AE**: 0.65-0.75 (parameter-efficient)
- **Isolation Forest**: 0.55-0.65
- **One-Class SVM**: 0.50-0.60

### Classification (Accuracy)
- Most models achieve high accuracy on normal vs abnormal classification
- **SS Classifier**: Maintains high performance with fewer parameters

## Novel Contributions

### Spectral Shape Models (ss_ae & ss_classifier)
Our proposed methods feature:
- **Parameter Efficiency**: Significantly fewer parameters than traditional approaches
- **Spectral Slice Processing**: Processes time-frequency slices with shared weights
- **Normalized Global Basis**: Learnable normalized basis for efficient feature extraction
- **Hierarchical Design**: Slice-level processing followed by global integration

**Key Advantages:**
- Fewer parameters (e.g., 16,193 vs 22,657 for simple baseline)
- Smaller model size (0.07MB vs 0.09MB)
- Competitive or superior performance
- Suitable for resource-constrained environments

## Requirements

### System Requirements
- Ubuntu 16.04+ / CentOS 7+ / Windows 10+
- Python 3.6+
- CUDA (optional, for GPU acceleration)

### Python Dependencies
```bash
pip install torch torchvision torchaudio
pip install scikit-learn librosa numpy matplotlib tqdm PyYAML
pip install pandas  # for results analysis
```

### Additional Software
- p7zip-full (for dataset extraction)
- FFmpeg (for audio processing)

## Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```bash
python src/exp_autoencoding.py --base_dir ./dataset --model simple_ae --batch_size 128
```

**Slow Training:**
```bash
python src/exp_autoencoding.py --base_dir ./dataset --model simple_ae --epochs 20
```

**Missing Dataset:**
- Ensure `--base_dir` points to extracted MIMII dataset
- Check dataset structure matches expected format

**Permission Errors:**
- Ensure write permissions for `cache/` and `results/` directories

### Debug Mode
```bash
python src/exp_autoencoding.py --base_dir ./dataset --model simple_ae 2>&1 | tee debug.log
```

## Legacy Baseline

The original baseline implementation is still available:
```bash
python src/baseline.py
```

This runs the original DAE-based anomaly detection from the MIMII dataset paper.

## Contributing

To add new models:
1. Implement model class following existing patterns
2. Add to `create_model()` factory function
3. Update model choices in argument parser
4. Test with single model run before full experiments

## File Structure Reference

### Key Files
- `src/exp_autoencoding.py`: Main anomaly detection experiments
- `src/exp_classification.py`: Main classification experiments
- `src/baseline.py`: Original MIMII baseline
- `requirements.txt`: Python dependencies
- `baseline.yaml`: Configuration for original baseline

### Generated Files
- `cache/`: All models and cached features
- `results/`: All experimental results (CSV files)
- `*.log`: Execution logs

## License

This project follows the same license terms as the original MIMII dataset baseline code.

## Acknowledgments

- Original MIMII dataset authors
- PyTorch and scikit-learn communities
- Contributors to the baseline implementations

---

For detailed information about specific components, see the individual documentation files in the repository.
