# MIMII Dataset Research Experiments Framework

This framework provides camera-ready baseline implementations for 10 different anomaly detection models on the MIMII dataset, designed for research paper comparisons.

## Overview

The research framework includes:
- **10 baseline models** covering different approaches to anomaly detection
- **Comprehensive evaluation metrics** (AUC, PR-AUC, efficiency metrics)
- **Single command execution** with argparse for easy experimentation
- **CSV output** with all metrics mapped to input parameters
- **Reproducible experiments** with seed control

## Supported Models

### Deep Learning Models (PyTorch)
1. **Simple Autoencoder** (`simple_ae`) - Basic 3-layer autoencoder
2. **Deep Autoencoder** (`deep_ae`) - Multi-layer deep autoencoder
3. **Variational Autoencoder** (`vae`) - VAE with KL divergence loss
4. **Convolutional Autoencoder** (`conv_ae`) - CNN-based for spectral features
5. **LSTM Autoencoder** (`lstm_ae`) - RNN-based for temporal patterns
6. **Attention Autoencoder** (`attention_ae`) - Transformer-style attention
7. **Residual Autoencoder** (`residual_ae`) - ResNet-style skip connections
8. **Denoising Autoencoder** (`denoising_ae`) - Noise-robust training

### Classical ML Models (Scikit-learn)
9. **Isolation Forest** (`isolation_forest`) - Tree-based anomaly detection
10. **One-Class SVM** (`one_class_svm`) - Support vector-based approach

## Quick Start

### Run All Models (Recommended for Research)
```bash
python research_experiments.py --base_dir ./dataset --model all
```

### Run Single Model
```bash
python research_experiments.py --base_dir ./dataset --model simple_ae
```

### Custom Hyperparameters
```bash
python research_experiments.py \
    --base_dir ./dataset \
    --model deep_ae \
    --epochs 100 \
    --batch_size 512 \
    --learning_rate 0.0001 \
    --n_mels 128 \
    --frames 10
```

## Command Line Arguments

### Required Arguments
- `--base_dir`: Path to MIMII dataset directory

### Model Selection
- `--model`: Choose from available models or 'all' (default: 'all')

### Feature Extraction Parameters
- `--n_mels`: Number of Mel-frequency bands (default: 64)
- `--frames`: Number of frames to concatenate (default: 5)
- `--n_fft`: FFT size for spectrogram (default: 1024)
- `--hop_length`: Hop length for STFT (default: 512)
- `--power`: Exponent for magnitude spectrogram (default: 2.0)

### Training Parameters
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 256)
- `--learning_rate`: Learning rate for optimizer (default: 0.001)
- `--val_split`: Fraction of training data for validation (default: 0.1)

### Output Control
- `--output_csv`: Output CSV filename (default: 'research_results.csv')
- `--pickle_dir`: Directory for cached features (default: './research_pickle_data')
- `--model_dir`: Directory for saved models (default: './research_models')
- `--result_dir`: Directory for results (default: './research_results')

### Experiment Control
- `--retrain`: Force retraining even if model exists
- `--seed`: Random seed for reproducibility (default: 42)

## Output Metrics

The framework generates comprehensive metrics saved to CSV:

### Task Performance Metrics
- **AUC**: Area Under ROC Curve
- **PR-AUC**: Area Under Precision-Recall Curve

### Model Efficiency Metrics
- **Training Time**: Time to train model (seconds)
- **Model Size**: Saved model file size (MB)
- **Parameters**: Number of trainable parameters
- **Inference Time**: Average inference time per sample (ms)
- **Throughput**: Samples processed per second
- **Memory Usage**: Peak GPU memory usage (MB)

### Experiment Parameters
All input parameters are logged for reproducibility:
- Model architecture and hyperparameters
- Feature extraction settings
- Training configuration
- Dataset statistics

## Example Usage Scenarios

### 1. Full Research Comparison
```bash
# Run all 10 models with default settings
python research_experiments.py --base_dir ./dataset --model all

# Results saved to: research_results/research_results.csv
```

### 2. Hyperparameter Study
```bash
# Test different learning rates for VAE
python research_experiments.py --base_dir ./dataset --model vae --learning_rate 0.01 --output_csv vae_lr001.csv
python research_experiments.py --base_dir ./dataset --model vae --learning_rate 0.001 --output_csv vae_lr0001.csv
python research_experiments.py --base_dir ./dataset --model vae --learning_rate 0.0001 --output_csv vae_lr00001.csv
```

### 3. Feature Engineering Study
```bash
# Test different mel-band configurations
python research_experiments.py --base_dir ./dataset --model conv_ae --n_mels 32 --output_csv conv_ae_32mels.csv
python research_experiments.py --base_dir ./dataset --model conv_ae --n_mels 64 --output_csv conv_ae_64mels.csv
python research_experiments.py --base_dir ./dataset --model conv_ae --n_mels 128 --output_csv conv_ae_128mels.csv
```

### 4. Quick Prototyping
```bash
# Fast experiments with reduced epochs
python research_experiments.py --base_dir ./dataset --model simple_ae --epochs 10 --retrain
```

## Dataset Structure

The framework expects the MIMII dataset in this structure:
```
dataset/
├── 6dB/
│   └── fan/
│       ├── id_00/
│       │   ├── normal/
│       │   │   ├── 00000000.wav
│       │   │   └── ...
│       │   └── abnormal/
│       │       ├── 00000000.wav
│       │       └── ...
│       ├── id_02/
│       └── ...
├── 0dB/
└── min6dB/
```

## Output Files

### CSV Results File
Contains all experimental results with columns:
- Experiment parameters (model, hyperparameters, etc.)
- Performance metrics (AUC, PR-AUC)
- Efficiency metrics (time, memory, model size)
- Dataset statistics

### Model Files
Trained PyTorch models saved as `.pth` files in `research_models/`

### Cached Features
Preprocessed audio features cached as `.pkl` files in `research_pickle_data/`

### Logs
Detailed execution logs saved to `research_experiments.log`

## Research Paper Usage

This framework is designed to provide camera-ready baselines for research papers:

1. **Reproducible Results**: Fixed random seeds and comprehensive parameter logging
2. **Standard Metrics**: AUC and PR-AUC commonly used in anomaly detection literature
3. **Efficiency Analysis**: Model size, inference time, and memory usage for practical considerations
4. **Multiple Approaches**: Covers major categories of anomaly detection methods
5. **Easy Comparison**: Single CSV output for direct comparison tables

### Example Research Workflow
```bash
# 1. Run full baseline comparison
python research_experiments.py --base_dir ./dataset --model all --epochs 50

# 2. Analyze results
python -c "
import pandas as pd
df = pd.read_csv('research_results/research_results.csv')
summary = df.groupby('model')['auc'].agg(['mean', 'std', 'count'])
print(summary.round(4))
"

# 3. Generate paper table
python -c "
import pandas as pd
df = pd.read_csv('research_results/research_results.csv')
table = df.groupby('model').agg({
    'auc': ['mean', 'std'],
    'training_time_sec': 'mean',
    'model_size_mb': 'mean',
    'num_parameters': 'mean'
}).round(4)
print(table.to_latex())
"
```

## Performance Expectations

Based on the existing baseline results, expected performance ranges:

- **Simple/Deep AE**: AUC 0.55-0.65
- **VAE**: AUC 0.60-0.70
- **Conv AE**: AUC 0.65-0.75
- **LSTM AE**: AUC 0.60-0.70
- **Attention AE**: AUC 0.65-0.75
- **Residual AE**: AUC 0.60-0.70
- **Denoising AE**: AUC 0.65-0.75
- **Isolation Forest**: AUC 0.55-0.65
- **One-Class SVM**: AUC 0.50-0.60

## Requirements

Install dependencies:
```bash
pip install torch scikit-learn librosa numpy matplotlib tqdm PyYAML
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `--batch_size`
2. **Slow Training**: Increase `--batch_size` or reduce `--epochs`
3. **Missing Dataset**: Ensure `--base_dir` points to correct MIMII dataset location
4. **Permission Errors**: Check write permissions for output directories

### Debug Mode
Add verbose logging:
```bash
python research_experiments.py --base_dir ./dataset --model simple_ae 2>&1 | tee debug.log
```

## Citation

If you use this framework in your research, please cite the original MIMII dataset papers and acknowledge this baseline implementation.

## Contributing

To add new models:
1. Implement model class following existing patterns
2. Add to `create_model()` factory function
3. Update model choices in argument parser
4. Test with single model run before full experiments
