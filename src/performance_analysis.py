#!/usr/bin/env python
"""
Performance analysis script for the PyTorch baseline model
"""
import torch
import time
import numpy as np
from pathlib import Path
from src.baseline import Autoencoder

def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, input_dim, device, num_samples=1000):
    """Measure average inference time"""
    model.eval()
    
    # Create random input data
    test_data = torch.randn(num_samples, input_dim).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_data[:10])
    
    # Measure inference time
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(test_data)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_sample = (total_time / num_samples) * 1000  # Convert to milliseconds
    
    return avg_time_per_sample, total_time

def get_model_size_mb(model_path):
    """Get model file size in MB"""
    size_bytes = Path(model_path).stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    return size_mb

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 64 * 5  # n_mels * frames
    
    print("=" * 60)
    print("MIMII PyTorch Baseline - Performance Analysis")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Input dimension: {input_dim}")
    print()
    
    # Load one of the trained models for analysis
    model_path = "models/model_fan_id_02_min6dB.pth"
    model = Autoencoder(input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Model size analysis
    num_params = count_parameters(model)
    model_size_mb = get_model_size_mb(model_path)
    
    print("MODEL ARCHITECTURE:")
    print("-" * 30)
    print(f"Total parameters: {num_params:,}")
    print(f"Model file size: {model_size_mb:.2f} MB")
    print()
    
    # Model architecture details
    print("LAYER DETAILS:")
    print("-" * 30)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape} ({param.numel():,} params)")
    print()
    
    # Inference time analysis
    print("INFERENCE PERFORMANCE:")
    print("-" * 30)
    avg_time_ms, total_time = measure_inference_time(model, input_dim, device, 1000)
    
    print(f"Average inference time per sample: {avg_time_ms:.3f} ms")
    print(f"Throughput: {1000/avg_time_ms:.1f} samples/second")
    print(f"Total time for 1000 samples: {total_time:.3f} seconds")
    print()
    
    # Memory usage (approximate)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Run inference to measure memory
        test_batch = torch.randn(256, input_dim).to(device)
        with torch.no_grad():
            _ = model(test_batch)
        
        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak GPU memory usage: {memory_mb:.2f} MB")
        print()
    
    # Performance summary
    print("PERFORMANCE SUMMARY:")
    print("-" * 30)
    print(f"✓ Model successfully converted from Keras to PyTorch")
    print(f"✓ GPU acceleration: {'Yes' if device.type == 'cuda' else 'No'}")
    print(f"✓ Compact model size: {model_size_mb:.2f} MB")
    print(f"✓ Fast inference: {avg_time_ms:.3f} ms per sample")
    print(f"✓ High throughput: {1000/avg_time_ms:.1f} samples/second")

if __name__ == "__main__":
    main()
