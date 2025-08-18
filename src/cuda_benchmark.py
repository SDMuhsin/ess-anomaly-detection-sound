import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from src.exp_autoencoding import SpectralShapeAutoencoder
import spectral_shape_ae_cuda

class SpectralShapeAutoencoderCUDA(SpectralShapeAutoencoder):
    """
    Wrapper for CUDA-optimized SpectralShapeAutoencoder 
    that uses the custom CUDA kernel during inference
    """
    def forward(self, x):
        """
        Override forward method to use CUDA kernel during inference
        """
        if not self.training and x.is_cuda:
            # Pre-normalize the basis once per call
            normalized_basis = torch.nn.functional.normalize(self.global_basis, p=2, dim=1)

            # Call the C++ extension's 'forward' function
            return spectral_shape_ae_cuda.forward(
                x.contiguous(),
                self.slice_encoder[0].weight.contiguous(),
                self.slice_encoder[0].bias.contiguous(),
                self.slice_encoder[1].weight.contiguous(),
                self.slice_encoder[1].bias.contiguous(),
                normalized_basis.contiguous(),
                self.bottleneck_mlp[0].weight.contiguous(),
                self.bottleneck_mlp[0].bias.contiguous(),
                self.bottleneck_mlp[1].weight.contiguous(),
                self.bottleneck_mlp[1].bias.contiguous(),
                self.slice_decoder[0].weight.contiguous(),
                self.slice_decoder[0].bias.contiguous()
            )
        else:
            # Fallback to the original Python implementation
            return super().forward(x)

def benchmark_inference(model, input_tensor, num_runs=100):
    """
    Benchmark inference time for a given model and input tensor
    
    Args:
        model (torch.nn.Module): Model to benchmark
        input_tensor (torch.Tensor): Input tensor for inference
        num_runs (int): Number of runs for averaging
    
    Returns:
        dict: Benchmark results including mean, std, and min inference time
    """
    model.eval()
    
    # Warm-up runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Actual benchmark runs
    inference_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.perf_counter()
            _ = model(input_tensor)
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    return {
        'mean_time_ms': np.mean(inference_times),
        'std_time_ms': np.std(inference_times),
        'min_time_ms': np.min(inference_times),
        'max_time_ms': np.max(inference_times)
    }

def generate_benchmark_report(batch_sizes=[32, 64, 128, 256, 512], 
                               input_dims=320, 
                               device='cuda'):
    """
    Generate comprehensive benchmarking report comparing native and CUDA implementations
    
    Args:
        batch_sizes (list): List of batch sizes to test
        input_dims (int): Input dimension for the model
        device (str): Device to run benchmarks on
    
    Returns:
        dict: Comprehensive benchmarking results
    """
    results = {
        'batch_sizes': batch_sizes,
        'native_results': [],
        'cuda_results': [],
        'speedup_ratios': []
    }
    
    for batch_size in batch_sizes:
        # Create input tensor
        input_tensor = torch.randn(batch_size, input_dims, device=device)
        
        # Native PyTorch Model (using standard forward pass)
        native_model = SpectralShapeAutoencoder(input_dims).to(device)
        native_result = benchmark_inference(native_model, input_tensor)
        
        # CUDA Kernel Model
        cuda_model = SpectralShapeAutoencoderCUDA(input_dims).to(device)
        # Copy weights from native model to ensure identical initialization
        cuda_model.load_state_dict(native_model.state_dict())
        cuda_result = benchmark_inference(cuda_model, input_tensor)
        
        results['native_results'].append(native_result)
        results['cuda_results'].append(cuda_result)
        
        # Calculate speedup
        speedup = native_result['mean_time_ms'] / cuda_result['mean_time_ms']
        results['speedup_ratios'].append(speedup)
    
    return results

def plot_benchmark_results(results):
    """
    Create visualization of benchmark results
    
    Args:
        results (dict): Benchmark results from generate_benchmark_report
    """
    plt.figure(figsize=(12, 5))
    
    # Inference Time Plot
    plt.subplot(1, 2, 1)
    plt.plot(results['batch_sizes'], 
             [r['mean_time_ms'] for r in results['native_results']], 
             label='Native PyTorch', marker='o')
    plt.plot(results['batch_sizes'], 
             [r['mean_time_ms'] for r in results['cuda_results']], 
             label='CUDA Kernel', marker='x')
    plt.title('Inference Time by Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Inference Time (ms)')
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    
    # Speedup Ratio Plot
    plt.subplot(1, 2, 2)
    plt.plot(results['batch_sizes'], results['speedup_ratios'], marker='s')
    plt.title('CUDA Kernel Speedup Ratio')
    plt.xlabel('Batch Size')
    plt.ylabel('Speedup (Native/CUDA)')
    plt.xscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cuda_benchmark_results.png')
    plt.close()

def main():
    """
    Main function to run benchmarks and generate report
    """
    print("Starting SpectralShapeAutoencoder CUDA Kernel Benchmarking...")
    
    # Run benchmarks
    benchmark_results = generate_benchmark_report()
    
    # Plot results
    plot_benchmark_results(benchmark_results)
    
    # Print detailed results
    print("\nBenchmark Results:")
    for i, batch_size in enumerate(benchmark_results['batch_sizes']):
        print(f"\nBatch Size: {batch_size}")
        print(f"Native PyTorch Mean Inference Time: {benchmark_results['native_results'][i]['mean_time_ms']:.4f} ms")
        print(f"CUDA Kernel Mean Inference Time: {benchmark_results['cuda_results'][i]['mean_time_ms']:.4f} ms")
        print(f"Speedup Ratio: {benchmark_results['speedup_ratios'][i]:.2f}x")

if __name__ == '__main__':
    main()
