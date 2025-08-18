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
import gc

class LargeSpectralShapeAutoencoder(SpectralShapeAutoencoder):
    """
    Large version of SpectralShapeAutoencoder for benchmarking CUDA performance
    """
    def __init__(self, input_dim, n_mels=64, frames=5, latent_dim=32, slice_latent_dim=64):
        super().__init__(input_dim, n_mels, frames, latent_dim, slice_latent_dim)

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
            try:
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
            except Exception as e:
                print(f"CUDA kernel error: {e}. Falling back to native implementation.")
        
        # Fallback to the original Python implementation
        return super().forward(x)

def get_precise_memory_usage():
    """Get precise GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
    return 0.0

def estimate_memory_usage_mb(batch_size, input_dim, latent_dim, slice_latent_dim, frames):
    """Estimate GPU memory usage for a model configuration in MB"""
    # Model parameters (more precise calculation)
    n_mels = input_dim // frames
    
    param_memory = (
        n_mels * slice_latent_dim +  # slice encoder LayerNorm weight
        n_mels +  # slice encoder LayerNorm bias
        n_mels * slice_latent_dim +  # slice encoder Linear weight
        slice_latent_dim +  # slice encoder Linear bias
        latent_dim * (frames * slice_latent_dim) +  # global basis
        latent_dim +  # bottleneck LayerNorm weight
        latent_dim +  # bottleneck LayerNorm bias
        latent_dim * latent_dim +  # bottleneck Linear weight
        latent_dim +  # bottleneck Linear bias
        slice_latent_dim * n_mels +  # slice decoder weight
        n_mels  # slice decoder bias
    ) * 4  # 4 bytes per float32
    
    # Activation memory (more precise calculation)
    activation_memory = batch_size * (
        input_dim +  # input tensor
        frames * n_mels +  # slice inputs
        frames * slice_latent_dim +  # encoded slices
        latent_dim +  # latent code
        frames * slice_latent_dim +  # reconstructed slices
        input_dim +  # output tensor
        # Additional intermediate tensors
        frames * n_mels * 2 +  # layer norm intermediates
        latent_dim * 2  # bottleneck intermediates
    ) * 4  # 4 bytes per float32
    
    total_memory_mb = (param_memory + activation_memory) / (1024 * 1024)
    return total_memory_mb

def benchmark_inference_precise(model, input_tensor, num_runs=100000):
    """
    Precise benchmark inference time with extensive averaging
    """
    model.eval()
    
    print(f"    Running {num_runs:,} iterations for precise measurement...")
    
    # Extended warm-up runs
    with torch.no_grad():
        for _ in range(1000):
            _ = model(input_tensor)
    
    # Synchronize GPU
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    # Actual benchmark runs with precise timing
    inference_times = []
    
    with torch.no_grad():
        # Use smaller chunks to avoid memory issues with large num_runs
        chunk_size = min(10000, num_runs)
        num_chunks = num_runs // chunk_size
        
        for chunk in range(num_chunks):
            chunk_times = []
            
            for _ in range(chunk_size):
                if input_tensor.is_cuda:
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                _ = model(input_tensor)
                
                if input_tensor.is_cuda:
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                chunk_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            inference_times.extend(chunk_times)
            
            # Progress indicator
            if (chunk + 1) % max(1, num_chunks // 10) == 0:
                progress = (chunk + 1) / num_chunks * 100
                print(f"      Progress: {progress:.0f}% ({chunk + 1}/{num_chunks} chunks)")
    
    # Calculate comprehensive statistics
    times_array = np.array(inference_times)
    
    return {
        'mean_time_ms': np.mean(times_array),
        'std_time_ms': np.std(times_array),
        'min_time_ms': np.min(times_array),
        'max_time_ms': np.max(times_array),
        'median_time_ms': np.median(times_array),
        'p95_time_ms': np.percentile(times_array, 95),
        'p99_time_ms': np.percentile(times_array, 99),
        'num_samples': len(times_array)
    }

def generate_precise_benchmark(model_configs, batch_sizes=[64, 128, 256, 512, 1024], num_runs=100000):
    """
    Generate precise benchmarking report with extensive averaging
    """
    results = []
    
    for config_name, config in model_configs.items():
        print(f"\n{'='*80}")
        print(f"Testing Configuration: {config_name}")
        print(f"Input Dim: {config['input_dim']}, Latent Dim: {config['latent_dim']}, Slice Latent Dim: {config['slice_latent_dim']}")
        print(f"{'='*80}")
        
        for batch_size in batch_sizes:
            # Estimate memory usage in MB
            memory_estimate_mb = estimate_memory_usage_mb(
                batch_size, config['input_dim'], config['latent_dim'], 
                config['slice_latent_dim'], config['frames']
            )
            
            if memory_estimate_mb > 25000:  # 25GB limit in MB
                print(f"Skipping batch size {batch_size} for {config_name} (estimated {memory_estimate_mb:.1f}MB > 25GB limit)")
                continue
            
            print(f"\nTesting batch size {batch_size} (estimated memory: {memory_estimate_mb:.1f}MB)")
            
            try:
                # Clear GPU memory before each test
                torch.cuda.empty_cache()
                gc.collect()
                
                # Measure baseline memory
                baseline_memory_mb = get_precise_memory_usage()
                
                # Create input tensor
                input_tensor = torch.randn(batch_size, config['input_dim'], device='cuda')
                
                # Native PyTorch Model
                native_model = LargeSpectralShapeAutoencoder(
                    config['input_dim'], 
                    config['n_mels'], 
                    config['frames'], 
                    config['latent_dim'], 
                    config['slice_latent_dim']
                ).to('cuda')
                
                # Measure actual memory usage
                actual_memory_mb = get_precise_memory_usage() - baseline_memory_mb
                
                print(f"  Native PyTorch Model:")
                native_result = benchmark_inference_precise(native_model, input_tensor, num_runs)
                
                # CUDA Kernel Model (now supports all model sizes)
                print(f"  CUDA Kernel Model:")
                cuda_model = SpectralShapeAutoencoderCUDA(
                    config['input_dim'], 
                    config['n_mels'], 
                    config['frames'], 
                    config['latent_dim'], 
                    config['slice_latent_dim']
                ).to('cuda')
                cuda_model.load_state_dict(native_model.state_dict())
                cuda_result = benchmark_inference_precise(cuda_model, input_tensor, num_runs)
                speedup = native_result['mean_time_ms'] / cuda_result['mean_time_ms']
                del cuda_model
                
                result = {
                    'config': config_name,
                    'batch_size': batch_size,
                    'input_dim': config['input_dim'],
                    'latent_dim': config['latent_dim'],
                    'slice_latent_dim': config['slice_latent_dim'],
                    'native_mean_ms': native_result['mean_time_ms'],
                    'native_std_ms': native_result['std_time_ms'],
                    'native_median_ms': native_result['median_time_ms'],
                    'cuda_mean_ms': cuda_result['mean_time_ms'] if cuda_result else 0.0,
                    'cuda_std_ms': cuda_result['std_time_ms'] if cuda_result else 0.0,
                    'speedup_ratio': speedup,
                    'memory_estimate_mb': memory_estimate_mb,
                    'actual_memory_mb': actual_memory_mb,
                    'throughput_samples_per_sec': batch_size / (native_result['mean_time_ms'] / 1000),
                    'num_runs': native_result['num_samples']
                }
                
                results.append(result)
                
                # Print detailed results
                print(f"    Native PyTorch: {native_result['mean_time_ms']:.3f} ± {native_result['std_time_ms']:.3f} ms")
                print(f"    Native Median: {native_result['median_time_ms']:.3f} ms (P95: {native_result['p95_time_ms']:.3f} ms)")
                
                if cuda_result:
                    print(f"    CUDA Kernel: {cuda_result['mean_time_ms']:.3f} ± {cuda_result['std_time_ms']:.3f} ms")
                    print(f"    Speedup: {speedup:.3f}x")
                
                print(f"    Throughput: {result['throughput_samples_per_sec']:.1f} samples/sec")
                print(f"    Actual Memory: {actual_memory_mb:.1f} MB")
                
                # Clean up GPU memory
                del native_model, input_tensor
                torch.cuda.empty_cache()
                gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Out of memory for batch size {batch_size}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    break
                else:
                    raise e
    
    return results

def plot_precise_results(results):
    """
    Create visualization of precise benchmark results with error bars
    """
    # Group results by configuration
    configs = {}
    for result in results:
        config_name = result['config']
        if config_name not in configs:
            configs[config_name] = []
        configs[config_name].append(result)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Inference Time vs Batch Size with error bars
    for config_name, config_results in configs.items():
        batch_sizes = [r['batch_size'] for r in config_results]
        native_means = [r['native_mean_ms'] for r in config_results]
        native_stds = [r['native_std_ms'] for r in config_results]
        
        ax1.errorbar(batch_sizes, native_means, yerr=native_stds, 
                    marker='o', label=f'{config_name} (Native)', capsize=3)
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_title('Inference Time vs Batch Size (Mean ± Std)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Throughput vs Batch Size
    for config_name, config_results in configs.items():
        batch_sizes = [r['batch_size'] for r in config_results]
        throughputs = [r['throughput_samples_per_sec'] for r in config_results]
        ax2.plot(batch_sizes, throughputs, marker='s', label=f'{config_name}', linewidth=2)
    
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (samples/sec)')
    ax2.set_title('Throughput vs Batch Size')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Actual vs Estimated Memory Usage
    actual_memory = [r['actual_memory_mb'] for r in results]
    estimated_memory = [r['memory_estimate_mb'] for r in results]
    config_names = [r['config'] for r in results]
    
    scatter = ax3.scatter(estimated_memory, actual_memory, 
                         c=range(len(results)), s=60, alpha=0.7, cmap='viridis')
    
    # Add diagonal line for perfect estimation
    max_mem = max(max(actual_memory), max(estimated_memory))
    ax3.plot([0, max_mem], [0, max_mem], 'r--', alpha=0.5, label='Perfect Estimation')
    
    ax3.set_xlabel('Estimated Memory Usage (MB)')
    ax3.set_ylabel('Actual Memory Usage (MB)')
    ax3.set_title('Memory Usage: Actual vs Estimated')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: CUDA Speedup with error bars (if available)
    original_results = [r for r in results if r['config'] == 'Original' and r['speedup_ratio'] > 0]
    if original_results:
        batch_sizes = [r['batch_size'] for r in original_results]
        speedups = [r['speedup_ratio'] for r in original_results]
        
        # Calculate speedup error propagation (simplified)
        speedup_errors = [
            r['speedup_ratio'] * np.sqrt(
                (r['native_std_ms'] / r['native_mean_ms'])**2 + 
                (r['cuda_std_ms'] / r['cuda_mean_ms'])**2
            ) for r in original_results
        ]
        
        ax4.errorbar(batch_sizes, speedups, yerr=speedup_errors,
                    marker='d', color='red', linewidth=2, capsize=5)
        ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No Speedup')
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Speedup Ratio (Native/CUDA)')
        ax4.set_title('CUDA Kernel Speedup (Mean ± Error)')
        ax4.set_xscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'CUDA Kernel\nSpeedup Data\nNot Available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('CUDA Kernel Speedup')
    
    plt.tight_layout()
    plt.savefig('precise_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to run precise benchmarks
    """
    print("Starting Precise SpectralShapeAutoencoder CUDA Benchmarking...")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Define model configurations
    model_configs = {
        "Original": {
            'input_dim': 320,
            'n_mels': 64,
            'frames': 5,
            'latent_dim': 32,
            'slice_latent_dim': 64
        },
        "Medium": {
            'input_dim': 1280,  # 4x larger
            'n_mels': 128,
            'frames': 10,
            'latent_dim': 128,
            'slice_latent_dim': 256
        },
        "Large": {
            'input_dim': 2560,  # 8x larger
            'n_mels': 256,
            'frames': 10,
            'latent_dim': 256,
            'slice_latent_dim': 512
        },
        "XLarge": {
            'input_dim': 5120,  # 16x larger
            'n_mels': 512,
            'frames': 10,
            'latent_dim': 512,
            'slice_latent_dim': 1024
        }
    }
    
    # Run precise benchmarks with extensive averaging
    batch_sizes = [64, 128, 256, 512, 1024, 2048]
    num_runs = 100000  # 100k runs for statistical reliability
    
    print(f"\nRunning {num_runs:,} iterations per test for maximum precision...")
    results = generate_precise_benchmark(model_configs, batch_sizes, num_runs)
    
    # Plot results
    plot_precise_results(results)
    
    # Print comprehensive summary
    print(f"\n{'='*100}")
    print("PRECISE BENCHMARK SUMMARY")
    print(f"{'='*100}")
    print(f"{'Config':<10} | {'Batch':<5} | {'Native (ms)':<15} | {'CUDA (ms)':<15} | {'Speedup':<8} | {'Memory (MB)':<12} | {'Throughput':<15}")
    print(f"{'-'*100}")
    
    for result in results:
        native_time = f"{result['native_mean_ms']:.3f}±{result['native_std_ms']:.3f}"
        cuda_time = f"{result['cuda_mean_ms']:.3f}±{result['cuda_std_ms']:.3f}" if result['cuda_mean_ms'] > 0 else "N/A"
        speedup = f"{result['speedup_ratio']:.3f}x" if result['speedup_ratio'] > 0 else "N/A"
        
        print(f"{result['config']:<10} | {result['batch_size']:<5} | {native_time:<15} | {cuda_time:<15} | "
              f"{speedup:<8} | {result['actual_memory_mb']:<12.1f} | {result['throughput_samples_per_sec']:<15.1f}")
    
    print(f"\nResults visualization saved to: precise_benchmark_results.png")
    print(f"Total benchmark runs per test: {num_runs:,}")
    print("Precise benchmarking completed!")

if __name__ == '__main__':
    main()
