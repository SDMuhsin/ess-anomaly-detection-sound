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

class LargeSpectralShapeAutoencoder(SpectralShapeAutoencoder):
    """
    Large version of SpectralShapeAutoencoder for benchmarking CUDA performance
    """
    def __init__(self, input_dim, n_mels=64, frames=5, latent_dim=32, slice_latent_dim=64, scale_factor=1):
        # Scale up the model dimensions
        scaled_latent_dim = latent_dim * scale_factor
        scaled_slice_latent_dim = slice_latent_dim * scale_factor
        
        super().__init__(input_dim, n_mels, frames, scaled_latent_dim, scaled_slice_latent_dim)

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

def estimate_memory_usage(batch_size, input_dim, latent_dim, slice_latent_dim, frames):
    """Estimate GPU memory usage for a model configuration"""
    # Model parameters
    param_memory = (
        input_dim * slice_latent_dim +  # slice encoder linear
        slice_latent_dim +  # slice encoder bias
        latent_dim * (frames * slice_latent_dim) +  # global basis
        latent_dim * 2 +  # bottleneck weights and biases
        slice_latent_dim * input_dim // frames +  # slice decoder
        slice_latent_dim  # slice decoder bias
    ) * 4  # 4 bytes per float32
    
    # Activation memory (rough estimate)
    activation_memory = batch_size * (
        input_dim +  # input
        frames * slice_latent_dim +  # encoded slices
        latent_dim +  # latent code
        frames * slice_latent_dim +  # reconstructed slices
        input_dim  # output
    ) * 4  # 4 bytes per float32
    
    total_memory_gb = (param_memory + activation_memory) / (1024**3)
    return total_memory_gb

def benchmark_inference(model, input_tensor, num_runs=100):
    """
    Benchmark inference time for a given model and input tensor
    """
    model.eval()
    
    # Warm-up runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Synchronize GPU
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    # Actual benchmark runs
    inference_times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if input_tensor.is_cuda:
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(input_tensor)
            
            if input_tensor.is_cuda:
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    return {
        'mean_time_ms': np.mean(inference_times),
        'std_time_ms': np.std(inference_times),
        'min_time_ms': np.min(inference_times),
        'max_time_ms': np.max(inference_times)
    }

def generate_large_model_benchmark(model_configs, batch_sizes=[64, 128, 256, 512, 1024]):
    """
    Generate comprehensive benchmarking report for large SpectralShapeAutoencoder models
    """
    results = []
    
    for config_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Testing Configuration: {config_name}")
        print(f"Input Dim: {config['input_dim']}, Latent Dim: {config['latent_dim']}, Slice Latent Dim: {config['slice_latent_dim']}")
        print(f"{'='*60}")
        
        for batch_size in batch_sizes:
            # Estimate memory usage
            memory_estimate = estimate_memory_usage(
                batch_size, config['input_dim'], config['latent_dim'], 
                config['slice_latent_dim'], config['frames']
            )
            
            if memory_estimate > 25:  # Leave some headroom
                print(f"Skipping batch size {batch_size} for {config_name} (estimated {memory_estimate:.1f}GB > 25GB limit)")
                continue
            
            print(f"Testing batch size {batch_size} (estimated memory: {memory_estimate:.1f}GB)")
            
            try:
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
                
                native_result = benchmark_inference(native_model, input_tensor, num_runs=10000)
                
                # CUDA Kernel Model (only for original size due to kernel constraints)
                if config_name == "Original":
                    cuda_model = SpectralShapeAutoencoderCUDA(
                        config['input_dim'], 
                        config['n_mels'], 
                        config['frames'], 
                        config['latent_dim'], 
                        config['slice_latent_dim']
                    ).to('cuda')
                    cuda_model.load_state_dict(native_model.state_dict())
                    cuda_result = benchmark_inference(cuda_model, input_tensor, num_runs=10000)
                    speedup = native_result['mean_time_ms'] / cuda_result['mean_time_ms']
                else:
                    cuda_result = {'mean_time_ms': 0.0}
                    speedup = 0.0
                
                result = {
                    'config': config_name,
                    'batch_size': batch_size,
                    'input_dim': config['input_dim'],
                    'latent_dim': config['latent_dim'],
                    'slice_latent_dim': config['slice_latent_dim'],
                    'native_time_ms': native_result['mean_time_ms'],
                    'cuda_time_ms': cuda_result['mean_time_ms'],
                    'speedup_ratio': speedup,
                    'memory_estimate_gb': memory_estimate,
                    'throughput_samples_per_sec': batch_size / (native_result['mean_time_ms'] / 1000)
                }
                
                results.append(result)
                
                print(f"  Native PyTorch: {native_result['mean_time_ms']:.2f}ms")
                if speedup > 0:
                    print(f"  CUDA Kernel: {cuda_result['mean_time_ms']:.2f}ms (Speedup: {speedup:.2f}x)")
                print(f"  Throughput: {result['throughput_samples_per_sec']:.1f} samples/sec")
                
                # Clean up GPU memory
                del native_model, input_tensor
                if config_name == "Original":
                    del cuda_model
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  Out of memory for batch size {batch_size}")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e
    
    return results

def plot_large_model_results(results):
    """
    Create visualization of large model benchmark results
    """
    # Group results by configuration
    configs = {}
    for result in results:
        config_name = result['config']
        if config_name not in configs:
            configs[config_name] = []
        configs[config_name].append(result)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Inference Time vs Batch Size
    for config_name, config_results in configs.items():
        batch_sizes = [r['batch_size'] for r in config_results]
        native_times = [r['native_time_ms'] for r in config_results]
        ax1.plot(batch_sizes, native_times, marker='o', label=f'{config_name} (Native)')
    
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Inference Time (ms)')
    ax1.set_title('Inference Time vs Batch Size')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Throughput vs Batch Size
    for config_name, config_results in configs.items():
        batch_sizes = [r['batch_size'] for r in config_results]
        throughputs = [r['throughput_samples_per_sec'] for r in config_results]
        ax2.plot(batch_sizes, throughputs, marker='s', label=f'{config_name}')
    
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (samples/sec)')
    ax2.set_title('Throughput vs Batch Size')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Memory Usage vs Model Size
    model_sizes = []
    memory_usage = []
    config_names = []
    
    for config_name, config_results in configs.items():
        if config_results:
            # Use the largest batch size result for each config
            largest_batch_result = max(config_results, key=lambda x: x['batch_size'])
            model_size = largest_batch_result['input_dim'] * largest_batch_result['latent_dim']
            model_sizes.append(model_size)
            memory_usage.append(largest_batch_result['memory_estimate_gb'])
            config_names.append(config_name)
    
    ax3.scatter(model_sizes, memory_usage, s=100)
    for i, name in enumerate(config_names):
        ax3.annotate(name, (model_sizes[i], memory_usage[i]), xytext=(5, 5), textcoords='offset points')
    
    ax3.set_xlabel('Model Size (input_dim × latent_dim)')
    ax3.set_ylabel('Memory Usage (GB)')
    ax3.set_title('Memory Usage vs Model Size')
    ax3.set_xscale('log')
    ax3.grid(True)
    
    # Plot 4: Speedup for Original Model (if available)
    original_results = configs.get('Original', [])
    if original_results and any(r['speedup_ratio'] > 0 for r in original_results):
        batch_sizes = [r['batch_size'] for r in original_results if r['speedup_ratio'] > 0]
        speedups = [r['speedup_ratio'] for r in original_results if r['speedup_ratio'] > 0]
        ax4.plot(batch_sizes, speedups, marker='d', color='red', linewidth=2)
        ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Speedup Ratio (Native/CUDA)')
        ax4.set_title('CUDA Kernel Speedup')
        ax4.set_xscale('log')
        ax4.grid(True)
    else:
        ax4.text(0.5, 0.5, 'CUDA Kernel\nOnly Available\nfor Original Model', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('CUDA Kernel Speedup')
    
    plt.tight_layout()
    plt.savefig('large_model_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to run large model benchmarks
    """
    print("Starting Large SpectralShapeAutoencoder CUDA Benchmarking...")
    print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # Define model configurations with increasing complexity
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
        },
        "XXLarge": {
            'input_dim': 10240,  # 32x larger
            'n_mels': 1024,
            'frames': 10,
            'latent_dim': 1024,
            'slice_latent_dim': 2048
        }
    }
    
    # Run benchmarks
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
    results = generate_large_model_benchmark(model_configs, batch_sizes)
    
    # Plot results
    plot_large_model_results(results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("LARGE MODEL BENCHMARK SUMMARY")
    print(f"{'='*80}")
    
    for result in results:
        print(f"{result['config']:10s} | Batch: {result['batch_size']:4d} | "
              f"Native: {result['native_time_ms']:6.2f}ms | "
              f"Throughput: {result['throughput_samples_per_sec']:8.1f} samples/sec | "
              f"Memory: {result['memory_estimate_gb']:5.1f}GB")
        if result['speedup_ratio'] > 0:
            print(f"           | CUDA: {result['cuda_time_ms']:6.2f}ms | Speedup: {result['speedup_ratio']:5.2f}x")
    
    print(f"\nResults visualization saved to: large_model_benchmark_results.png")
    print("Large model benchmarking completed!")

if __name__ == '__main__':
    main()
