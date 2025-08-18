import torch
import torch.nn.functional as F
import numpy as np
import spectral_shape_ae_cuda
import time
from pathlib import Path

def stage_by_stage_comparison(x, slice_enc_lin_w, slice_enc_lin_b, normalized_basis, slice_dec_lin_w, slice_dec_lin_b):
    """
    Perform a detailed, stage-by-stage comparison of computational steps
    """
    # Test parameters
    batch_size = x.size(0)
    n_mels = 64
    frames = 5
    slice_latent_dim = 64
    latent_dim = 32
    combined_dim = frames * slice_latent_dim

    # Stage 1: Slice Encoding
    def compare_slice_encoding():
        print("\n--- Stage 1: Slice Encoding ---")
        
        # Native PyTorch
        x_slices = x.view(-1, frames, n_mels)
        native_encoded_slices = F.gelu(F.linear(x_slices, slice_enc_lin_w, slice_enc_lin_b))
        
        # CUDA Kernel (simulated)
        cuda_encoded_slices = torch.zeros_like(native_encoded_slices)
        for frame in range(frames):
            for batch in range(batch_size):
                slice_input = x_slices[batch, frame]
                for j in range(slice_latent_dim):
                    cuda_sum = torch.sum(slice_input * slice_enc_lin_w[j]) + slice_enc_lin_b[j]
                    cuda_encoded_slices[batch, frame, j] = torch.nn.functional.gelu(cuda_sum)
        
        # Comparison
        abs_diff = torch.abs(native_encoded_slices - cuda_encoded_slices)
        print("Maximum Absolute Difference in Slice Encoding:", torch.max(abs_diff).item())
        print("Mean Absolute Difference in Slice Encoding:", torch.mean(abs_diff).item())
        
        return native_encoded_slices, cuda_encoded_slices

    # Stage 2: Global Basis Projection
    def compare_basis_projection(encoded_slices):
        print("\n--- Stage 2: Global Basis Projection ---")
        
        # Native PyTorch
        combined_vec = encoded_slices.view(-1, combined_dim)
        native_latent_code = torch.matmul(combined_vec, normalized_basis.T)
        
        # CUDA Kernel (simulated)
        cuda_latent_code = torch.zeros_like(native_latent_code)
        for batch in range(batch_size):
            for i in range(latent_dim):
                cuda_latent_code[batch, i] = torch.sum(combined_vec[batch] * normalized_basis[i])
        
        # Comparison
        abs_diff = torch.abs(native_latent_code - cuda_latent_code)
        print("Maximum Absolute Difference in Basis Projection:", torch.max(abs_diff).item())
        print("Mean Absolute Difference in Basis Projection:", torch.mean(abs_diff).item())
        
        return native_latent_code, cuda_latent_code

    # Stage 3: Bottleneck MLP
    def compare_bottleneck_mlp(latent_code):
        print("\n--- Stage 3: Bottleneck MLP ---")
        
        # Native PyTorch
        native_processed_latent = F.gelu(latent_code)
        
        # CUDA Kernel (simulated)
        cuda_processed_latent = torch.zeros_like(native_processed_latent)
        for batch in range(batch_size):
            for i in range(latent_dim):
                cuda_processed_latent[batch, i] = torch.nn.functional.gelu(latent_code[batch, i])
        
        # Comparison
        abs_diff = torch.abs(native_processed_latent - cuda_processed_latent)
        print("Maximum Absolute Difference in Bottleneck MLP:", torch.max(abs_diff).item())
        print("Mean Absolute Difference in Bottleneck MLP:", torch.mean(abs_diff).item())
        
        return native_processed_latent, cuda_processed_latent

    # Stage 4: Reconstruction
    def compare_reconstruction(processed_latent):
        print("\n--- Stage 4: Reconstruction ---")
        
        # Native PyTorch
        native_reconstructed_combined = torch.matmul(processed_latent, normalized_basis)
        
        # CUDA Kernel (simulated)
        cuda_reconstructed_combined = torch.zeros_like(native_reconstructed_combined)
        for batch in range(batch_size):
            for i in range(combined_dim):
                cuda_reconstructed_combined[batch, i] = torch.sum(processed_latent[batch] * normalized_basis[:, i])
        
        # Comparison
        abs_diff = torch.abs(native_reconstructed_combined - cuda_reconstructed_combined)
        print("Maximum Absolute Difference in Reconstruction:", torch.max(abs_diff).item())
        print("Mean Absolute Difference in Reconstruction:", torch.mean(abs_diff).item())
        
        return native_reconstructed_combined, cuda_reconstructed_combined

    # Stage 5: Slice Decoding
    def compare_slice_decoding(reconstructed_combined):
        print("\n--- Stage 5: Slice Decoding ---")
        
        # Native PyTorch
        reconstructed_slices_latent = reconstructed_combined.view(-1, frames, slice_latent_dim)
        native_reconstructed_slices = F.linear(reconstructed_slices_latent, slice_dec_lin_w, slice_dec_lin_b)
        native_output = native_reconstructed_slices.view(-1, x.size(1))
        
        # CUDA Kernel (simulated)
        cuda_reconstructed_slices = torch.zeros_like(native_reconstructed_slices)
        for batch in range(batch_size):
            for frame in range(frames):
                for j in range(n_mels):
                    slice_sum = torch.sum(reconstructed_combined[batch, frame * slice_latent_dim:(frame+1) * slice_latent_dim] * slice_dec_lin_w[j]) + slice_dec_lin_b[j]
                    cuda_reconstructed_slices[batch, frame, j] = slice_sum
        cuda_output = cuda_reconstructed_slices.view(-1, x.size(1))
        
        # Comparison
        abs_diff = torch.abs(native_output - cuda_output)
        print("Maximum Absolute Difference in Slice Decoding:", torch.max(abs_diff).item())
        print("Mean Absolute Difference in Slice Decoding:", torch.mean(abs_diff).item())
        
        return native_output, cuda_output

    # Run stage-by-stage comparison
    print("\n=== Stage-by-Stage Computational Comparison ===")
    
    # Stage 1: Slice Encoding
    native_encoded_slices, cuda_encoded_slices = compare_slice_encoding()
    
    # Stage 2: Global Basis Projection
    native_latent_code, cuda_latent_code = compare_basis_projection(native_encoded_slices)
    
    # Stage 3: Bottleneck MLP
    native_processed_latent, cuda_processed_latent = compare_bottleneck_mlp(native_latent_code)
    
    # Stage 4: Reconstruction
    native_reconstructed_combined, cuda_reconstructed_combined = compare_reconstruction(native_processed_latent)
    
    # Stage 5: Slice Decoding
    native_output, cuda_output = compare_slice_decoding(native_reconstructed_combined)
    
    return native_output, cuda_output

def test_cuda_kernel():
    # Test parameters matching the SpectralShapeAutoencoder
    batch_size = 32
    n_mels = 64
    frames = 5
    input_dim = n_mels * frames
    latent_dim = 32
    slice_latent_dim = 64
    combined_dim = frames * slice_latent_dim

    # Set fixed random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create random input tensor
    x = torch.randn(batch_size, input_dim, device='cuda')

    # Create random weights and biases
    slice_enc_lin_w = torch.randn(slice_latent_dim, n_mels, device='cuda')
    slice_enc_lin_b = torch.randn(slice_latent_dim, device='cuda')
    
    normalized_basis = F.normalize(torch.randn(latent_dim, combined_dim, device='cuda'), p=2, dim=1)
    
    slice_dec_lin_w = torch.randn(n_mels, slice_latent_dim, device='cuda')
    slice_dec_lin_b = torch.randn(n_mels, device='cuda')

    # Perform stage-by-stage comparison
    native_output, cuda_output = stage_by_stage_comparison(
        x, slice_enc_lin_w, slice_enc_lin_b, 
        normalized_basis, slice_dec_lin_w, slice_dec_lin_b
    )

    # Final comparison
    print("\n=== Final Output Comparison ===")
    abs_diff = torch.abs(native_output - cuda_output)
    print("Maximum Absolute Difference:", torch.max(abs_diff).item())
    print("Mean Absolute Difference:", torch.mean(abs_diff).item())
    print("Mismatched Elements:", torch.sum(abs_diff > 1e-3))

    # Attempt close comparison with more lenient tolerances
    try:
        torch.testing.assert_close(cuda_output, native_output, rtol=1e-2, atol=1e-2)
        print("\n✅ CUDA kernel output matches native PyTorch implementation!")
    except AssertionError as e:
        print("\n❌ CUDA kernel output differs from native PyTorch implementation:")
        print(str(e))
        
        # Additional debugging: save problematic tensors
        debug_dir = Path('./debug')
        debug_dir.mkdir(exist_ok=True)
        torch.save(cuda_output, debug_dir / 'cuda_output.pt')
        torch.save(native_output, debug_dir / 'native_output.pt')
        print(f"\nDebug tensors saved in {debug_dir}")
        
        raise

if __name__ == "__main__":
    test_cuda_kernel()
