#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_THREADS = 256;

// Fast GELU approximation
__device__ __forceinline__ float gelu_fast(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Parameterizable SpectralShapeAutoencoder kernel
__global__ void spectral_shape_autoencoder_parameterizable_kernel(
    const float* __restrict__ x_in,
    float* __restrict__ x_out,
    const float* __restrict__ slice_enc_ln_w, 
    const float* __restrict__ slice_enc_ln_b,
    const float* __restrict__ slice_enc_lin_w, 
    const float* __restrict__ slice_enc_lin_b,
    const float* __restrict__ normalized_basis,
    const float* __restrict__ bottle_ln_w, 
    const float* __restrict__ bottle_ln_b,
    const float* __restrict__ bottle_lin_w, 
    const float* __restrict__ bottle_lin_b,
    const float* __restrict__ slice_dec_lin_w, 
    const float* __restrict__ slice_dec_lin_b,
    const int batch_size,
    const int n_mels,
    const int frames,
    const int latent_dim,
    const int slice_latent_dim
) {
    // Calculate derived dimensions
    const int combined_dim = frames * slice_latent_dim;
    const int input_dim = n_mels * frames;
    
    // Shared memory allocation (dynamically sized)
    extern __shared__ float shmem[];
    
    // Partition shared memory efficiently
    float* sh_encoded_slices = shmem; // combined_dim
    float* sh_latent = shmem + combined_dim; // latent_dim
    float* sh_reconstructed = shmem + combined_dim + latent_dim; // combined_dim
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    if (batch_idx >= batch_size) return;
    
    const float* batch_input = x_in + batch_idx * input_dim;
    float* batch_output = x_out + batch_idx * input_dim;
    
    // ===================================================================
    // STAGE 1: Slice Encoding with Shared Weights
    // Process each frame's slice through the shared encoder
    // ===================================================================
    
    // Process all slices in parallel
    for (int slice_idx = 0; slice_idx < frames; slice_idx++) {
        const float* slice_input = batch_input + slice_idx * n_mels;
        float* slice_output = sh_encoded_slices + slice_idx * slice_latent_dim;
        
        // Layer Normalization with warp-level reduction
        if (tid < n_mels) {
            // Compute mean using warp reduction
            float sum = 0.0f;
            for (int i = lane_id; i < n_mels; i += WARP_SIZE) {
                sum += slice_input[i];
            }
            sum = warp_reduce_sum(sum);
            float mean = __shfl_sync(0xffffffff, sum, 0) / n_mels;
            
            // Compute variance using warp reduction
            float sum_sq = 0.0f;
            for (int i = lane_id; i < n_mels; i += WARP_SIZE) {
                float diff = slice_input[i] - mean;
                sum_sq += diff * diff;
            }
            sum_sq = warp_reduce_sum(sum_sq);
            float variance = __shfl_sync(0xffffffff, sum_sq, 0) / n_mels;
            float inv_std = rsqrtf(variance + 1e-5f);
            
            // Apply layer normalization and linear transformation
            if (tid < slice_latent_dim) {
                float sum_linear = 0.0f;
                
                // Vectorized matrix multiplication
                #pragma unroll 4
                for (int k = 0; k < n_mels; k += 4) {
                    if (k + 3 < n_mels) {
                        float norm_val0 = (slice_input[k] - mean) * inv_std * slice_enc_ln_w[k] + slice_enc_ln_b[k];
                        float norm_val1 = (slice_input[k+1] - mean) * inv_std * slice_enc_ln_w[k+1] + slice_enc_ln_b[k+1];
                        float norm_val2 = (slice_input[k+2] - mean) * inv_std * slice_enc_ln_w[k+2] + slice_enc_ln_b[k+2];
                        float norm_val3 = (slice_input[k+3] - mean) * inv_std * slice_enc_ln_w[k+3] + slice_enc_ln_b[k+3];
                        
                        sum_linear += norm_val0 * slice_enc_lin_w[tid * n_mels + k] +
                                     norm_val1 * slice_enc_lin_w[tid * n_mels + k + 1] +
                                     norm_val2 * slice_enc_lin_w[tid * n_mels + k + 2] +
                                     norm_val3 * slice_enc_lin_w[tid * n_mels + k + 3];
                    } else {
                        for (int j = k; j < n_mels; j++) {
                            float norm_val = (slice_input[j] - mean) * inv_std * slice_enc_ln_w[j] + slice_enc_ln_b[j];
                            sum_linear += norm_val * slice_enc_lin_w[tid * n_mels + j];
                        }
                        break;
                    }
                }
                
                sum_linear += slice_enc_lin_b[tid];
                slice_output[tid] = gelu_fast(sum_linear);
            }
        }
        __syncthreads();
    }
    
    // ===================================================================
    // STAGE 2: Global Basis Projection
    // Project combined encoded slices onto normalized basis
    // ===================================================================
    
    for (int latent_idx = tid; latent_idx < latent_dim; latent_idx += BLOCK_THREADS) {
        float sum = 0.0f;
        
        // Optimized dot product with loop unrolling
        #pragma unroll 8
        for (int i = 0; i < combined_dim; i += 8) {
            if (i + 7 < combined_dim) {
                sum += sh_encoded_slices[i] * normalized_basis[latent_idx * combined_dim + i] +
                       sh_encoded_slices[i + 1] * normalized_basis[latent_idx * combined_dim + i + 1] +
                       sh_encoded_slices[i + 2] * normalized_basis[latent_idx * combined_dim + i + 2] +
                       sh_encoded_slices[i + 3] * normalized_basis[latent_idx * combined_dim + i + 3] +
                       sh_encoded_slices[i + 4] * normalized_basis[latent_idx * combined_dim + i + 4] +
                       sh_encoded_slices[i + 5] * normalized_basis[latent_idx * combined_dim + i + 5] +
                       sh_encoded_slices[i + 6] * normalized_basis[latent_idx * combined_dim + i + 6] +
                       sh_encoded_slices[i + 7] * normalized_basis[latent_idx * combined_dim + i + 7];
            } else {
                for (int j = i; j < combined_dim; j++) {
                    sum += sh_encoded_slices[j] * normalized_basis[latent_idx * combined_dim + j];
                }
                break;
            }
        }
        
        sh_latent[latent_idx] = sum;
    }
    
    __syncthreads();
    
    // ===================================================================
    // STAGE 3: Bottleneck MLP
    // Apply layer normalization and linear transformation
    // ===================================================================
    
    // Layer normalization on latent code
    if (tid == 0) {
        float sum = 0.0f, sum_sq = 0.0f;
        for (int i = 0; i < latent_dim; i++) {
            sum += sh_latent[i];
            sum_sq += sh_latent[i] * sh_latent[i];
        }
        float mean = sum / latent_dim;
        float variance = sum_sq / latent_dim - mean * mean;
        float inv_std = rsqrtf(variance + 1e-5f);
        
        // Apply bottleneck MLP
        for (int i = 0; i < latent_dim; i++) {
            float normalized = (sh_latent[i] - mean) * inv_std * bottle_ln_w[i] + bottle_ln_b[i];
            float linear_out = normalized * bottle_lin_w[i] + bottle_lin_b[i];
            sh_latent[i] = gelu_fast(linear_out);
        }
    }
    
    __syncthreads();
    
    // ===================================================================
    // STAGE 4: Reconstruction from Global Basis
    // Reconstruct combined vector using the same normalized basis
    // ===================================================================
    
    for (int recon_idx = tid; recon_idx < combined_dim; recon_idx += BLOCK_THREADS) {
        float sum = 0.0f;
        
        // Optimized dot product with loop unrolling
        #pragma unroll 8
        for (int k = 0; k < latent_dim; k += 8) {
            if (k + 7 < latent_dim) {
                sum += sh_latent[k] * normalized_basis[k * combined_dim + recon_idx] +
                       sh_latent[k + 1] * normalized_basis[(k + 1) * combined_dim + recon_idx] +
                       sh_latent[k + 2] * normalized_basis[(k + 2) * combined_dim + recon_idx] +
                       sh_latent[k + 3] * normalized_basis[(k + 3) * combined_dim + recon_idx] +
                       sh_latent[k + 4] * normalized_basis[(k + 4) * combined_dim + recon_idx] +
                       sh_latent[k + 5] * normalized_basis[(k + 5) * combined_dim + recon_idx] +
                       sh_latent[k + 6] * normalized_basis[(k + 6) * combined_dim + recon_idx] +
                       sh_latent[k + 7] * normalized_basis[(k + 7) * combined_dim + recon_idx];
            } else {
                for (int j = k; j < latent_dim; j++) {
                    sum += sh_latent[j] * normalized_basis[j * combined_dim + recon_idx];
                }
                break;
            }
        }
        
        sh_reconstructed[recon_idx] = sum;
    }
    
    __syncthreads();
    
    // ===================================================================
    // STAGE 5: Slice Decoding
    // Decode each slice using the shared decoder
    // ===================================================================
    
    for (int slice_idx = 0; slice_idx < frames; slice_idx++) {
        const float* slice_latent = sh_reconstructed + slice_idx * slice_latent_dim;
        float* slice_output = batch_output + slice_idx * n_mels;
        
        if (tid < n_mels) {
            float sum = 0.0f;
            
            // Optimized linear transformation
            #pragma unroll 8
            for (int k = 0; k < slice_latent_dim; k += 8) {
                if (k + 7 < slice_latent_dim) {
                    sum += slice_latent[k] * slice_dec_lin_w[tid * slice_latent_dim + k] +
                           slice_latent[k + 1] * slice_dec_lin_w[tid * slice_latent_dim + k + 1] +
                           slice_latent[k + 2] * slice_dec_lin_w[tid * slice_latent_dim + k + 2] +
                           slice_latent[k + 3] * slice_dec_lin_w[tid * slice_latent_dim + k + 3] +
                           slice_latent[k + 4] * slice_dec_lin_w[tid * slice_latent_dim + k + 4] +
                           slice_latent[k + 5] * slice_dec_lin_w[tid * slice_latent_dim + k + 5] +
                           slice_latent[k + 6] * slice_dec_lin_w[tid * slice_latent_dim + k + 6] +
                           slice_latent[k + 7] * slice_dec_lin_w[tid * slice_latent_dim + k + 7];
                } else {
                    for (int j = k; j < slice_latent_dim; j++) {
                        sum += slice_latent[j] * slice_dec_lin_w[tid * slice_latent_dim + j];
                    }
                    break;
                }
            }
            
            sum += slice_dec_lin_b[tid];
            slice_output[tid] = sum;
        }
        __syncthreads();
    }
}

} // anonymous namespace

// C++ Wrapper to Launch the CUDA Kernel
void spectral_shape_autoencoder_forward_cuda(
    const at::Tensor& x, at::Tensor& out,
    const at::Tensor& slice_enc_ln_w, const at::Tensor& slice_enc_ln_b,
    const at::Tensor& slice_enc_lin_w, const at::Tensor& slice_enc_lin_b,
    const at::Tensor& normalized_basis,
    const at::Tensor& bottle_ln_w, const at::Tensor& bottle_ln_b,
    const at::Tensor& bottle_lin_w, const at::Tensor& bottle_lin_b,
    const at::Tensor& slice_dec_lin_w, const at::Tensor& slice_dec_lin_b
) {
    const int batch_size = x.size(0);
    const int input_dim = x.size(1);
    
    // Extract model dimensions from tensor shapes
    const int n_mels = slice_enc_ln_w.size(0);
    const int slice_latent_dim = slice_enc_lin_w.size(0);
    const int frames = input_dim / n_mels;
    const int latent_dim = normalized_basis.size(0);
    const int combined_dim = frames * slice_latent_dim;
    
    // Optimized kernel launch configuration
    const dim3 threads(BLOCK_THREADS);
    const dim3 blocks(batch_size);

    // Calculate shared memory size dynamically
    const int shared_mem_size = 
        combined_dim +  // encoded slices
        latent_dim +    // latent code
        combined_dim;   // reconstructed slices

    // Launch the parameterizable kernel
    spectral_shape_autoencoder_parameterizable_kernel<<<blocks, threads, shared_mem_size * sizeof(float)>>>(
        x.data_ptr<float>(), out.data_ptr<float>(),
        slice_enc_ln_w.data_ptr<float>(), slice_enc_ln_b.data_ptr<float>(),
        slice_enc_lin_w.data_ptr<float>(), slice_enc_lin_b.data_ptr<float>(),
        normalized_basis.data_ptr<float>(),
        bottle_ln_w.data_ptr<float>(), bottle_ln_b.data_ptr<float>(),
        bottle_lin_w.data_ptr<float>(), bottle_lin_b.data_ptr<float>(),
        slice_dec_lin_w.data_ptr<float>(), slice_dec_lin_b.data_ptr<float>(),
        batch_size, n_mels, frames, latent_dim, slice_latent_dim
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
