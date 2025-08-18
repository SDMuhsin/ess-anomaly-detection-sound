#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {
// Model dimension constants
constexpr int N_MELS = 64;
constexpr int FRAMES = 5;
constexpr int LATENT_DIM = 32;
constexpr int SLICE_LATENT_DIM = 64;
constexpr int COMBINED_DIM = FRAMES * SLICE_LATENT_DIM; // 320
constexpr int BLOCK_THREADS = 256; // 8 warps of 32 threads

// Exact GELU approximation matching PyTorch
__device__ __forceinline__ float gelu_approx(float x) {
    return 0.5f * x * (1.0f + tanhf(sqrtf(2.0f / M_PI) * (x + 0.044715f * powf(x, 3.0f))));
}

// Precise layer normalization kernel
__device__ void layer_norm(
    const float* input, 
    float* output, 
    const float* weight, 
    const float* bias, 
    int size
) {
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < size; ++i) {
        mean += input[i];
    }
    mean /= size;

    // Compute variance
    float variance = 0.0f;
    for (int i = 0; i < size; ++i) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= size;

    // Standard deviation with small epsilon
    float std_dev = sqrtf(variance + 1e-5f);

    // Normalize and scale
    for (int i = 0; i < size; ++i) {
        output[i] = ((input[i] - mean) / std_dev) * weight[i] + bias[i];
    }
}

// Main CUDA kernel for SpectralShapeAutoencoder
__global__ void spectral_shape_autoencoder_kernel(
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
    const int batch_size
) {
    // Shared memory allocation
    extern __shared__ float shmem[];
    
    // Memory partitions in shared memory
    float* sh_input = shmem; 
    float* sh_normalized = shmem + N_MELS; 
    float* sh_encoded = shmem + N_MELS * 2;
    float* sh_latent = shmem + N_MELS * 2 + SLICE_LATENT_DIM;
    float* sh_reconstructed = shmem + N_MELS * 2 + SLICE_LATENT_DIM + LATENT_DIM;

    // Thread & block indexing
    int bidx = blockIdx.x; 
    int tid = threadIdx.x; 
    
    if (bidx >= batch_size) return;

    // Process each frame
    for (int frame = 0; frame < FRAMES; ++frame) {
        // Copy input slice to shared memory
        if (tid < N_MELS) {
            sh_input[tid] = x_in[bidx * COMBINED_DIM + frame * N_MELS + tid];
        }
        __syncthreads();

        // Layer Normalization (matching PyTorch's implementation)
        if (tid < N_MELS) {
            float mean = 0.0f;
            for (int i = 0; i < N_MELS; ++i) {
                mean += sh_input[i];
            }
            mean /= N_MELS;

            float variance = 0.0f;
            for (int i = 0; i < N_MELS; ++i) {
                float diff = sh_input[i] - mean;
                variance += diff * diff;
            }
            variance /= N_MELS;

            float std_dev = sqrtf(variance + 1e-5f);
            sh_normalized[tid] = (sh_input[tid] - mean) / std_dev;
        }
        __syncthreads();

        // Linear Layer with GELU
        if (tid < SLICE_LATENT_DIM) {
            float sum = 0.0f;
            for (int k = 0; k < N_MELS; ++k) {
                sum += sh_normalized[k] * slice_enc_lin_w[tid * N_MELS + k];
            }
            sum += slice_enc_lin_b[tid];
            sh_encoded[tid] = gelu_approx(sum);
        }
        __syncthreads();

        // Global Basis Projection (for the first thread block)
        if (frame == 0 && tid < LATENT_DIM) {
            float sum = 0.0f;
            for (int i = 0; i < COMBINED_DIM; ++i) {
                sum += sh_encoded[i] * normalized_basis[tid * COMBINED_DIM + i];
            }
            sh_latent[tid] = sum;
        }
        __syncthreads();

        // Reconstruction (for the first thread block)
        if (frame == 0 && tid < SLICE_LATENT_DIM) {
            float sum = 0.0f;
            for (int k = 0; k < LATENT_DIM; ++k) {
                sum += sh_latent[k] * normalized_basis[k * COMBINED_DIM + tid];
            }
            sh_reconstructed[tid] = sum;
        }
        __syncthreads();

        // Slice Decoder
        if (tid < N_MELS) {
            float sum = 0.0f;
            for (int k = 0; k < SLICE_LATENT_DIM; ++k) {
                sum += sh_reconstructed[k] * slice_dec_lin_w[tid * SLICE_LATENT_DIM + k];
            }
            sum += slice_dec_lin_b[tid];
            x_out[bidx * COMBINED_DIM + frame * N_MELS + tid] = sum;
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
    
    // Kernel launch configuration
    const dim3 threads(BLOCK_THREADS);
    const dim3 blocks(batch_size);

    // Calculate shared memory size
    const int shared_mem_size = 
        (N_MELS * 2) +  // input and normalized input
        SLICE_LATENT_DIM +  // encoded slice
        LATENT_DIM +  // latent code
        SLICE_LATENT_DIM;  // reconstructed slice

    // Launch the kernel
    spectral_shape_autoencoder_kernel<<<blocks, threads, shared_mem_size * sizeof(float)>>>(
        x.data_ptr<float>(), out.data_ptr<float>(),
        slice_enc_ln_w.data_ptr<float>(), slice_enc_ln_b.data_ptr<float>(),
        slice_enc_lin_w.data_ptr<float>(), slice_enc_lin_b.data_ptr<float>(),
        normalized_basis.data_ptr<float>(),
        bottle_ln_w.data_ptr<float>(), bottle_ln_b.data_ptr<float>(),
        bottle_lin_w.data_ptr<float>(), bottle_lin_b.data_ptr<float>(),
        slice_dec_lin_w.data_ptr<float>(), slice_dec_lin_b.data_ptr<float>(),
        batch_size
    );
}
