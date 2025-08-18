#include <torch/extension.h>
#include <vector>

// Forward declaration of the CUDA kernel launcher function
void spectral_shape_autoencoder_forward_cuda(
    const at::Tensor& x, at::Tensor& out,
    const at::Tensor& slice_enc_ln_w, const at::Tensor& slice_enc_ln_b,
    const at::Tensor& slice_enc_lin_w, const at::Tensor& slice_enc_lin_b,
    const at::Tensor& normalized_basis,
    const at::Tensor& bottle_ln_w, const at::Tensor& bottle_ln_b,
    const at::Tensor& bottle_lin_w, const at::Tensor& bottle_lin_b,
    const at::Tensor& slice_dec_lin_w, const at::Tensor& slice_dec_lin_b
);

// Python-exposed forward function
at::Tensor spectral_shape_autoencoder_forward(
    const at::Tensor& x,
    const at::Tensor& slice_enc_ln_w, const at::Tensor& slice_enc_ln_b,
    const at::Tensor& slice_enc_lin_w, const at::Tensor& slice_enc_lin_b,
    const at::Tensor& normalized_basis,
    const at::Tensor& bottle_ln_w, const at::Tensor& bottle_ln_b,
    const at::Tensor& bottle_lin_w, const at::Tensor& bottle_lin_b,
    const at::Tensor& slice_dec_lin_w, const at::Tensor& slice_dec_lin_b
) {
    // Perform input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor 'x' must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input tensor 'x' must be contiguous");
    
    // Create an output tensor with the same shape and type as the input
    auto out = at::empty_like(x);

    // Call the CUDA kernel launcher function
    spectral_shape_autoencoder_forward_cuda(
        x, out,
        slice_enc_ln_w, slice_enc_ln_b, 
        slice_enc_lin_w, slice_enc_lin_b,
        normalized_basis,
        bottle_ln_w, bottle_ln_b, 
        bottle_lin_w, bottle_lin_b,
        slice_dec_lin_w, slice_dec_lin_b
    );

    return out;
}

// PYBIND11 module definition to expose the forward function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &spectral_shape_autoencoder_forward, 
          "Spectral Shape Autoencoder forward pass (CUDA)");
}
