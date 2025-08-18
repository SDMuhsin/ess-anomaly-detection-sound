from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='spectral_shape_ae_cuda',
    ext_modules=[
        CUDAExtension(
            'spectral_shape_ae_cuda',
            [
                'spectral_shape_ae_cuda.cpp',
                'spectral_shape_ae_kernel.cu',
            ],
            # Add compiler flags for optimization and architecture targeting
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    # The -gencode flag specifies a "virtual" architecture (arch)
                    # and a "real" architecture (code). This creates optimized
                    # machine code (SASS) for the specified real architectures.
                    
                    # --- Target Architectures ---
                    
                    # Ampere Architecture
                    '-gencode=arch=compute_80,code=sm_80',  # For NVIDIA A100
                    '-gencode=arch=compute_86,code=sm_86',  # For NVIDIA A40, RTX 30-series

                    # Hopper Architecture
                    '-gencode=arch=compute_90,code=sm_90',  # For NVIDIA H100
                    
                    # --- Forward-Compatibility ---
                    
                    # This last flag generates PTX intermediate code for the newest
                    # virtual architecture. It allows forward-compatibility, enabling
                    # the driver to Just-In-Time (JIT) compile the kernel for
                    # future GPUs that are newer than Hopper.
                    '-gencode=arch=compute_90,code=compute_90'
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
