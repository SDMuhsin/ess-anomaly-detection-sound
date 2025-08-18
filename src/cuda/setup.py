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
            ]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
