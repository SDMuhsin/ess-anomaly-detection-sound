#!/bin/bash

# Activate virtual environment
source ../../env/bin/activate

# Set CUDA device to use
export CUDA_VISIBLE_DEVICES=1

# Build the CUDA extension
echo "Building CUDA extension..."
python setup.py install

# Run the test script
echo "Running CUDA kernel test..."
python test_cuda_kernel.py

# Deactivate virtual environment
deactivate
