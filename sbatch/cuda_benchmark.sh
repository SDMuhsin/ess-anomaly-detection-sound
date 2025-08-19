#!/bin/bash


sbatch \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --gpus=1 \
    --mem=8000M \
    --time=1-00:00:00 \
    --chdir=/scratch/sdmuhsin/ess-anomaly-detection-sound \
    --output=./logs/cuda-benchmark-%N-%j.out \
    --wrap="
	export TORCH_HOME=\"./cache\"
	module load python/3.11 gcc arrow cuda
	source ./env/bin/activate
	echo 'Environment loaded'
	which python3
	python3 src/cuda_large_benchmark.py
    "
