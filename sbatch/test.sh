#!/bin/sh

sbatch \
    --nodes=1 \
    --gpus-per-node=1 \
    --mem=1G \
    --time=00:05:00 \
    --output=./logs/env-test-%j.out \
    --chdir=/scratch/sdmuhsin/ess-anomaly-detection-sound \
    --wrap="
        module load gcc arrow cuda/11.8
	nvidia-smi
        source /scratch/sdmuhsin/ess-anomaly-detection-sound/env/bin/activate
        python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"Device name: {torch.cuda.get_device_name(0)}\")'
    "
