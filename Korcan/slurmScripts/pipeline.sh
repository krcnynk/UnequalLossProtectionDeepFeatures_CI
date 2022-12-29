#!/bin/bash
#SBATCH --account=def-ibajic
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G               # memory per node
#SBATCH --time=00:05:00

module load python/3.10
module load scipy-stack
source ../pipelineStack/bin/activate &&
#nvidia-smi
python Korcan/pipeline.py
