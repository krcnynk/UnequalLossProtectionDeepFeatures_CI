#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --array=1-11
#SBATCH --mem=50G
#SBATCH --account=def-ibajic

module restore uneq
# source /project/6008756/foniks/Project_1/uneqENV/bin/activate
# mkdir -p /project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/Korcan/Plots/resnet/Bot
cd ..

mkdir -p ./Korcan/Plots/resnet/Least\ important/

python Korcan/pipeline.py $(($(($SLURM_ARRAY_TASK_ID-1))*10)) 2 0 0
# python Korcan/pipeline.py $(($(($SLURM_ARRAY_TASK_ID-1))*10)) 16 0 0