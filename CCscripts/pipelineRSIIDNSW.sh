#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=1-11
#SBATCH --mem=50G
#SBATCH --account=def-ibajic

module restore uneq
# source /project/6008756/foniks/Project_1/uneqENV/bin/activate
# mkdir -p /project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/Korcan/Plots/resnet/Random_RSCorrected_20_80
cd ..

# python Korcan/pipeline.py $(($(($SLURM_ARRAY_TASK_ID-1))+40)) 11 40 60
# python Korcan/pipeline.py $(($(($SLURM_ARRAY_TASK_ID-1))*10)) 11 40 60
python Korcan/pipeline.py $(($(($SLURM_ARRAY_TASK_ID-1))*10)) 20 50 50
python Korcan/pipeline.py $(($(($SLURM_ARRAY_TASK_ID-1))+50)) 20 50 50
