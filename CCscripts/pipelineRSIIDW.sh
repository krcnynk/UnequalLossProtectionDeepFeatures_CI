#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --array=1-11
#SBATCH --mem=50G
#SBATCH --account=def-ibajic

module restore uneq
# source /project/6008756/foniks/Project_1/uneqENV/bin/activate
# mkdir -p /project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/Korcan/Plots/resnet/Random_RSCorrected_FECRemovesBOT_20_80
cd ..

mkdir -p ./Korcan/Plots/resnet/FEC\ \(IID\)\ Weighted/
# python Korcan/pipeline.py $(($(($SLURM_ARRAY_TASK_ID-1))+40)) 5 40 60

python Korcan/pipeline.py $(($(($SLURM_ARRAY_TASK_ID-1))*10)) 19 30 70
python Korcan/pipeline.py $(($(($SLURM_ARRAY_TASK_ID-1))+30)) 19 30 70