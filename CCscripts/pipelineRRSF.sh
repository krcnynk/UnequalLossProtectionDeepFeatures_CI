#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --array=1-11
#SBATCH --mem=50G
#SBATCH --account=def-ibajic

module restore uneq
source /project/6008756/foniks/Project_1/uneqENV/bin/activate
mkdir -p /project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/Korcan/Plots/resnet/Random_RSCorrected_FECRemovesBOT_70_30
cd ..

python Korcan/pipeline.py $(($(($i-1))*10)) Random_RSCorrected_FECRemovesBOT 70 30