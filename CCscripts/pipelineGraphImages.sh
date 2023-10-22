#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --mem=50G
#SBATCH --account=def-ibajic

module restore uneq
# source /project/6008756/foniks/Project_1/uneqENV/bin/activate
# mkdir -p /project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/Korcan/Plots/resnet/Random_RSCorrected_20_80
cd ..

python Korcan/pipeline.py 0 10 0 0