#!/bin/bash

git pull
rm -r /project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/Korcan/Plots
mkdir -p /project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/Korcan/Plots/resnet

# sbatch pipelineT.sh
# sbatch pipelineB.sh

# sbatch pipelineRIID.sh
# for i in $(seq 0 10 100)
# do
#     # echo "$i"
#     sbatch --export=arg1=$i pipelineRIIDEN.sh
# done

for i in {0..100..5}
do
    # sbatch --export=arg1=$i pipelineRIIDEN.sh
    sbatch --export=arg1=$i pipelineNLEN.sh
done

for i in {1..1}
do
    # sbatch pipelineRBUR.sh
    # sbatch pipelineRIID.sh

    # sbatch --export=arg1=80 pipelineRIIDEN.sh

    # sbatch pipelineRSBUR.sh
    # sbatch pipelineRSIID.sh

    # sbatch pipelineRBURNS.sh
    # sbatch pipelineRIIDNS.sh
    # sbatch pipelineRSBURNS.sh
    # sbatch pipelineRSIIDNS.sh
done