#!/bin/bash

#module restore uneq
#source /project/6008756/foniks/Project_1/uneqENV/bin/activate
git pull
rm -r /project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/Korcan/Plots

sbatch pipelineT.sh
sbatch pipelineB.sh
for i in {1..2}
do
    sbatch pipelineRBUR.sh
    sbatch pipelineRIID.sh


    sbatch pipelineRBURNS.sh
    sbatch pipelineRIIDNS.sh
    sbatch pipelineRIIDEN.sh


    sbatch pipelineRSBUR.sh
    sbatch pipelineRSIID.sh

    sbatch pipelineRSBURNS.sh
    sbatch pipelineRSIIDNS.sh
done