#!/bin/bash

git pull
rm -r /project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/Korcan/Plots
mkdir -p /project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/Korcan/Plots/resnet

sbatch pipelineT.sh
sbatch pipelineB.sh
sbatch pipelineTNS.sh
sbatch pipelineBNS.sh
# sbatch pipelineRSBUR.sh
# sbatch pipelineRSBURNS.sh

for i in {1..10}
do
    sbatch pipelineRSIID.sh
    sbatch pipelineRSIIDNS.sh
    sbatch pipelineRSIIDW.sh
    sbatch pipelineRSIIDNSW.sh
done

sbatch pipelineRIID.sh
sbatch pipelineRIIDNS.sh

# sbatch pipelineRBUR.sh
# sbatch pipelineRBURNS.sh

# sbatch pipelineRIID.sh
# for i in $(seq 0 10 100)
# do
#     # echo "$i"
#     sbatch --export=arg1=$i pipelineRIIDEN.sh
# done


# Hi my name is Korcan, I joined the group during my undergrad which is since 2021. I am doing error resilience in transmission of tensor data.
# Give score to packets based on channel similarity, played with the weight

# then similar to only first half because where the importance is

# then i copied first half to second half to see if that would work not much improvemnt, i am still playing with the parameters,
# not sure if this is ok basically sending twice

# sbatch pipelineNLEN.sh

for i in {1..1}
do
    # # sbatch pipelineRBUR.sh
    # sbatch pipelineRIID.sh

    # # sbatch --export=arg1=80 pipelineRIIDEN.sh

    # # sbatch pipelineRSBUR.sh
    # sbatch pipelineRSIID.sh

    # # sbatch pipelineRBURNS.sh
    # sbatch pipelineRIIDNS.sh
    
    # # sbatch pipelineRSBURNS.sh
    # sbatch pipelineRSIIDNS.sh
done