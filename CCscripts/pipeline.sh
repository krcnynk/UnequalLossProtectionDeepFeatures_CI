#!/bin/bash

#module restore uneq
#source /project/6008756/foniks/Project_1/uneqENV/bin/activate
rm -r /project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/Korcan/Plots
source pipelineT.sh
source pipelineB.sh
for i in {1..10}
do
    source pipelineR.sh
    source pipelineRRS.sh
    source pipelineRRSF.sh
done

#python Korcan/pipeline.py