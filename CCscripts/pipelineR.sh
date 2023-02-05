#!/bin/bash
#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --array=1-11
#SBATCH --mem=50G
#SBATCH --account=def-ibajic

module restore uneq
source /project/6008756/foniks/Project_1/uneqENV/bin/activate
mkdir -p /project/6008756/foniks/Project_1/UnequalLossProtectionDeepFeatures_CI/Korcan/Plots/resnet/Random
cd ..

python Korcan/pipeline.py $(($(($i-1))*10)) Random 0 0