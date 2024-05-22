#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --mem=20G
#SBATCH --account=def-ibajic

module restore uneq
# source /project/6008756/foniks/Project_1/uneqENV/bin/activate
cd ..
python Korcan/pipeline.py 0 makeplot 0 0