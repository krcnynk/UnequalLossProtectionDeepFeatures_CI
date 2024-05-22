#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --account=def-ibajic
#SBATCH --mem=1G

python downloadModels.py
