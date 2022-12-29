#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --account=def-ibajic
#SBATCH --gpus-per-node=1
#SBATCH --mem=6000M
#SBATCH --mail-user=kuyanik@sfu.ca
#SBATCH --mail-type=ALL

search_dir=./halrtc_yml/efficientnetb0/block2b_add/
for entry in "$search_dir"*
do
   srun --ntasks=1 python main.py -p "$entry" &
done

wait
