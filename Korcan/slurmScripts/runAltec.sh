#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --account=def-ibajic
#SBATCH --ntasks=150
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --mail-user=kuyanik@sfu.ca
#SBATCH --mail-type=ALL


search_dir=./altec_yml/efficientnetb0/block2b_add/
for entry in "$search_dir"*
do
    # printf "%s\n" "$entry"
    srun --exclusive -n 1 -N 1 python main.py -p "$entry" &
done
wait
