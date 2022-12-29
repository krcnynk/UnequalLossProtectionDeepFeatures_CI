#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --account=def-ibajic
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=30000M
#SBATCH --mail-user=kuyanik@sfu.ca
#SBATCH --mail-type=ALL
python create_loss_maps_yml.py
python create_altec_yml.py
python create_caltec_yml.py
python create_inpaint_yml.py
python create_tc_HaLRTC_yml.py
python create_tc_SiLRTC_yml.py
python create_linear_yml.py
#python create_quant_yml.py

search_dir=./lossMaps_yml/efficientnetb0/
for entry in "$search_dir"*
do
    #printf "%s\n" "$entry"
    srun --exclusive --ntasks 1 python main.py -p "$entry" &
done
wait
