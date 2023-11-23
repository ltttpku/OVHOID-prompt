#!/bin/bash
#SBATCH --job-name=vision-language    # create a short name for your job
#SBATCH --partition=HGX
#SBATCH --account=research
#SBATCH --qos=lv0b
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G               # total memory (RAM) per node
#SBATCH --time=0-24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=8      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --output=/scratch/generalvision/ting/OVHOID-prompt/logs/out-%j.out      # output format
#SBATCH --error=/scratch/generalvision/ting/OVHOID-prompt/logs/out-%j.err      # error output file
### SBATCH --exclude=lambda-hyperplane[00]
### SBATCH --export=ALL

#--------------------task  part-------------------------
## clean env
module purge

## load environment need by this
#module load slurm/slurm/20.11.8

## load conda
source /etc/profile
module load slurm/BigAI/23.02.2
module load cuda11.7

cd /scratch/generalvision/ting/OVHOID-prompt
# source activate
# conda activate thid


## Training
/home/leiting/scratch/.conda/envs/thid/bin/python -m torch.distributed.launch --nproc_per_node=2 --master_port 3995 --use_env main.py \
    --batch_size 64 \
    --output_dir checkpoints/hico_bs2x64_lr1e-4_token25_ms5+8+11_cm5_AuxText_focal_alpha0.5_gamma0.5_UO_noninter \
    --epochs 100 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 25 \
    --enable_dec \
    --dataset_file hico --multi_scale true --f_idxs 5 8 11 --set_cost_hoi_type 5 --use_aux_text true \
    --enable_focal_loss --zero_shot_type unseen_object --description_file_path hico_hoi_descriptions.json --focal_alpha 0.5 --focal_gamma 0.5 \
    --ignore_non_interaction false

## Evaluation
