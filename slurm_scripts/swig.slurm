#!/bin/bash
#SBATCH --job-name=vision-language    # create a short name for your job
#SBATCH --partition=gpu
#SBATCH --account=research
#SBATCH --qos=level0
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100G               # total memory (RAM) per node
#SBATCH --time=0-10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --cpus-per-task=8      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --output=/scratch/generalvision/ting/OVHOID-prompt/logs/out-%j.out      # output format
#SBATCH --error=/scratch/generalvision/ting/OVHOID-prompt/logs/out-%j.err      # error output file
#SBATCH --exclude=lambda-hyperplane[00]
#SBATCH --export=ALL

#--------------------task  part-------------------------
## clean env
module purge

## load environment need by this
#module load slurm/slurm/20.11.8

## load conda
module load anaconda3/2021.11

cd /scratch/generalvision/ting/OVHOID-prompt
source activate
conda activate thid


## Training
python -m torch.distributed.launch --nproc_per_node=2 --master_port 2113 --use_env main.py \
    --batch_size 32 \
    --output_dir checkpoints/swig_bs2x32_lr1e-4_token5_ms \
    --epochs 125 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 5 \
    --enable_dec \
    --dataset_file swig --multi_scale true


## Inference
# export MASTER_ADDR=localhost
# export MASTER_PORT=5679

# python -m torch.distributed.launch --nproc_per_node=1 --master_port 2113 --use_env main.py \
#     --batch_size 8 \
#     --output_dir checkpoints/swig_bs2x64_lr1e-4_token5_ms \
#     --epochs 150 \
#     --lr 1e-4 --min-lr 1e-7 \
#     --hoi_token_length 5 \
#     --enable_dec \
#     --dataset_file swig --multi_scale true --eval --test_score_thresh 1e-4 \
#     --pretrained checkpoints/swig_bs2x64_lr1e-4_token5_ms/checkpoint.pth
