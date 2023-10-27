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
/home/leiting/scratch/.conda/envs/thid/bin/python -m torch.distributed.launch --nproc_per_node=2 --master_port 2992 --use_env main.py \
    --batch_size 64 \
    --output_dir checkpoints/swig_bs2x64_lr1e-4_token10_ms5+8+11_cm5_su100_fmdrop7e-1 \
    --epochs 125 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 10 \
    --enable_dec \
    --dataset_file swig --multi_scale true --f_idxs 5 8 11 --set_cost_hoi_type 5 \
    --semantic_query true --semantic_units_file semantic_units_files/su_word_100.pkl --feature_map_dropout_weight 0.7


# ## Inference
# export MASTER_ADDR=localhost
# export MASTER_PORT=5679

# /home/leiting/scratch/.conda/envs/thid/bin/python -m torch.distributed.launch --nproc_per_node=1 --master_port 2990 --use_env main.py \
#     --batch_size 64 \
#     --output_dir checkpoints/swig_bs1x64_lr1e-4_token10_ms5+8+11 \
#     --epochs 125 \
#     --lr 1e-4 --min-lr 1e-7 \
#     --hoi_token_length 10 \
#     --enable_dec \
#     --dataset_file swig --multi_scale true --f_idxs 5 8 11 --eval --test_score_thresh 1e-4 \
#     --pretrained checkpoints/swig_bs1x64_lr1e-4_token10_ms5+8+11/checkpoint0100.pth
