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
python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port 12234 main.py \
    --batch_size 64 \
    --output_dir checkpoints/swig_bs2x64_lr1e-4_token20_prompt_hint \
    --epochs 100 \
    --lr 1e-4 --min-lr 1e-7 \
    --hoi_token_length 20 \
    --enable_dec \
    --dataset_file swig --use_prompt_hint true

## Inference
# export MASTER_ADDR=localhost
# export MASTER_PORT=5679

# python main.py --eval \
#     --batch_size 8 \
#     --output_dir outputs/ \
#     --hoi_token_length 10 \
#     --enable_dec \
#     --pretrained checkpoints/swig_bs8x2_lr1e-4_token10/checkpoint0089.pth \
#     --eval_size 256 \
#     --test_score_thresh 1e-4 \
#     --dataset_file swig

# python main.py --eval \
#     --batch_size 8 \
#     --output_dir outputs/ \
#     --hoi_token_length 10 \
#     --enable_dec \
#     --pretrained checkpoints/swig_bs8x2_lr1e-4_token10/checkpoint0079.pth \
#     --eval_size 256 \
#     --test_score_thresh 1e-4 \
#     --dataset_file swig

# python main.py --eval \
#     --batch_size 8 \
#     --output_dir outputs/ \
#     --hoi_token_length 10 \
#     --enable_dec \
#     --pretrained checkpoints/swig_bs8x2_lr1e-4_token10/checkpoint0069.pth \
#     --eval_size 256 \
#     --test_score_thresh 1e-4 \
#     --dataset_file swig