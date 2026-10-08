#!/bin/bash
#SBATCH --job-name=au_stage2
#SBATCH --partition=gpu-h200-141g-ellis    
#SBATCH --gres=gpu:h200:1
#SBATCH --account=ellis_users
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/train2.out
#SBATCH --error=logs/train2.err

module load scicomp-python-env
ACCELERATE_BIN=/scratch/work/weim3/conda/envs/considgen/bin/accelerate



$ACCELERATE_BIN launch --num_processes 2 train_s2.py 
