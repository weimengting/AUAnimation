#!/bin/bash
#SBATCH --job-name=au_stage1
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/train1.out
#SBATCH --error=logs/train1.err

module load scicomp-python-env
ACCELERATE_BIN=/scratch/work/weim3/conda/envs/considgen/bin/accelerate



$ACCELERATE_BIN launch --num_processes 2 train_s1.py 
