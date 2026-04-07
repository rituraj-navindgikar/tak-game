#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --output=finetune_%j.log

python train_alpha_zero_rl_model.py