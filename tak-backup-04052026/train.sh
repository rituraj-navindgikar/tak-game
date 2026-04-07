#!/bin/bash
#SBATCH --job-name=tak-az
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=tak_az_%j.out
#SBATCH --error=tak_az_%j.err

set -euo pipefail

cd /home/navindgikar.r/cs5180_final_project

mkdir -p logs tak_checkpoints

echo "==== Job started at $(date) ===="
echo "Node: $(hostname)"

PYTHON=/home/navindgikar.r/.conda/envs/takrl/bin/python

echo "Python: $PYTHON"
$PYTHON --version

echo "GPU before run:"
nvidia-smi

$PYTHON -u train_alpha_zero_rl_model.py | tee logs/train_${SLURM_JOB_ID}.log

echo "==== Job finished at $(date) ===="
