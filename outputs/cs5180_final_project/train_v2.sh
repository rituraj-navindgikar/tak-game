#!/bin/bash
#SBATCH --job-name=tak_az_v2
#SBATCH --partition=short
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=tak_train_v2_%j.log
#SBATCH --error=tak_train_v2_%j.err

set -euo pipefail

cd /home/navindgikar.r/cs5180_final_project

mkdir -p logs tak_checkpoints_v2

echo "==== Job started at $(date) ===="
echo "Node: $(hostname)"

PYTHON=/home/navindgikar.r/.conda/envs/takrl/bin/python

echo "Python: $PYTHON"
$PYTHON --version


$PYTHON -u train_alpha_zero_rl_model_v2.py | tee logs/train_v2_${SLURM_JOB_ID}.log

echo "==== Job finished at $(date) ===="
