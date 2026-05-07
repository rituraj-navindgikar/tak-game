#!/bin/bash
#SBATCH --job-name=tak_alphazero
#SBATCH --partition=short
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=tak_train_%j.log
#SBATCH --error=tak_train_%j.err

set -euo pipefail

cd /home/navindgikar.r/cs5180_final_project

mkdir -p logs tak_checkpoints

echo "==== Job started at $(date) ===="
echo "Node: $(hostname)"

PYTHON=/home/navindgikar.r/.conda/envs/takrl/bin/python

echo "Python: $PYTHON"
$PYTHON --version


$PYTHON -u train_alpha_zero_rl_model.py | tee logs/train_${SLURM_JOB_ID}.log

echo "==== Job finished at $(date) ===="
