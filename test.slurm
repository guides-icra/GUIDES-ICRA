#!/bin/bash
#SBATCH --job-name=eef
#SBATCH --output=/weka/scratch/tshu2/dzhang98/log/eef_%j.out
#SBATCH --error=/weka/scratch/tshu2/dzhang98/log/eef_%j.err
#SBATCH --partition=a100
#SBATCH --exclude=c003
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=70:00:00
source ~/.bashrc
conda activate sivlm

nvidia-smi
export TORCH_USE_CUDA_DSA=1
cd /weka/scratch/tshu2/dzhang98/code/completion-infuse-robot
python robomimic/scripts/train.py --config robomimic/scripts/run_configs/experiment-proof-completion-infuse/align_dj.json

conda deactivate
