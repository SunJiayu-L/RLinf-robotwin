#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH -J rgb-frames
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/extract_blocks_rgb_training_frames-%j.out
#SBATCH --time=01:00:00

set -euo pipefail

cd /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
./.venv/bin/python scripts/extract_blocks_rgb_training_frames.py
