#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH -J submit-rgb5k
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/submit_corrected_rgb5k_eval-%j.out
#SBATCH --time=00:10:00

set -euo pipefail
cd /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
sbatch examples/embodiment/slurm/blocks_ranking_rgb_h16_sft5k_eval_8gpu.sbatch
