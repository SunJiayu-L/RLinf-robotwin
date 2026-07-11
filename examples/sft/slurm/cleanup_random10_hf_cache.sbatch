#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH -J clean-r10-hf
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/cleanup_random10_hf_cache-%j.out
#SBATCH --time=04:00:00

set -euo pipefail

TARGET=/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/random10_h16_9805726
EXPECTED=/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/random10_h16_9805726
RESOLVED=$(realpath -e -- "${TARGET}")

echo "target=${TARGET}"
echo "resolved=${RESOLVED}"
du -sh "${RESOLVED}"

if [[ "${RESOLVED}" != "${EXPECTED}" || ! -d "${RESOLVED}/hf_datasets" ]]; then
    echo "Refusing to delete unexpected cache path"
    exit 1
fi

rm -rf -- "${RESOLVED}"
echo "cache removed"
