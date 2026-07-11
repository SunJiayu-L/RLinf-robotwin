#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=80G
#SBATCH -J merge-a07
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/merge_ppo_adj200_plus_ca_tv_a07-%j.out
#SBATCH --time=1:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-117,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-58,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-7,SH-IDCA1404-10-140-54-46,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-36,SH-IDCA1404-10-140-54-59,SH-IDCA1404-10-140-54-50,SH-IDCA1404-10-140-54-25

set -eo pipefail

SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin

if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec \
        --cleanenv --nv \
        --bind /mnt:/mnt --bind /nvme:/nvme \
        --pwd "${REPO_PATH}" "${SIF_IMAGE}" \
        env IN_RLINF_ROBOTWIN_CONTAINER=1 \
            HOME="${HOME}" \
            SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
            REPO_PATH="${REPO_PATH}" \
        bash "${REPO_PATH}/examples/embodiment/slurm/merge_ppo_adj200_plus_ca_tv_a07.sbatch"
fi

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export PATH=$GCC_HOME/bin:$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}

cd "${REPO_PATH}"
python scripts/merge_ppo_adj200_plus_ca_tv_a07.py
