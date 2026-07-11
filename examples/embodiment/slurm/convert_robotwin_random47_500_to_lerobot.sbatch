#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -J rw47-convert
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/convert_robotwin_random47_500_to_lerobot-%j.out
#SBATCH --time=2-00:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-69

set -euo pipefail

ROBOWIN_PI05_DIR=/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05
PYTHON=${ROBOWIN_PI05_DIR}/.venv/bin/python
SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}

RAW_DIR=/mnt/petrelfs/muyao/jiayusun/robotwin-random-47-500
REPO_ID=robotwin_random_47_500_pi05
CACHE_DIR=/mnt/petrelfs/muyao/jiayusun/store/cache

if [[ "${IN_PI05_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec \
        --nv \
        --bind /mnt/petrelfs:/mnt/petrelfs \
        --pwd "${ROBOWIN_PI05_DIR}" \
        "${SIF_IMAGE}" \
        env IN_PI05_CONTAINER=1 \
            HOME=/mnt/petrelfs/muyao/jiayusun \
            SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
            SLURM_NODELIST="${SLURM_NODELIST:-}" \
        bash "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/examples/embodiment/slurm/convert_robotwin_random47_500_to_lerobot.sbatch"
fi

# === inside container from here ===

export HOME=/mnt/petrelfs/muyao/jiayusun
export XDG_CACHE_HOME=${CACHE_DIR}
export HF_HOME=${CACHE_DIR}/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets

cd "${ROBOWIN_PI05_DIR}"

printf "[convert] host=%s\n" "$(hostname)"
printf "[convert] job_id=%s\n" "${SLURM_JOB_ID:-none}"
printf "[convert] raw_dir=%s\n" "${RAW_DIR}"
printf "[convert] repo_id=%s\n" "${REPO_ID}"
printf "[convert] output=%s\n" "${HF_HOME}/lerobot/${REPO_ID}"

"${PYTHON}" examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
    --raw_dir "${RAW_DIR}" \
    --repo_id "${REPO_ID}"

printf "[convert] done: %s/lerobot/%s\n" "${HF_HOME}" "${REPO_ID}"
