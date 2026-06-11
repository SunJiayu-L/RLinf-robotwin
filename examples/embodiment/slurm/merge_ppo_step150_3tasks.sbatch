#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH -J rw-merge-3task
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/merge_ppo_step150_3tasks-%j.out
#SBATCH --time=0:30:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18

set -eo pipefail

SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}

export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin

if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec \
        --cleanenv \
        --bind /mnt:/mnt \
        --bind /nvme:/nvme \
        "${SIF_IMAGE}" \
        env IN_RLINF_ROBOTWIN_CONTAINER=1 \
            HOME="${HOME}" \
            SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
            REPO_PATH="${REPO_PATH}" \
        bash "${REPO_PATH}/examples/embodiment/slurm/merge_ppo_step150_3tasks.sbatch"
fi

export PATH=$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}

BASE_CKPT="/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260609-10:17:50-robotwin_sft_pi05_8gpu/robotwin_sft_openpi_pi05/checkpoints/global_step_20000/actor/model_state_dict/full_weights.pt"

ADJ_CKPT="/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260610-10:56:32-robotwin_adjust_bottle_ppo_openpi_pi05/robotwin_ppo_openpi_pi05/checkpoints/global_step_150/actor/model_state_dict/full_weights.pt"

PCP_CKPT="/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260610-18:33:28-robotwin_place_container_plate_ppo_openpi_pi05/robotwin_ppo_openpi_pi05_place_container_plate/checkpoints/global_step_150/actor/model_state_dict/full_weights.pt"

PEC_CKPT="/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260610-22:58:11-robotwin_place_empty_cup_ppo_openpi_pi05/robotwin_ppo_openpi_pi05_place_empty_cup/checkpoints/global_step_150/actor/model_state_dict/full_weights.pt"

OUTPUT="/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/merged/sft20k_ppo150_adj_pcp_pec_a035/full_weights.pt"

python "${REPO_PATH}/scripts/task_arithmetic_rlinf.py" \
    --base   "${BASE_CKPT}" \
    --ft     "${ADJ_CKPT}" "${PCP_CKPT}" "${PEC_CKPT}" \
    --output "${OUTPUT}" \
    --alpha  0.35

echo "Merged checkpoint saved to: ${OUTPUT}"
