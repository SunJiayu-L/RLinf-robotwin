#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH -J pi05-hb1to1-sft
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/finetune_moe_demo_clean_hb1to1_pi05-%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-20,SH-IDCA1404-10-140-54-21,SH-IDCA1404-10-140-54-25,SH-IDCA1404-10-140-54-28,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-35,SH-IDCA1404-10-140-54-37,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-45,SH-IDCA1404-10-140-54-57,SH-IDCA1404-10-140-54-61,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-73,SH-IDCA1404-10-140-54-79,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-118

set -euo pipefail

REPO_DIR=/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05
PYTHON=${REPO_DIR}/.venv/bin/python
CONFIG_NAME=${CONFIG_NAME:-pi05_aloha_moe_demo_clean_hb1to1}
EXP_NAME=${EXP_NAME:-robotwin_moe_demo_clean_hb1to1}
REPO_ID=${REPO_ID:-robotwin_moe_demo_clean_hb1to1}
DATASET_DIR=/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/${REPO_ID}
CACHE_DIR=/mnt/petrelfs/muyao/jiayusun/store/cache
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}

if [[ "${IN_PI05_CONTAINER:-0}" != "1" ]]; then
    if [[ ! -f "${SIF_IMAGE}" ]]; then
        echo "[sbatch] ERROR: Apptainer image not found: ${SIF_IMAGE}" >&2
        exit 1
    fi
    echo "[sbatch] entering Apptainer image: ${SIF_IMAGE}"
    exec "${APPTAINER_BIN}" exec --nv --bind /mnt/petrelfs:/mnt/petrelfs --pwd "${REPO_DIR}" "${SIF_IMAGE}" \
        env IN_PI05_CONTAINER=1 \
            HOME=/mnt/petrelfs/muyao/jiayusun \
            SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
            SLURM_NODELIST="${SLURM_NODELIST:-}" \
        bash "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/examples/embodiment/slurm/finetune_moe_demo_clean_hb1to1_pi05.sbatch"
fi

cd "${REPO_DIR}"

export HOME=/mnt/petrelfs/muyao/jiayusun
export XDG_CACHE_HOME=${CACHE_DIR}
export HF_HOME=${CACHE_DIR}/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export WANDB_MODE=${WANDB_MODE:-offline}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-1}

TRAIN_FSDP_DEVICES=${TRAIN_FSDP_DEVICES:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-8}

printf "[sbatch] host=%s\n" "$(hostname)"
printf "[sbatch] job_id=%s\n" "${SLURM_JOB_ID:-none}"
printf "[sbatch] cuda_visible_devices=%s\n" "${CUDA_VISIBLE_DEVICES:-unset}"
printf "[sbatch] config=%s exp=%s repo_id=%s\n" "${CONFIG_NAME}" "${EXP_NAME}" "${REPO_ID}"
printf "[sbatch] dataset=%s\n" "${DATASET_DIR}"

nvidia-smi || true

if [[ ! -x "${PYTHON}" ]]; then
    echo "[sbatch] ERROR: python not found: ${PYTHON}" >&2
    exit 1
fi

if [[ ! -f "${DATASET_DIR}/meta/info.json" ]]; then
    echo "[sbatch] ERROR: LeRobot dataset not found: ${DATASET_DIR}" >&2
    echo "[sbatch] Run convert_moe_demo_clean_hb1to1_to_lerobot.sbatch first." >&2
    exit 1
fi

"${PYTHON}" - <<PY
import json
from pathlib import Path
info = json.loads(Path("${DATASET_DIR}/meta/info.json").read_text())
print("[dataset] total_episodes=", info.get("total_episodes"))
print("[dataset] total_frames=", info.get("total_frames"))
PY

# Compute norm stats if not present (reuse existing robotwin_moe_demo_clean stats as asset_id)
NORM_DIR=${REPO_DIR}/assets/${CONFIG_NAME}/${REPO_ID}
if [[ -d "${NORM_DIR}" ]]; then
    echo "[norm] Found existing norm stats: ${NORM_DIR}"
else
    echo "[norm] Computing norm stats for ${CONFIG_NAME}"
    "${PYTHON}" scripts/compute_norm_stats.py --config-name "${CONFIG_NAME}"
fi

CKPT_DIR=${REPO_DIR}/checkpoints/${CONFIG_NAME}/${EXP_NAME}
TRAIN_FLAGS=("${CONFIG_NAME}" "--exp-name=${EXP_NAME}" "--fsdp-devices=${TRAIN_FSDP_DEVICES}" "--batch-size=${TRAIN_BATCH_SIZE}" "--num-workers=${TRAIN_NUM_WORKERS}")
if [[ -d "${CKPT_DIR}" ]]; then
    echo "[train] Existing checkpoint dir found, resuming: ${CKPT_DIR}"
    TRAIN_FLAGS+=("--resume")
else
    echo "[train] Starting new run: ${CKPT_DIR}"
    TRAIN_FLAGS+=("--overwrite")
fi

"${PYTHON}" scripts/train.py "${TRAIN_FLAGS[@]}"
