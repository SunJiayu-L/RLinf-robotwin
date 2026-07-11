#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -J moe-sft20k-pt
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/convert_moe_sft20k_to_pytorch-%j.out
#SBATCH --time=04:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-117,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-58,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-20,SH-IDCA1404-10-140-54-28,SH-IDCA1404-10-140-54-61,SH-IDCA1404-10-140-54-57,SH-IDCA1404-10-140-54-21,SH-IDCA1404-10-140-54-37

set -euo pipefail

REPO_DIR=/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05
CONVERT_SCRIPT=/mnt/petrelfs/muyao/jiayusun/openpi/examples/convert_jax_model_to_pytorch.py
CONFIG_NAME=pi05_aloha_moe_demo_clean
CHECKPOINT_DIR=${REPO_DIR}/checkpoints/${CONFIG_NAME}/robotwin_moe_demo_clean_full/20000
OUTPUT_PATH=${REPO_DIR}/checkpoints/${CONFIG_NAME}/robotwin_moe_demo_clean_full_pytorch/20000
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}

if [[ "${IN_PI05_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec \
        --nv \
        --bind /mnt/petrelfs:/mnt/petrelfs \
        --pwd "${REPO_DIR}" \
        "${SIF_IMAGE}" \
        env IN_PI05_CONTAINER=1 \
        bash "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/examples/embodiment/slurm/convert_moe_sft20k_to_pytorch.sbatch"
fi

cd "${REPO_DIR}"
export HOME=/mnt/petrelfs/muyao/jiayusun
export XDG_CACHE_HOME=/mnt/petrelfs/muyao/jiayusun/store/cache
export HF_HOME=${XDG_CACHE_HOME}/huggingface
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

PYTHON=/mnt/petrelfs/muyao/jiayusun/openpi/.venv/bin/python
PYTHON_VERSION=$("${PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
TRANSFORMERS_SRC=${REPO_DIR}/src/openpi/models_pytorch/transformers_replace
TRANSFORMERS_DST=/mnt/petrelfs/muyao/jiayusun/openpi/.venv/lib/python${PYTHON_VERSION}/site-packages/transformers

if [[ ! -d "${CHECKPOINT_DIR}/params" ]]; then
    echo "Missing JAX checkpoint: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

cp -Lr "${TRANSFORMERS_SRC}/"* "${TRANSFORMERS_DST}/"
mkdir -p "${OUTPUT_PATH}"

"${PYTHON}" -u "${CONVERT_SCRIPT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --config-name "${CONFIG_NAME}" \
    --output-path "${OUTPUT_PATH}" \
    --precision bfloat16

if [[ -d "${CHECKPOINT_DIR}/assets" ]]; then
    rm -rf "${OUTPUT_PATH}/assets"
    cp -a "${CHECKPOINT_DIR}/assets" "${OUTPUT_PATH}/assets"
fi

# create norm_stats symlink
ln -sfn "${OUTPUT_PATH}/assets/robotwin_moe_demo_clean" "${OUTPUT_PATH}/robotwin_moe_demo_clean"

echo "[convert] done: ${OUTPUT_PATH}"
