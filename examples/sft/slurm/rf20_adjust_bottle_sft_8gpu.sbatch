#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J rf20-adj-sft
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/rf20_adjust_bottle_sft_8gpu-%j.out
#SBATCH --time=48:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-117,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-58,SH-IDCA1404-10-140-54-12

set -eo pipefail

SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}

export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin

if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec \
        --cleanenv \
        --nv \
        --bind /mnt:/mnt \
        --bind /nvme:/nvme \
        --pwd "${REPO_PATH}" \
        "${SIF_IMAGE}" \
        env IN_RLINF_ROBOTWIN_CONTAINER=1 \
            HOME="${HOME}" \
            SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
            SLURM_NODELIST="${SLURM_NODELIST:-}" \
            CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
            REPO_PATH="${REPO_PATH}" \
        bash "${REPO_PATH}/examples/sft/slurm/rf20_adjust_bottle_sft_8gpu.sbatch"
fi

# === inside container from here ===

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"

SRC_DATASET="/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_random_47_500_pi05"
DST_DATASET="/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_adjust_bottle_500_pi05"

cd "${REPO_PATH}"

# ---- Step 1: Filter adjust_bottle episodes (0-499) ----
echo "=== [step1] filter adjust_bottle dataset: $(date) ==="
if [[ -d "${DST_DATASET}" ]]; then
    echo "  ${DST_DATASET} already exists, skipping"
else
    python toolkits/lerobot/filter_lerobot_dataset.py \
        --source-dir "${SRC_DATASET}" \
        --output-dir "${DST_DATASET}" \
        --episodes "0-499"
fi
echo "=== [step1] done: $(date) ==="

# ---- Step 2: SFT training ----
echo "=== [step2] SFT training: $(date) ==="
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-rf20_adjust_bottle_sft_8gpu"
mkdir -p "${LOG_DIR}"

ray stop --force 2>/dev/null || true
ray start --head --num-cpus=64 --num-gpus=8

python examples/sft/train_vla_sft.py \
    --config-path "${REPO_PATH}/examples/sft/config" \
    --config-name rf20_adjust_bottle_sft_pi05 \
    runner.logger.log_path="${LOG_DIR}" 2>&1 | tee "${LOG_DIR}/sft.log"

ray stop
echo "=== [step2] done: $(date) ==="
