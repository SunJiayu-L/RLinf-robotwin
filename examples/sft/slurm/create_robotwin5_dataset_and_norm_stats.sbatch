#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH -J create-rw5
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/create_robotwin5-%j.out
#SBATCH --time=36:00:00

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
            REPO_PATH="${REPO_PATH}" \
        bash "${REPO_PATH}/examples/sft/slurm/create_robotwin5_dataset_and_norm_stats.sbatch"
fi

# === inside container from here ===

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}

# First 5 tasks from the robotwin15 top-15 selection (alphabetical order):
#   adjust_bottle(0):       ep 0-99
#   beat_block_hammer(1):   ep 100-199
#   click_alarmclock(4):    ep 400-499
#   dump_bin_bigbin(6):     ep 600-699
#   handover_mic(8):        ep 800-899

SRC_DATASET="/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_47_pi05"
DST_DATASET="/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_5_pi05"
EPISODE_RANGES="0-99,100-199,400-499,600-699,800-899"

NORM_STATS_DIR="/mnt/petrelfs/muyao/jiayusun/pi05_base/physical-intelligence/robotwin5"
CONFIG_NAME="pi05_aloha_robotwin5"

cd "${REPO_PATH}"

# ---- Step 1: Filter dataset ----
echo "=== [step1] filter dataset: $(date) ==="
if [[ -d "${DST_DATASET}" ]]; then
    echo "  ${DST_DATASET} already exists, skipping filter step"
else
    python toolkits/lerobot/filter_lerobot_dataset.py \
        --source-dir "${SRC_DATASET}" \
        --output-dir "${DST_DATASET}" \
        --episodes "${EPISODE_RANGES}"
fi
echo "=== [step1] done: $(date) ==="

# ---- Step 2: Calculate norm stats ----
echo "=== [step2] calc norm stats: $(date) ==="
mkdir -p "${NORM_STATS_DIR}"
python toolkits/lerobot/calculate_norm_stats.py \
    --config-name "${CONFIG_NAME}" \
    --repo-id "${DST_DATASET}" \
    --output-path "${NORM_STATS_DIR}"
echo "=== [step2] done: $(date) ==="
echo "norm_stats saved to: ${NORM_STATS_DIR}/norm_stats.json"
