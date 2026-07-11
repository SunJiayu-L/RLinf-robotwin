#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH -J create-rw15
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/create_robotwin15-%j.out
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
        bash "${REPO_PATH}/examples/sft/slurm/create_robotwin15_dataset_and_norm_stats.sbatch"
fi

# === inside container from here ===

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}

# Top 15 tasks (by JAX SFT success rate) → episode ranges in robotwin_47_pi05
# Tasks in alphabetical order, 100 episodes each:
#   adjust_bottle(0):       ep 0-99      (54.5%)
#   beat_block_hammer(1):   ep 100-199   (73.5%)
#   click_alarmclock(4):    ep 400-499   (57.0%)
#   dump_bin_bigbin(6):     ep 600-699   (74.0%)
#   handover_mic(8):        ep 800-899   (53.0%)
#   open_microwave(15):     ep 1500-1599 (78.0%)
#   place_container_plate(25): ep 2500-2599 (78.0%)
#   place_empty_cup(27):    ep 2700-2799 (79.5%)
#   place_object_stand(32): ep 3200-3299 (57.0%)
#   place_shoe(34):         ep 3400-3499 (51.0%)
#   press_stapler(35):      ep 3500-3599 (76.5%)
#   put_bottles_dustbin(36): ep 3600-3699 (47.0%)
#   shake_bottle(39):       ep 3900-3999 (84.5%)
#   shake_bottle_horizontally(40): ep 4000-4099 (91.0%)
#   stack_bowls_two(44):    ep 4400-4499 (82.0%)

SRC_DATASET="/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_47_pi05"
DST_DATASET="/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_15_pi05"
EPISODE_RANGES="0-99,100-199,400-499,600-699,800-899,1500-1599,2500-2599,2700-2799,3200-3299,3400-3499,3500-3599,3600-3699,3900-3999,4000-4099,4400-4499"

NORM_STATS_DIR="/mnt/petrelfs/muyao/jiayusun/pi05_base/physical-intelligence/robotwin15"
CONFIG_NAME="pi05_aloha_robotwin15"

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
