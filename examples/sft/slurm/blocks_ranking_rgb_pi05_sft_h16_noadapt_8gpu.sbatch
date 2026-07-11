#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J rgb-h16-sft
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/blocks_ranking_rgb_pi05_sft_h16_noadapt_8gpu-%j.out
#SBATCH --time=48:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-1,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-117,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-58,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-79

set -eo pipefail

SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin

if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec --cleanenv --nv --bind /mnt:/mnt --bind /nvme:/nvme --pwd "${REPO_PATH}" "${SIF_IMAGE}" env IN_RLINF_ROBOTWIN_CONTAINER=1 HOME="${HOME}" SLURM_JOB_ID="${SLURM_JOB_ID:-}" SLURM_NODELIST="${SLURM_NODELIST:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" REPO_PATH="${REPO_PATH}" bash "${REPO_PATH}/examples/sft/slurm/blocks_ranking_rgb_pi05_sft_h16_noadapt_8gpu.sbatch"
fi

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"

export SRC_DATASET="/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_random_47_500_pi05"
export DST_DATASET="/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_blocks_ranking_rgb_500_pi05_h16"
export CONFIG_NAME="pi05_aloha_robotwin_blocks_ranking_rgb_h16"
export BASE_CKPT="/mnt/petrelfs/muyao/jiayusun/pi05_base_adapt_false"
export NORM_STATS_DIR="${BASE_CKPT}/physical-intelligence/robotwin_blocks_ranking_rgb_h16"
export RLINF_JOB_SCRATCH="/nvme/muyao/rlinf_rgb_h16_${SLURM_JOB_ID:-manual}"
export RAY_TMPDIR="${RLINF_JOB_SCRATCH}/ray"

cleanup_local_scratch() {
    ray stop --force 2>/dev/null || true
    case "${RLINF_JOB_SCRATCH}" in
        /nvme/muyao/rlinf_rgb_h16_*) rm -rf -- "${RLINF_JOB_SCRATCH}" ;;
        *) echo "Refusing to clean unexpected scratch path: ${RLINF_JOB_SCRATCH}" ;;
    esac
}
trap cleanup_local_scratch EXIT

mkdir -p "${RLINF_JOB_SCRATCH}/hf_datasets" "${RLINF_JOB_SCRATCH}/hf_home" "${RLINF_JOB_SCRATCH}/hf_hub" "${RLINF_JOB_SCRATCH}/xdg" "${RAY_TMPDIR}" "${REPO_PATH}/slurm_logs"
export HF_HOME="${RLINF_JOB_SCRATCH}/hf_home"
export HF_DATASETS_CACHE="${RLINF_JOB_SCRATCH}/hf_datasets"
export HUGGINGFACE_HUB_CACHE="${RLINF_JOB_SCRATCH}/hf_hub"
export XDG_CACHE_HOME="${RLINF_JOB_SCRATCH}/xdg"
export TMPDIR="${RAY_TMPDIR}"
echo "local scratch: ${RLINF_JOB_SCRATCH}"

cd "${REPO_PATH}"

echo "=== [preflight] verify episodes 1000-1499: $(date) ==="
python - <<'PY'
import json
import os
from pathlib import Path

src = Path(os.environ["SRC_DATASET"])
with (src / "meta" / "episodes.jsonl").open() as f:
    episodes = {row["episode_index"]: row for row in map(json.loads, f) if row}
missing = [i for i in range(1000, 1500) if i not in episodes]
if missing:
    raise SystemExit(f"missing blocks_ranking_rgb episodes: {missing[:10]}")
texts = [task for i in range(1000, 1500) for task in episodes[i]["tasks"]]
if not all("red" in text.lower() and "green" in text.lower() and "blue" in text.lower() for text in texts):
    raise SystemExit("episode range contains an unexpected task description")
print(f"verified {len(texts)} blocks_ranking_rgb episodes")
PY

echo "=== [step1] build single-task dataset: $(date) ==="
if [[ -d "${DST_DATASET}" ]]; then
    EPISODE_COUNT=$(wc -l < "${DST_DATASET}/meta/episodes.jsonl" 2>/dev/null || echo 0)
    if [[ "${EPISODE_COUNT}" == "500" ]]; then
        echo "dataset already complete, skipping filter"
    else
        echo "dataset incomplete (${EPISODE_COUNT}), rebuilding"
        rm -rf "${DST_DATASET}"
        python toolkits/lerobot/filter_lerobot_dataset.py --source-dir "${SRC_DATASET}" --output-dir "${DST_DATASET}" --episodes "1000-1499"
    fi
else
    python toolkits/lerobot/filter_lerobot_dataset.py --source-dir "${SRC_DATASET}" --output-dir "${DST_DATASET}" --episodes "1000-1499"
fi

echo "=== [step2] calculate norm stats: $(date) ==="
mkdir -p "${NORM_STATS_DIR}"
if [[ -s "${NORM_STATS_DIR}/norm_stats.json" ]]; then
    echo "norm_stats already exists, skipping calculation"
else
    python toolkits/lerobot/calculate_norm_stats.py --config-name "${CONFIG_NAME}" --repo-id "${DST_DATASET}" --output-path "${NORM_STATS_DIR}"
fi
ln -sf "${NORM_STATS_DIR}/norm_stats.json" "${DST_DATASET}/norm_stats.json"

echo "=== [step3] SFT training: $(date) ==="
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-blocks_ranking_rgb_pi05_sft_h16_noadapt_8gpu"
mkdir -p "${LOG_DIR}"
ray stop --force 2>/dev/null || true
ray start --head --temp-dir="${RAY_TMPDIR}" --num-cpus=64 --num-gpus=8
python examples/sft/train_vla_sft.py --config-path "${REPO_PATH}/examples/sft/config" --config-name blocks_ranking_rgb_sft_openpi_pi05_h16_noadapt runner.logger.log_path="${LOG_DIR}" 2>&1 | tee "${LOG_DIR}/sft.log"
echo "=== done: $(date) ==="
