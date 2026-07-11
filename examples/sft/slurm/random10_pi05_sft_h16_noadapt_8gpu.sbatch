#!/bin/bash

#SBATCH -p wam_data

#SBATCH -N 1

#SBATCH -n 1

#SBATCH --cpus-per-task=64

#SBATCH --gres=gpu:8

#SBATCH -J r10-h16-sft

#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/random10_pi05_sft_h16_noadapt_8gpu-%j.out

#SBATCH --time=48:00:00

#SBATCH --exclude=SH-IDCA1404-10-140-54-1,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-117,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-58,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-79



set -eo pipefail



SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}

APPTAINER_BIN=${APPTAINER_BIN:-apptainer}

export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin



if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then

    exec "${APPTAINER_BIN}" exec --cleanenv --nv --bind /mnt:/mnt --bind /nvme:/nvme --pwd "${REPO_PATH}" "${SIF_IMAGE}" env IN_RLINF_ROBOTWIN_CONTAINER=1 HOME="${HOME}" SLURM_JOB_ID="${SLURM_JOB_ID:-}" SLURM_NODELIST="${SLURM_NODELIST:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" REPO_PATH="${REPO_PATH}" bash "${REPO_PATH}/examples/sft/slurm/random10_pi05_sft_h16_noadapt_8gpu.sbatch"

fi



# === inside container from here ===



export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0

export CUDA_HOME=/mnt/lustre/share/cuda-12.4

export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH

export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"



export RLINF_JOB_SCRATCH="/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/random10_h16_9805726"

mkdir -p "${RLINF_JOB_SCRATCH}/hf_datasets" "${RLINF_JOB_SCRATCH}/hf_home" "${RLINF_JOB_SCRATCH}/hf_hub" "${RLINF_JOB_SCRATCH}/xdg" "${RLINF_JOB_SCRATCH}/tmp"

export HF_HOME="${RLINF_JOB_SCRATCH}/hf_home"

export HF_DATASETS_CACHE="${RLINF_JOB_SCRATCH}/hf_datasets"

export HUGGINGFACE_HUB_CACHE="${RLINF_JOB_SCRATCH}/hf_hub"

export XDG_CACHE_HOME="${RLINF_JOB_SCRATCH}/xdg"

export RAY_TMPDIR="/tmp/r10_${SLURM_JOB_ID:-manual}"
mkdir -p "${RAY_TMPDIR}"
export TMPDIR="${RAY_TMPDIR}"

echo "hf scratch cache: ${RLINF_JOB_SCRATCH}"



export SRC_DATASET="/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_random_47_500_pi05"

export DST_DATASET="/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_random_10_500_pi05_h16"

export EPISODE_RANGES="0-4999"

export BASE_CKPT="/mnt/petrelfs/muyao/jiayusun/pi05_base_adapt_false"

export CONFIG_NAME="pi05_aloha_robotwin_random10_h16"

export NORM_STATS_DIR="${BASE_CKPT}/physical-intelligence/robotwin_random10_h16"



cd "${REPO_PATH}"

mkdir -p "${REPO_PATH}/slurm_logs"



echo "=== [preflight] check random dataset completeness: $(date) ==="

python - <<'PRECHECK'

import json

import os

from pathlib import Path

src = Path(os.environ["SRC_DATASET"])

if not src.exists():

    raise SystemExit(f"missing source dataset: {src}")

episodes = []

with (src / "meta" / "episodes.jsonl").open() as f:

    for line in f:

        if line.strip():

            episodes.append(json.loads(line)["episode_index"])

present = set(episodes)

missing = [i for i in range(5000) if i not in present]

if missing:

    raise SystemExit(f"source dataset is missing first-10-task episodes: {missing[:10]}")

print(f"source episodes={len(episodes)}, max_episode={max(episodes)}, first_10_tasks=OK")

PRECHECK



echo "=== [step1] filter first 10 random tasks: $(date) ==="

if [[ -d "${DST_DATASET}" ]]; then

    PARQUET_COUNT=$(find "${DST_DATASET}" -maxdepth 3 -type f -name '*.parquet' 2>/dev/null | wc -l)

    if [[ "${PARQUET_COUNT}" == "5000" && -f "${DST_DATASET}/meta/info.json" && -f "${DST_DATASET}/meta/episodes.jsonl" ]]; then

        echo "  ${DST_DATASET} already complete (${PARQUET_COUNT} parquet files), skipping filter step"

    else

        echo "  ${DST_DATASET} exists but is incomplete (${PARQUET_COUNT} parquet files), rebuilding"

        rm -rf "${DST_DATASET}"

        python toolkits/lerobot/filter_lerobot_dataset.py --source-dir "${SRC_DATASET}" --output-dir "${DST_DATASET}" --episodes "${EPISODE_RANGES}"

    fi

else

    python toolkits/lerobot/filter_lerobot_dataset.py --source-dir "${SRC_DATASET}" --output-dir "${DST_DATASET}" --episodes "${EPISODE_RANGES}"

fi

echo "=== [step1] done: $(date) ==="



echo "=== [step2] calc norm stats: $(date) ==="

mkdir -p "${NORM_STATS_DIR}"

if [[ -s "${NORM_STATS_DIR}/norm_stats.json" ]]; then
    echo "norm_stats already exists, skipping calculation"
else
    python toolkits/lerobot/calculate_norm_stats.py --config-name "${CONFIG_NAME}" --repo-id "${DST_DATASET}" --output-path "${NORM_STATS_DIR}"
fi

echo "norm_stats saved to: ${NORM_STATS_DIR}/norm_stats.json"

ln -sf "${NORM_STATS_DIR}/norm_stats.json" "${DST_DATASET}/norm_stats.json"
echo "dataset norm_stats link: ${DST_DATASET}/norm_stats.json"
echo "=== [step2] done: $(date) ==="



echo "=== [step3] SFT training: $(date) ==="

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-random10_pi05_sft_h16_noadapt_8gpu"

mkdir -p "${LOG_DIR}"

ray stop --force 2>/dev/null || true

ray start --head --temp-dir="${RAY_TMPDIR}" --num-cpus=64 --num-gpus=8

python examples/sft/train_vla_sft.py --config-path "${REPO_PATH}/examples/sft/config" --config-name random10_sft_openpi_pi05_h16_noadapt runner.logger.log_path="${LOG_DIR}" 2>&1 | tee "${LOG_DIR}/sft.log"

ray stop

echo "=== [step3] done: $(date) ==="

