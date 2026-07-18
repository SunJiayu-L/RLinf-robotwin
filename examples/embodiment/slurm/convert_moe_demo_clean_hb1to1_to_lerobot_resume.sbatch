#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -J moe-convert-hb1to1-resume
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/convert_moe_demo_clean_hb1to1_to_lerobot_resume-%j.out
#SBATCH --time=2-00:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-20,SH-IDCA1404-10-140-54-21,SH-IDCA1404-10-140-54-25,SH-IDCA1404-10-140-54-28,SH-IDCA1404-10-140-54-31,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-35,SH-IDCA1404-10-140-54-37,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-45,SH-IDCA1404-10-140-54-57,SH-IDCA1404-10-140-54-61,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-73,SH-IDCA1404-10-140-54-79,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-118

set -euo pipefail

ROBOWIN_PI05_DIR=/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05
PYTHON=${ROBOWIN_PI05_DIR}/.venv/bin/python
SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}

ORIG_RAW_DIR=/mnt/petrelfs/muyao/jiayusun/moe-demo-clean
MIXED_RAW_DIR=/mnt/petrelfs/muyao/jiayusun/moe-demo-clean-hb1to1
REPO_ID=robotwin_moe_demo_clean_hb1to1
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
        bash "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/examples/embodiment/slurm/convert_moe_demo_clean_hb1to1_to_lerobot_resume.sbatch"
fi

# === inside container from here ===

export HOME=/mnt/petrelfs/muyao/jiayusun
export XDG_CACHE_HOME=${CACHE_DIR}
export HF_HOME=${CACHE_DIR}/huggingface
export HF_DATASETS_CACHE=${HF_HOME}/datasets

printf "[resume] host=%s\n" "$(hostname)"
printf "[resume] job_id=%s\n" "${SLURM_JOB_ID:-none}"
printf "[resume] output repo_id=%s\n" "${REPO_ID}"
printf "[resume] output=%s\n" "${HF_HOME}/lerobot/${REPO_ID}"

# Ensure symlink raw dir exists (recreate if missing, skip rm -rf to preserve on-disk dataset)
if [[ ! -d "${MIXED_RAW_DIR}" ]]; then
    printf "[setup] Recreating symlink raw dir at %s\n" "${MIXED_RAW_DIR}"
    mkdir -p "${MIXED_RAW_DIR}"

    for task_dir in "${ORIG_RAW_DIR}"/*/; do
        task_name=$(basename "${task_dir}")
        if [[ "${task_name}" == "handover_block-demo_clean-200" ]]; then
            continue
        fi
        ln -s "${task_dir%/}" "${MIXED_RAW_DIR}/${task_name}"
        printf "[setup]   linked: %s\n" "${task_name}"
    done

    for i in $(seq -w 0 18); do
        ln -s "${ORIG_RAW_DIR}/handover_block-demo_clean-200" \
            "${MIXED_RAW_DIR}/handover_block_copy${i}-demo_clean-200"
        printf "[setup]   linked: handover_block_copy%s\n" "${i}"
    done
fi

printf "[setup] Total dirs in mixed raw dir: %d\n" "$(ls "${MIXED_RAW_DIR}" | wc -l)"

# Print existing dataset state
"${PYTHON}" - <<PY
import json
from pathlib import Path
p = Path("${HF_HOME}/lerobot/${REPO_ID}/meta/info.json")
info = json.loads(p.read_text())
print(f"[resume] existing total_episodes: {info['total_episodes']}")
print(f"[resume] existing total_frames:   {info['total_frames']}")
PY

# Run resume conversion (appends from total_episodes onward)
cd "${ROBOWIN_PI05_DIR}"
"${PYTHON}" examples/aloha_real/convert_aloha_data_to_lerobot_robotwin_resume.py \
    --raw_dir "${MIXED_RAW_DIR}" \
    --repo_id "${REPO_ID}"

printf "[resume] done: %s/lerobot/%s\n" "${HF_HOME}" "${REPO_ID}"

# Verify final state
"${PYTHON}" - <<PY
import json
from pathlib import Path

lerobot_path = Path("${HF_HOME}/lerobot/${REPO_ID}")
info = json.loads((lerobot_path / "meta/info.json").read_text())
print(f"[verify] total_episodes: {info['total_episodes']}")
print(f"[verify] total_frames:   {info['total_frames']}")

n_parquet = len(list(lerobot_path.rglob("*.parquet")))
print(f"[verify] parquet files on disk: {n_parquet}")
assert n_parquet == info["total_episodes"], f"Mismatch: {n_parquet} parquets vs {info['total_episodes']} episodes"
print("[verify] OK")
PY
