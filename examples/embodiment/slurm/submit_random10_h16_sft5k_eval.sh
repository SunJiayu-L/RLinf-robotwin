#!/bin/bash
set -euo pipefail

REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
SBATCH_SCRIPT="${REPO_PATH}/examples/embodiment/slurm/random10_h16_sft5k_eval_task.sbatch"
LOG_DIR="${REPO_PATH}/slurm_logs"
SEEDS_PATH="${REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json"

# A800-aligned task horizons where available; the remaining tasks retain the
# task-specific PJLab limits. Every task below has fixed seeds in eval_seeds.json.
TASK_SPECS=(
    "adjust_bottle:208"
    "beat_block_hammer:400"
    "blocks_ranking_rgb:208"
    "click_bell:400"
    "handover_block:400"
    "handover_mic:608"
)

mkdir -p "${LOG_DIR}"

python3 - "${SEEDS_PATH}" "${TASK_SPECS[@]}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
for spec in sys.argv[2:]:
    task = spec.split(":", 1)[0]
    if task not in data or not data[task].get("success_seeds"):
        raise SystemExit(f"{task}: fixed eval seeds unavailable; refusing submission")
    print(f"{task}: {len(data[task]['success_seeds'])} fixed seeds available")
PY

for SPEC in "${TASK_SPECS[@]}"; do
    TASK=${SPEC%%:*}
    STEP_LIM=${SPEC##*:}
    SHORT=${TASK:0:13}
    JOB_ID=$(sbatch \
        --job-name="r10e5k-${SHORT}" \
        --output="${LOG_DIR}/random10_h16_sft5k_eval_${TASK}-%j.out" \
        --export=ALL,TASK_NAME="${TASK}",STEP_LIM="${STEP_LIM}" \
        "${SBATCH_SCRIPT}" | awk '{print $NF}')
    printf '%-24s step_lim=%-4s job=%s\n' "${TASK}" "${STEP_LIM}" "${JOB_ID}"
done
