#!/bin/bash
set -euo pipefail

CKPT_STEP=${1:-5000}
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

CKPT_FILE="${REPO_PATH}/logs/20260620-23:01:03-random10_pi05_sft_h16_noadapt_8gpu/random10_pi05_sft_h16_noadapt/checkpoints/global_step_${CKPT_STEP}/actor/model_state_dict/full_weights.pt"
[[ -s "${CKPT_FILE}" ]] || { echo "missing checkpoint step ${CKPT_STEP}: ${CKPT_FILE}"; exit 1; }
echo "checkpoint step ${CKPT_STEP}: ready"

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
        --job-name="r10e${CKPT_STEP}-${SHORT}" \
        --output="${LOG_DIR}/random10_h16_sft${CKPT_STEP}_eval_${TASK}-%j.out" \
        --export=ALL,TASK_NAME="${TASK}",STEP_LIM="${STEP_LIM}",CKPT_STEP="${CKPT_STEP}" \
        "${SBATCH_SCRIPT}" | awk '{print $NF}')
    printf '%-24s step=%-5s step_lim=%-4s job=%s\n' "${TASK}" "${CKPT_STEP}" "${STEP_LIM}" "${JOB_ID}"
done
