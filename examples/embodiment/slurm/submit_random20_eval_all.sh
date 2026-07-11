#!/bin/bash
# Submit eval jobs for all 20 tasks using random_first20_pi05_sft weights.
# Requires manifests to exist in seeds/ (run collect_manifests_20tasks.sbatch first).
# Usage: bash examples/embodiment/slurm/submit_random20_eval_all.sh

REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
SBATCH_SCRIPT="${REPO_PATH}/examples/embodiment/slurm/random20_pi05_eval_task.sbatch"
LOG_DIR="${REPO_PATH}/slurm_logs"
SEEDS_DIR="${REPO_PATH}/rlinf/envs/robotwin/seeds"

TASKS=(
    adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
    click_alarmclock click_bell dump_bin_bigbin handover_block
    handover_mic hanging_mug lift_pot move_can_pot
    move_pillbottle_pad move_playingcard_away move_stapler_pad open_microwave
    pick_diverse_bottles pick_dual_bottles place_a2b_left place_a2b_right
)

# Verify all manifests exist before submitting
MISSING=0
for TASK in "${TASKS[@]}"; do
    MANIFEST="${SEEDS_DIR}/${TASK}_manifest_96ep.json"
    if [[ ! -f "${MANIFEST}" ]]; then
        echo "MISSING manifest: ${MANIFEST}"
        MISSING=1
    fi
done
if [[ "${MISSING}" -eq 1 ]]; then
    echo ""
    echo "Run collect_manifests_20tasks.sbatch first to generate missing manifests."
    exit 1
fi

echo "Submitting 20 eval jobs for random_first20_pi05_sft..."
echo ""
for TASK in "${TASKS[@]}"; do
    MANIFEST="${SEEDS_DIR}/${TASK}_manifest_96ep.json"
    SHORT="${TASK:0:12}"
    JOB_ID=$(sbatch \
        --job-name="rw20-${SHORT}" \
        --output="${LOG_DIR}/random20_pi05_eval_${TASK}-%j.out" \
        --export=ALL,TASK_NAME="${TASK}",HAS_SEEDS="true",SEEDS_PATH="${MANIFEST}" \
        "${SBATCH_SCRIPT}" | awk '{print $NF}')
    printf "  %-30s manifest(seed+instr)   job=%s\n" "${TASK}" "${JOB_ID}"
done
echo ""
echo "All 20 jobs submitted. Monitor with: squeue -u muyao"
