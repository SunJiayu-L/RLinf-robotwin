#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J sft20k-e19
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/moe_sft20k_eval_all19-%j.out
#SBATCH --time=24:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-7,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-20,SH-IDCA1404-10-140-54-21,SH-IDCA1404-10-140-54-25,SH-IDCA1404-10-140-54-28,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-35,SH-IDCA1404-10-140-54-36,SH-IDCA1404-10-140-54-37,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-45,SH-IDCA1404-10-140-54-46,SH-IDCA1404-10-140-54-50,SH-IDCA1404-10-140-54-57,SH-IDCA1404-10-140-54-59,SH-IDCA1404-10-140-54-61,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-73,SH-IDCA1404-10-140-54-79,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-118

set -eo pipefail

SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}

export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
export ROBOTWIN_PATH=/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf

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
            ROBOTWIN_PATH="${ROBOTWIN_PATH}" \
        bash "${REPO_PATH}/examples/embodiment/slurm/moe_sft20k_eval_all19.sbatch"
fi

# === inside container ===

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

export ROBOT_PLATFORM=ALOHA
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export VK_DRIVER_FILES="${REPO_PATH}/nvidia_icd.json"
export VK_ICD_FILENAMES="${REPO_PATH}/nvidia_icd.json"
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}
export ASSETS_PATH="${ROBOTWIN_PATH}"
export HYDRA_FULL_ERROR=1
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"

cd "${REPO_PATH}"

# ============================================================
# SFT-20k base 权重，顺序评测 moe-demo-clean 20 个任务中已有 eval seed 的 19 个
# （pick_diverse_bottles 的 eval seed 还在生成中，本轮先跳过）
# 每个任务独立 ray stop -> ray start，避免 Ray 状态污染同一 job 崩溃
# ============================================================

TASKS=(
    beat_block_hammer
    click_alarmclock
    click_bell
    handover_block
    move_can_pot
    move_playingcard_away
    pick_dual_bottles
    place_a2b_left
    place_a2b_right
    place_can_basket
    place_cans_plasticbox
    place_dual_shoes
    place_object_stand
    place_phone_stand
    put_bottles_dustbin
    rotate_qrcode
    stack_blocks_two
    stack_bowls_three
    turn_switch
)

declare -A RESULTS

for task in "${TASKS[@]}"; do
    cfg_name="moe_sft20k_eval_${task}"

    echo ""
    echo "======================================================="
    echo "  Evaluating [SFT-20k base] on: ${task}  ($(date))"
    echo "======================================================="

    LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${cfg_name}"
    mkdir -p "${LOG_DIR}"

    ray stop --force 2>/dev/null || true
    ray start --head --num-cpus=64 --num-gpus=8

    set +e
    python examples/embodiment/eval_embodied_agent.py \
        --config-path "${EMBODIED_PATH}/config/" \
        --config-name "${cfg_name}" \
        "runner.logger.log_path=${LOG_DIR}" \
        2>&1 | tee "${LOG_DIR}/eval.log"
    exit_code=$?
    set -e

    ray stop --force 2>/dev/null || true

    if [[ $exit_code -ne 0 ]]; then
        echo "  [FAIL] ${task} exited with code ${exit_code}"
        RESULTS[$task]="FAILED"
    else
        success=$(grep "eval/success_once" "${LOG_DIR}/eval.log" 2>/dev/null | tail -1 | \
            grep -oP "'eval/success_once': array\(\K[0-9.]+")
        RESULTS[$task]="${success:-UNKNOWN}"
        echo "  [DONE] ${task}: success_once=${success:-UNKNOWN}"
    fi
done

ray stop --force 2>/dev/null || true

# ============================================================
# Summary
# ============================================================
echo ""
echo "======================================================="
echo "  moe_sft20k_base EVAL SUMMARY (19/20 tasks) — $(date)"
echo "======================================================="
printf "%-30s %s\n" "Task" "Success"
printf "%-30s %s\n" "----" "-------"
for task in "${TASKS[@]}"; do
    printf "%-30s %s\n" "${task}" "${RESULTS[$task]:-N/A}"
done
echo "======================================================="
