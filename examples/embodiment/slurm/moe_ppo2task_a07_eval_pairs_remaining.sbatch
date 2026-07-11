#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J ppo2t-rem
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/moe_ppo2task_a07_eval_pairs_remaining-%j.out
#SBATCH --time=48:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-7,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-20,SH-IDCA1404-10-140-54-21,SH-IDCA1404-10-140-54-25,SH-IDCA1404-10-140-54-28,SH-IDCA1404-10-140-54-29,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-35,SH-IDCA1404-10-140-54-36,SH-IDCA1404-10-140-54-37,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-45,SH-IDCA1404-10-140-54-46,SH-IDCA1404-10-140-54-50,SH-IDCA1404-10-140-54-57,SH-IDCA1404-10-140-54-59,SH-IDCA1404-10-140-54-61,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-73,SH-IDCA1404-10-140-54-79,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-118

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
        bash "${REPO_PATH}/examples/embodiment/slurm/moe_ppo2task_a07_eval_pairs_remaining.sbatch"
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
# 5 任务两两融合（task arithmetic, alpha=0.7, N=2）冲突性分析
# 每对任务：merge(CPU) -> eval task1 -> eval task2 -> 删除merge权重 -> 下一对
# 顺序执行，避免磁盘空间同时被多个 merge 权重占满
# ============================================================

SFT_BASE="/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/20000/model.safetensors"

declare -A CKPT
CKPT[beat_block_hammer]="${REPO_PATH}/logs/20260625-11:37:02-moe_sft20k_beat_block_hammer_ppo100/moe_sft20k_beat_block_hammer_ppo100/checkpoints/global_step_300/actor/model_state_dict/full_weights.pt"
CKPT[place_phone_stand]="${REPO_PATH}/logs/20260703-21:07:14-moe_sft20k_place_phone_stand_ppo100/moe_sft20k_place_phone_stand_ppo100/checkpoints/global_step_300/actor/model_state_dict/full_weights.pt"
CKPT[place_a2b_left]="${REPO_PATH}/logs/20260703-21:07:14-moe_sft20k_place_a2b_left_ppo100/moe_sft20k_place_a2b_left_ppo100/checkpoints/global_step_300/actor/model_state_dict/full_weights.pt"
CKPT[place_a2b_right]="${REPO_PATH}/logs/20260705-10:45:17-moe_sft20k_place_a2b_right_ppo100/moe_sft20k_place_a2b_right_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"
CKPT[move_can_pot]="${REPO_PATH}/logs/20260703-00:38:41-moe_sft20k_move_can_pot_ppo100/moe_sft20k_move_can_pot_ppo100/checkpoints/global_step_400/actor/model_state_dict/full_weights.pt"

PAIRS=(
    "place_a2b_left move_can_pot"
    "place_a2b_right move_can_pot"
)

declare -A RESULTS

for pair in "${PAIRS[@]}"; do
    read -r t1 t2 <<< "${pair}"
    MERGED="/mnt/petrelfs/muyao/jiayusun/merged_weights/pair_${t1}__${t2}_a07/full_weights.pt"

    echo ""
    echo "======================================================="
    echo "  Merging pair: ${t1} + ${t2}  ($(date))"
    echo "======================================================="

    python scripts/task_arithmetic_rlinf.py \
        --base "${SFT_BASE}" \
        --ft "${CKPT[$t1]}" "${CKPT[$t2]}" \
        --output "${MERGED}" \
        --alpha 0.7

    for task in "${t1}" "${t2}"; do
        cfg_name="moe_ppo2task_a07_eval_${t1}__${t2}_on_${task}"

        echo ""
        echo "  --- Evaluating [${t1}+${t2}] on: ${task}  ($(date)) ---"

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

        key="${t1}+${t2}|${task}"
        if [[ $exit_code -ne 0 ]]; then
            echo "  [FAIL] ${t1}+${t2} on ${task} exited with code ${exit_code}"
            RESULTS[$key]="FAILED"
        else
            success=$(grep "eval/success_once" "${LOG_DIR}/eval.log" 2>/dev/null | tail -1 | \
                grep -oP "'eval/success_once': array\(\K[0-9.]+")
            RESULTS[$key]="${success:-UNKNOWN}"
            echo "  [DONE] ${t1}+${t2} on ${task}: success_once=${success:-UNKNOWN}"
        fi
    done

    echo "  Removing merged weight to free space: ${MERGED}"
    rm -rf "$(dirname "${MERGED}")"
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "======================================================="
echo "  PAIRWISE MERGE (alpha=0.7) EVAL SUMMARY — $(date)"
echo "======================================================="
printf "%-45s %s\n" "Pair | Task" "Success"
printf "%-45s %s\n" "----" "-------"
for pair in "${PAIRS[@]}"; do
    read -r t1 t2 <<< "${pair}"
    for task in "${t1}" "${t2}"; do
        key="${t1}+${t2}|${task}"
        printf "%-45s %s\n" "${t1}+${t2} | ${task}" "${RESULTS[$key]:-N/A}"
    done
done
echo "======================================================="
