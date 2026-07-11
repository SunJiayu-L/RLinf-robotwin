#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J ppo5x5-eval
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/eval_5ppo_on_5task-%j.out
#SBATCH --time=12:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-7,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-20,SH-IDCA1404-10-140-54-21,SH-IDCA1404-10-140-54-25,SH-IDCA1404-10-140-54-26,SH-IDCA1404-10-140-54-28,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-34,SH-IDCA1404-10-140-54-35,SH-IDCA1404-10-140-54-36,SH-IDCA1404-10-140-54-37,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-45,SH-IDCA1404-10-140-54-46,SH-IDCA1404-10-140-54-47,SH-IDCA1404-10-140-54-50,SH-IDCA1404-10-140-54-57,SH-IDCA1404-10-140-54-59,SH-IDCA1404-10-140-54-61,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-73,SH-IDCA1404-10-140-54-78,SH-IDCA1404-10-140-54-79,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-82,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-118

set -eo pipefail

SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}

export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
export ROBOTWIN_PATH=/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf

if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec \
        --cleanenv --nv \
        --bind /mnt:/mnt --bind /nvme:/nvme \
        --pwd "${REPO_PATH}" "${SIF_IMAGE}" \
        env IN_RLINF_ROBOTWIN_CONTAINER=1 \
            HOME="${HOME}" \
            SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
            SLURM_NODELIST="${SLURM_NODELIST:-}" \
            CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
            REPO_PATH="${REPO_PATH}" \
            ROBOTWIN_PATH="${ROBOTWIN_PATH}" \
            SELECT_PPO="${SELECT_PPO:-}" \
            START_TASK="${START_TASK:-}" \
            END_TASK="${END_TASK:-}" \
            EVAL_TIMEOUT="${EVAL_TIMEOUT:-3600}" \
        bash "${REPO_PATH}/examples/embodiment/slurm/eval_5ppo_on_5task.sbatch"
fi

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

TASKS=(
    beat_block_hammer
    click_alarmclock
    click_bell
    move_can_pot
    move_playingcard_away
)
PPOS=(
    ppo_bbh_s50
    ppo_ca_s50
    ppo_cb_s50
    ppo_mcp_s100
    ppo_mpa_s50
)

declare -A CKPT
CKPT[ppo_bbh_s50]="${REPO_PATH}/logs/20260707-22:22:21-moe_sft20k_beat_block_hammer_ppo100/moe_sft20k_beat_block_hammer_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"
CKPT[ppo_ca_s50]="${REPO_PATH}/logs/20260708-13:50:33-moe_sft20k_click_alarmclock_ppo100/moe_sft20k_click_alarmclock_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"
CKPT[ppo_cb_s50]="${REPO_PATH}/logs/20260708-13:56:28-moe_sft20k_click_bell_ppo100/moe_sft20k_click_bell_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"
CKPT[ppo_mcp_s100]="${REPO_PATH}/logs/20260706-23:28:23-moe_sft20k_move_can_pot_ppo100/moe_sft20k_move_can_pot_ppo100/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt"
CKPT[ppo_mpa_s50]="${REPO_PATH}/logs/20260706-11:28:57-moe_sft20k_move_playingcard_away_ppo100/moe_sft20k_move_playingcard_away_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"

SELECT_PPO=${SELECT_PPO:-}
START_TASK=${START_TASK:-}
END_TASK=${END_TASK:-}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-3600}
STARTED=0
FINISHED=0
if [[ -z "${START_TASK}" ]]; then
    STARTED=1
fi

if [[ -z "${SELECT_PPO}" ]]; then
    echo "[ERROR] SELECT_PPO must be one of: ${PPOS[*]}"
    exit 2
fi
if [[ -z "${CKPT[$SELECT_PPO]:-}" ]]; then
    echo "[ERROR] unknown SELECT_PPO=${SELECT_PPO}; expected one of: ${PPOS[*]}"
    exit 2
fi
if [[ ! -s "${CKPT[$SELECT_PPO]}" ]]; then
    echo "[MISSING] ${SELECT_PPO}: ${CKPT[$SELECT_PPO]}"
    exit 2
fi

declare -A RESULTS

for task in "${TASKS[@]}"; do
    key="${SELECT_PPO}/${task}"
    if [[ "${FINISHED}" == "1" ]]; then
        RESULTS[$key]="SKIPPED"
        continue
    fi
    if [[ "${STARTED}" == "0" ]]; then
        if [[ "${task}" == "${START_TASK}" ]]; then
            STARTED=1
        else
            RESULTS[$key]="SKIPPED"
            echo "[SKIP] ${key}; waiting for ${START_TASK}"
            continue
        fi
    fi

    cfg_name="moe_sft20k_eval_${task}"
    exp_name="${SELECT_PPO}_eval_${task}"
    ckpt="${CKPT[$SELECT_PPO]}"

    echo ""
    echo "======================================================="
    echo "  Evaluating [${SELECT_PPO}] on: ${task}  ($(date))"
    echo "  ckpt: ${ckpt}"
    echo "======================================================="

    ray stop --force 2>/dev/null || true
    ray start --head --num-cpus=64 --num-gpus=8

    set +e
    timeout "${EVAL_TIMEOUT}" python examples/embodiment/eval_embodied_agent.py \
        --config-path "${EMBODIED_PATH}/config/" \
        --config-name "${cfg_name}" \
        runner.ckpt_path="${ckpt}" \
        runner.logger.experiment_name="${exp_name}" \
        env.eval.video_cfg.save_video=False \
        2>&1
    exit_code=$?
    set -e

    ray stop --force 2>/dev/null || true

    if [[ $exit_code -ne 0 ]]; then
        echo "  [FAIL] ${key} exited with code ${exit_code}"
        RESULTS[$key]="FAILED"
    else
        RESULTS[$key]="DONE"
        echo "  [DONE] ${key}"
    fi

    if [[ -n "${END_TASK}" && "${task}" == "${END_TASK}" ]]; then
        FINISHED=1
    fi
done

ray stop --force 2>/dev/null || true

echo ""
echo "======================================================="
echo "  5 PPO x 5 task EVAL SUMMARY -- $(date)"
echo "======================================================="
printf "%-16s %-28s %s\n" "PPO" "Task" "Status"
for task in "${TASKS[@]}"; do
    key="${SELECT_PPO}/${task}"
    printf "%-16s %-28s %s\n" "${SELECT_PPO}" "${task}" "${RESULTS[$key]:-N/A}"
done
echo "======================================================="
