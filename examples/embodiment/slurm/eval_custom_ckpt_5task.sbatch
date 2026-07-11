#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J custom5-eval
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/eval_custom_ckpt_5task-%j.out
#SBATCH --time=8:00:00
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
            CANDIDATE_LABEL="${CANDIDATE_LABEL:-custom}" \
            CKPT_PATH="${CKPT_PATH:-}" \
            START_TASK="${START_TASK:-}" \
            END_TASK="${END_TASK:-}" \
            EVAL_TIMEOUT="${EVAL_TIMEOUT:-3600}" \
        bash "${REPO_PATH}/examples/embodiment/slurm/eval_custom_ckpt_5task.sbatch"
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

if [[ -z "${CKPT_PATH}" || ! -s "${CKPT_PATH}" ]]; then
    echo "[MISSING] CKPT_PATH=${CKPT_PATH}"
    exit 2
fi

STARTED=0
FINISHED=0
if [[ -z "${START_TASK}" ]]; then
    STARTED=1
fi
EVAL_TIMEOUT=${EVAL_TIMEOUT:-3600}

declare -A RESULTS

for task in "${TASKS[@]}"; do
    key="${CANDIDATE_LABEL}/${task}"
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
    exp_name="${CANDIDATE_LABEL}_eval_${task}"

    echo ""
    echo "======================================================="
    echo "  Evaluating [${CANDIDATE_LABEL}] on: ${task}  ($(date))"
    echo "  ckpt: ${CKPT_PATH}"
    echo "======================================================="

    ray stop --force 2>/dev/null || true
    ray start --head --num-cpus=64 --num-gpus=8

    set +e
    timeout "${EVAL_TIMEOUT}" python examples/embodiment/eval_embodied_agent.py \
        --config-path "${EMBODIED_PATH}/config/" \
        --config-name "${cfg_name}" \
        runner.ckpt_path="${CKPT_PATH}" \
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
echo "  custom checkpoint 5-task EVAL SUMMARY -- $(date)"
echo "======================================================="
printf "%-28s %-28s %s\n" "Candidate" "Task" "Status"
for task in "${TASKS[@]}"; do
    key="${CANDIDATE_LABEL}/${task}"
    printf "%-28s %-28s %s\n" "${CANDIDATE_LABEL}" "${task}" "${RESULTS[$key]:-N/A}"
done
echo "======================================================="
