#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J grid-bpp-eval
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/eval_bbh_pal_par_s50_grid-%j.out
#SBATCH --time=12:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-7,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-20,SH-IDCA1404-10-140-54-21,SH-IDCA1404-10-140-54-25,SH-IDCA1404-10-140-54-26,SH-IDCA1404-10-140-54-28,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-34,SH-IDCA1404-10-140-54-35,SH-IDCA1404-10-140-54-36,SH-IDCA1404-10-140-54-37,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-45,SH-IDCA1404-10-140-54-46,SH-IDCA1404-10-140-54-47,SH-IDCA1404-10-140-54-50,SH-IDCA1404-10-140-54-57,SH-IDCA1404-10-140-54-59,SH-IDCA1404-10-140-54-61,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-73,SH-IDCA1404-10-140-54-78,SH-IDCA1404-10-140-54-79,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-82,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-118

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
            START_CANDIDATE="${START_CANDIDATE:-}" \
            START_TASK="${START_TASK:-}" \
            END_CANDIDATE="${END_CANDIDATE:-}" \
            END_TASK="${END_TASK:-}" \
            EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}" \
        bash "${REPO_PATH}/examples/embodiment/slurm/eval_bbh_pal_par_s50_grid.sbatch"
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

MERGED_DIR=/mnt/petrelfs/muyao/jiayusun/merged_weights
TASKS=(beat_block_hammer place_a2b_left place_a2b_right)
CANDIDATES=(
    grid_s50_bbh05_pal05_par05
    grid_s50_bbh07_pal05_par05
    grid_s50_bbh05_pal07_par05
    grid_s50_bbh09_pal05_par05
    grid_s50_bbh07_pal07_par05
    grid_s50_bbh05_pal05_par07
    grid_s50_bbh07_pal05_par07
    grid_s50_bbh09_pal07_par05
    grid_s50_bbh05_pal07_par07
    grid_s50_bbh09_pal05_par07
    grid_s50_bbh07_pal07_par07
    grid_s50_bbh09_pal07_par07
)

declare -A CKPT
CKPT[grid_s50_bbh05_pal05_par05]="${MERGED_DIR}/grid_s50_bbh05_pal05_par05/full_weights.pt"
CKPT[grid_s50_bbh07_pal05_par05]="${MERGED_DIR}/grid_s50_bbh07_pal05_par05/full_weights.pt"
CKPT[grid_s50_bbh05_pal07_par05]="${MERGED_DIR}/grid_s50_bbh05_pal07_par05/full_weights.pt"
CKPT[grid_s50_bbh09_pal05_par05]="${MERGED_DIR}/grid_s50_bbh09_pal05_par05/full_weights.pt"
CKPT[grid_s50_bbh07_pal07_par05]="${MERGED_DIR}/grid_s50_bbh07_pal07_par05/full_weights.pt"
CKPT[grid_s50_bbh05_pal05_par07]="${MERGED_DIR}/grid_s50_bbh05_pal05_par07/full_weights.pt"
CKPT[grid_s50_bbh07_pal05_par07]="${MERGED_DIR}/grid_s50_bbh07_pal05_par07/full_weights.pt"
CKPT[grid_s50_bbh09_pal07_par05]="${MERGED_DIR}/grid_s50_bbh09_pal07_par05/full_weights.pt"
CKPT[grid_s50_bbh05_pal07_par07]="${MERGED_DIR}/grid_s50_bbh05_pal07_par07/full_weights.pt"
CKPT[grid_s50_bbh09_pal05_par07]="${MERGED_DIR}/grid_s50_bbh09_pal05_par07/full_weights.pt"
CKPT[grid_s50_bbh07_pal07_par07]="${MERGED_DIR}/moe_3task_bbh_pal_par_step50_a07/full_weights.pt"
CKPT[grid_s50_bbh09_pal07_par07]="${MERGED_DIR}/moe_3task_bbh09_pal07_par07_step50/full_weights.pt"

declare -A RESULTS
START_CANDIDATE=${START_CANDIDATE:-}
START_TASK=${START_TASK:-}
END_CANDIDATE=${END_CANDIDATE:-}
END_TASK=${END_TASK:-}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-1800}
STARTED=0
FINISHED=0

if [[ -z "${START_CANDIDATE}" ]]; then
    STARTED=1
fi

for candidate in "${CANDIDATES[@]}"; do
    ckpt="${CKPT[$candidate]}"
    if [[ ! -s "${ckpt}" ]]; then
        echo "[MISSING] ${candidate}: ${ckpt}"
        exit 2
    fi
done

for candidate in "${CANDIDATES[@]}"; do
    ckpt="${CKPT[$candidate]}"
    for task in "${TASKS[@]}"; do
        if [[ "${FINISHED}" == "1" ]]; then
            key="${candidate}/${task}"
            RESULTS[$key]="SKIPPED"
            continue
        fi

        if [[ "${STARTED}" == "0" ]]; then
            if [[ "${candidate}" == "${START_CANDIDATE}" && "${task}" == "${START_TASK}" ]]; then
                STARTED=1
            else
                key="${candidate}/${task}"
                RESULTS[$key]="SKIPPED"
                echo "[SKIP] ${key}; waiting for ${START_CANDIDATE}/${START_TASK}"
                continue
            fi
        fi

        cfg_name="moe_3task_bbh_pal_par_s50_a07_eval_${task}"
        exp_name="${candidate}_eval_${task}"

        echo ""
        echo "======================================================="
        echo "  Evaluating [${candidate}] on: ${task}  ($(date))"
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

        key="${candidate}/${task}"
        if [[ $exit_code -ne 0 ]]; then
            echo "  [FAIL] ${key} exited with code ${exit_code}"
            RESULTS[$key]="FAILED"
        else
            RESULTS[$key]="DONE"
            echo "  [DONE] ${key}"
        fi

        if [[ -n "${END_CANDIDATE}" && "${candidate}" == "${END_CANDIDATE}" && "${task}" == "${END_TASK}" ]]; then
            FINISHED=1
        fi
    done
done

ray stop --force 2>/dev/null || true

echo ""
echo "======================================================="
echo "  bbh/pal/par step50 grid EVAL SUMMARY -- $(date)"
echo "======================================================="
printf "%-35s %-25s %s\n" "Candidate" "Task" "Status"
for candidate in "${CANDIDATES[@]}"; do
    for task in "${TASKS[@]}"; do
        key="${candidate}/${task}"
        printf "%-35s %-25s %s\n" "${candidate}" "${task}" "${RESULTS[$key]:-N/A}"
    done
done
echo "======================================================="
