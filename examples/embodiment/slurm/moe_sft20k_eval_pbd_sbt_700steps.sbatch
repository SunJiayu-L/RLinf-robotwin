#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J sft20k-pbd-sbt
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/moe_sft20k_eval_pbd_sbt_700steps-%j.out
#SBATCH --time=8:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-7,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-20,SH-IDCA1404-10-140-54-21,SH-IDCA1404-10-140-54-25,SH-IDCA1404-10-140-54-26,SH-IDCA1404-10-140-54-28,SH-IDCA1404-10-140-54-29,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-34,SH-IDCA1404-10-140-54-35,SH-IDCA1404-10-140-54-36,SH-IDCA1404-10-140-54-37,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-45,SH-IDCA1404-10-140-54-46,SH-IDCA1404-10-140-54-47,SH-IDCA1404-10-140-54-50,SH-IDCA1404-10-140-54-57,SH-IDCA1404-10-140-54-59,SH-IDCA1404-10-140-54-61,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-73,SH-IDCA1404-10-140-54-78,SH-IDCA1404-10-140-54-79,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-118

set -eo pipefail
SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
export ROBOTWIN_PATH=/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf

if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec --cleanenv --nv --bind /mnt:/mnt --bind /nvme:/nvme --pwd "${REPO_PATH}" "${SIF_IMAGE}" env IN_RLINF_ROBOTWIN_CONTAINER=1 HOME="${HOME}" SLURM_JOB_ID="${SLURM_JOB_ID:-}" SLURM_NODELIST="${SLURM_NODELIST:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" REPO_PATH="${REPO_PATH}" ROBOTWIN_PATH="${ROBOTWIN_PATH}" bash "${REPO_PATH}/examples/embodiment/slurm/moe_sft20k_eval_pbd_sbt_700steps.sbatch"
fi

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export ROBOT_PLATFORM=ALOHA; export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl
export VK_DRIVER_FILES="${REPO_PATH}/nvidia_icd.json"
export VK_ICD_FILENAMES="${REPO_PATH}/nvidia_icd.json"
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}
export ASSETS_PATH="${ROBOTWIN_PATH}"
export HYDRA_FULL_ERROR=1
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"

cd "${REPO_PATH}"

TASKS=(put_bottles_dustbin stack_bowls_three)
declare -A RESULTS

for task in "${TASKS[@]}"; do
    cfg_name="moe_sft20k_eval_${task}"
    echo ""
    echo "======================================================="
    echo "  Evaluating (700 steps): ${task}  ($(date))"
    echo "======================================================="
    ray stop --force 2>/dev/null || true
    ray start --head --num-cpus=64 --num-gpus=8
    set +e
    timeout 3600 python examples/embodiment/eval_embodied_agent.py         --config-path "${EMBODIED_PATH}/config/"         --config-name "${cfg_name}" 2>&1
    exit_code=$?
    set -e
    ray stop --force 2>/dev/null || true
    if [[ $exit_code -eq 124 ]]; then
        RESULTS[$task]="TIMEOUT"
    elif [[ $exit_code -ne 0 ]]; then
        RESULTS[$task]="FAILED($exit_code)"
    else
        RESULTS[$task]="DONE"
    fi
done

echo ""
echo "======================================================="
echo "  EVAL SUMMARY (700 steps) — $(date)"
echo "======================================================="
for task in "${TASKS[@]}"; do
    printf "%-30s %s\n" "${task}" "${RESULTS[$task]:-N/A}"
done
echo "======================================================="
