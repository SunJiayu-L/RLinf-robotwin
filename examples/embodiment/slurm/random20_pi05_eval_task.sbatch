#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-69
# TASK_NAME and HAS_SEEDS are passed via --export from submit_random20_eval_all.sh

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
            TASK_NAME="${TASK_NAME}" \
            HAS_SEEDS="${HAS_SEEDS}" \
            SEEDS_PATH="${SEEDS_PATH}" \
        bash "${REPO_PATH}/examples/embodiment/slurm/random20_pi05_eval_task.sbatch"
fi

# === inside container from here ===

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
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export HYDRA_FULL_ERROR=1

cd "${REPO_PATH}"

if [[ "${HAS_SEEDS:-true}" == "true" ]]; then
    USE_FIXED="True"
else
    USE_FIXED="False"
fi

# Default seeds_path (eval_seeds.json, success_seeds format)
DEFAULT_SEEDS="${REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json"
EVAL_SEEDS_PATH="${SEEDS_PATH:-${DEFAULT_SEEDS}}"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-random20_eval_${TASK_NAME}"
mkdir -p "${LOG_DIR}"

ray stop --force 2>/dev/null || true
ray start --head --num-cpus=64 --num-gpus=8

python examples/embodiment/eval_embodied_agent.py \
    --config-path "${EMBODIED_PATH}/config/" \
    --config-name random20_eval_template \
    runner.logger.log_path="${LOG_DIR}" \
    runner.logger.experiment_name="random20_eval_${TASK_NAME}" \
    env.train.task_config.task_name="${TASK_NAME}" \
    env.eval.task_config.task_name="${TASK_NAME}" \
    env.eval.use_fixed_reset_state_ids="${USE_FIXED}" \
    env.eval.seeds_path="${EVAL_SEEDS_PATH}" \
    2>&1 | tee "${LOG_DIR}/eval_embodiment.log"

ray stop
