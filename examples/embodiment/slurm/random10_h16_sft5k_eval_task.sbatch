#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=8:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-1,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-117,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-58,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-79

set -euo pipefail

: "${TASK_NAME:?TASK_NAME is required}"
: "${STEP_LIM:?STEP_LIM is required}"

SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
export ROBOTWIN_PATH=/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf

if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec --cleanenv --nv \
        --bind /mnt:/mnt --bind /nvme:/nvme \
        --pwd "${REPO_PATH}" "${SIF_IMAGE}" \
        env IN_RLINF_ROBOTWIN_CONTAINER=1 \
            HOME="${HOME}" SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
            SLURM_NODELIST="${SLURM_NODELIST:-}" \
            CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
            REPO_PATH="${REPO_PATH}" ROBOTWIN_PATH="${ROBOTWIN_PATH}" \
            TASK_NAME="${TASK_NAME}" STEP_LIM="${STEP_LIM}" \
        bash "${REPO_PATH}/examples/embodiment/slurm/random10_h16_sft5k_eval_task.sbatch"
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
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR="/tmp/r10e_${SLURM_JOB_ID:-manual}"

CKPT_DIR="${REPO_PATH}/logs/20260620-23:01:03-random10_pi05_sft_h16_noadapt_8gpu/random10_pi05_sft_h16_noadapt/checkpoints/global_step_5000/actor"
DATASET_DIR="/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_random_10_500_pi05_h16"
NORM_STATS="${DATASET_DIR}/norm_stats.json"
SEEDS_PATH="${REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json"

cd "${REPO_PATH}"
mkdir -p "${RAY_TMPDIR}" "${REPO_PATH}/eval_logs" "${REPO_PATH}/slurm_logs"

[[ -s "${CKPT_DIR}/model_state_dict/full_weights.pt" ]] || { echo "missing 5k checkpoint"; exit 1; }
[[ -s "${NORM_STATS}" ]] || { echo "missing norm stats"; exit 1; }
python3 - "${SEEDS_PATH}" "${TASK_NAME}" <<'PY'
import json
import sys

path, task = sys.argv[1:]
data = json.load(open(path))
if task not in data:
    raise SystemExit(f"{task} is absent from eval_seeds.json; refusing random fallback")
seeds = data[task].get("success_seeds")
if not seeds:
    raise SystemExit(f"{task} has no success_seeds; refusing random fallback")
print(f"fixed eval seeds: task={task}, available={len(seeds)}")
PY

mkdir -p "${CKPT_DIR}/physical-intelligence/robotwin_random10_h16"
ln -sfn "${NORM_STATS}" "${CKPT_DIR}/physical-intelligence/robotwin_random10_h16/norm_stats.json"

STAMP=$(date +'%Y%m%d-%H%M%S')
LOG_DIR="${REPO_PATH}/eval_logs/${STAMP}-random10-h16-sft5k-${TASK_NAME}-n56"
mkdir -p "${LOG_DIR}"

ray stop --force 2>/dev/null || true
ray start --head --temp-dir="${RAY_TMPDIR}" --num-cpus=64 --num-gpus=8

python examples/embodiment/eval_embodied_agent.py \
    --config-path "${EMBODIED_PATH}/config/" \
    --config-name random10_h16_sft5k_eval \
    runner.logger.log_path="${LOG_DIR}" \
    runner.logger.experiment_name="random10_h16_sft5k_${TASK_NAME}" \
    env.train.task_config.task_name="${TASK_NAME}" \
    env.eval.task_config.task_name="${TASK_NAME}" \
    env.train.task_config.step_lim="${STEP_LIM}" \
    env.eval.task_config.step_lim="${STEP_LIM}" \
    env.train.max_episode_steps="${STEP_LIM}" \
    env.eval.max_episode_steps="${STEP_LIM}" \
    env.train.max_steps_per_rollout_epoch="${STEP_LIM}" \
    env.eval.max_steps_per_rollout_epoch="${STEP_LIM}" \
    2>&1 | tee "${LOG_DIR}/run_eval.log"

ray stop
