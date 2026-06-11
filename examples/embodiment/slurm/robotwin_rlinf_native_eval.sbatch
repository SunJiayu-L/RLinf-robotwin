#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -J rw-native-eval
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/robotwin_rlinf_native_eval_%x-%j.out
#SBATCH --time=6:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18

# 必传环境变量：
#   TASK_NAME      e.g. adjust_bottle
#   CKPT_PATH      full_weights.pt 路径
#   CKPT_SETTING   标识字符串，用于结果目录命名
# 可选：
#   NUM_STEPS      默认 10
#   SEED           默认 0
#   TASK_CONFIG    默认 demo_clean_no_video

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
        --pwd "${ROBOTWIN_PATH}" \
        "${SIF_IMAGE}" \
        env IN_RLINF_ROBOTWIN_CONTAINER=1 \
            HOME="${HOME}" \
            SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
            SLURM_NODELIST="${SLURM_NODELIST:-}" \
            CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
            REPO_PATH="${REPO_PATH}" \
            ROBOTWIN_PATH="${ROBOTWIN_PATH}" \
            TASK_NAME="${TASK_NAME}" \
            CKPT_PATH="${CKPT_PATH}" \
            CKPT_SETTING="${CKPT_SETTING}" \
            NUM_STEPS="${NUM_STEPS:-10}" \
            SEED="${SEED:-0}" \
            TASK_CONFIG="${TASK_CONFIG:-demo_clean_no_video}" \
        bash "${REPO_PATH}/examples/embodiment/slurm/robotwin_rlinf_native_eval.sbatch"
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
# 禁用 Python stdout 缓冲，使 Success!/Fail!/Success rate 实时写入 log
export PYTHONUNBUFFERED=1

: "${TASK_NAME:?TASK_NAME must be set}"
: "${CKPT_PATH:?CKPT_PATH must be set}"
: "${CKPT_SETTING:?CKPT_SETTING must be set}"

NUM_STEPS=${NUM_STEPS:-10}
SEED=${SEED:-0}
TASK_CONFIG=${TASK_CONFIG:-demo_clean_no_video}

echo "=== RoboTwin Native Eval ==="
echo "  TASK:         ${TASK_NAME}"
echo "  CKPT_SETTING: ${CKPT_SETTING}"
echo "  CKPT_PATH:    ${CKPT_PATH}"
echo "  NUM_STEPS:    ${NUM_STEPS}"
echo "  TASK_CONFIG:  ${TASK_CONFIG}"
echo "============================"

cd "${ROBOTWIN_PATH}"

python -u script/eval_policy.py \
    --config "${ROBOTWIN_PATH}/policy/pi05_rlinf/deploy_policy.yml" \
    --overrides \
    task_name "${TASK_NAME}" \
    task_config "${TASK_CONFIG}" \
    ckpt_setting "${CKPT_SETTING}" \
    policy_name pi05_rlinf \
    seed "${SEED}" \
    ckpt_path "${CKPT_PATH}" \
    num_steps "${NUM_STEPS}"
