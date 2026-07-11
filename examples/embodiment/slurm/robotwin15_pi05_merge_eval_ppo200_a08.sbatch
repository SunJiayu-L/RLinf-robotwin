#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J rw15-merge-eval-a08
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/robotwin15_pi05_merge_eval_ppo200_a08-%j.out
#SBATCH --time=4:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-69

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
        bash "${REPO_PATH}/examples/embodiment/slurm/robotwin15_pi05_merge_eval_ppo200_a08.sbatch"
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

cd "${REPO_PATH}"

BASE_WEIGHTS="/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260614-02:45:23-robotwin15_sft_pi05_8gpu/robotwin15_sft_openpi_pi05/checkpoints/global_step_20000/actor/model_state_dict/full_weights.pt"
PPO_WEIGHTS="/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260615-15:12:15-robotwin15_adjust_bottle_ppo_openpi_pi05/robotwin15_ppo_adjust_bottle_openpi_pi05/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt"
MERGED_OUTPUT="/mnt/petrelfs/muyao/jiayusun/merged_weights/sft20k_ppo200_a08_adjust_bottle/full_weights.pt"

echo "=== Step 1: Weight merging (alpha=0.8) ==="
python scripts/task_arithmetic_rlinf.py \
    --base "${BASE_WEIGHTS}" \
    --ft   "${PPO_WEIGHTS}" \
    --output "${MERGED_OUTPUT}" \
    --alpha 0.8

echo "=== Step 2: Eval ==="
ray stop --force 2>/dev/null || true
ray start --head --num-cpus=64 --num-gpus=8

bash examples/embodiment/eval_embodiment.sh robotwin15_adjust_bottle_ppo200_a08_eval ALOHA

ray stop
