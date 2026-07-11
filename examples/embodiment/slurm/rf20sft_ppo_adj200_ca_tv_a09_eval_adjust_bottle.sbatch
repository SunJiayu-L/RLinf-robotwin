#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -J a09-adj
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/rf20sft_ppo_adj200_ca_tv_a09_eval_adjust_bottle-%j.out
#SBATCH --time=4:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-117,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-58,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-7,SH-IDCA1404-10-140-54-46,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-36,SH-IDCA1404-10-140-54-59,SH-IDCA1404-10-140-54-50,SH-IDCA1404-10-140-54-25

set -eo pipefail
SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
export ROBOTWIN_PATH=/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf

if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec --cleanenv --nv         --bind /mnt:/mnt --bind /nvme:/nvme         --pwd "${REPO_PATH}" "${SIF_IMAGE}"         env IN_RLINF_ROBOTWIN_CONTAINER=1             HOME="${HOME}" SLURM_JOB_ID="${SLURM_JOB_ID:-}"             SLURM_NODELIST="${SLURM_NODELIST:-}"             CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"             REPO_PATH="${REPO_PATH}" ROBOTWIN_PATH="${ROBOTWIN_PATH}"         bash "${REPO_PATH}/examples/embodiment/slurm/rf20sft_ppo_adj200_ca_tv_a09_eval_adjust_bottle.sbatch"
fi

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export ROBOT_PLATFORM=ALOHA; export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl
export VK_DRIVER_FILES="${REPO_PATH}/nvidia_icd.json"
export VK_ICD_FILENAMES="${REPO_PATH}/nvidia_icd.json"
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}
cd "${REPO_PATH}"
ray stop --force 2>/dev/null || true
ray start --head --num-cpus=32 --num-gpus=4
bash examples/embodiment/eval_embodiment.sh rf20sft_ppo_adj200_ca_tv_a09_eval_adjust_bottle ALOHA
ray stop
