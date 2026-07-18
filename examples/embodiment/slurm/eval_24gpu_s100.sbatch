#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J eval-24gpu-s100
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/eval_24gpu_s100-%j.out
#SBATCH --time=48:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-20,SH-IDCA1404-10-140-54-21,SH-IDCA1404-10-140-54-25,SH-IDCA1404-10-140-54-28,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-35,SH-IDCA1404-10-140-54-37,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-45,SH-IDCA1404-10-140-54-57,SH-IDCA1404-10-140-54-61,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-73,SH-IDCA1404-10-140-54-79,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-118

set -eo pipefail
SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
export ROBOTWIN_PATH=/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf

if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec --cleanenv --nv --bind /mnt:/mnt --bind /nvme:/nvme --pwd "${REPO_PATH}" "${SIF_IMAGE}" env IN_RLINF_ROBOTWIN_CONTAINER=1 HOME="${HOME}" SLURM_JOB_ID="${SLURM_JOB_ID:-}" SLURM_NODELIST="${SLURM_NODELIST:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" REPO_PATH="${REPO_PATH}" ROBOTWIN_PATH="${ROBOTWIN_PATH}" bash "${REPO_PATH}/examples/embodiment/slurm/eval_24gpu_s100.sbatch"
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

run_eval() {
    local cfg="$1"
    echo "===== Evaluating ${cfg} ====="
    ray stop --force 2>/dev/null || true
    sleep 5
    ray start --head --num-cpus=64 --num-gpus=8
    bash examples/embodiment/eval_embodiment.sh "${cfg}" ALOHA
    echo "===== Done: ${cfg} ====="
}

run_eval eval_24gpu_s100_bbh
run_eval eval_24gpu_s100_ca
run_eval eval_24gpu_s100_cb
run_eval eval_24gpu_s100_hb
run_eval eval_24gpu_s100_mpa

ray stop --force 2>/dev/null || true
echo "===== All step_100 evals done ====="
