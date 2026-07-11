#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J rw15-sft-resume
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/robotwin15_pi05_sft_8gpu-%j.out
#SBATCH --time=48:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18

set -eo pipefail

SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}

export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin

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
        bash "${REPO_PATH}/examples/sft/slurm/robotwin15_pi05_sft_8gpu_resume15k.sbatch"
fi

# === inside container from here ===

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"

cd "${REPO_PATH}"

# Resume into the original log dir so the 20k checkpoint lands at the path PPO config expects
LOG_DIR="${REPO_PATH}/logs/20260614-02:45:23-robotwin15_sft_pi05_8gpu"
RESUME_DIR="${LOG_DIR}/robotwin15_sft_openpi_pi05/checkpoints/global_step_15000"

ray stop --force 2>/dev/null || true
ray start --head --num-cpus=64 --num-gpus=8

python examples/sft/train_vla_sft.py \
    --config-path "${REPO_PATH}/examples/sft/config" \
    --config-name robotwin15_sft_openpi_pi05 \
    runner.logger.log_path="${LOG_DIR}" \
    runner.resume_dir="${RESUME_DIR}" 2>&1 | tee "${LOG_DIR}/sft_resume.log"

ray stop
