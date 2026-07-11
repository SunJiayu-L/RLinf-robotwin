#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -J hidden-state-analysis
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/analyze_hidden_states-%j.out
#SBATCH --time=2:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-117,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-58,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-7,SH-IDCA1404-10-140-54-46,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-36,SH-IDCA1404-10-140-54-59,SH-IDCA1404-10-140-54-50,SH-IDCA1404-10-140-54-25

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
        bash "${REPO_PATH}/examples/embodiment/slurm/analyze_hidden_states.sbatch"
fi

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}

export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}

cd "${REPO_PATH}"

RF20SFT_WEIGHTS="/mnt/petrelfs/muyao/jiayusun/random_first20_pi05_sft/full_weights.pt"
CA_SFT15K_WEIGHTS="logs/20260617-20:39:17-rf20_click_alarmclock_sft_8gpu/rf20_click_alarmclock_sft_pi05/checkpoints/global_step_15000/actor/model_state_dict/full_weights.pt"
MERGED_A07_WEIGHTS="/mnt/petrelfs/muyao/jiayusun/merged_weights/ppo_adj200_plus_ca100_tv_a07/full_weights.pt"
PPO_STEP50_WEIGHTS="logs/20260619-16:05:12-merged_a07_ppo_adjust_bottle/merged_a07_ppo_adjust_bottle/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt"
PPO_CA_WEIGHTS="logs/20260617-13:27:58-rf20sft_ppo_click_alarmclock/rf20sft_ppo_click_alarmclock/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt"
PPO_ADJ_WEIGHTS="logs/20260615-15:45:43-robotwin15_adjust_bottle_ppo_random20/robotwin15_ppo_adjust_bottle_random20/checkpoints/global_step_200/actor/model_state_dict/full_weights.pt"
ADJ_SFT15K_WEIGHTS="logs/20260617-20:00:24-rf20_adjust_bottle_sft_8gpu/rf20_adjust_bottle_sft_pi05/checkpoints/global_step_15000/actor/model_state_dict/full_weights.pt"

echo "===== 开始隐藏状态分析 ====="
echo "节点: $(hostname), GPU: ${CUDA_VISIBLE_DEVICES:-all}"
echo "时间: $(date)"

python scripts/analyze_hidden_states.py \
    --rf20sft      "${RF20SFT_WEIGHTS}" \
    --ca-sft15k    "${CA_SFT15K_WEIGHTS}" \
    --adj-sft15k   "${ADJ_SFT15K_WEIGHTS}" \
    --merged-a07   "${MERGED_A07_WEIGHTS}" \
    --ppo-step50   "${PPO_STEP50_WEIGHTS}" \
    --ppo-ca       "${PPO_CA_WEIGHTS}" \
    --ppo-adj      "${PPO_ADJ_WEIGHTS}" \
    --output-dir results/hidden_states/ \
    --device cuda \
    --batch-size 4

echo "===== 完成 ====="
echo "时间: $(date)"
