#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=96G
#SBATCH -J merge-bcam-040405
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/merge_5task_bcam_custom_0404040505-%j.out
#SBATCH --time=1:00:00

set -eo pipefail

SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin

if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec \
        --cleanenv --nv \
        --bind /mnt:/mnt --bind /nvme:/nvme \
        --pwd "${REPO_PATH}" "${SIF_IMAGE}" \
        env IN_RLINF_ROBOTWIN_CONTAINER=1 \
            HOME="${HOME}" \
            SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
            REPO_PATH="${REPO_PATH}" \
        bash "${REPO_PATH}/examples/embodiment/slurm/merge_5task_bcam_custom_0404040505.sbatch"
fi

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export PATH=$GCC_HOME/bin:$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO_PATH}:${PYTHONPATH:-}

cd "${REPO_PATH}"

OUTPUT=/mnt/petrelfs/muyao/jiayusun/merged_weights/moe_5task_bcam_custom_0404040505/full_weights.pt
if [[ -s "${OUTPUT}" ]]; then
    echo "[SKIP] exists: ${OUTPUT}"
    exit 0
fi

python scripts/task_arithmetic_rlinf.py \
  --base /mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full_pytorch/20000/model.safetensors \
  --ft \
    logs/20260707-22:22:21-moe_sft20k_beat_block_hammer_ppo100/moe_sft20k_beat_block_hammer_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt \
    logs/20260708-13:50:33-moe_sft20k_click_alarmclock_ppo100/moe_sft20k_click_alarmclock_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt \
    logs/20260708-13:56:28-moe_sft20k_click_bell_ppo100/moe_sft20k_click_bell_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt \
    logs/20260706-23:28:23-moe_sft20k_move_can_pot_ppo100/moe_sft20k_move_can_pot_ppo100/checkpoints/global_step_100/actor/model_state_dict/full_weights.pt \
    logs/20260706-11:28:57-moe_sft20k_move_playingcard_away_ppo100/moe_sft20k_move_playingcard_away_ppo100/checkpoints/global_step_50/actor/model_state_dict/full_weights.pt \
  --alphas 0.4 0.4 0.4 0.5 0.5 \
  --output "${OUTPUT}"

ls -lh "${OUTPUT}"
