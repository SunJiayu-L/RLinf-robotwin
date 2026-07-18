#!/bin/bash
exec "apptainer" exec --cleanenv --nv \
    --bind /mnt:/mnt --bind /nvme:/nvme \
    --pwd "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin" "/mnt/petrelfs/muyao/shared_img/new_app.sif" \
    env IN_RLINF_ROBOTWIN_CONTAINER=1 \
        HOME="/mnt/petrelfs/muyao/jiayusun" \
        SLURM_JOB_ID="10230874" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
        RLINF_NODE_RANK="${SLURM_PROCID}" \
        REPO_PATH="/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin" \
        ROBOTWIN_PATH="/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf" \
    bash "/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/examples/embodiment/slurm/moe_ppo_5task_24gpu_bbh_ca_cb_hb_mpa.sbatch"
