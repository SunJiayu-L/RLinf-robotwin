#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J moe-5task-24gpu
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/moe_ppo_5task_24gpu_bbh_ca_cb_hb_mpa-%j.out
#SBATCH --time=168:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-20,SH-IDCA1404-10-140-54-21,SH-IDCA1404-10-140-54-28,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-35,SH-IDCA1404-10-140-54-37,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-45,SH-IDCA1404-10-140-54-57,SH-IDCA1404-10-140-54-61,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-73,SH-IDCA1404-10-140-54-79,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-118

set -eo pipefail

SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
export ROBOTWIN_PATH=/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf

# ── Inside-container entrypoint ───────────────────────────────────────────────
if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" == "1" ]]; then
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

    RAY_HEAD_IP_FILE="${REPO_PATH}/ray_utils/ray_head_ip_${SLURM_JOB_ID}.txt"
    TRAINING_DONE_FILE="${REPO_PATH}/ray_utils/training_done_${SLURM_JOB_ID}.flag"
    RAY_PORT=6379

    if [[ "${RLINF_NODE_RANK:-0}" == "0" ]]; then
        # ── Head node: start Ray head, run training ───────────────────────────
        HEAD_IP=$(hostname -I | awk '{print $1}')
        echo "Head node: rank=0 ip=${HEAD_IP}"
        rm -f "$RAY_HEAD_IP_FILE" "$TRAINING_DONE_FILE"
        echo "$HEAD_IP" > "$RAY_HEAD_IP_FILE"

        ray stop --force 2>/dev/null || true
        ray start --head --port=$RAY_PORT --node-ip-address="$HEAD_IP" \
            --num-cpus=64 --num-gpus=8
        echo "Ray head started; waiting 20s for workers to join..."
        sleep 20

        bash examples/embodiment/run_embodiment.sh \
            moe_ppo_5task_24gpu_bbh_ca_cb_hb_mpa ALOHA
        echo "Training complete; signalling workers..."
        touch "$TRAINING_DONE_FILE"
        sleep 15
        ray stop --force 2>/dev/null || true
        rm -f "$RAY_HEAD_IP_FILE" "$TRAINING_DONE_FILE"
    else
        # ── Worker node: join Ray cluster, wait for head to finish ────────────
        echo "Worker node: rank=${RLINF_NODE_RANK}, waiting for head IP..."
        HEAD_IP=""
        for i in $(seq 1 120); do
            if [[ -f "$RAY_HEAD_IP_FILE" ]]; then
                HEAD_IP=$(cat "$RAY_HEAD_IP_FILE")
                [[ -n "$HEAD_IP" ]] && break
            fi
            sleep 2
        done
        if [[ -z "$HEAD_IP" ]]; then
            echo "ERROR: head IP not available after 240s"
            exit 1
        fi
        echo "Worker node: connecting to head at ${HEAD_IP}:${RAY_PORT}"
        ray stop --force 2>/dev/null || true
        ray start --address="${HEAD_IP}:${RAY_PORT}" --num-cpus=64 --num-gpus=8

        while [[ ! -f "$TRAINING_DONE_FILE" ]]; do
            sleep 10
        done
        echo "Worker node: training done flag detected, exiting."
        ray stop --force 2>/dev/null || true
    fi
    exit 0
fi

# ── Outside container: write per-node launcher, srun on all 3 nodes ──────────
LAUNCH_SCRIPT="${REPO_PATH}/ray_utils/launch_node_${SLURM_JOB_ID:-local}.sh"
cat > "${LAUNCH_SCRIPT}" << EOF
#!/bin/bash
exec "${APPTAINER_BIN}" exec --cleanenv --nv \\
    --bind /mnt:/mnt --bind /nvme:/nvme \\
    --pwd "${REPO_PATH}" "${SIF_IMAGE}" \\
    env IN_RLINF_ROBOTWIN_CONTAINER=1 \\
        HOME="${HOME}" \\
        SLURM_JOB_ID="${SLURM_JOB_ID:-}" \\
        CUDA_VISIBLE_DEVICES="\${CUDA_VISIBLE_DEVICES:-}" \\
        RLINF_NODE_RANK="\${SLURM_PROCID}" \\
        REPO_PATH="${REPO_PATH}" \\
        ROBOTWIN_PATH="${ROBOTWIN_PATH}" \\
    bash "${REPO_PATH}/examples/embodiment/slurm/moe_ppo_5task_24gpu_bbh_ca_cb_hb_mpa.sbatch"
EOF
chmod +x "${LAUNCH_SCRIPT}"

srun --ntasks-per-node=1 bash "${LAUNCH_SCRIPT}"
rm -f "${LAUNCH_SCRIPT}"
