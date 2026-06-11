#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH -J rw-native-eval-8g
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/robotwin_rlinf_native_eval_8gpu-%j.out
#SBATCH --time=8:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18

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
            REPO_PATH="${REPO_PATH}" \
            ROBOTWIN_PATH="${ROBOTWIN_PATH}" \
        bash "${REPO_PATH}/examples/embodiment/slurm/robotwin_rlinf_native_eval_8gpu.sbatch"
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
export PYTHONUNBUFFERED=1

SFT20K="/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260609-10:17:50-robotwin_sft_pi05_8gpu/robotwin_sft_openpi_pi05/checkpoints/global_step_20000/actor/model_state_dict/full_weights.pt"
SFT30K="/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/20260609-10:17:50-robotwin_sft_pi05_8gpu/robotwin_sft_openpi_pi05/checkpoints/global_step_30000/actor/model_state_dict/full_weights.pt"
MERGED="/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs/merged/sft20k_ppo150_adj_pcp_pec_a035/full_weights.pt"

LOG_DIR="${REPO_PATH}/slurm_logs/native_eval_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

cd "${ROBOTWIN_PATH}"

# 9 个评测任务：(GPU_ID, TASK, CKPT_PATH, CKPT_SETTING)
run_eval() {
    local GPU_ID=$1
    local TASK=$2
    local CKPT=$3
    local CKPT_SETTING=$4
    local LOGFILE="${LOG_DIR}/${TASK}_${CKPT_SETTING}.log"

    echo "[$(date '+%H:%M:%S')] START GPU${GPU_ID}: ${TASK} | ${CKPT_SETTING}" | tee -a "${LOG_DIR}/summary.log"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u script/eval_policy.py \
        --config "${ROBOTWIN_PATH}/policy/pi05_rlinf/deploy_policy.yml" \
        --overrides \
        task_name "${TASK}" \
        task_config "demo_clean_no_video" \
        ckpt_setting "${CKPT_SETTING}" \
        policy_name pi05_rlinf \
        seed 0 \
        ckpt_path "${CKPT}" \
        num_steps 10 \
        > "${LOGFILE}" 2>&1

    local EXIT_CODE=$?
    # 提取最终成功率
    local FINAL=$(grep "Success rate:" "${LOGFILE}" | tail -1)
    echo "[$(date '+%H:%M:%S')] DONE  GPU${GPU_ID}: ${TASK} | ${CKPT_SETTING} | ${FINAL}" | tee -a "${LOG_DIR}/summary.log"
    return ${EXIT_CODE}
}

# 并行启动 9 个任务，GPU 0-7 循环分配
run_eval 0 adjust_bottle          "${SFT20K}"  sft20k       &
run_eval 1 adjust_bottle          "${SFT30K}"  sft30k       &
run_eval 2 adjust_bottle          "${MERGED}"  merged_a035  &
run_eval 3 place_container_plate  "${SFT20K}"  sft20k       &
run_eval 4 place_container_plate  "${SFT30K}"  sft30k       &
run_eval 5 place_container_plate  "${MERGED}"  merged_a035  &
run_eval 6 place_empty_cup        "${SFT20K}"  sft20k       &
run_eval 7 place_empty_cup        "${SFT30K}"  sft30k       &

# 等待前 8 个完成后，用 GPU 0 跑第 9 个（wait || true 防止某个任务失败时 set -e 终止脚本）
wait || true
run_eval 0 place_empty_cup        "${MERGED}"  merged_a035

echo ""
echo "===== ALL DONE ====="
echo "Individual logs: ${LOG_DIR}/"
echo "Summary:"
cat "${LOG_DIR}/summary.log"
