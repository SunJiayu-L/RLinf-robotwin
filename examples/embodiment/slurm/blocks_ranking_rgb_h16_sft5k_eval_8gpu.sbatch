#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J rgb5k-eval
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/blocks_ranking_rgb_h16_sft5k_eval_8gpu-%j.out
#SBATCH --time=12:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-1,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-117,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-58,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-79

set -euo pipefail

SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
export ROBOTWIN_PATH=/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf

if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec --cleanenv --nv --bind /mnt:/mnt --bind /nvme:/nvme --pwd "${REPO_PATH}" "${SIF_IMAGE}" env IN_RLINF_ROBOTWIN_CONTAINER=1 HOME="${HOME}" SLURM_JOB_ID="${SLURM_JOB_ID:-}" SLURM_NODELIST="${SLURM_NODELIST:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" REPO_PATH="${REPO_PATH}" ROBOTWIN_PATH="${ROBOTWIN_PATH}" bash "${REPO_PATH}/examples/embodiment/slurm/blocks_ranking_rgb_h16_sft5k_eval_8gpu.sbatch"
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
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export HYDRA_FULL_ERROR=1
export RAY_TMPDIR="/nvme/muyao/rgb5k_eval_${SLURM_JOB_ID:-manual}"

CKPT_DIR="${REPO_PATH}/logs/20260622-00:45:51-blocks_ranking_rgb_pi05_sft_h16_noadapt_8gpu/blocks_ranking_rgb_pi05_sft_h16_noadapt/checkpoints/global_step_5000/actor"
DATASET_DIR="/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_blocks_ranking_rgb_500_pi05_h16"
NORM_STATS="${DATASET_DIR}/norm_stats.json"
SEEDS_PATH="${REPO_PATH}/rlinf/envs/robotwin/seeds/blocks_ranking_rgb_manifest_96ep.json"
ASSET_ID="physical-intelligence/robotwin_blocks_ranking_rgb_h16"

cleanup_local_scratch() {
    ray stop --force 2>/dev/null || true
    case "${RAY_TMPDIR}" in
        /nvme/muyao/rgb5k_eval_*) rm -rf -- "${RAY_TMPDIR}" ;;
        *) echo "Refusing to clean unexpected path: ${RAY_TMPDIR}" ;;
    esac
}
trap cleanup_local_scratch EXIT

cd "${REPO_PATH}"
mkdir -p "${RAY_TMPDIR}" "${REPO_PATH}/eval_logs" "${REPO_PATH}/slurm_logs"

[[ -s "${CKPT_DIR}/model_state_dict/full_weights.pt" ]] || { echo "missing 5k checkpoint"; exit 1; }
[[ -s "${NORM_STATS}" ]] || { echo "missing norm stats"; exit 1; }
python3 - "${SEEDS_PATH}" <<'PY'
import json
import sys

data = json.load(open(sys.argv[1]))
task = "blocks_ranking_rgb"
episodes = data.get(task, {}).get("episodes")
if not episodes or len(episodes) != 96:
    raise SystemExit(f"{task}: expected a 96-episode manifest")
print(f"fixed eval manifest: task={task}, episodes={len(episodes)}")
PY

mkdir -p "${CKPT_DIR}/${ASSET_ID}"
ln -sfn "${NORM_STATS}" "${CKPT_DIR}/${ASSET_ID}/norm_stats.json"

STAMP=$(date +'%Y%m%d-%H%M%S')
LOG_DIR="${REPO_PATH}/eval_logs/${STAMP}-blocks-ranking-rgb-h16-sft5k-n96-step800"
mkdir -p "${LOG_DIR}"

ray stop --force 2>/dev/null || true
ray start --head --temp-dir="${RAY_TMPDIR}" --num-cpus=64 --num-gpus=8

python examples/embodiment/eval_embodied_agent.py \
    --config-path "${EMBODIED_PATH}/config/" \
    --config-name random10_h16_sft5k_eval \
    runner.logger.log_path="${LOG_DIR}" \
    runner.logger.experiment_name="blocks_ranking_rgb_h16_sft5k_n96_step800" \
    algorithm.eval_rollout_epoch=3 \
    actor.model.model_path="${CKPT_DIR}" \
    actor.model.openpi_data.repo_id="${DATASET_DIR}" \
    actor.model.openpi.config_name="pi05_aloha_robotwin_blocks_ranking_rgb_h16" \
    env.train.seeds_path="${SEEDS_PATH}" \
    env.eval.seeds_path="${SEEDS_PATH}" \
    env.train.total_num_envs=32 \
    env.eval.total_num_envs=32 \
    env.train.task_config.task_name="blocks_ranking_rgb" \
    env.eval.task_config.task_name="blocks_ranking_rgb" \
    env.train.task_config.step_lim=800 \
    env.eval.task_config.step_lim=800 \
    env.train.max_episode_steps=800 \
    env.eval.max_episode_steps=800 \
    env.train.max_steps_per_rollout_epoch=800 \
    env.eval.max_steps_per_rollout_epoch=800 \
    2>&1 | tee "${LOG_DIR}/run_eval.log"
