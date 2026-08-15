#!/usr/bin/env bash
# Resume stage-2 handover_block PPO from step 25 with a lower environment
# count to avoid the EnvWorker CPU-memory OOM seen with 256 environments.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <image> [job-name]" >&2
    exit 2
fi

IMAGE="$1"
JOB_NAME="${2:-handover-block-stage2-r25-128env-8g-$(date +%m%d-%H%M)}"
NODE_TAG="${NODE_TAG:-gpu-l-lg-cmc-h-h200-0260.host.h.pjlab.org.cn}"

case "$NODE_TAG" in
    gpu-l-lg-cmc-h-h200-0256.host.h.pjlab.org.cn|\
    gpu-l-lg-cmc-h-h200-0257.host.h.pjlab.org.cn|\
    gpu-l-lg-cmc-h-h200-0258.host.h.pjlab.org.cn|\
    gpu-l-lg-cmc-h-h200-0260.host.h.pjlab.org.cn)
        ;;
    *)
        echo "NODE_TAG must be one of the four approved H200 nodes." >&2
        exit 2
        ;;
esac

if [[ ! "$JOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
    echo "Job name must contain only lowercase letters, digits, and hyphens." >&2
    exit 2
fi

CONTAINER_GPFS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin"
HOST_GPFS_ROOT="/mnt/ailab-eailabagent-gpfs/chenzanxin"
RESUME_REL="rlinf-robotwin-runs/handover-block-stage2-kl300-256env-8g-n0258-0809-1145/handover_block_stage2_kl300_256env_8gpu/checkpoints/global_step_25"
RESUME_CKPT="${CONTAINER_GPFS_ROOT}/${RESUME_REL}"
HOST_RESUME_CKPT="${HOST_GPFS_ROOT}/${RESUME_REL}"
RUN_ROOT="${CONTAINER_GPFS_ROOT}/rlinf-robotwin-runs/${JOB_NAME}"
HOST_RUN_ROOT="${HOST_GPFS_ROOT}/rlinf-robotwin-runs/${JOB_NAME}"
REF_WEIGHTS="/mnt/shared-storage-user/chenzanxin/jiayusun/five_task_ppo_step100/full_weights.pt"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
export NO_PROXY="$no_proxy"

test -f "${HOST_RESUME_CKPT}/actor/dcp_checkpoint/.metadata"
test -f "${HOST_RESUME_CKPT}/actor/model_state_dict/full_weights.pt"
test -f "$REF_WEIGHTS"
mkdir -p "$HOST_RUN_ROOT"

rjob submit \
    --name="$JOB_NAME" \
    --task-type=normal \
    --charged-group=eailabagent_gpu \
    --private-machine=group \
    --gpu=8 \
    --cpu=64 \
    --memory=1400000 \
    --image="$IMAGE" \
    --mount=gpfs://gpfs1/chenzanxin:/mnt/shared-storage-user/chenzanxin \
    --mount=gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs \
    --positive-tags "node/${NODE_TAG}" \
    -e DISTRIBUTED_JOB=true \
    -e RLINF_RUN_ROOT="$RUN_ROOT" \
    -e RESUME_CKPT="$RESUME_CKPT" \
    -e REF_WEIGHTS="$REF_WEIGHTS" \
    -- bash -exc '
        set -euo pipefail
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
        export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
        export NO_PROXY="$no_proxy"
        export REPO_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RLinf-robotwin
        export ROBOTWIN_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RoboTwin-RLinf
        export VENV_PATH=/opt/venvs/openpi-robotwin-cu124
        source "${VENV_PATH}/bin/activate"
        export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
        export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
        export ROBOT_PLATFORM=ALOHA
        export MUJOCO_GL=egl
        export PYOPENGL_PLATFORM=egl
        export VK_DRIVER_FILES="${REPO_PATH}/nvidia_icd.json"
        export VK_ICD_FILENAMES="${REPO_PATH}/nvidia_icd.json"
        export TORCH_DISTRIBUTED_DEBUG=OFF
        export RLINF_NODE_RANK=0
        export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

        test -f "${RESUME_CKPT}/actor/dcp_checkpoint/.metadata"
        test -f "${RESUME_CKPT}/actor/model_state_dict/full_weights.pt"
        test -f "$REF_WEIGHTS"
        cd "$REPO_PATH"
        ray stop --force 2>/dev/null || true
        ray start --head --num-cpus=64 --num-gpus=8
        trap "status=\$?; ray stop --force 2>/dev/null || true; exit \$status" EXIT
        mkdir -p "$RLINF_RUN_ROOT"

        CMD=(python "${REPO_PATH}/examples/embodiment/train_embodied_agent.py"
             --config-path "${EMBODIED_PATH}/config/"
             --config-name moe_sft20k_handover_block_ppo150_128env
             "runner.logger.log_path=${RLINF_RUN_ROOT}"
             "runner.logger.experiment_name=handover_block_stage2_resume25_128env_8gpu"
             "runner.resume_dir=${RESUME_CKPT}"
             "runner.ckpt_path=null"
             "runner.max_steps=150"
             "runner.val_check_interval=25"
             "runner.save_interval=25"
             "env.train.total_num_envs=128"
             "env.eval.total_num_envs=32"
             "env.train.task_config.clear_cache_freq=1"
             "env.eval.task_config.clear_cache_freq=1"
             "algorithm.rollout_epoch=4"
             "algorithm.eval_rollout_epoch=3"
             "algorithm.update_epoch=3"
             "algorithm.kl_penalty=low_var_kl"
             "algorithm.kl_beta=0.0"
             "+algorithm.ref_kl_beta=0.001"
             "+algorithm.ref_policy_path=${REF_WEIGHTS}"
             "algorithm.sampling_params.temperature_train=0.8"
             "algorithm.sampling_params.temperature_eval=-1.0"
             "actor.global_batch_size=2048"
             "actor.micro_batch_size=32"
             "actor.optim.lr=2.0e-6"
             "actor.optim.value_lr=5.0e-5"
             "weight_syncer.patch.init_sync.enabled=true"
             "env.train.video_cfg.save_video=False"
             "env.eval.video_cfg.save_video=False"
             "env.train.task_config.eval_video_log=false"
             "env.eval.task_config.eval_video_log=false"
             "env.train.task_config.collect_data=false"
             "env.eval.task_config.collect_data=false"
             "env.train.task_config.save_freq=null"
             "env.eval.task_config.save_freq=null")

        printf "%q " "${CMD[@]}" > "${RLINF_RUN_ROOT}/run_embodiment.log"
        printf "\n" >> "${RLINF_RUN_ROOT}/run_embodiment.log"
        "${CMD[@]}" 2>&1 | tee -a "${RLINF_RUN_ROOT}/run_embodiment.log"
    '
