#!/usr/bin/env bash
# Re-evaluate Stage3 handover_block checkpoints after the PPO job releases node 0258.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <image> [job-name]" >&2
    exit 2
fi

IMAGE="$1"
JOB_NAME="${2:-eval-hb-stage3-s25-75-96ep-4g-$(date +%m%d-%H%M)}"
NODE_TAG="gpu-l-lg-cmc-h-h200-0258.host.h.pjlab.org.cn"
CONTAINER_GPFS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin"
HOST_GPFS_ROOT="/mnt/ailab-eailabagent-gpfs/chenzanxin"
CKPT_REL="rlinf-robotwin-runs/handover-block-stage3-s125-lr1e6-128env-8g-n0258-0811-2324/handover_block_stage3_from125_lr1e6_128env_8gpu/checkpoints"
CKPT_ROOT="${CONTAINER_GPFS_ROOT}/${CKPT_REL}"
HOST_CKPT_ROOT="${HOST_GPFS_ROOT}/${CKPT_REL}"
RUN_ROOT="${CONTAINER_GPFS_ROOT}/rlinf-robotwin-runs/${JOB_NAME}"
HOST_RUN_ROOT="${HOST_GPFS_ROOT}/rlinf-robotwin-runs/${JOB_NAME}"

for STEP in 25 50; do
    test -f "${HOST_CKPT_ROOT}/global_step_${STEP}/actor/model_state_dict/full_weights.pt"
done
mkdir -p "$HOST_RUN_ROOT"

source /etc/profile.d/ssh-init.sh >/dev/null 2>&1 || true
rjob submit \
    --name="$JOB_NAME" \
    --task-type=normal \
    --charged-group=eailabagent_gpu \
    --private-machine=group \
    --gpu=4 \
    --cpu=32 \
    --memory=500000 \
    --image="$IMAGE" \
    --mount=gpfs://gpfs1/chenzanxin:/mnt/shared-storage-user/chenzanxin \
    --mount=gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs \
    --positive-tags "node/${NODE_TAG}" \
    -e RLINF_RUN_ROOT="$RUN_ROOT" \
    -e CKPT_ROOT="$CKPT_ROOT" \
    -- bash -exc '
        set -uo pipefail
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
        export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
        export NO_PROXY="$no_proxy"
        export REPO_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RLinf-robotwin
        export ROBOTWIN_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RoboTwin-RLinf
        source /opt/venvs/openpi-robotwin-cu124/bin/activate
        export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}"
        export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
        export ROBOT_PLATFORM=ALOHA
        export MUJOCO_GL=egl
        export PYOPENGL_PLATFORM=egl
        export VK_DRIVER_FILES="${REPO_PATH}/nvidia_icd.json"
        export VK_ICD_FILENAMES="${REPO_PATH}/nvidia_icd.json"
        export RLINF_NODE_RANK=0
        export PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:${PATH}"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"

        cd "$REPO_PATH"
        mkdir -p "$RLINF_RUN_ROOT"
        printf "step\tsuccess_once\tnum_trajectories\tepisode_len\tstatus\n" > "$RLINF_RUN_ROOT/summary.tsv"
        FAILED=0

        STEP75="${CKPT_ROOT}/global_step_75/actor/model_state_dict/full_weights.pt"
        for _ in $(seq 1 120); do
            test -f "$STEP75" && break
            sleep 30
        done
        if ! test -f "$STEP75"; then
            echo "Stage3 global_step_75 full weights did not appear" >&2
            exit 4
        fi

        for STEP in 25 50 75; do
            CKPT="${CKPT_ROOT}/global_step_${STEP}/actor/model_state_dict/full_weights.pt"
            STEP_DIR="${RLINF_RUN_ROOT}/step_${STEP}"
            LOG_FILE="${STEP_DIR}/run_embodiment.log"
            mkdir -p "$STEP_DIR"
            test -f "$CKPT"

            echo "=== Evaluating Stage3 global_step_${STEP} in a fresh process ===" | tee -a "$RLINF_RUN_ROOT/run_embodiment.log"
            ray stop --force 2>/dev/null || true
            ray start --head --num-cpus=32 --num-gpus=4

            if python "${REPO_PATH}/examples/embodiment/eval_embodied_agent.py" \
                --config-path "${EMBODIED_PATH}/config/" \
                --config-name eval_kl001_s300_hb \
                "runner.logger.log_path=${STEP_DIR}" \
                "runner.logger.experiment_name=eval_hb_stage3_step_${STEP}_96ep" \
                "runner.ckpt_path=${CKPT}" \
                "algorithm.eval_rollout_epoch=3" \
                "algorithm.sampling_params.temperature_eval=-1.0" \
                "env.eval.total_num_envs=32" \
                "env.eval.video_cfg.save_video=False" \
                "env.eval.task_config.eval_video_log=false" \
                "env.eval.task_config.collect_data=false" \
                "env.eval.task_config.save_freq=null" \
                2>&1 | tee "$LOG_FILE"; then
                if python - "$STEP" "$LOG_FILE" "$RLINF_RUN_ROOT/summary.tsv" <<'PY'
import re
import sys

step, log_path, summary_path = sys.argv[1:]
text = open(log_path, encoding="utf-8", errors="replace").read()
lines = [line for line in text.splitlines() if "eval/success_once" in line]
if not lines:
    raise SystemExit("missing eval metrics")
line = lines[-1]

def extract(pattern, name):
    match = re.search(pattern, line)
    if not match:
        raise SystemExit(f"missing {name}: {line}")
    return match.group(1)

success = float(extract(r"eval/success_once[^\n]*?array\(([0-9.eE+-]+)", "success_once"))
trajectories = int(extract(r"eval/num_trajectories[^\n]*?:\s*([0-9]+)", "num_trajectories"))
episode_len = float(extract(r"eval/episode_len[^\n]*?array\(([0-9.eE+-]+)", "episode_len"))
status = "valid" if trajectories == 96 and abs(episode_len - 400.0) < 1e-4 else "invalid"
with open(summary_path, "a", encoding="utf-8") as stream:
    stream.write(f"{step}\t{success:.8f}\t{trajectories}\t{episode_len:.4f}\t{status}\n")
if status != "valid":
    raise SystemExit(3)
PY
                then
                    echo "global_step_${STEP}: valid 96-episode evaluation" | tee -a "$RLINF_RUN_ROOT/run_embodiment.log"
                else
                    FAILED=1
                    echo "global_step_${STEP}: metric validation failed" | tee -a "$RLINF_RUN_ROOT/run_embodiment.log"
                fi
            else
                FAILED=1
                printf "%s\tNA\tNA\tNA\tfailed\n" "$STEP" >> "$RLINF_RUN_ROOT/summary.tsv"
            fi
            ray stop --force 2>/dev/null || true
        done

        cat "$RLINF_RUN_ROOT/summary.tsv" | tee -a "$RLINF_RUN_ROOT/run_embodiment.log"
        exit "$FAILED"
    '
