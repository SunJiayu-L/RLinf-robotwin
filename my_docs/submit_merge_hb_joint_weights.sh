#!/usr/bin/env bash
# Produce 9 merged-weight candidates from handover_block PPO (step 400, no-KL)
# and 5-task joint PPO (step 100), using scripts/task_arithmetic_rlinf.py.
#
# Family A (mode 1, SFT base, two task vectors):
#   mode1-a03, mode1-a05, mode1-a07, mode1-hb07-joint04
# Family B (hb<->joint line, single invocation form, base=hb):
#   hbline-r-025, hbline-r025, hbline-r050, hbline-r075, hbline-r125
#
# CPU-only job (no GPU needed for tensor arithmetic). Idempotent: skips a
# candidate if its output file already exists and is non-empty.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 <image> [job-name]" >&2
    exit 2
fi

IMAGE="$1"
JOB_NAME="${2:-wm-merge-hb-joint-$(date +%m%d-%H%M)}"
GPFS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin"
OUT_ROOT="${GPFS_ROOT}/merged_weights/wm-hb-joint-$(date +%m%d)"
RUN_ROOT="${GPFS_ROOT}/rlinf-robotwin-runs/${JOB_NAME}"

SFT="/mnt/shared-storage-user/chenzanxin/jiayusun/sft20k/model.safetensors"
HB_PPO="${GPFS_ROOT}/rlinf-robotwin-runs/handover-block-ppo250resume-novideo-4g-m1400-0803-1155/handover_block_ppo250resume_novideo_128env_4gpu/checkpoints/global_step_400/actor/model_state_dict/full_weights.pt"
JOINT_PPO="/mnt/shared-storage-user/chenzanxin/jiayusun/five_task_ppo_step100/full_weights.pt"

if [[ ! "$JOB_NAME" =~ ^[a-z0-9]([-a-z0-9]*[a-z0-9])?$ ]]; then
    echo "Job name must contain only lowercase letters, digits, and hyphens." >&2
    exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
export NO_PROXY="$no_proxy"
mkdir -p "$RUN_ROOT"

rjob submit \
    --name="$JOB_NAME" \
    --task-type=normal \
    --charged-group=eailabagent_gpu \
    --private-machine=group \
    --gpu=1 \
    --cpu=16 \
    --memory=131072 \
    --image="$IMAGE" \
    --mount=gpfs://gpfs1/chenzanxin:/mnt/shared-storage-user/chenzanxin \
    --mount=gpfs://gpfs2/ailab-eailabagent-gpfs:/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs \
    -e DISTRIBUTED_JOB=true \
    -e RLINF_RUN_ROOT="$RUN_ROOT" \
    -- bash -exc '
        set -euo pipefail
        unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
        export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
        export NO_PROXY="$no_proxy"
        export REPO_PATH=/mnt/shared-storage-user/chenzanxin/jiayusun/RLinf-robotwin
        export VENV_PATH=/opt/venvs/openpi-robotwin-cu124
        source "${VENV_PATH}/bin/activate"
        export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
        cd "${REPO_PATH}"

        SFT="/mnt/shared-storage-user/chenzanxin/jiayusun/sft20k/model.safetensors"
        HB_PPO="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/handover-block-ppo250resume-novideo-4g-m1400-0803-1155/handover_block_ppo250resume_novideo_128env_4gpu/checkpoints/global_step_400/actor/model_state_dict/full_weights.pt"
        JOINT_PPO="/mnt/shared-storage-user/chenzanxin/jiayusun/five_task_ppo_step100/full_weights.pt"
        OUT_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/merged_weights/wm-hb-joint-$(date +%m%d)"

        test -f "${SFT}"
        test -f "${HB_PPO}"
        test -f "${JOINT_PPO}"
        mkdir -p "${RLINF_RUN_ROOT}"
        MANIFEST="${RLINF_RUN_ROOT}/MERGE_MANIFEST.md"

        echo "# Merge manifest ($(date -Iseconds))" > "${MANIFEST}"
        echo "" >> "${MANIFEST}"
        echo "- SFT base: ${SFT}" >> "${MANIFEST}"
        echo "- handover_block PPO (step400, no-KL): ${HB_PPO}" >> "${MANIFEST}"
        echo "- 5-task joint PPO (step100): ${JOINT_PPO}" >> "${MANIFEST}"
        echo "" >> "${MANIFEST}"

        echo "[Phase0] Verifying key sets match..." | tee -a "${RLINF_RUN_ROOT}/run_embodiment.log"
        python3 -c "
import torch
sft = torch.load(\"${SFT}\", map_location=\"cpu\") if not \"${SFT}\".endswith(\".safetensors\") else None
if sft is None:
    from safetensors.torch import load_file
    sft = load_file(\"${SFT}\")
hb = torch.load(\"${HB_PPO}\", map_location=\"cpu\")
joint = torch.load(\"${JOINT_PPO}\", map_location=\"cpu\")
assert set(hb.keys()) == set(joint.keys()), \"hb/joint key mismatch\"
print(f\"OK: sft={len(sft)} keys, hb={len(hb)} keys, joint={len(joint)} keys\")
del sft, hb, joint
" 2>&1 | tee -a "${RLINF_RUN_ROOT}/run_embodiment.log"

        run_merge() {
            local name="$1"; shift
            local out="${OUT_ROOT}/${name}/full_weights.pt"
            if [[ -s "${out}" ]]; then
                echo "[SKIP] ${name} already exists: ${out}" | tee -a "${RLINF_RUN_ROOT}/run_embodiment.log"
                echo "- SKIPPED (already existed): ${name} -> ${out}" >> "${MANIFEST}"
                return
            fi
            mkdir -p "$(dirname "${out}")"
            echo "=== Merging ${name} ===" | tee -a "${RLINF_RUN_ROOT}/run_embodiment.log"
            echo "CLI: python scripts/task_arithmetic_rlinf.py $*" | tee -a "${RLINF_RUN_ROOT}/run_embodiment.log"
            python scripts/task_arithmetic_rlinf.py "$@" --output "${out}" 2>&1 | tee -a "${RLINF_RUN_ROOT}/run_embodiment.log"
            local size
            size=$(stat -c%s "${out}")
            echo "- ${name}: python scripts/task_arithmetic_rlinf.py $* --output ${out} (size=${size} bytes)" >> "${MANIFEST}"
        }

        # Family A: SFT base, two task vectors (hb, joint)
        run_merge "mode1-a03"          --base "${SFT}" --ft "${HB_PPO}" "${JOINT_PPO}" --alphas 0.3 0.3
        run_merge "mode1-a05"          --base "${SFT}" --ft "${HB_PPO}" "${JOINT_PPO}" --alphas 0.5 0.5
        run_merge "mode1-a07"          --base "${SFT}" --ft "${HB_PPO}" "${JOINT_PPO}" --alphas 0.7 0.7
        run_merge "mode1-hb07-joint04" --base "${SFT}" --ft "${HB_PPO}" "${JOINT_PPO}" --alphas 0.7 0.4

        # Family B: hb<->joint line, base=hb, single ft=joint, alpha=r (fraction toward joint)
        run_merge "hbline-r-025" --base "${HB_PPO}" --ft "${JOINT_PPO}" --alpha -0.25
        run_merge "hbline-r025"  --base "${HB_PPO}" --ft "${JOINT_PPO}" --alpha 0.25
        run_merge "hbline-r050"  --base "${HB_PPO}" --ft "${JOINT_PPO}" --alpha 0.50
        run_merge "hbline-r075"  --base "${HB_PPO}" --ft "${JOINT_PPO}" --alpha 0.75
        run_merge "hbline-r125"  --base "${HB_PPO}" --ft "${JOINT_PPO}" --alpha 1.25

        echo "All merges complete." | tee -a "${RLINF_RUN_ROOT}/run_embodiment.log"
        cat "${MANIFEST}"
    '
