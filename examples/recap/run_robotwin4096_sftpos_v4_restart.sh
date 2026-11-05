#!/usr/bin/env bash
set -Eeuo pipefail
repo=/home/tanhe/sjy/RLinf-robotwin
root=/data2/sjy/recap/beat_block_hammer/rollout4096_sftpos_v4_restart
python=$repo/.venv/bin/python
stage=${1:?stage required}
case "$stage" in
  policy|eval_baseline|eval_policy) ;;
  *) echo 'This restart only supports policy and evaluation; labels are already verified.'; exit 2 ;;
esac
exec 9>"$root/stage.lock"
flock -n 9 || { echo 'Another 4096 experiment stage is running'; exit 3; }
test ! -e "$root/USER_PAUSED" || exit 4
export REPO_PATH=$repo EMBODIED_PATH=$repo/examples/embodiment
export ROBOTWIN_PATH=/data2/sjy/RoboTwin-RLinf ROBOT_PLATFORM=ALOHA
export PYTHONPATH=$repo:$ROBOTWIN_PATH
export USE_TF=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export RES_OPTIONS='timeout:1 attempts:1' HYDRA_FULL_ERROR=1 RLINF_NODE_RANK=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export VK_DRIVER_FILES=$repo/nvidia_icd.json VK_ICD_FILENAMES=$repo/nvidia_icd.json
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RAY_AUTH_MODE=disabled
unset CUDA_VISIBLE_DEVICES
cd "$repo"
case "$stage" in
  returns_smoke|returns)
    suffix=4096_v2; if [[ "$stage" == returns_smoke ]]; then suffix=4096_smoke; fi
    printf '%s\tSTART\t%s\n' "$(date -Is)" "$stage" >> "$root/status.tsv"
    "$python" examples/recap/process/compute_returns.py --config-path ../config --config-name "robotwin_returns_$suffix" >"$root/logs/$stage.log" 2>&1
    printf '%s\tEND\t%s\texit=0\n' "$(date -Is)" "$stage" >> "$root/status.tsv"
    exit 0
    ;;
  adv)
    args=(--config-path ../config --config-name robotwin_adv_4096_sftpos_v3)
    args+=("advantage.value_checkpoint=${RECAP_VALUE_CHECKPOINT:?Selected critic required}")
    printf '%s\tSTART\t%s\n' "$(date -Is)" "$stage" >> "$root/status.tsv"
    set +e
    "$python" -m torch.distributed.run --standalone --nproc_per_node=8 examples/recap/process/compute_advantages.py "${args[@]}" >"$root/logs/adv.log" 2>&1
    rc=$?
    set -e
    printf '%s\tEND\t%s\texit=%s\n' "$(date -Is)" "$stage" "$rc" >> "$root/status.tsv"
    exit "$rc"
    ;;
  collect|collect_smoke|eval_baseline|eval_policy)
    entry=examples/recap/collect_robotwin.py
    args=(--config-name robotwin_collect_4096_sftpos_v4_restart)
    if [[ "$stage" == collect_smoke ]]; then
      args+=(collect.episodes_per_worker=4 collect.validation_episodes_per_worker=2
             "collect.output=$root/datasets/rollout_smoke" "env.train.seeds_path=$root/protocol/smoke_pool.json")
    elif [[ "$stage" == eval* ]]; then
      args+=(collect.num_envs_per_worker=1 collect.episodes_per_worker=12 collect.validation_episodes_per_worker=0
             "collect.output=$root/$stage" env.train.seeds_path=/data2/sjy/rl_protocol/robotwin6/eval_manifest_96ep.json)
      if [[ "$stage" == eval_policy ]]; then
        args+=("collect.policy_checkpoint=$root/runs/policy/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt"
               actor.model.openpi.guidance_type=positive)
      fi
    fi
    ;;
  value_smoke|value)
    suffix=4096_v2; if [[ "$stage" == value_smoke ]]; then suffix=4096_smoke; fi
    entry=examples/recap/value/train_value.py
    args=(--config-path ../config --config-name "robotwin_value_$suffix")
    ;;
  policy_smoke)
    entry=examples/recap/cfg/train_cfg.py
    args=(--config-path ../config --config-name robotwin_cfg_4096_sftpos_v4_restart
          runner.max_steps=2 runner.max_epochs=2 runner.save_interval=2
          runner.logger.experiment_name=policy_smoke)
    ;;
  policy)
    entry=examples/recap/cfg/train_cfg.py
    args=(--config-path ../config --config-name robotwin_cfg_4096_sftpos_v4_restart)
    ;;
  *) echo "Unknown stage $stage"; exit 2 ;;
esac
"$python" - <<'PY'
import shutil,subprocess
from pathlib import Path
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
assert len(rows)==8
for row in rows:
    index,free,util=map(int,row.split(','))
    assert free>16000 and util<30,(index,free,util)
mem={line.split(':')[0]:int(line.split(':')[1].split()[0]) for line in Path('/proc/meminfo').read_text().splitlines()}
assert mem['MemAvailable']>160*1024*1024
assert shutil.disk_usage('/data2').free>300*1024**3
PY
export RAY_TMPDIR=/data2/sjy/rt/r4v_$(date +%H%M%S)
export RAY_ADDRESS=127.0.0.1:7811
cleanup() {
 "$python" - "$RAY_TMPDIR" <<'PY'
import os,signal,sys,time
from pathlib import Path
needle=sys.argv[1].encode()
for sig in [signal.SIGTERM,signal.SIGKILL]:
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            pid=int(path.parent.name); cmd=path.read_bytes()
            if pid!=os.getpid() and path.parent.stat().st_uid==os.getuid() and needle in cmd and any(token in cmd for token in [b'raylet',b'gcs_server',b'ray/dashboard',b'ray/_private',b'ray/autoscaler']):
                os.kill(pid,sig)
        except (OSError,ProcessLookupError):pass
    time.sleep(3)
PY
}
trap cleanup EXIT
"$repo/.venv/bin/ray" start --head --port=7811 --num-cpus=64 --num-gpus=8 --temp-dir="$RAY_TMPDIR" --object-store-memory=8589934592 --include-dashboard=false --disable-usage-stats >"$root/logs/ray_$stage.log" 2>&1
printf '%s\tSTART\t%s\n' "$(date -Is)" "$stage" >> "$root/status.tsv"
set +e
"$python" -u "$entry" "${args[@]}" >"$root/logs/$stage.log" 2>&1
rc=$?
set -e
printf '%s\tEND\t%s\texit=%s\n' "$(date -Is)" "$stage" "$rc" >> "$root/status.tsv"
exit "$rc"
