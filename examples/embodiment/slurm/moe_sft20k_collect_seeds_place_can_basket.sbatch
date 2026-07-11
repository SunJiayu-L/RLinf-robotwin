#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J moe20k-seeds-place_ca
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/moe_sft20k_collect_seeds_place_can_basket-%j.out
#SBATCH --time=4:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-20,SH-IDCA1404-10-140-54-21,SH-IDCA1404-10-140-54-28,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-35,SH-IDCA1404-10-140-54-37,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-45,SH-IDCA1404-10-140-54-57,SH-IDCA1404-10-140-54-61,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-73,SH-IDCA1404-10-140-54-79,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-26,SH-IDCA1404-10-140-54-78,SH-IDCA1404-10-140-54-36,SH-IDCA1404-10-140-54-22,SH-IDCA1404-10-140-54-49,SH-IDCA1404-10-140-54-70,SH-IDCA1404-10-140-54-56,SH-IDCA1404-10-140-54-46,SH-IDCA1404-10-140-54-51

set -eo pipefail
SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
export ROBOTWIN_PATH=/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf

if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec --cleanenv --nv --bind /mnt:/mnt --bind /nvme:/nvme --pwd "${REPO_PATH}" "${SIF_IMAGE}" env IN_RLINF_ROBOTWIN_CONTAINER=1 HOME="${HOME}" SLURM_JOB_ID="${SLURM_JOB_ID:-}" SLURM_NODELIST="${SLURM_NODELIST:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" REPO_PATH="${REPO_PATH}" ROBOTWIN_PATH="${ROBOTWIN_PATH}" bash "${REPO_PATH}/examples/embodiment/slurm/moe_sft20k_collect_seeds_place_can_basket.sbatch"
fi

export GCC_HOME=/mnt/lustre/share/gcc/gcc-11.2.0
export CUDA_HOME=/mnt/lustre/share/cuda-12.4
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$REPO_PATH/.venv/bin:$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$GCC_HOME/lib64:$CUDA_HOME/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export ROBOT_PLATFORM=ALOHA; export MUJOCO_GL=egl; export PYOPENGL_PLATFORM=egl
export VK_DRIVER_FILES="${REPO_PATH}/nvidia_icd.json"
export VK_ICD_FILENAMES="${REPO_PATH}/nvidia_icd.json"
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}
cd "${REPO_PATH}"
ray stop --force 2>/dev/null || true
ray start --head --num-cpus=64 --num-gpus=8

bash examples/embodiment/run_embodiment.sh moe_sft20k_collect_seeds_place_can_basket ALOHA

ray stop

TRAIN_SEEDS_JSON="${REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json"
EPISODE_LOG_DIR="$(ls -dt "${REPO_PATH}/logs/"*"-moe_sft20k_collect_seeds_place_can_basket/episode_log" 2>/dev/null | head -1)"

if [[ -z "${EPISODE_LOG_DIR}" ]]; then
    echo "[ERROR] Could not find episode_log directory"
    exit 1
fi
echo "[INFO] Using episode_log dir: ${EPISODE_LOG_DIR}"

python3 - "${EPISODE_LOG_DIR}" "${TRAIN_SEEDS_JSON}" <<'PYEOF'
import json, sys, pathlib

episode_log_dir = sys.argv[1]
train_seeds_json = sys.argv[2]
target_count = 1000
task_name = "place_can_basket"

files = sorted(pathlib.Path(episode_log_dir).glob("episodes.worker-*.jsonl"))
if not files:
    print(f"[ERROR] No episode log files in {episode_log_dir}")
    sys.exit(1)

records = []
for f in files:
    for line in f.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Total episodes: {len(records)}")
success_seeds = sorted(set(r["seed"] for r in records if r.get("success", False)))
print(f"Unique success seeds: {len(success_seeds)}")

if len(success_seeds) < 100:
    print("[WARNING] Very few success seeds — check model performance")

kept = success_seeds[:target_count]
print(f"Writing {len(kept)} seeds (range: {kept[0] if kept else 'N/A'} - {kept[-1] if kept else 'N/A'})")

with open(train_seeds_json) as f:
    data = json.load(f)

if task_name in data:
    print(f"[WARNING] Overwriting existing {task_name} entry")

data[task_name] = {"task_name": task_name, "success_seeds": kept}

with open(train_seeds_json, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"[OK] Done -> {train_seeds_json}")
PYEOF

echo "=== Seed collection complete for place_can_basket ==="
