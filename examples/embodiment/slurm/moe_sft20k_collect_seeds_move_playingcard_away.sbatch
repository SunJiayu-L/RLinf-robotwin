#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -J moe20k-seeds-mpa
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/moe_sft20k_collect_seeds_move_playingcard_away-%j.out
#SBATCH --time=4:00:00
#SBATCH --exclude=SH-IDCA1404-10-140-54-10,SH-IDCA1404-10-140-54-12,SH-IDCA1404-10-140-54-13,SH-IDCA1404-10-140-54-18,SH-IDCA1404-10-140-54-19,SH-IDCA1404-10-140-54-20,SH-IDCA1404-10-140-54-21,SH-IDCA1404-10-140-54-28,SH-IDCA1404-10-140-54-33,SH-IDCA1404-10-140-54-35,SH-IDCA1404-10-140-54-37,SH-IDCA1404-10-140-54-44,SH-IDCA1404-10-140-54-45,SH-IDCA1404-10-140-54-57,SH-IDCA1404-10-140-54-61,SH-IDCA1404-10-140-54-69,SH-IDCA1404-10-140-54-73,SH-IDCA1404-10-140-54-79,SH-IDCA1404-10-140-54-81,SH-IDCA1404-10-140-54-89,SH-IDCA1404-10-140-54-115,SH-IDCA1404-10-140-54-118,SH-IDCA1404-10-140-54-26,SH-IDCA1404-10-140-54-78,SH-IDCA1404-10-140-54-36,SH-IDCA1404-10-140-54-22,SH-IDCA1404-10-140-54-49,SH-IDCA1404-10-140-54-70,SH-IDCA1404-10-140-54-56,SH-IDCA1404-10-140-54-46,SH-IDCA1404-10-140-54-51

set -eo pipefail
SIF_IMAGE=${SIF_IMAGE:-/mnt/petrelfs/muyao/shared_img/new_app.sif}
APPTAINER_BIN=${APPTAINER_BIN:-apptainer}
export REPO_PATH=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin
export ROBOTWIN_PATH=/mnt/petrelfs/muyao/jiayusun/RoboTwin-RLinf

if [[ "${IN_RLINF_ROBOTWIN_CONTAINER:-0}" != "1" ]]; then
    exec "${APPTAINER_BIN}" exec --cleanenv --nv --bind /mnt:/mnt --bind /nvme:/nvme --pwd "${REPO_PATH}" "${SIF_IMAGE}" env IN_RLINF_ROBOTWIN_CONTAINER=1 HOME="${HOME}" SLURM_JOB_ID="${SLURM_JOB_ID:-}" SLURM_NODELIST="${SLURM_NODELIST:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" REPO_PATH="${REPO_PATH}" ROBOTWIN_PATH="${ROBOTWIN_PATH}" bash "${REPO_PATH}/examples/embodiment/slurm/moe_sft20k_collect_seeds_move_playingcard_away.sbatch"
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

# === Step 1: 运行 8 steps，收集 per-episode 数据 ===
# 8 steps × rollout_epoch=4 × 64 envs × ~1 ep/env/epoch ≈ 2048 episodes
# 期望 SFT-20k 成功率 ~72% → ~1475 success seeds，取前 1000 写入 train_seeds.json
bash examples/embodiment/run_embodiment.sh moe_sft20k_collect_seeds_move_playingcard_away ALOHA

ray stop

# === Step 2: 从 episode_log JSONL 提取 success seeds，写入 train_seeds.json ===
TRAIN_SEEDS_JSON="${REPO_PATH}/rlinf/envs/robotwin/seeds/train_seeds.json"
# log_path 实际解析到 logs/<timestamp>-moe_sft20k_collect_seeds_move_playingcard_away/
# 取最新的一个目录
EPISODE_LOG_DIR="$(ls -dt "${REPO_PATH}/logs/"*"-moe_sft20k_collect_seeds_move_playingcard_away/episode_log" 2>/dev/null | head -1)"

if [[ -z "${EPISODE_LOG_DIR}" ]]; then
    echo "[ERROR] Could not find episode_log directory under ${REPO_PATH}/logs/"
    exit 1
fi
echo "[INFO] Using episode_log dir: ${EPISODE_LOG_DIR}"

python3 - "${EPISODE_LOG_DIR}" "${TRAIN_SEEDS_JSON}" <<'PYEOF'
import json, glob, sys, pathlib

episode_log_dir = sys.argv[1]
train_seeds_json = sys.argv[2]
target_count = 1000

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

print(f"Total episodes logged: {len(records)}")
success_seeds = sorted(set(r["seed"] for r in records if r.get("success", False)))
print(f"Unique success seeds: {len(success_seeds)}")

if len(success_seeds) < 100:
    print("[WARNING] Very few success seeds — check model performance and episode log")

kept = success_seeds[:target_count]
print(f"Writing {len(kept)} seeds to train_seeds.json (target={target_count})")
if kept:
    print(f"  seed range: {kept[0]} - {kept[-1]}")

with open(train_seeds_json) as f:
    data = json.load(f)

if "move_playingcard_away" in data:
    print("[WARNING] Overwriting existing move_playingcard_away entry in train_seeds.json")

data["move_playingcard_away"] = {
    "task_name": "move_playingcard_away",
    "success_seeds": kept,
}

with open(train_seeds_json, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"[OK] Done → {train_seeds_json}")
PYEOF

echo "=== Seed collection complete. Now run PPO: ==="
echo "    sbatch examples/embodiment/slurm/moe_sft20k_move_playingcard_away_ppo100.sbatch"
