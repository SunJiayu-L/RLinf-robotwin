#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J move-back-from-nvme
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/move_back_from_nvme-%j.out
#SBATCH --time=2-00:00:00
#SBATCH --nodelist=SH-IDCA1404-10-140-54-10

set -euo pipefail

NODE=$(hostname)
echo "[restore] Running on node: ${NODE}"
echo "[restore] Job ID: ${SLURM_JOB_ID}"
echo "[restore] Start: $(date)"

NVME_BASE=/nvme/jiayusun/data_backup

if [[ ! -d "${NVME_BASE}" ]]; then
    echo "[restore] ERROR: backup dir not found: ${NVME_BASE}" >&2
    exit 1
fi

echo "[restore] Source: ${NVME_BASE}"
du -sh "${NVME_BASE}"/*/  2>/dev/null
df -h /nvme

# ── 1. Restore lerobot dataset ───────────────────────────────────────────────
SRC1=${NVME_BASE}/lerobot_robotwin_moe_demo_clean
DST1=/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_moe_demo_clean
if [[ -d "${SRC1}" ]]; then
    echo "[restore] [1/3] Restoring lerobot dataset → ${DST1}"
    mkdir -p "$(dirname ${DST1})"
    rsync -a --info=progress2 "${SRC1}/" "${DST1}/"
    echo "[restore] [1/3] Done lerobot dataset"
else
    echo "[restore] [1/3] SKIP: ${SRC1} not found"
fi

# ── 2. Restore SFT intermediate checkpoints ──────────────────────────────────
SRC2_BASE=${NVME_BASE}/pi05_ckpt_intermediate
DST2_BASE=/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full
if [[ -d "${SRC2_BASE}" ]]; then
    for step in 5000 10000 15000; do
        SRC="${SRC2_BASE}/${step}"
        DST="${DST2_BASE}/${step}"
        if [[ -d "${SRC}" ]]; then
            echo "[restore] [2/3] Restoring SFT step ${step} → ${DST}"
            mkdir -p "${DST2_BASE}"
            rsync -a --info=progress2 "${SRC}/" "${DST}/"
            echo "[restore] [2/3] Done step ${step}"
        fi
    done
else
    echo "[restore] [2/3] SKIP: ${SRC2_BASE} not found"
fi

# ── 3. Restore merged_weights ────────────────────────────────────────────────
SRC3=${NVME_BASE}/merged_weights
DST3=/mnt/petrelfs/muyao/jiayusun/merged_weights
if [[ -d "${SRC3}" ]]; then
    echo "[restore] [3/3] Restoring merged_weights → ${DST3}"
    mkdir -p "${DST3}"
    rsync -a --info=progress2 "${SRC3}/" "${DST3}/"
    echo "[restore] [3/3] Done merged_weights"
else
    echo "[restore] [3/3] SKIP: ${SRC3} not found"
fi

echo ""
echo "[restore] ALL DONE at $(date)"
echo ""
echo "[restore] Verify restored files:"
du -sh /mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_moe_demo_clean/ 2>/dev/null && echo "  lerobot: OK" || echo "  lerobot: MISSING"
du -sh /mnt/petrelfs/muyao/jiayusun/merged_weights/ 2>/dev/null && echo "  merged_weights: OK" || echo "  merged_weights: MISSING"
ls /mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full/ 2>/dev/null
echo ""
echo "======================================================================"
echo "After verifying restore, optionally clean /nvme backup:"
echo "  Submit job to node-10 and run: rm -rf ${NVME_BASE}"
echo "======================================================================"
