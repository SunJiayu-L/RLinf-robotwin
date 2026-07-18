#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J move-to-nvme
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/move_to_nvme-%j.out
#SBATCH --time=2-00:00:00
#SBATCH --nodelist=SH-IDCA1404-10-140-54-10

set -euo pipefail

NODE=$(hostname)
echo "[move] Running on node: ${NODE}"
echo "[move] Job ID: ${SLURM_JOB_ID}"
echo "[move] Start: $(date)"

NVME_BASE=/nvme/jiayusun/data_backup
mkdir -p "${NVME_BASE}"

echo "[move] Target: ${NVME_BASE}"
df -h /nvme

# ── 1. lerobot dataset: robotwin_moe_demo_clean (218 GB) ───────────────────
SRC1=/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_moe_demo_clean
DST1=${NVME_BASE}/lerobot_robotwin_moe_demo_clean
echo "[move] [1/3] Copying lerobot dataset → ${DST1}"
rsync -a --info=progress2 "${SRC1}/" "${DST1}/"
echo "[move] [1/3] Done lerobot dataset"

# ── 2. SFT intermediate checkpoints (steps 5000, 10000, 15000 only) ────────
SRC2_BASE=/mnt/petrelfs/muyao/jiayusun/RoboTwin/policy/pi05/checkpoints/pi05_aloha_moe_demo_clean/robotwin_moe_demo_clean_full
DST2_BASE=${NVME_BASE}/pi05_ckpt_intermediate
mkdir -p "${DST2_BASE}"
for step in 5000 10000 15000; do
    SRC="${SRC2_BASE}/${step}"
    DST="${DST2_BASE}/${step}"
    if [[ -d "${SRC}" ]]; then
        echo "[move] [2/3] Copying SFT step ${step} → ${DST}"
        rsync -a --info=progress2 "${SRC}/" "${DST}/"
        echo "[move] [2/3] Done step ${step}"
    else
        echo "[move] [2/3] WARN: SFT step ${step} not found, skipping"
    fi
done

# ── 3. merged_weights (157 GB) ──────────────────────────────────────────────
SRC3=/mnt/petrelfs/muyao/jiayusun/merged_weights
DST3=${NVME_BASE}/merged_weights
echo "[move] [3/3] Copying merged_weights → ${DST3}"
rsync -a --info=progress2 "${SRC3}/" "${DST3}/"
echo "[move] [3/3] Done merged_weights"

# ── Verify ───────────────────────────────────────────────────────────────────
echo ""
echo "[verify] /nvme usage after copy:"
df -h /nvme
echo ""
echo "[verify] Backup contents:"
du -sh "${NVME_BASE}"/*/  2>/dev/null
echo ""
echo "[verify] Checksums (file counts):"
echo "  lerobot episodes: $(find ${DST1}/data -name '*.parquet' 2>/dev/null | wc -l)"
echo "  merged_weights dirs: $(ls ${DST3}/ 2>/dev/null | wc -l)"
echo ""
echo "[move] ALL DONE at $(date)"
echo ""
echo "======================================================================"
echo "NEXT STEPS (run manually after verifying backup):"
echo "  1. Delete from petrelfs:"
echo "     rm -rf ${SRC1}"
echo "     rm -rf ${SRC2_BASE}/5000 ${SRC2_BASE}/10000 ${SRC2_BASE}/15000"
echo "     rm -rf ${SRC3}"
echo "  2. Submit conversion:"
echo "     sbatch examples/embodiment/slurm/convert_moe_demo_clean_hb1to1_to_lerobot.sbatch"
echo "  3. After conversion, submit SFT:"
echo "     sbatch examples/embodiment/slurm/finetune_moe_demo_clean_hb1to1_pi05.sbatch"
echo "  4. AFTER SFT is done, restore from /nvme/jiayusun/data_backup on node-10:"
echo "     sbatch examples/embodiment/slurm/move_data_back_from_nvme_node10.sbatch"
echo "======================================================================"
