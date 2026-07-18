#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -J move-clean-to-nvme
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/move_moe_demo_clean_to_nvme-%j.out
#SBATCH --time=4:00:00
#SBATCH --nodelist=SH-IDCA1404-10-140-54-10

set -euo pipefail

SRC=/mnt/petrelfs/muyao/jiayusun/store/cache/huggingface/lerobot/robotwin_moe_demo_clean
DST=/nvme/jiayusun/data_backup/lerobot_robotwin_moe_demo_clean
NVME_BASE=/nvme/jiayusun/data_backup

echo "[move] host=$(hostname)  job=${SLURM_JOB_ID}  $(date)"
echo "[move] /nvme before:"
df -h /nvme
du -sh ${NVME_BASE}/*/ 2>/dev/null || true
echo ""

if [[ ! -d "${SRC}" ]]; then
    echo "[move] ERROR: source not found: ${SRC}" >&2
    exit 1
fi

echo "[move] Copying ${SRC} → ${DST}"
mkdir -p "${DST}"
rsync -a --info=progress2 "${SRC}/" "${DST}/"
echo "[move] rsync done"
echo ""

echo "[move] Verify:"
SRC_COUNT=$(find "${SRC}/data" -name "*.parquet" 2>/dev/null | wc -l)
DST_COUNT=$(find "${DST}/data" -name "*.parquet" 2>/dev/null | wc -l)
echo "  petrelfs parquets: ${SRC_COUNT}"
echo "  nvme parquets:     ${DST_COUNT}"

if [[ "${SRC_COUNT}" -ne "${DST_COUNT}" ]]; then
    echo "[move] ERROR: parquet count mismatch!" >&2
    exit 1
fi
echo "[move] Verify OK (${DST_COUNT} parquets match)"
echo ""

echo "[move] /nvme after:"
df -h /nvme
du -sh ${NVME_BASE}/*/ 2>/dev/null

echo ""
echo "======================================================================"
echo "NEXT STEPS — after verifying counts above:"
echo "  Delete from petrelfs to reclaim ~218 GB:"
echo "  rm -rf ${SRC}"
echo "======================================================================"
echo "[move] DONE at $(date)"
