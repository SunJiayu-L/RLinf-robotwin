#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -J delete-old-logs
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/delete_backed_logs-%j.out
#SBATCH --time=2:00:00
#SBATCH --nodelist=SH-IDCA1404-10-140-54-10

set -euo pipefail

LOGS_DIR=/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs
NVME_LOGS=/nvme/jiayusun/data_backup/logs

echo "[delete] job=${SLURM_JOB_ID}  host=$(hostname)  $(date)"
echo "[delete] Verifying nvme backup exists..."
if [[ ! -d "${NVME_LOGS}" ]]; then
    echo "[delete] ERROR: nvme backup not found: ${NVME_LOGS}" >&2
    exit 1
fi

NVME_COUNT=$(ls "${NVME_LOGS}" | wc -l)
echo "[delete] nvme backup has ${NVME_COUNT} directories"
df -h /nvme

echo ""
echo "[delete] Deleting from petrelfs (only dirs confirmed in nvme)..."

DELETED=0
ERRORS=0
for d in $(ls "${NVME_LOGS}/"); do
    SRC="${LOGS_DIR}/${d}"
    if [[ -d "${SRC}" ]]; then
        rm -rf "${SRC}"
        echo "[del] ${d}"
        ((DELETED++)) || true
    else
        echo "[skip] already gone: ${d}"
    fi
done

echo ""
echo "[delete] Deleted ${DELETED} dirs, ${ERRORS} errors"
echo ""
echo "[delete] petrelfs logs/ after deletion:"
du -sh "${LOGS_DIR}/" 2>/dev/null
echo ""
echo "[delete] petrelfs quota after deletion:"
petrelfs-ctl --getquota --uid 200000967 2>/dev/null || true
echo ""
echo "[delete] DONE at $(date)"
