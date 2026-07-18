#!/bin/bash
#SBATCH -p wam_data
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH -J migrate-logs
#SBATCH -o /mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/slurm_logs/migrate_logs_july17_to_node10-%j.out
#SBATCH --time=4:00:00
#SBATCH --nodelist=SH-IDCA1404-10-140-54-10

set -eo pipefail

PETRELFS_LOGS="/mnt/petrelfs/muyao/jiayusun/RLinf-robotwin/logs"
NVME_BACKUP="/nvme/jiayusun/data_backup/logs_july17"

mkdir -p "${NVME_BACKUP}"

DIRS=(
    "20260715-05:51:55-moe_sft20k_handover_block_ppo150_128env"
    "20260715-12:29:28-moe_ppo_5task_bbh_ca_cb_mpa_mcp"
    "20260715-17:21:21-moe_ppo_4task_pal_par_mcp_pps"
    "20260716-21:39:48-moe_sft20k_turn_switch_ppo100"
    "20260716-21:41:25-moe_sft20k_rotate_qrcode_ppo100"
    "20260716-21:41:25-moe_sft20k_place_phone_stand_ppo100"
)

echo "========================================"
echo "  Migration to node-10 /nvme"
echo "  Start: $(date)"
echo "  Target: ${NVME_BACKUP}"
echo "========================================"
df -h /nvme
echo ""

declare -A RESULTS

for dirname in "${DIRS[@]}"; do
    src="${PETRELFS_LOGS}/${dirname}"
    dst="${NVME_BACKUP}/${dirname}"

    if [[ ! -d "$src" ]]; then
        echo "[SKIP] ${dirname} — not found"
        RESULTS[$dirname]="SKIP"
        continue
    fi

    src_size_h=$(du -sh "$src" 2>/dev/null | awk '{print $1}')
    echo ""
    echo "---------------------------------------"
    echo "  rsync: ${dirname}  (${src_size_h})  $(date)"

    rsync -a --stats "${src}/" "${dst}/"

    src_count=$(find "$src" -type f | wc -l)
    dst_count=$(find "$dst" -type f | wc -l)
    echo "  files: src=${src_count}  dst=${dst_count}"

    if [[ "$src_count" -eq "$dst_count" && "$src_count" -gt 0 ]]; then
        echo "  [OK] Verified. Deleting petrelfs copy..."
        rm -rf "$src"
        RESULTS[$dirname]="OK+DELETED"
        echo "  [DONE]"
    else
        echo "  [FAIL] File count mismatch (src=${src_count} dst=${dst_count}). Keeping petrelfs copy."
        RESULTS[$dirname]="VERIFY_FAIL"
    fi
done

echo ""
echo "========================================"
echo "  MIGRATION SUMMARY — $(date)"
echo "========================================"
for dirname in "${DIRS[@]}"; do
    printf "  %-60s %s\n" "$dirname" "${RESULTS[$dirname]:-N/A}"
done
echo ""
df -h /nvme
echo "========================================"
