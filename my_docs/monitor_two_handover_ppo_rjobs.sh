#!/usr/bin/env bash

set -uo pipefail

INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-3600}"
MONITOR_DIR="${MONITOR_DIR:-/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chenzanxin/rlinf-robotwin-runs/monitoring}"
LOCK_FILE="/tmp/rlinf-two-handover-ppo.monitor.lock"

if [[ "$#" -eq 0 ]]; then
    echo "Usage: $0 <rjob-name> [<rjob-name> ...]" >&2
    exit 2
fi

mkdir -p "${MONITOR_DIR}"
exec 9>"${LOCK_FILE}"
flock -n 9 || exit 0

if [[ -r /etc/profile.d/ssh-init.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/ssh-init.sh
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
export NO_PROXY="${no_proxy}"

check_job() {
    local job_name="$1"
    local checked_at log_file latest_file status_output events_output logs_output
    checked_at=$(date '+%F %T %Z')
    log_file="${MONITOR_DIR}/${job_name}.log"
    latest_file="${MONITOR_DIR}/${job_name}.latest"
    status_output=$(rjob get "${job_name}" 2>&1)
    events_output=$(rjob events "${job_name}" 2>&1 | tail -40)

    {
        printf '\n===== %s =====\n' "${checked_at}"
        printf '%s\n' "${status_output}"
        printf '%s\n' '--- events ---'
        printf '%s\n' "${events_output}"
    } >>"${log_file}"

    {
        printf 'checked_at=%s\n' "${checked_at}"
        printf '%s\n' "${status_output}"
    } >"${latest_file}"

    if grep -Eqi 'rjob .+: Running$|replica .+: RUNNING([,[:space:]]|$)|active.: [1-9]' <<<"${status_output}"; then
        logs_output=$(rjob logs job "${job_name}" -n 200 2>&1)
        {
            printf '%s\n' '--- rjob logs ---'
            printf '%s\n' "${logs_output}"
        } >>"${log_file}"
    fi
}

while true; do
    for job_name in "$@"; do
        check_job "${job_name}"
    done
    sleep "${INTERVAL_SECONDS}"
done
