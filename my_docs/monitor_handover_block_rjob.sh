#!/usr/bin/env bash

set -uo pipefail

JOB_NAME="${1:-handover-block-ppo8-c32-m512-0723-59615777}"
INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-3600}"
MONITOR_DIR=/mnt/shared-storage-user/chenzanxin/jiayusun/rjob-monitor
LOG_FILE="${MONITOR_DIR}/${JOB_NAME}.log"
LATEST_FILE="${MONITOR_DIR}/${JOB_NAME}.latest"
LOCK_FILE="/tmp/${JOB_NAME}.monitor.lock"

mkdir -p "${MONITOR_DIR}"
exec 9>"${LOCK_FILE}"
flock -n 9 || exit 0

# rjob/brainpp needs the Kubebrain variables inherited by PID 1. A user
# systemd service does not receive them unless they are imported explicitly.
if [[ -r /etc/profile.d/ssh-init.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/ssh-init.sh
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,100.96.0.0/12,172.16.0.0/12,192.168.0.0/16,.pjlab.org.cn,.pjh-service.org.cn,.svc,.cluster.local,kubernetes.default.svc"
export NO_PROXY="${no_proxy}"

check_job() {
    local checked_at status_output events_output logs_output
    checked_at=$(date '+%F %T %Z')
    status_output=$(rjob get "${JOB_NAME}" 2>&1)
    events_output=$(rjob events "${JOB_NAME}" 2>&1 | tail -40)

    {
        printf '\n===== %s =====\n' "${checked_at}"
        printf '%s\n' "${status_output}"
        printf '%s\n' '--- events ---'
        printf '%s\n' "${events_output}"
    } >>"${LOG_FILE}"

    {
        printf 'checked_at=%s\n' "${checked_at}"
        printf '%s\n' "${status_output}"
    } >"${LATEST_FILE}"

    if grep -Eqi 'rjob .+: Running$|replica .+: RUNNING([,[:space:]]|$)|active.: [1-9]' <<<"${status_output}"; then
        logs_output=$(rjob logs job "${JOB_NAME}" -n 200 2>&1)
        {
            printf '%s\n' '--- logs ---'
            printf '%s\n' "${logs_output}"
        } >>"${LOG_FILE}"
    fi

    if grep -Eqi 'rjob .+: (Succeed(ed)?|Failed|Stopped|Completed)$|replica .+: (SUCCEED(ED)?|FAILED|STOPPED|COMPLETED)([,[:space:]]|$)' <<<"${status_output}"; then
        return 10
    fi
    return 0
}

while true; do
    check_job
    status=$?
    if [[ "${status}" -eq 10 ]]; then
        exit 0
    fi
    sleep "${INTERVAL_SECONDS}"
done
