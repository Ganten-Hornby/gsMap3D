#!/usr/bin/env bash

configure_ldsc_worker_split() {
    local -r total_workers="${1:?total worker count is required}"
    local -r worker_ratio_parts=2

    if ((total_workers < worker_ratio_parts || total_workers % worker_ratio_parts != 0)); then
        printf 'Worker count must be a positive multiple of %s, found %s\n' \
            "${worker_ratio_parts}" "${total_workers}" >&2
        return 2
    fi

    declare -gr LDSC_READ_WORKERS=$((total_workers / worker_ratio_parts))
    declare -gr LDSC_COMPUTE_WORKERS=$((total_workers - LDSC_READ_WORKERS))
    export GSMAP_LDSC_FORCE_COMPUTE_WORKERS="${LDSC_COMPUTE_WORKERS}"
}
