#!/usr/bin/env bash
#SBATCH --job-name=gsmap-exact-mmap-preflight
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly PROBE_DIR="${PROJECT_ROOT}/analysis/revision_graph_3d/tmp/exact_memmap_probe_${SLURM_JOB_ID}"
readonly REPORT="${PROJECT_ROOT}/analysis/revision_graph_3d/validation/exact_memmap_probe_${SLURM_JOB_ID}.json"
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"

cleanup() {
    rm -f "${PROBE_DIR}/marker_scores.dat" "${PROBE_DIR}/marker_scores.meta.json"
    rmdir "${PROBE_DIR}" 2>/dev/null || true
}
trap cleanup EXIT

mkdir -p "${PROBE_DIR}" "${PROJECT_ROOT}/analysis/revision_graph_3d/logs" "$(dirname "${REPORT}")"
lfs setstripe -c 4 "${PROBE_DIR}"

printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
printf 'host=%s\n' "$(hostname)"
printf 'job_id=%s\n' "${SLURM_JOB_ID}"
printf 'cpus=%s\n' "${SLURM_CPUS_PER_TASK}"
printf 'commit=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"

"${PYTHON}" "${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/validate_exact_marker_memmap.py" \
    --project-root "${PROJECT_ROOT}" \
    --probe-dir "${PROBE_DIR}" \
    --report "${REPORT}" \
    --workers "${SLURM_CPUS_PER_TASK}"

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
printf 'report=%s\n' "${REPORT}"
