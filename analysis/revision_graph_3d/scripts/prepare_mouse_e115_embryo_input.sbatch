#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-embryo-prepare
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly SOURCE=/storage/yangjianLab/lilounan/data/3dst/Mouse_E11.5_embryo.h5ad
readonly OUTPUT="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/Mouse_E11.5_embryo_graph_ready.h5ad"
readonly AUDIT="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/Mouse_E11.5_embryo_graph_ready.audit.json"
readonly STATUS_DIR="${PROJECT_ROOT}/analysis/revision_graph_3d/runs/mouse_e115_embryo_input_preparation"

mkdir -p "${STATUS_DIR}" "${PROJECT_ROOT}/analysis/revision_graph_3d/logs"

record_exit() {
    local status=$?
    {
        printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
        printf 'exit_code=%s\n' "${status}"
        printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    } >"${STATUS_DIR}/slurm_status.txt"
    exit "${status}"
}
trap record_exit EXIT

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
printf 'host=%s\n' "$(hostname)"
printf 'job_id=%s\n' "${SLURM_JOB_ID}"

"${PROJECT_ROOT}/.venv/bin/python" \
    "${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/prepare_mouse_e115_embryo_input.py" \
    --source "${SOURCE}" \
    --output "${OUTPUT}" \
    --audit "${AUDIT}"
