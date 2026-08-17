#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-2d-3d-report-recover
#SBATCH --partition=test
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly RUN_DIR="${PROJECT_ROOT}/analysis/revision_graph_3d/runs/mouse_e115_embryo_sectionwise_2d_k10"
readonly RECOVERY_SCRIPT="${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/recover_spatial_3d_report.py"
readonly N_SPOTS=6993667
readonly SCRATCH_DIR="/tmp/gsmap-${USER}/${SLURM_JOB_ID}"

cleanup() {
    case "${SCRATCH_DIR}" in
        "/tmp/gsmap-${USER}/${SLURM_JOB_ID}")
            rm -rf -- "${SCRATCH_DIR}"
            ;;
        *)
            printf 'Refusing to clean unexpected scratch path: %s\n' "${SCRATCH_DIR}" >&2
            ;;
    esac
}
trap cleanup EXIT

mkdir -p "${PROJECT_ROOT}/analysis/revision_graph_3d/logs" "${SCRATCH_DIR}"
export TMPDIR="${SCRATCH_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
uv run --extra cuda128 --locked python "${RECOVERY_SCRIPT}" \
    --run-dir "${RUN_DIR}" \
    --expected-observations "${N_SPOTS}"

readonly SPATIAL_3D_DIR="${RUN_DIR}/gsmap_web_report/spatial_3d"
[[ -s "${SPATIAL_3D_DIR}/spatial_3d_anno_ManualAnnotation.html" ]]
[[ -s "${SPATIAL_3D_DIR}/spatial_3d_trait_Height.html" ]]
[[ -s "${SPATIAL_3D_DIR}/spatial_3d_trait_IQ.html" ]]
[[ -s "${SPATIAL_3D_DIR}/spatial_3d_trait_CBX.html" ]]
[[ -s "${SPATIAL_3D_DIR}/spatial_3d_trait_CTX.html" ]]
[[ -s "${SPATIAL_3D_DIR}/spatial_3d_export_audit.json" ]]
