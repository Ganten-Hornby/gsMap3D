#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-trait-compare
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly WORKDIR="${PROJECT_ROOT}/analysis/revision_graph_3d/runs"
readonly OUTPUT="${PROJECT_ROOT}/analysis/revision_graph_3d/comparisons/mouse_e115_supervised_trait_mapping_2d_vs_3d"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT}"

"${PROJECT_ROOT}/.venv/bin/python" \
    "${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/compare_mouse_e115_supervised_trait_mapping.py" \
    --workdir "${WORKDIR}" \
    --project-2d mouse_e115_heart_supervised_trait_mapping_2d \
    --project-3d mouse_e115_heart_supervised_trait_mapping_3d \
    --trait-manifest "${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/mouse_e115_heart_gwas_traits.yaml" \
    --annotation heart_regions \
    --output-dir "${OUTPUT}"
