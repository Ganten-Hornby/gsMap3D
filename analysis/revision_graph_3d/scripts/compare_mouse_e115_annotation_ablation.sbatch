#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-ann-ablation
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly RUNS="${PROJECT_ROOT}/analysis/revision_graph_3d/runs"
readonly OUTPUT="${PROJECT_ROOT}/analysis/revision_graph_3d/comparisons/mouse_e115_heart_annotation_ablation"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUTPUT}"

"${PROJECT_ROOT}/.venv/bin/python" \
    "${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/compare_mouse_e115_annotation_ablation.py" \
    --supervised-2d "${RUNS}/mouse_e115_heart_sectionwise_2d_k10/find_latent_representations/Mouse_E11.5_heart_2d_add_latent.h5ad" \
    --supervised-3d "${RUNS}/mouse_e115_heart_section_aware_3d_k10/find_latent_representations/Mouse_E11.5_heart_add_latent.h5ad" \
    --unannotated-2d "${RUNS}/mouse_e115_heart_sectionwise_2d_k10_no_annotation/find_latent_representations/Mouse_E11.5_heart_2d_add_latent.h5ad" \
    --unannotated-3d "${RUNS}/mouse_e115_heart_section_aware_3d_k10_no_annotation/find_latent_representations/Mouse_E11.5_heart_add_latent.h5ad" \
    --output-dir "${OUTPUT}" \
    --neighbors 15 \
    --spatial-neighbors 10 \
    --threads "${SLURM_CPUS_PER_TASK}" \
    --seed 2024
