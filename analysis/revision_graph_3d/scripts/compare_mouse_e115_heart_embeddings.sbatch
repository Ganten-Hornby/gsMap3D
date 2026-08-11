#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-heart-compare
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly EMBEDDING_2D="${PROJECT_ROOT}/analysis/revision_graph_3d/runs/mouse_e115_heart_sectionwise_2d_k10/find_latent_representations/Mouse_E11.5_heart_2d_add_latent.h5ad"
readonly EMBEDDING_3D="${PROJECT_ROOT}/analysis/revision_graph_3d/runs/mouse_e115_heart_section_aware_3d_k10/find_latent_representations/Mouse_E11.5_heart_add_latent.h5ad"
readonly OUTPUT_DIR="${PROJECT_ROOT}/analysis/revision_graph_3d/comparisons/mouse_e115_heart_2d_vs_3d"

mkdir -p "${OUTPUT_DIR}" "${PROJECT_ROOT}/analysis/revision_graph_3d/logs"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/gsmap-matplotlib-${SLURM_JOB_ID}"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/gsmap-numba-${SLURM_JOB_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"${PROJECT_ROOT}/.venv/bin/python" \
    "${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/compare_mouse_e115_heart_embeddings.py" \
    --embedding-2d "${EMBEDDING_2D}" \
    --embedding-3d "${EMBEDDING_3D}" \
    --output-dir "${OUTPUT_DIR}" \
    --neighbors 15 \
    --spatial-neighbors 10 \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    --seed 2024 \
    --graph-edges-2d 1141174 \
    --graph-edges-3d 2468990
