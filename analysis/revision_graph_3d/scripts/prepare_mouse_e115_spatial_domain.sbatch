#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-spatial-domain
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly H5AD="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/Mouse_E11.5_embryo_graph_ready.h5ad"
readonly METADATA=/storage/yangjianLab/lilounan/data/3dst/Mouse_E11.5_embryo.meta.csv
readonly AUDIT="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/Mouse_E11.5_embryo_spatial_domain.audit.json"

mkdir -p "${PROJECT_ROOT}/analysis/revision_graph_3d/logs"
"${PROJECT_ROOT}/.venv/bin/python" \
    "${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/attach_mouse_e115_spatial_domain.py" \
    --h5ad "${H5AD}" \
    --metadata "${METADATA}" \
    --audit "${AUDIT}"
