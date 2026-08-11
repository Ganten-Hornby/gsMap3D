#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e95-2d3d-compare
#SBATCH --output=analysis/revision_graph_3d/logs/gsmap-e95-2d3d-compare-%j.out
#SBATCH --error=analysis/revision_graph_3d/logs/gsmap-e95-2d3d-compare-%j.err

set -euo pipefail

project_root=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
baseline_dir=/storage/yangjianLab/lilounan/gsMap3D-scripts/20260107_gsmap_3d_report_all_collected/mouse_embryo/E9.5/find_latent_representations
section_aware_h5ad="$project_root/analysis/revision_graph_3d/runs/mouse_e95_section_aware_3d_k10/find_latent_representations/Mouse_E9.5_embryo_sectioned_add_latent.h5ad"
output_dir="$project_root/analysis/revision_graph_3d/comparisons/mouse_e95_2d_vs_3d"

cd "$project_root"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/gsmap-matplotlib-$SLURM_JOB_ID"
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/gsmap-numba-$SLURM_JOB_ID"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

"$project_root/.venv/bin/python" \
  analysis/revision_graph_3d/scripts/compare_mouse_e95_embeddings.py \
  --baseline-dir "$baseline_dir" \
  --section-aware-h5ad "$section_aware_h5ad" \
  --output-dir "$output_dir" \
  --threads "${SLURM_CPUS_PER_TASK:-8}"
