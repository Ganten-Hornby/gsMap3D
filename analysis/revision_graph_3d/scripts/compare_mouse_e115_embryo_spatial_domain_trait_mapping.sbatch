#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-spatial-domain-compare
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly WORKDIR="${PROJECT_ROOT}/analysis/revision_graph_3d/runs"
readonly PROJECT_2D=mouse_e115_embryo_sectionwise_2d_k10_spatial_domain
readonly PROJECT_3D=mouse_e115_embryo_section_aware_3d_k10_spatial_domain
readonly TRAIT_MANIFEST="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/mouse_e115_embryo_gwas_traits.yaml"
readonly OUTPUT="${PROJECT_ROOT}/analysis/revision_graph_3d/comparisons/mouse_e115_embryo_spatial_domain_2d_vs_3d"
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly COMPARISON_SCRIPT="${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/compare_mouse_e115_supervised_trait_mapping.py"

mkdir -p "${OUTPUT}" "${PROJECT_ROOT}/analysis/revision_graph_3d/logs"

record_exit() {
    local status=$?
    trap - EXIT
    {
        printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
        printf 'exit_code=%s\n' "${status}"
        printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    } >"${OUTPUT}/slurm_status.txt"
    exit "${status}"
}
trap record_exit EXIT

upstream_succeeded() {
    local status_file=$1
    local line
    [[ -f "${status_file}" ]] || return 1
    while IFS= read -r line; do
        [[ "${line}" == "exit_code=0" ]] && return 0
    done <"${status_file}"
    return 1
}

readonly STATUS_2D="${WORKDIR}/${PROJECT_2D}/report_status.txt"
readonly STATUS_3D="${WORKDIR}/${PROJECT_3D}/report_status.txt"
if ! upstream_succeeded "${STATUS_2D}" || ! upstream_succeeded "${STATUS_3D}"; then
    printf 'Both spatial-domain report jobs must complete successfully.\n' >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
printf 'host=%s\n' "$(hostname)"
printf 'job_id=%s\n' "${SLURM_JOB_ID}"
printf 'commit=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"

"${PYTHON}" "${COMPARISON_SCRIPT}" \
    --workdir "${WORKDIR}" \
    --project-2d "${PROJECT_2D}" \
    --project-3d "${PROJECT_3D}" \
    --trait-manifest "${TRAIT_MANIFEST}" \
    --annotation spatial_domain \
    --output-dir "${OUTPUT}"

readonly REPORT="${OUTPUT}/mouse_e115_spatial_domain_trait_mapping_2d_vs_3d_report.md"
readonly AUDIT="${OUTPUT}/comparison_audit.json"
[[ -s "${REPORT}" ]]
[[ -s "${AUDIT}" ]]
[[ -s "${OUTPUT}/per_trait_per_spot_correlations.tsv" ]]
[[ -s "${OUTPUT}/per_trait_per_spot_aligned_scores.parquet" ]]
[[ -s "${OUTPUT}/per_trait_spatial_domain_correlations.tsv" ]]
[[ -s "${OUTPUT}/per_trait_spatial_domain_aligned_scores.tsv" ]]
for stem in \
    per_trait_per_spot_2d_vs_3d_scatter \
    per_trait_spatial_domain_cauchy_scatter \
    per_trait_spatial_domain_rankflow; do
    [[ -s "${OUTPUT}/${stem}.png" ]]
    [[ -s "${OUTPUT}/${stem}.pdf" ]]
done

AUDIT_PATH="${AUDIT}" "${PYTHON}" - <<'PY'
import json
import os

with open(os.environ["AUDIT_PATH"], encoding="utf-8") as handle:
    audit = json.load(handle)
if audit.get("annotation") != "spatial_domain":
    raise RuntimeError(f"Unexpected comparison annotation: {audit.get('annotation')}")
if audit.get("n_traits") != 5:
    raise RuntimeError(f"Expected five traits, found {audit.get('n_traits')}")
sampling = audit.get("sampling_audit", {})
if sampling.get("metrics_subsampling") or sampling.get("visualization_subsampling"):
    raise RuntimeError(f"Comparison unexpectedly downsampled data: {sampling}")
if not sampling.get("all_matched_spots_plotted"):
    raise RuntimeError(f"Not all matched spots were plotted: {sampling}")
if audit.get("raster_dpi", 0) < 300:
    raise RuntimeError(f"Raster DPI is below 300: {audit.get('raster_dpi')}")
PY
