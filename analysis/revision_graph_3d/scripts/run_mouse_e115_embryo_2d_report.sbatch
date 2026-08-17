#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-embryo-2d-report
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly WORKDIR="${PROJECT_ROOT}/analysis/revision_graph_3d/runs"
readonly PROJECT_NAME=mouse_e115_embryo_sectionwise_2d_k10
readonly RUN_DIR="${WORKDIR}/${PROJECT_NAME}"
readonly GSMAP="${PROJECT_ROOT}/.venv/bin/gsmap"
readonly TRAIT_MANIFEST="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/mouse_e115_embryo_gwas_traits.yaml"
readonly SNP_GENE_WEIGHTS=/storage/yangjianLab/lilounan/resources/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad
readonly W_LD_DIR=/storage/yangjianLab/lilounan/resources/gsMap_resource/LDSC_resource/weights_hm3_no_hla
readonly N_SPOTS=6993667
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly REPORT_TMP_DIR="/tmp/gsmap-${USER}/${SLURM_JOB_ID}"

mkdir -p "${PROJECT_ROOT}/analysis/revision_graph_3d/logs" "${RUN_DIR}"

record_exit() {
    local status=$?
    trap - EXIT
    case "${REPORT_TMP_DIR}" in
        "/tmp/gsmap-${USER}/${SLURM_JOB_ID}")
            rm -rf -- "${REPORT_TMP_DIR}"
            ;;
        *)
            printf 'Refusing to clean unexpected report scratch path: %s\n' "${REPORT_TMP_DIR}" >&2
            ;;
    esac
    {
        printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
        printf 'exit_code=%s\n' "${status}"
        printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    } >"${RUN_DIR}/report_status.txt"
    exit "${status}"
}
trap record_exit EXIT

mkdir -p "${REPORT_TMP_DIR}"
[[ -w "${REPORT_TMP_DIR}" ]]
findmnt -T "${REPORT_TMP_DIR}"
df -h "${REPORT_TMP_DIR}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
printf 'host=%s\n' "$(hostname)"
printf 'job_id=%s\n' "${SLURM_JOB_ID}"
printf 'commit=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
printf 'project=%s\n' "${PROJECT_NAME}"
printf 'visualization_spots=%s (no downsampling)\n' "${N_SPOTS}"

[[ -s "${RUN_DIR}/latent_to_gene/marker_score_blocks/manifest.json" ]]
[[ -s "${RUN_DIR}/cauchy_combination/${PROJECT_NAME}_combined_ldsc.parquet" ]]

"${GSMAP}" quick-mode \
    --workdir "${WORKDIR}" \
    --project-name "${PROJECT_NAME}" \
    --start-step report \
    --stop-step report \
    --dataset-type spatial3D \
    --annotation ManualAnnotation \
    --data-layer X \
    --latent-representation-niche emb_niche \
    --latent-representation-cell emb_cell \
    --spatial-key z_correction \
    --sumstats-config-file "${TRAIT_MANIFEST}" \
    --w-ld-dir "${W_LD_DIR}" \
    --snp-gene-weight-adata-path "${SNP_GENE_WEIGHTS}" \
    --marker-score-format gene_block \
    --memmap-tmp-dir "${REPORT_TMP_DIR}" \
    --downsampling-n-spots-pcc 100000 \
    --downsampling-n-spots-2d "${N_SPOTS}" \
    --downsampling-n-spots-3d "${N_SPOTS}" \
    --subsample-n-points "${N_SPOTS}" \
    --no-gpu

readonly REPORT_DATA="${RUN_DIR}/report_data"
readonly WEB_REPORT="${RUN_DIR}/gsmap_web_report"
[[ -s "${WEB_REPORT}/index.html" ]]
[[ -s "${WEB_REPORT}/report_meta.json" ]]
[[ -s "${REPORT_DATA}/spot_metadata.csv" ]]
[[ -s "${REPORT_DATA}/umap_data.csv" ]]
[[ -s "${REPORT_DATA}/spatial_3d/spatial_3d.h5ad" ]]
[[ -s "${WEB_REPORT}/spatial_3d/spatial_3d_anno_ManualAnnotation.html" ]]
[[ -s "${WEB_REPORT}/spatial_3d/spatial_3d_export_audit.json" ]]
[[ -s "${WEB_REPORT}/umap_scanpy/umap_audit.json" ]]
[[ "$(wc -l <"${REPORT_DATA}/spot_metadata.csv")" -eq $((N_SPOTS + 1)) ]]
[[ "$(wc -l <"${REPORT_DATA}/umap_data.csv")" -eq $((N_SPOTS + 1)) ]]
EXPECTED_REPORT_SPOTS="${N_SPOTS}" \
SPATIAL_3D_H5AD="${REPORT_DATA}/spatial_3d/spatial_3d.h5ad" \
SPATIAL_3D_AUDIT="${WEB_REPORT}/spatial_3d/spatial_3d_export_audit.json" \
UMAP_AUDIT="${WEB_REPORT}/umap_scanpy/umap_audit.json" \
    "${PYTHON}" - <<'PY'
import json
import os

import anndata as ad

path = os.environ["SPATIAL_3D_H5AD"]
expected = int(os.environ["EXPECTED_REPORT_SPOTS"])
adata = ad.read_h5ad(path, backed="r")
try:
    if adata.n_obs != expected:
        raise RuntimeError(f"Expected {expected} report observations, found {adata.n_obs}")
finally:
    adata.file.close()

with open(os.environ["SPATIAL_3D_AUDIT"], encoding="utf-8") as f:
    audit = json.load(f)
expected_assets = {
    "spatial_3d_anno_ManualAnnotation.html",
    "spatial_3d_trait_Height.html",
    "spatial_3d_trait_IQ.html",
    "spatial_3d_trait_CBX.html",
    "spatial_3d_trait_CTX.html",
}
if audit.get("input_observations") != expected:
    raise RuntimeError(f"Unexpected 3D input count: {audit}")
if audit.get("exported_observations") != expected:
    raise RuntimeError(f"3D visualization was downsampled: {audit}")
if set(audit.get("assets", {})) != expected_assets:
    raise RuntimeError(f"Unexpected 3D assets: {audit.get('assets')}")
for asset, count in audit["assets"].items():
    if count != expected:
        raise RuntimeError(f"{asset} contains {count}, expected {expected}")

with open(os.environ["UMAP_AUDIT"], encoding="utf-8") as f:
    umap_audit = json.load(f)
if umap_audit.get("observations") != expected:
    raise RuntimeError(f"UMAP visualization was downsampled: {umap_audit}")
if len(umap_audit.get("images", {})) != 10:
    raise RuntimeError(f"Expected 10 Scanpy UMAP panels: {umap_audit.get('images')}")
for asset, image in umap_audit["images"].items():
    if image.get("observations") != expected:
        raise RuntimeError(f"{asset} contains {image.get('observations')}, expected {expected}")
    if min(image.get("stored_dpi_x", 0), image.get("stored_dpi_y", 0)) < 319.9:
        raise RuntimeError(f"{asset} is below 320-DPI export tolerance: {image}")
    if min(image.get("pixel_width", 0), image.get("pixel_height", 0)) <= 0:
        raise RuntimeError(f"{asset} has invalid pixel dimensions: {image}")
PY
for trait in Height IQ CBX CTX; do
    [[ -s "${WEB_REPORT}/spatial_3d/spatial_3d_trait_${trait}.html" ]]
done
