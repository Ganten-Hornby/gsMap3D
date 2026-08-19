#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-spatial-domain-report
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly ARM="${1:?usage: run_mouse_e115_embryo_spatial_domain_report.sbatch 2d|3d [workdir] [marker-score-format] [run|audit-only]}"
readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly INPUT_H5AD="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/Mouse_E11.5_embryo_graph_ready.h5ad"
readonly WORKDIR="${2:-${PROJECT_ROOT}/analysis/revision_graph_3d/runs}"
readonly MARKER_SCORE_FORMAT="${3:-memmap}"
readonly REPORT_ACTION="${4:-run}"
readonly GSMAP="${PROJECT_ROOT}/.venv/bin/gsmap"
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly HOMOLOG_FILE=/storage/yangjianLab/lilounan/resources/mouse_human_homologs.txt
readonly TRAIT_MANIFEST="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/mouse_e115_embryo_gwas_traits.yaml"
readonly SNP_GENE_WEIGHTS=/storage/yangjianLab/lilounan/resources/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad
readonly W_LD_DIR=/storage/yangjianLab/lilounan/resources/gsMap_resource/LDSC_resource/weights_hm3_no_hla
readonly N_SPOTS=6993667
readonly REPORT_TMP_DIR="/tmp/gsmap-${USER}/${SLURM_JOB_ID}"

case "${WORKDIR}" in
    "${PROJECT_ROOT}/analysis/revision_graph_3d/runs" | \
        "${PROJECT_ROOT}/analysis/revision_graph_3d/runs/corrected_training_staging")
        ;;
    *)
        printf 'Unsupported workdir: %s\n' "${WORKDIR}" >&2
        exit 2
        ;;
esac

case "${MARKER_SCORE_FORMAT}" in
    memmap | gene_block)
        ;;
    *)
        printf 'Unsupported marker-score format: %s\n' "${MARKER_SCORE_FORMAT}" >&2
        exit 2
        ;;
esac

case "${REPORT_ACTION}" in
    run | audit-only)
        ;;
    *)
        printf 'Unsupported report action: %s\n' "${REPORT_ACTION}" >&2
        exit 2
        ;;
esac

case "${ARM}" in
    2d)
        readonly PROJECT_NAME=mouse_e115_embryo_sectionwise_2d_k10_spatial_domain
        readonly GRAPH_MODE=sectionwise_2d
        ;;
    3d)
        readonly PROJECT_NAME=mouse_e115_embryo_section_aware_3d_k10_spatial_domain
        readonly GRAPH_MODE=section_aware_3d
        ;;
    *)
        printf 'Unknown arm: %s (expected 2d or 3d)\n' "${ARM}" >&2
        exit 2
        ;;
esac

readonly RUN_DIR="${WORKDIR}/${PROJECT_NAME}"
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
        printf 'arm=%s\n' "${ARM}"
        printf 'exit_code=%s\n' "${status}"
        printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    } >"${RUN_DIR}/report_status.txt"
    exit "${status}"
}
trap record_exit EXIT

mkdir -p "${REPORT_TMP_DIR}"
chmod 700 "${REPORT_TMP_DIR}"
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
printf 'arm=%s\n' "${ARM}"
printf 'commit=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
printf 'project=%s\n' "${PROJECT_NAME}"
printf 'report_action=%s\n' "${REPORT_ACTION}"
printf 'source_spots=%s\n' "${N_SPOTS}"

if [[ "${MARKER_SCORE_FORMAT}" == "gene_block" ]]; then
    readonly MARKER_SCORE_METADATA="${RUN_DIR}/latent_to_gene/marker_score_blocks/manifest.json"
else
    [[ -s "${RUN_DIR}/latent_to_gene/marker_scores.dat" ]]
    readonly MARKER_SCORE_METADATA="${RUN_DIR}/latent_to_gene/marker_scores.meta.json"
fi
[[ -s "${MARKER_SCORE_METADATA}" ]]
[[ -s "${RUN_DIR}/cauchy_combination/${PROJECT_NAME}_combined_ldsc.parquet" ]]

EXPECTED_REPORT_SPOTS="$("${PYTHON}" -c \
    'import json, sys; print(int(json.load(open(sys.argv[1], encoding="utf-8"))["shape"][0]))' \
    "${MARKER_SCORE_METADATA}")"
readonly EXPECTED_REPORT_SPOTS
printf 'report_spots=%s (all scored observations; no visualization downsampling)\n' \
    "${EXPECTED_REPORT_SPOTS}"

if [[ "${REPORT_ACTION}" == "run" ]]; then
    "${GSMAP}" quick-mode \
        --workdir "${WORKDIR}" \
        --project-name "${PROJECT_NAME}" \
        --h5ad-path "${INPUT_H5AD}" \
        --start-step report \
        --stop-step report \
        --dataset-type spatial3D \
        --data-layer X \
        --spatial-key z_correction \
        --annotation spatial_domain \
        --latent-representation-cell emb_cell \
        --latent-representation-niche emb_niche \
        --homolog-file "${HOMOLOG_FILE}" \
        --no-high-quality-cell-qc \
        --graph-mode "${GRAPH_MODE}" \
        --graph-section-key slice_id \
        --graph-z-scale 1.0 \
        --inference-chunk-size 100000 \
        --two-stage \
        --n-cell-training 100000 \
        --n-adjacent-slices 1 \
        --cross-slice-selection-mode physical_window \
        --cross-slice-max-distance 5.000001 \
        --cross-slice-z-coordinate 2 \
        --cross-slice-z-scale 1.0 \
        --sumstats-config-file "${TRAIT_MANIFEST}" \
        --w-ld-dir "${W_LD_DIR}" \
        --snp-gene-weight-adata-path "${SNP_GENE_WEIGHTS}" \
        --marker-score-format "${MARKER_SCORE_FORMAT}" \
        --memmap-tmp-dir "${REPORT_TMP_DIR}" \
        --downsampling-n-spots-pcc 100000 \
        --downsampling-n-spots-2d "${EXPECTED_REPORT_SPOTS}" \
        --downsampling-n-spots-3d "${EXPECTED_REPORT_SPOTS}" \
        --subsample-n-points "${EXPECTED_REPORT_SPOTS}" \
        --no-gpu
fi

readonly REPORT_DATA="${RUN_DIR}/report_data"
readonly WEB_REPORT="${RUN_DIR}/gsmap_web_report"
[[ -s "${WEB_REPORT}/index.html" ]]
[[ -s "${WEB_REPORT}/report_meta.json" ]]
[[ -s "${REPORT_DATA}/spot_metadata.csv" ]]
[[ -s "${REPORT_DATA}/umap_data.csv" ]]
[[ -s "${REPORT_DATA}/spatial_3d/spatial_3d_manifest.json" ]]
[[ -s "${REPORT_DATA}/spatial_3d/spatial_3d_display_spots.parquet" ]]
[[ -s "${WEB_REPORT}/js_data/sample_spatial_manifest.json" ]]
[[ -s "${WEB_REPORT}/js_data/sample_index.js" ]]
[[ -s "${WEB_REPORT}/umap_scanpy/umap_audit.json" ]]
[[ "$(wc -l <"${REPORT_DATA}/spot_metadata.csv")" -eq $((EXPECTED_REPORT_SPOTS + 1)) ]]
[[ "$(wc -l <"${REPORT_DATA}/umap_data.csv")" -eq $((EXPECTED_REPORT_SPOTS + 1)) ]]

EXPECTED_REPORT_SPOTS="${EXPECTED_REPORT_SPOTS}" \
SPATIAL_3D_MANIFEST="${REPORT_DATA}/spatial_3d/spatial_3d_manifest.json" \
SPATIAL_3D_DISPLAY="${REPORT_DATA}/spatial_3d/spatial_3d_display_spots.parquet" \
SAMPLE_SPATIAL_MANIFEST="${WEB_REPORT}/js_data/sample_spatial_manifest.json" \
SAMPLE_INDEX="${WEB_REPORT}/js_data/sample_index.js" \
WEB_REPORT="${WEB_REPORT}" \
UMAP_AUDIT="${WEB_REPORT}/umap_scanpy/umap_audit.json" \
    "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

expected = int(os.environ["EXPECTED_REPORT_SPOTS"])
web_report = Path(os.environ["WEB_REPORT"])

with open(os.environ["SPATIAL_3D_MANIFEST"], encoding="utf-8") as handle:
    spatial_manifest = json.load(handle)
signature = spatial_manifest.get("signature", {})
if signature.get("n_total") != expected or signature.get("n_display") != expected:
    raise RuntimeError(f"3D visualization was downsampled: {signature}")
if signature.get("annotations") != ["spatial_domain"]:
    raise RuntimeError(f"Unexpected 3D annotations: {signature.get('annotations')}")
expected_traits = {"Height", "IQ", "MCHC", "CBX", "CTX"}
if set(signature.get("traits", [])) != expected_traits:
    raise RuntimeError(f"Unexpected 3D traits: {signature.get('traits')}")
expected_spatial_outputs = {
    "js_data/spatial_3d/spatial_3d_base.js",
    "js_data/spatial_3d/spatial_3d_index.js",
    "js_data/spatial_3d/annotation_spatial_domain.js",
    *(f"js_data/spatial_3d/trait_{trait}.js" for trait in expected_traits),
}
if set(spatial_manifest.get("outputs", [])) != expected_spatial_outputs:
    raise RuntimeError(f"Unexpected 3D outputs: {spatial_manifest.get('outputs')}")
for relative_path in expected_spatial_outputs:
    output_path = web_report / relative_path
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Missing 3D payload: {output_path}")
display_rows = pq.ParquetFile(os.environ["SPATIAL_3D_DISPLAY"]).metadata.num_rows
if display_rows != expected:
    raise RuntimeError(f"3D display parquet contains {display_rows}, expected {expected}")

with open(os.environ["SAMPLE_SPATIAL_MANIFEST"], encoding="utf-8") as handle:
    sample_manifest = json.load(handle)
sample_index_text = Path(os.environ["SAMPLE_INDEX"]).read_text(encoding="utf-8")
prefix = "window.GSMAP_SAMPLE_INDEX="
if not sample_index_text.startswith(prefix):
    raise RuntimeError("Unexpected sample_index.js prefix")
sample_index = json.loads(sample_index_text[len(prefix) :].split(";", 1)[0])
if len(sample_index) != 84:
    raise RuntimeError(f"Expected 84 sample payloads, found {len(sample_index)}")
expected_sample_outputs = {"sample_index.js"}
sample_spots = 0
for sample, info in sample_index.items():
    if info.get("n_spots") != info.get("n_spots_original"):
        raise RuntimeError(f"Sample {sample} was downsampled: {info}")
    sample_spots += info["n_spots"]
    expected_sample_outputs.add(info["base_file"])
    if set(info.get("trait_files", {})) != expected_traits:
        raise RuntimeError(f"Unexpected traits for sample {sample}: {info.get('trait_files')}")
    expected_sample_outputs.update(item["file"] for item in info["trait_files"].values())
if sample_spots != expected:
    raise RuntimeError(f"Sample payloads contain {sample_spots} spots, expected {expected}")
if set(sample_manifest.get("outputs", [])) != expected_sample_outputs:
    raise RuntimeError("Sample payload manifest does not match sample_index.js")
for relative_path in expected_sample_outputs:
    output_path = web_report / "js_data" / relative_path
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Missing sample payload: {output_path}")

with open(os.environ["UMAP_AUDIT"], encoding="utf-8") as handle:
    umap_audit = json.load(handle)
if umap_audit.get("observations") != expected:
    raise RuntimeError(f"UMAP visualization was downsampled: {umap_audit}")
if len(umap_audit.get("images", {})) != 12:
    raise RuntimeError(f"Expected 12 Scanpy UMAP panels: {umap_audit.get('images')}")
for asset, image in umap_audit["images"].items():
    if image.get("observations") != expected:
        raise RuntimeError(f"{asset} contains {image.get('observations')}, expected {expected}")
    if min(image.get("stored_dpi_x", 0), image.get("stored_dpi_y", 0)) < 319.9:
        raise RuntimeError(f"{asset} is below 320-DPI export tolerance: {image}")
PY
