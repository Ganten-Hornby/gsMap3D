#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-trait
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly ARM="${1:?usage: run_mouse_e115_supervised_trait_mapping.sbatch 2d|3d}"
readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly WORKDIR="${PROJECT_ROOT}/analysis/revision_graph_3d/runs"
readonly INPUT_ROOT="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/mouse_e115_supervised_trait_mapping"
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly GSMAP="${PROJECT_ROOT}/.venv/bin/gsmap"
readonly HOMOLOG_FILE=/storage/yangjianLab/lilounan/resources/mouse_human_homologs.txt
readonly SNP_GENE_WEIGHTS=/storage/yangjianLab/lilounan/resources/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad
readonly W_LD_DIR=/storage/yangjianLab/lilounan/resources/gsMap_resource/LDSC_resource/weights_hm3_no_hla
readonly TRAIT_MANIFEST="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/mouse_e115_heart_gwas_traits.yaml"

LATENT_ARGS=()
case "${ARM}" in
    2d)
        readonly PROJECT_NAME=mouse_e115_heart_supervised_trait_mapping_2d
        readonly SOURCE_LATENT="${WORKDIR}/mouse_e115_heart_sectionwise_2d_k10/find_latent_representations/Mouse_E11.5_heart_2d_add_latent.h5ad"
        readonly DATASET_TYPE=spatial2D
        readonly SPATIAL_KEY=aligned_spatial_2D_mm
        ;;
    3d)
        readonly PROJECT_NAME=mouse_e115_heart_supervised_trait_mapping_3d
        readonly SOURCE_LATENT="${WORKDIR}/mouse_e115_heart_section_aware_3d_k10/find_latent_representations/Mouse_E11.5_heart_add_latent.h5ad"
        readonly DATASET_TYPE=spatial3D
        readonly SPATIAL_KEY=aligned_spatial_3D_mm
        LATENT_ARGS=(
            --adjacent-slice-spatial-neighbors 200
            --n-adjacent-slices 1
            --cross-slice-selection-mode physical_window
            --cross-slice-max-distance 0.010001
            --cross-slice-z-coordinate 2
            --cross-slice-z-scale 1.0
            --cross-slice-marker-score-strategy hierarchical_pool
        )
        ;;
    *)
        printf 'Unknown arm: %s (expected 2d or 3d)\n' "${ARM}" >&2
        exit 2
        ;;
esac
readonly -a LATENT_ARGS

readonly ARM_INPUT_DIR="${INPUT_ROOT}/${ARM}"
readonly H5AD_MANIFEST="${INPUT_ROOT}/${ARM}_sections.yaml"
readonly INPUT_AUDIT="${INPUT_ROOT}/${ARM}_preparation_audit.json"
readonly RUN_DIR="${WORKDIR}/${PROJECT_NAME}"
mkdir -p "${RUN_DIR}" "${PROJECT_ROOT}/analysis/revision_graph_3d/logs"

record_exit() {
    local status=$?
    {
        printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
        printf 'arm=%s\n' "${ARM}"
        printf 'exit_code=%s\n' "${status}"
        printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    } >"${RUN_DIR}/slurm_status.txt"
    exit "${status}"
}
trap record_exit EXIT

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
printf 'host=%s\n' "$(hostname)"
printf 'job_id=%s\n' "${SLURM_JOB_ID}"
printf 'arm=%s\n' "${ARM}"
printf 'commit=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv

"${PYTHON}" "${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/prepare_mouse_e115_supervised_trait_inputs.py" \
    --source "${SOURCE_LATENT}" \
    --homolog-file "${HOMOLOG_FILE}" \
    --output-dir "${ARM_INPUT_DIR}" \
    --manifest "${H5AD_MANIFEST}" \
    --audit "${INPUT_AUDIT}"

"${GSMAP}" latent-to-gene \
    --workdir "${WORKDIR}" \
    --project-name "${PROJECT_NAME}" \
    --dataset-type "${DATASET_TYPE}" \
    --h5ad-yaml "${H5AD_MANIFEST}" \
    --annotation heart_regions \
    --data-layer X \
    --latent-representation-niche emb_niche \
    --latent-representation-cell emb_cell \
    --spatial-key "${SPATIAL_KEY}" \
    --spatial-neighbors 301 \
    --homogeneous-neighbors 21 \
    --cell-embedding-similarity-threshold 0.0 \
    --spatial-domain-similarity-threshold 0.6 \
    --homogeneous-selection-mode soft_priority \
    --no-high-quality-filter \
    --use-gpu \
    --mkscore-batch-size 500 \
    --rank-read-workers 6 \
    --mkscore-compute-workers 2 \
    --mkscore-write-workers 2 \
    "${LATENT_ARGS[@]}"

"${GSMAP}" spatial-ldsc \
    --workdir "${WORKDIR}" \
    --project-name "${PROJECT_NAME}" \
    --sumstats-config-file "${TRAIT_MANIFEST}" \
    --w-ld-dir "${W_LD_DIR}" \
    --snp-gene-weight-adata-path "${SNP_GENE_WEIGHTS}" \
    --use-gpu \
    --ldsc-read-workers 5 \
    --ldsc-compute-workers 5 \
    --spots-per-chunk-quick-mode 50

"${GSMAP}" cauchy-combination \
    --workdir "${WORKDIR}" \
    --project-name "${PROJECT_NAME}" \
    --sumstats-config-file "${TRAIT_MANIFEST}" \
    --annotation heart_regions
