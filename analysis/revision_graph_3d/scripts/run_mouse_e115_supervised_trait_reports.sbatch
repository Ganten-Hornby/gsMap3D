#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-trait-reports
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly WORKDIR="${PROJECT_ROOT}/analysis/revision_graph_3d/runs"
readonly INPUT_ROOT="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/mouse_e115_supervised_trait_mapping"
readonly GSMAP="${PROJECT_ROOT}/.venv/bin/gsmap"
readonly SNP_GENE_WEIGHTS=/storage/yangjianLab/lilounan/resources/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad
readonly W_LD_DIR=/storage/yangjianLab/lilounan/resources/gsMap_resource/LDSC_resource/weights_hm3_no_hla
readonly TRAIT_MANIFEST="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/mouse_e115_heart_gwas_traits.yaml"
readonly REPORT_RUN_DIR="${WORKDIR}/mouse_e115_heart_supervised_trait_reports"

mkdir -p "${REPORT_RUN_DIR}" "${PROJECT_ROOT}/analysis/revision_graph_3d/logs"

record_exit() {
    local status=$?
    {
        printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
        printf 'exit_code=%s\n' "${status}"
        printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    } >"${REPORT_RUN_DIR}/slurm_status.txt"
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

readonly STATUS_2D="${WORKDIR}/mouse_e115_heart_supervised_trait_mapping_2d/slurm_status.txt"
readonly STATUS_3D="${WORKDIR}/mouse_e115_heart_supervised_trait_mapping_3d/slurm_status.txt"
if ! upstream_succeeded "${STATUS_2D}" || ! upstream_succeeded "${STATUS_3D}"; then
    printf 'Both upstream trait-mapping jobs must complete successfully before report generation.\n' >&2
    printf '2D status: %s\n3D status: %s\n' "${STATUS_2D}" "${STATUS_3D}" >&2
    exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

run_report() {
    local arm=$1
    local project_name dataset_type h5ad_manifest spatial_key
    local -a spatial_args=()

    case "${arm}" in
        2d)
            project_name=mouse_e115_heart_supervised_trait_mapping_2d
            dataset_type=spatial2D
            h5ad_manifest="${INPUT_ROOT}/2d_sections.yaml"
            spatial_key=aligned_spatial_2D_mm
            ;;
        3d)
            project_name=mouse_e115_heart_supervised_trait_mapping_3d
            dataset_type=spatial3D
            h5ad_manifest="${INPUT_ROOT}/3d_sections.yaml"
            spatial_key=aligned_spatial_3D_mm
            spatial_args=(
                --n-adjacent-slices 1
                --cross-slice-marker-score-strategy hierarchical_pool
            )
            ;;
        *)
            printf 'Unknown arm: %s\n' "${arm}" >&2
            return 2
            ;;
    esac

    "${GSMAP}" quick-mode \
        --workdir "${WORKDIR}" \
        --project-name "${project_name}" \
        --start-step report \
        --stop-step report \
        --dataset-type "${dataset_type}" \
        --h5ad-yaml "${h5ad_manifest}" \
        --annotation heart_regions \
        --data-layer X \
        --latent-representation-niche emb_niche \
        --latent-representation-cell emb_cell \
        --spatial-key "${spatial_key}" \
        "${spatial_args[@]}" \
        --sumstats-config-file "${TRAIT_MANIFEST}" \
        --w-ld-dir "${W_LD_DIR}" \
        --snp-gene-weight-adata-path "${SNP_GENE_WEIGHTS}" \
        --no-gpu
}

printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
printf 'host=%s\n' "$(hostname)"
printf 'job_id=%s\n' "${SLURM_JOB_ID}"
printf 'commit=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"

run_report 2d
run_report 3d
