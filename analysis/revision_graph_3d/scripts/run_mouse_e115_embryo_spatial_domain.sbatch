#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-spatial-domain-embed
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err
# 11950209/11950210 reached the 160G ceiling; this doubles the 16-worker trial.
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G

set -Eeuo pipefail

readonly ARM="${1:?usage: run_mouse_e115_embryo_spatial_domain.sbatch 2d|3d}"
readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly LDSC_WORKER_HELPER="${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/ldsc_worker_split.sh"
readonly INPUT_H5AD="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/Mouse_E11.5_embryo_graph_ready.h5ad"
readonly WORKDIR="${PROJECT_ROOT}/analysis/revision_graph_3d/runs"
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly GSMAP="${PROJECT_ROOT}/.venv/bin/gsmap"
readonly HOMOLOG_FILE=/storage/yangjianLab/lilounan/resources/mouse_human_homologs.txt
readonly TRAIT_MANIFEST="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/mouse_e115_embryo_gwas_traits.yaml"
readonly SNP_GENE_WEIGHTS=/storage/yangjianLab/lilounan/resources/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad
readonly W_LD_DIR=/storage/yangjianLab/lilounan/resources/gsMap_resource/LDSC_resource/weights_hm3_no_hla
readonly LOCAL_TMP_ROOT="/tmp/gsmap-${USER}"
readonly LOCAL_TMP_DIR="${LOCAL_TMP_ROOT}/${SLURM_JOB_ID}"

source "${LDSC_WORKER_HELPER}"
configure_ldsc_worker_split "${SLURM_CPUS_PER_TASK}"

case "${ARM}" in
    2d)
        readonly PROJECT_NAME=mouse_e115_embryo_sectionwise_2d_k10_spatial_domain
        readonly GRAPH_MODE=sectionwise_2d
        readonly MARKER_SCORE_GENE_BLOCK_SIZE=2048
        ;;
    3d)
        readonly PROJECT_NAME=mouse_e115_embryo_section_aware_3d_k10_spatial_domain
        readonly GRAPH_MODE=section_aware_3d
        readonly MARKER_SCORE_GENE_BLOCK_SIZE=2400
        ;;
    *)
        printf 'Unknown arm: %s (expected 2d or 3d)\n' "${ARM}" >&2
        exit 2
        ;;
esac

readonly RUN_DIR="${WORKDIR}/${PROJECT_NAME}"
readonly LATENT2GENE_DIR="${RUN_DIR}/latent_to_gene"
mkdir -p "${RUN_DIR}" "${LATENT2GENE_DIR}" "${PROJECT_ROOT}/analysis/revision_graph_3d/logs"

record_exit() {
    local status=$?
    {
        printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
        printf 'arm=%s\n' "${ARM}"
        printf 'annotation=spatial_domain\n'
        printf 'exit_code=%s\n' "${status}"
        printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    } >"${RUN_DIR}/slurm_status.txt"
    case "${LOCAL_TMP_DIR}" in
        "/tmp/gsmap-${USER}/${SLURM_JOB_ID}")
            rm -rf -- "${LOCAL_TMP_DIR}" || printf 'Failed to clean %s\n' "${LOCAL_TMP_DIR}" >&2
            ;;
        *)
            printf 'Refusing to clean unexpected temp path: %s\n' "${LOCAL_TMP_DIR}" >&2
            ;;
    esac
    exit "${status}"
}
trap record_exit EXIT

if [[ ! -d /tmp || ! -w /tmp ]]; then
    printf '/tmp is not writable on host %s\n' "$(hostname)" >&2
    exit 1
fi
mkdir -p "${LOCAL_TMP_DIR}"
chmod 700 "${LOCAL_TMP_ROOT}" "${LOCAL_TMP_DIR}"

if command -v lfs >/dev/null 2>&1; then
    stripe_count="$(lfs getstripe -d -c "${LATENT2GENE_DIR}" | tr -d '[:space:]')"
    if [[ "${stripe_count}" != "4" ]]; then
        lfs setstripe -c 4 "${LATENT2GENE_DIR}"
    fi
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
printf 'host=%s\n' "$(hostname)"
printf 'job_id=%s\n' "${SLURM_JOB_ID}"
printf 'arm=%s\n' "${ARM}"
printf 'annotation=spatial_domain\n'
printf 'ldsc_workers=read:%s compute:%s\n' \
    "${LDSC_READ_WORKERS}" "${LDSC_COMPUTE_WORKERS}"
printf 'commit=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
printf 'memmap_tmp_dir=%s\n' "${LOCAL_TMP_DIR}"
findmnt -T "${LOCAL_TMP_DIR}"
df -h "${LOCAL_TMP_DIR}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv

"${PYTHON}" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
assert torch.version.cuda is not None and torch.version.cuda.startswith("12.")
assert torch.cuda.is_available()
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

[[ -s "${INPUT_H5AD}" ]]

"${GSMAP}" quick-mode \
    --workdir "${WORKDIR}" \
    --project-name "${PROJECT_NAME}" \
    --h5ad-path "${INPUT_H5AD}" \
    --dataset-type spatial3D \
    --data-layer X \
    --spatial-key z_correction \
    --annotation spatial_domain \
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
    --marker-score-format gene_block \
    --marker-score-gene-block-size "${MARKER_SCORE_GENE_BLOCK_SIZE}" \
    --ldsc-read-workers "${LDSC_READ_WORKERS}" \
    --ldsc-compute-workers "${LDSC_COMPUTE_WORKERS}" \
    --memmap-tmp-dir "${LOCAL_TMP_DIR}" \
    --use-gpu \
    --start-step find_latent \
    --stop-step cauchy
