#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-2d-ldsc-32c
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err
# 11950209 Height: 15:12:49 at ~120 cells/s with the old 1-compute-worker cap.
# 11950209/11950210 peaked at 159.970/159.968 GiB; 2x workers * 1.20 -> 400G.
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly RUNS_DIR="${PROJECT_ROOT}/analysis/revision_graph_3d/runs"
readonly SOURCE_PROJECT=mouse_e115_embryo_sectionwise_2d_k10_spatial_domain
readonly SOURCE_PROJECT_DIR="${RUNS_DIR}/${SOURCE_PROJECT}"
readonly BENCHMARK_PROJECT="mouse_e115_embryo_2d_ldsc_32c_benchmark_${SLURM_JOB_ID}"
readonly BENCHMARK_DIR="${RUNS_DIR}/${BENCHMARK_PROJECT}"
readonly LDSC_WORKER_HELPER="${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/ldsc_worker_split.sh"
readonly HEIGHT_SUMSTATS=/storage/yangjianLab/lilounan/resources/GWAS_trait/LDSC/GIANT_EUR_Height_2022_Nature.sumstats.gz
readonly SNP_GENE_WEIGHTS=/storage/yangjianLab/lilounan/resources/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad
readonly W_LD_DIR=/storage/yangjianLab/lilounan/resources/gsMap_resource/LDSC_resource/weights_hm3_no_hla
readonly LOCAL_TMP_ROOT="/tmp/gsmap-${USER}"
readonly LOCAL_TMP_DIR="${LOCAL_TMP_ROOT}/${SLURM_JOB_ID}"

source "${LDSC_WORKER_HELPER}"
configure_ldsc_worker_split "${SLURM_CPUS_PER_TASK}"

if [[ -e "${BENCHMARK_DIR}" ]]; then
    printf 'Refusing to reuse benchmark output directory: %s\n' "${BENCHMARK_DIR}" >&2
    exit 2
fi
[[ -s "${SOURCE_PROJECT_DIR}/latent_to_gene/marker_score_blocks/manifest.json" ]]
[[ -s "${SOURCE_PROJECT_DIR}/latent_to_gene/concatenated_latent_adata.h5ad" ]]

mkdir -p "${BENCHMARK_DIR}" "${LOCAL_TMP_DIR}" "${PROJECT_ROOT}/analysis/revision_graph_3d/logs"
chmod 700 "${LOCAL_TMP_ROOT}" "${LOCAL_TMP_DIR}"
ln -s -- "${SOURCE_PROJECT_DIR}/latent_to_gene" "${BENCHMARK_DIR}/latent_to_gene"

record_exit() {
    local status=$?
    printf 'benchmark_exit_code=%s\n' "${status}"
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
printf 'host=%s\n' "$(hostname)"
printf 'job_id=%s\n' "${SLURM_JOB_ID}"
printf 'source_project=%s\n' "${SOURCE_PROJECT_DIR}"
printf 'benchmark_project=%s\n' "${BENCHMARK_DIR}"
printf 'ldsc_workers=read:%s compute:%s forced_compute:%s\n' \
    "${LDSC_READ_WORKERS}" "${LDSC_COMPUTE_WORKERS}" \
    "${GSMAP_LDSC_FORCE_COMPUTE_WORKERS}"
printf 'commit=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
findmnt -T "${LOCAL_TMP_DIR}"
df -h "${LOCAL_TMP_DIR}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv

uv run --extra cuda128 --locked python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
assert torch.version.cuda is not None and torch.version.cuda.startswith("12.")
assert torch.cuda.is_available()
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

uv run --extra cuda128 --locked gsmap spatial-ldsc \
    --workdir "${RUNS_DIR}" \
    --project-name "${BENCHMARK_PROJECT}" \
    --trait-name Height \
    --sumstats-file "${HEIGHT_SUMSTATS}" \
    --w-ld-dir "${W_LD_DIR}" \
    --snp-gene-weight-adata-path "${SNP_GENE_WEIGHTS}" \
    --marker-score-format gene_block \
    --ldsc-read-workers "${LDSC_READ_WORKERS}" \
    --ldsc-compute-workers "${LDSC_COMPUTE_WORKERS}" \
    --spots-per-chunk-quick-mode 50 \
    --memmap-tmp-dir "${LOCAL_TMP_DIR}" \
    --use-gpu

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
printf 'result_dir=%s\n' "${BENCHMARK_DIR}/spatial_ldsc"
