#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-gb-ssd-bench
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly SOURCE_PROJECT=mouse_e115_embryo_sectionwise_2d_k10_spatial_domain
readonly SOURCE_LATENT="${PROJECT_ROOT}/analysis/revision_graph_3d/runs/${SOURCE_PROJECT}/latent_to_gene"
readonly LOCAL_TMP_ROOT="/tmp/gsmap-${USER}"
readonly LOCAL_TMP_DIR="${LOCAL_TMP_ROOT}/${SLURM_JOB_ID}"
readonly BENCH_WORKDIR="${LOCAL_TMP_DIR}/workdir"
readonly BENCH_PROJECT=mouse_e115_gene_block_ssd_benchmark
readonly BENCH_PROJECT_DIR="${BENCH_WORKDIR}/${BENCH_PROJECT}"
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly GSMAP="${PROJECT_ROOT}/.venv/bin/gsmap"
readonly WORKER_HELPER="${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/ldsc_worker_split.sh"
readonly SNP_GENE_WEIGHTS=/storage/yangjianLab/lilounan/resources/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad
readonly W_LD_DIR=/storage/yangjianLab/lilounan/resources/gsMap_resource/LDSC_resource/weights_hm3_no_hla
readonly SUMSTATS=/storage/yangjianLab/lilounan/resources/GWAS_trait/LDSC/CBX_h_0.3.P1.sumstats.gz

source "${WORKER_HELPER}"
configure_ldsc_worker_split "${SLURM_CPUS_PER_TASK}"

cleanup() {
    local status=$?
    case "${LOCAL_TMP_DIR}" in
        "/tmp/gsmap-${USER}/${SLURM_JOB_ID}")
            rm -rf -- "${LOCAL_TMP_DIR}" || true
            ;;
        *)
            printf 'Refusing to clean unexpected path: %s\n' "${LOCAL_TMP_DIR}" >&2
            ;;
    esac
    exit "${status}"
}
trap cleanup EXIT

mkdir -p "${LOCAL_TMP_DIR}" "${BENCH_PROJECT_DIR}"
chmod 700 "${LOCAL_TMP_ROOT}" "${LOCAL_TMP_DIR}"
ln -s "${SOURCE_LATENT}" "${BENCH_PROJECT_DIR}/latent_to_gene"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
printf 'host=%s\n' "$(hostname)"
printf 'job_id=%s\n' "${SLURM_JOB_ID}"
printf 'commit=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
printf 'source_format=gene_block\n'
printf 'storage=job-local-ssd:%s\n' "${LOCAL_TMP_DIR}"
printf 'cell_range=0:250000\n'
printf 'ldsc_workers=read:%s compute:%s\n' \
    "${LDSC_READ_WORKERS}" "${LDSC_COMPUTE_WORKERS}"
findmnt -T "${LOCAL_TMP_DIR}"
df -h "${LOCAL_TMP_DIR}"

"${PYTHON}" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
assert torch.version.cuda is not None and torch.version.cuda.startswith("12.")
assert torch.cuda.is_available()
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

"${GSMAP}" spatial-ldsc \
    --workdir "${BENCH_WORKDIR}" \
    --project-name "${BENCH_PROJECT}" \
    --marker-score-format gene_block \
    --memmap-tmp-dir "${LOCAL_TMP_DIR}" \
    --ldsc-read-workers "${LDSC_READ_WORKERS}" \
    --ldsc-compute-workers "${LDSC_COMPUTE_WORKERS}" \
    --spots-per-chunk-quick-mode 50 \
    --trait-name CBX \
    --sumstats-file "${SUMSTATS}" \
    --w-ld-dir "${W_LD_DIR}" \
    --snp-gene-weight-adata-path "${SNP_GENE_WEIGHTS}" \
    --cell-indices-range 0 250000 \
    --use-gpu
