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
readonly LOCAL_LATENT="${LOCAL_TMP_DIR}/latent_to_gene"
readonly LOCAL_BLOCKS="${LOCAL_LATENT}/marker_score_blocks"
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly GSMAP="${PROJECT_ROOT}/.venv/bin/gsmap"
readonly SNP_GENE_WEIGHTS=/storage/yangjianLab/lilounan/resources/1000GP3_GRCh37_gencode_v46_protein_coding_ldscore_weights.h5ad
readonly W_LD_DIR=/storage/yangjianLab/lilounan/resources/gsMap_resource/LDSC_resource/weights_hm3_no_hla
readonly SUMSTATS=/storage/yangjianLab/lilounan/resources/GWAS_trait/LDSC/CBX_h_0.3.P1.sumstats.gz
readonly -a WORKER_SPLITS=("4:28" "8:24" "16:16" "24:8" "28:4" "30:2" "31:1")

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

mkdir -p "${LOCAL_TMP_DIR}" "${BENCH_WORKDIR}" "${LOCAL_BLOCKS}"
chmod 700 "${LOCAL_TMP_ROOT}" "${LOCAL_TMP_DIR}"

for source_path in "${SOURCE_LATENT}"/*; do
    if [[ "$(basename "${source_path}")" != marker_score_blocks ]]; then
        ln -s "${source_path}" "${LOCAL_LATENT}/$(basename "${source_path}")"
    fi
done

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
printf 'worker_splits=%s\n' "${WORKER_SPLITS[*]}"
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

required_bytes=$(find "${SOURCE_LATENT}/marker_score_blocks" -maxdepth 1 -type f \
    -name 'marker_scores_genes_*.dat' -printf '%s\n' | awk '{total += $1} END {print total + 0}')
available_bytes=$(df --output=avail -B1 "${LOCAL_TMP_DIR}" | tail -n 1 | tr -d ' ')
reserve_bytes=$((20 * 1024 * 1024 * 1024))
if ((available_bytes < required_bytes + reserve_bytes)); then
    printf 'Insufficient SSD: need %s bytes plus %s reserve, found %s\n' \
        "${required_bytes}" "${reserve_bytes}" "${available_bytes}" >&2
    exit 1
fi

printf 'staging_started_at=%s\n' "$(date --iso-8601=seconds)"
cp --reflink=never -- "${SOURCE_LATENT}/marker_score_blocks/manifest.json" "${LOCAL_BLOCKS}/"
cp --reflink=never -- \
    "${SOURCE_LATENT}"/marker_score_blocks/marker_scores_genes_*.dat \
    "${LOCAL_BLOCKS}/"
printf 'staging_finished_at=%s\n' "$(date --iso-8601=seconds)"
du -sh "${LOCAL_BLOCKS}"

readonly SUMMARY_PATH="${LOCAL_TMP_DIR}/worker_split_summary.tsv"
printf 'read_workers\tcompute_workers\tstatus\twall_seconds\tcells_per_second\tGPU_OOM\tmean_gpu_util_pct\tmax_gpu_util_pct\tmax_gpu_mem_mib\n' > "${SUMMARY_PATH}"

for worker_split in "${WORKER_SPLITS[@]}"; do
    IFS=: read -r read_workers compute_workers <<< "${worker_split}"
    benchmark_project="mouse_e115_gene_block_ssd_r${read_workers}_c${compute_workers}"
    benchmark_project_dir="${BENCH_WORKDIR}/${benchmark_project}"
    run_log="${LOCAL_TMP_DIR}/ratio_r${read_workers}_c${compute_workers}.log"
    gpu_log="${LOCAL_TMP_DIR}/ratio_r${read_workers}_c${compute_workers}.gpu.csv"
    mkdir -p "${benchmark_project_dir}"
    ln -s "${LOCAL_LATENT}" "${benchmark_project_dir}/latent_to_gene"

    export GSMAP_LDSC_FORCE_COMPUTE_WORKERS="${compute_workers}"
    printf 'ratio_started_at=%s read=%s compute=%s\n' \
        "$(date --iso-8601=seconds)" "${read_workers}" "${compute_workers}"
    nvidia-smi \
        --query-gpu=timestamp,utilization.gpu,memory.used \
        --format=csv,noheader,nounits \
        --loop=1 > "${gpu_log}" &
    gpu_monitor_pid=$!
    run_started=$(date +%s)
    set +e
    "${GSMAP}" spatial-ldsc \
        --workdir "${BENCH_WORKDIR}" \
        --project-name "${benchmark_project}" \
        --marker-score-format gene_block \
        --ldsc-read-workers "${read_workers}" \
        --ldsc-compute-workers "${compute_workers}" \
        --spots-per-chunk-quick-mode 50 \
        --trait-name CBX \
        --sumstats-file "${SUMSTATS}" \
        --w-ld-dir "${W_LD_DIR}" \
        --snp-gene-weight-adata-path "${SNP_GENE_WEIGHTS}" \
        --cell-indices-range 0 250000 \
        --use-gpu 2>&1 | tee "${run_log}"
    run_status=${PIPESTATUS[0]}
    set -e
    run_finished=$(date +%s)
    kill "${gpu_monitor_pid}" 2>/dev/null || true
    wait "${gpu_monitor_pid}" 2>/dev/null || true

    cells_per_second=$( \
        rg -o 'Overall speed: [0-9,]+ cells/s' "${run_log}" \
            | tail -n 1 | sed -E 's/.*: ([0-9,]+) cells\/s/\1/' | tr -d ',' \
            || true
    )
    cells_per_second=${cells_per_second:-0}
    gpu_oom=$(rg -c 'CUDA_ERROR_OUT_OF_MEMORY' "${run_log}" || true)
    read -r mean_gpu max_gpu max_gpu_mem < <(
        awk -F',' '
            {
                gsub(/ /, "", $2); gsub(/ /, "", $3)
                util_sum += $2; util_count += 1
                if ($2 > util_max) util_max = $2
                if ($3 > memory_max) memory_max = $3
            }
            END {
                mean = util_count ? util_sum / util_count : 0
                printf "%.1f %.0f %.0f\n", mean, util_max, memory_max
            }
        ' "${gpu_log}"
    )
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${read_workers}" "${compute_workers}" "${run_status}" \
        "$((run_finished - run_started))" "${cells_per_second}" "${gpu_oom}" \
        "${mean_gpu}" "${max_gpu}" "${max_gpu_mem}" >> "${SUMMARY_PATH}"
    printf 'ratio_finished_at=%s read=%s compute=%s status=%s\n' \
        "$(date --iso-8601=seconds)" "${read_workers}" "${compute_workers}" "${run_status}"
done

printf '\nworker_split_summary\n'
column -t -s $'\t' "${SUMMARY_PATH}"
