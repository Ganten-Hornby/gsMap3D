#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-full-graph-train
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G

set -Eeuo pipefail

readonly ARM="${1:?usage: run_mouse_e115_full_graph_training_embedding.sbatch 2d|3d}"
readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly INPUT_H5AD="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/Mouse_E11.5_embryo_graph_ready.h5ad"
readonly WORKDIR="${PROJECT_ROOT}/analysis/revision_graph_3d/runs/corrected_training_staging"
readonly HOMOLOG_FILE=/storage/yangjianLab/lilounan/resources/mouse_human_homologs.txt
TRAINING_CONTEXT_CONFIRMED=0

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
mkdir -p "${RUN_DIR}" "${PROJECT_ROOT}/analysis/revision_graph_3d/logs"

record_exit() {
    local status=$?
    {
        printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
        printf 'arm=%s\n' "${ARM}"
        printf 'annotation=spatial_domain\n'
        printf 'requested_training_graph_context=full_source_graph_sampled_focal\n'
        if [[ "${TRAINING_CONTEXT_CONFIRMED}" == 1 ]]; then
            printf 'confirmed_training_graph_context=full_source_graph_sampled_focal\n'
        fi
        printf 'exit_code=%s\n' "${status}"
        printf 'finished_at=%s\n' "$(date --iso-8601=seconds)"
    } >"${RUN_DIR}/slurm_status.txt"
    exit "${status}"
}
trap record_exit EXIT

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
printf 'host=%s\n' "$(hostname)"
printf 'job_id=%s\n' "${SLURM_JOB_ID}"
printf 'arm=%s\n' "${ARM}"
printf 'commit=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
printf 'output=%s\n' "${RUN_DIR}"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv

cd "${PROJECT_ROOT}"
uv run --extra cuda128 --locked --no-sync python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
assert torch.version.cuda is not None and torch.version.cuda.startswith("12.")
assert torch.cuda.is_available()
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

[[ -s "${INPUT_H5AD}" ]]

uv run --extra cuda128 --locked --no-sync gsmap quick-mode \
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
    --use-gpu \
    --start-step find_latent \
    --stop-step find_latent

metadata="${RUN_DIR}/find_latent_representations/find_latent_metadata.yaml"
[[ -s "${metadata}" ]]
grep -q 'graph_context_mode: full_source_graph_sampled_focal' "${metadata}"
TRAINING_CONTEXT_CONFIRMED=1
