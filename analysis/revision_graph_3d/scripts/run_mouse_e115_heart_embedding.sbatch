#!/usr/bin/env bash
#SBATCH --job-name=gsmap-e115-heart-embed
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly ARM="${1:?usage: run_mouse_e115_heart_embedding.sbatch 2d|3d [annotated|no_annotation]}"
readonly TRAINING_MODE="${2:-annotated}"
readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly SOURCE_H5AD=/storage/yangjianLab/lilounan/data/3dst/Mouse_E11.5_heart.h5ad
readonly PREPARED_2D_H5AD="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/Mouse_E11.5_heart_2d.h5ad"
readonly WORKDIR="${PROJECT_ROOT}/analysis/revision_graph_3d/runs"
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly GSMAP="${PROJECT_ROOT}/.venv/bin/gsmap"

case "${ARM}" in
    2d)
        readonly BASE_PROJECT_NAME=mouse_e115_heart_sectionwise_2d_k10
        readonly INPUT_H5AD="${PREPARED_2D_H5AD}"
        readonly SPATIAL_KEY=aligned_spatial_2D_mm
        readonly GRAPH_MODE=sectionwise_2d
        ;;
    3d)
        readonly BASE_PROJECT_NAME=mouse_e115_heart_section_aware_3d_k10
        readonly INPUT_H5AD="${SOURCE_H5AD}"
        readonly SPATIAL_KEY=aligned_spatial_3D_mm
        readonly GRAPH_MODE=section_aware_3d
        ;;
    *)
        printf 'Unknown arm: %s (expected 2d or 3d)\n' "${ARM}" >&2
        exit 2
        ;;
esac

TRAINING_ARGS=()
case "${TRAINING_MODE}" in
    annotated)
        readonly PROJECT_NAME="${BASE_PROJECT_NAME}"
        TRAINING_ARGS=(--annotation heart_regions --two-stage)
        ;;
    no_annotation)
        readonly PROJECT_NAME="${BASE_PROJECT_NAME}_no_annotation"
        TRAINING_ARGS=(--single-stage)
        ;;
    *)
        printf 'Unknown training mode: %s (expected annotated or no_annotation)\n' \
            "${TRAINING_MODE}" >&2
        exit 2
        ;;
esac
readonly -a TRAINING_ARGS

readonly RUN_DIR="${WORKDIR}/${PROJECT_NAME}"
mkdir -p "${RUN_DIR}" "${PROJECT_ROOT}/analysis/revision_graph_3d/logs"

record_exit() {
    local status=$?
    {
        printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
        printf 'arm=%s\n' "${ARM}"
        printf 'training_mode=%s\n' "${TRAINING_MODE}"
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
printf 'training_mode=%s\n' "${TRAINING_MODE}"
printf 'commit=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
git -C "${PROJECT_ROOT}" diff -- src/gsMap/find_latent/gnn/gcn.py \
    src/gsMap/find_latent/st_process.py src/gsMap/config/find_latent_config.py | sha256sum
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv

"${PYTHON}" -c 'import torch; print(f"torch={torch.__version__}"); print(f"torch_cuda_build={torch.version.cuda}"); print(f"cuda_available={torch.cuda.is_available()}"); assert torch.cuda.is_available(); print(f"cuda_device={torch.cuda.get_device_name(0)}")'

if [[ "${ARM}" == 2d ]]; then
    "${PYTHON}" "${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/prepare_mouse_e115_heart_input.py" \
        --source "${SOURCE_H5AD}" \
        --output "${PREPARED_2D_H5AD}"
fi

"${GSMAP}" find-latent \
    --workdir "${WORKDIR}" \
    --project-name "${PROJECT_NAME}" \
    --h5ad-path "${INPUT_H5AD}" \
    --data-layer X \
    --spatial-key "${SPATIAL_KEY}" \
    --no-high-quality-cell-qc \
    --feat-cell 2000 \
    --n-neighbors 10 \
    --graph-mode "${GRAPH_MODE}" \
    --graph-section-key slices \
    --graph-z-scale 1.0 \
    --k 3 \
    --hidden-size 128 \
    --embedding-size 32 \
    --batch-size 1024 \
    --itermax 100 \
    --patience 10 \
    --random-seed 2024 \
    --no-sampling \
    --n-cell-training 100000 \
    "${TRAINING_ARGS[@]}"
