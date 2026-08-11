#!/usr/bin/env bash
#SBATCH --job-name=gsmap3d-e95-embed
#SBATCH --partition=l40s
#SBATCH --qos=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly SOURCE_H5AD=/storage/yangjianLab/lilounan/data/3dst/E9.5/Mouse_E9.5_embryo.h5ad
readonly PREPARED_H5AD="${PROJECT_ROOT}/analysis/revision_graph_3d/inputs/Mouse_E9.5_embryo_sectioned.h5ad"
readonly WORKDIR="${PROJECT_ROOT}/analysis/revision_graph_3d/runs"
readonly PROJECT_NAME=mouse_e95_section_aware_3d_k10
readonly RUN_DIR="${WORKDIR}/${PROJECT_NAME}"
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly GSMAP="${PROJECT_ROOT}/.venv/bin/gsmap"

mkdir -p "${RUN_DIR}"

record_exit() {
    local status=$?
    {
        printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
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
printf 'commit=%s\n' "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv

"${PYTHON}" - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError("Slurm allocation does not expose a usable CUDA GPU")
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

"${PYTHON}" "${PROJECT_ROOT}/analysis/revision_graph_3d/scripts/prepare_mouse_e95_input.py" \
    --source "${SOURCE_H5AD}" \
    --output "${PREPARED_H5AD}" \
    --spatial-key 3d_align_spatial \
    --section-key section_id

"${GSMAP}" find-latent \
    --workdir "${WORKDIR}" \
    --project-name "${PROJECT_NAME}" \
    --h5ad-path "${PREPARED_H5AD}" \
    --data-layer X \
    --spatial-key 3d_align_spatial \
    --annotation transf_anno_E95 \
    --no-high-quality-cell-qc \
    --feat-cell 2000 \
    --n-neighbors 10 \
    --graph-mode section_aware_3d \
    --graph-section-key section_id \
    --graph-z-scale 1.0 \
    --k 3 \
    --hidden-size 128 \
    --embedding-size 32 \
    --batch-size 1024 \
    --itermax 100 \
    --patience 10 \
    --random-seed 2024 \
    --single-stage \
    --do-sampling \
    --n-cell-training 100000
