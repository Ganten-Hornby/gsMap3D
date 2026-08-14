#!/usr/bin/env bash
#SBATCH --job-name=gsmap-gene-block-smoke
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly LOCAL_TMP="/data/${USER}/gsmap-gene-block-smoke/${SLURM_JOB_ID}"

cleanup() {
    local status=$?
    case "${LOCAL_TMP}" in
        "/data/${USER}/gsmap-gene-block-smoke/${SLURM_JOB_ID}")
            rm -rf -- "${LOCAL_TMP}" || true
            ;;
        *)
            printf 'Refusing to clean unexpected path: %s\n' "${LOCAL_TMP}" >&2
            ;;
    esac
    exit "${status}"
}
trap cleanup EXIT

mkdir -p "${PROJECT_ROOT}/analysis/revision_graph_3d/logs" "${LOCAL_TMP}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

"${PYTHON}" - <<'PY'
import jax
import torch

print(f"jax_backend={jax.default_backend()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
assert jax.default_backend() == "gpu"
assert torch.version.cuda is not None and torch.version.cuda.startswith("12.")
assert torch.cuda.is_available()
PY

"${PYTHON}" -m pytest -q \
    --basetemp "${LOCAL_TMP}/pytest" \
    tests/test_gene_block_marker_scores.py \
    tests/test_ldscore_backend.py
