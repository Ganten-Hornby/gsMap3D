#!/usr/bin/env bash
#SBATCH --job-name=gsmap-venv-import-health
#SBATCH --chdir=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
#SBATCH --output=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.out
#SBATCH --error=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning/analysis/revision_graph_3d/logs/%x-%j.err

set -Eeuo pipefail

readonly PROJECT_ROOT=/storage/yangjianLab/lilounan/gsMap3D-3d-graph-learning
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

printf 'host=%s\n' "$(hostname)"
printf 'job_id=%s\n' "${SLURM_JOB_ID}"
printf 'python=%s\n' "${PYTHON}"

"${PYTHON}" -c '
import importlib

for module_name in ("numpy", "h5py", "pandas", "sklearn"):
    print(f"import_start={module_name}", flush=True)
    module = importlib.import_module(module_name)
    print(f"import_ok={module_name} version={module.__version__}", flush=True)

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

print(f"import_ok=sklearn.cluster.KMeans object={KMeans.__name__}", flush=True)
print(
    f"import_ok=sklearn.metrics.adjusted_rand_score "
    f"object={adjusted_rand_score.__name__}",
    flush=True,
)
print("PROJECT_VENV_IMPORT_HEALTH_OK", flush=True)
'
