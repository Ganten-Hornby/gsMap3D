"""Tutorial validation respects explicitly assigned execution resources."""

from pathlib import Path

from tests.helpers.validate_examples import validation_environment


def test_validation_preserves_thread_budget():
    env = validation_environment(
        Path("repo"), {"SLURM_CPUS_PER_TASK": "16", "OMP_NUM_THREADS": "16"}
    )
    assert env["OMP_NUM_THREADS"] == env["NUMBA_NUM_THREADS"] == "16"
    assert env["OPENBLAS_NUM_THREADS"] == "1"
    env = validation_environment(
        Path("repo"), {"SLURM_CPUS_PER_TASK": "2", "NUMBA_NUM_THREADS": "64"}
    )
    assert env["NUMBA_NUM_THREADS"] == env["OMP_NUM_THREADS"] == "2"
    assert validation_environment(Path("repo"), {})["OMP_NUM_THREADS"] == "8"
