"""Package boundaries must not initialize GPU runtimes in the coordinator."""

import os
import subprocess
import sys
from pathlib import Path

from gsMap.spatial_ldsc.execution.multi_gpu import worker_environment


def test_subpackages_are_lightweight():
    code = """
import sys
import gsMap.spatial_ldsc.data
import gsMap.spatial_ldsc.backends
import gsMap.spatial_ldsc.kernels
import gsMap.spatial_ldsc.execution
import gsMap.spatial_ldsc.execution.multi_gpu
import gsMap.spatial_ldsc.execution.multi_gpu_worker
assert not {'jax', 'torch', 'triton', 'numba'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True)


def test_worker_environment_prepends_actual_source_root(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unrelated/package")
    env = worker_environment("0", 1)
    source, old = env["PYTHONPATH"].split(os.pathsep, 1)
    assert old == "/unrelated/package"
    assert (Path(source) / "gsMap/spatial_ldsc/execution/multi_gpu_worker.py").is_file()


def test_report_reader_import_surface_is_preserved():
    from gsMap.spatial_ldsc import io
    from gsMap.spatial_ldsc.data import loading

    for name in io.__all__:
        assert getattr(io, name) is getattr(loading, name)
