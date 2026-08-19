"""Unit tests for the core-aware LDSC compute-worker resolver.

The resolver (`SpatialLDSCEngine._resolve_compute_workers`) decides how many
regression worker threads to launch. It must:
  * pin to 1 on GPU (cuSolver/device-memory contention),
  * cap to ``cpu_count // 2`` on CPU so a low-core host (e.g. 2-core CI) cannot
    starve XLA's intra-op pool and deadlock,
  * honor the ``GSMAP_LDSC_FORCE_COMPUTE_WORKERS`` override.
"""

import types

import pytest

from gsMap.spatial_ldsc.ldscore_quick_mode import SpatialLDSCEngine


def _stub(configured):
    """Minimal object exposing only what the resolver reads."""
    s = types.SimpleNamespace()
    s.config = types.SimpleNamespace(ldsc_compute_workers=configured)
    return s


def _resolve(stub, backend):
    return SpatialLDSCEngine._resolve_compute_workers(stub, backend)


def test_gpu_pins_to_single_worker(monkeypatch):
    monkeypatch.delenv("GSMAP_LDSC_FORCE_COMPUTE_WORKERS", raising=False)
    assert _resolve(_stub(10), "gpu") == 1


def test_cpu_caps_at_half_cores(monkeypatch):
    monkeypatch.delenv("GSMAP_LDSC_FORCE_COMPUTE_WORKERS", raising=False)
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(8)))
    # configured (10) above cap (8//2=4) -> capped to 4
    assert _resolve(_stub(10), "cpu") == 4
    # configured below cap is left untouched
    assert _resolve(_stub(2), "cpu") == 2


def test_cpu_two_cores_collapses_to_one(monkeypatch):
    """The deadlock guard: 2-core host must run a single compute worker."""
    monkeypatch.delenv("GSMAP_LDSC_FORCE_COMPUTE_WORKERS", raising=False)
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: {0, 1})
    assert _resolve(_stub(10), "cpu") == 1


def test_force_env_overrides_cap(monkeypatch):
    monkeypatch.setenv("GSMAP_LDSC_FORCE_COMPUTE_WORKERS", "7")
    # override beats both the GPU pin and the CPU cap
    assert _resolve(_stub(10), "gpu") == 7
    assert _resolve(_stub(10), "cpu") == 7


def test_force_env_floored_at_one(monkeypatch):
    monkeypatch.setenv("GSMAP_LDSC_FORCE_COMPUTE_WORKERS", "0")
    assert _resolve(_stub(10), "cpu") == 1


@pytest.mark.parametrize(("cores", "expected"), [(1, 1), (2, 1), (4, 2), (24, 10)])
def test_cpu_cap_never_exceeds_configured(monkeypatch, cores, expected):
    """With configured=10, effective = min(10, cores//2) (and >=1)."""
    monkeypatch.delenv("GSMAP_LDSC_FORCE_COMPUTE_WORKERS", raising=False)
    monkeypatch.setattr("os.sched_getaffinity", lambda _pid: set(range(cores)))
    assert _resolve(_stub(10), "cpu") == expected
