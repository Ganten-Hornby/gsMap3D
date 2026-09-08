import os
import sys
from types import ModuleType, SimpleNamespace

import pytest

from gsMap.config.utils import configure_jax_platform


@pytest.mark.parametrize("existing_value", [None, "caller-value"])
def test_cpu_jax_discovery_temporarily_skips_cuda_constraint_probe(monkeypatch, existing_value):
    observations: list[tuple[str, str | None]] = []
    config_updates: list[tuple[str, str]] = []
    fake_jax = ModuleType("jax")
    fake_jax.config = SimpleNamespace(
        update=lambda key, value: config_updates.append((key, value))
    )

    def devices(platform: str) -> list[object]:
        observations.append((platform, os.environ.get("JAX_SKIP_CUDA_CONSTRAINTS_CHECK")))
        return [object()]

    fake_jax.devices = devices
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    if existing_value is None:
        monkeypatch.delenv("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", raising=False)
    else:
        monkeypatch.setenv("JAX_SKIP_CUDA_CONSTRAINTS_CHECK", existing_value)

    configure_jax_platform(use_accelerator=False)

    assert config_updates == [("jax_platform_name", "cpu")]
    assert observations == [("cpu", "1")]
    assert os.environ.get("JAX_SKIP_CUDA_CONSTRAINTS_CHECK") == existing_value
