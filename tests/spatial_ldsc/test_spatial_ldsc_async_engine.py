"""Self-contained full-engine tests for asynchronous result persistence."""

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from gsMap.spatial_ldsc.execution import engine as module


@pytest.mark.parametrize("failure", [None, "writer", "source_init", "compute_init"])
def test_engine_drains_output_or_surfaces_async_error(tmp_path, monkeypatch, caplog, failure):
    rng = np.random.default_rng(31)
    snps, cells = 400, 11
    direct = ad.AnnData(
        rng.uniform(0.1, 2, (snps, cells)).astype(np.float32),
        obs=pd.DataFrame(index=[f"rs{i}" for i in range(snps)]),
        var=pd.DataFrame(index=[f"cell{i}" for i in range(cells)]),
    )
    baseline = pd.DataFrame(rng.uniform(0.1, 2, (snps, 2)).astype(np.float32))
    data = {
        "baseline_ld": baseline,
        "baseline_ld_sum": baseline.sum(axis=1).to_numpy(),
        "w_ld": rng.uniform(1, 2, snps).astype(np.float32),
        "chisq": rng.uniform(1, 2.5, snps).astype(np.float32),
        "N": np.full(snps, 100_000, dtype=np.float32),
        "Nbar": 100_000.0,
        "snp_positions": np.arange(snps),
    }
    monkeypatch.setattr(module, "prepare_trait_data", lambda *args: (data, direct.obs_names))
    config = SimpleNamespace(
        project_name="toy",
        sample_filter=None,
        cell_indices_range=None,
        spots_per_chunk_quick_mode=3,
        n_blocks=4,
        ldsc_read_workers=2,
        ldsc_compute_workers=1,
    )
    engine = module.SpatialLDSCEngine(
        config, tmp_path, None, None, baseline, None, ldscore_adata=direct
    )
    engine.setup_trait("trait", "unused")
    engine.result_writer.output_batch_rows = 3
    caplog.set_level("INFO")
    if failure:

        def fail(*args, **kwargs):
            raise OSError("injected output failure")

        if failure == "writer":
            monkeypatch.setattr(engine.result_writer, "_write_batch", fail)
        else:
            target = "LDScoreStream" if failure == "source_init" else "LDScoreComputePool"
            monkeypatch.setattr(module, target, fail)
        with pytest.raises((RuntimeError, OSError), match="injected output failure"):
            engine.process_all_chunks()
        assert not list(tmp_path.glob("*.csv.gz"))
        assert "Processing complete:" not in caplog.text
    else:
        result = pd.read_csv(engine.process_all_chunks())
        assert result.spot.tolist() == direct.var_names.tolist()
        assert np.isfinite(result.drop(columns="spot")).all().all()
        assert "including output finalization" in caplog.text
    assert not engine.result_writer._writer_thread.is_alive()
    assert not list(tmp_path.glob("*.partial.csv.gz"))
