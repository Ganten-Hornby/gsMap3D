"""Real CPU CLI/API execution: files, regression, writer and resource logging."""

import json

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from scipy import sparse
from typer.testing import CliRunner

from gsMap.cli import app
from gsMap.config import SpatialLDSCConfig
from gsMap.spatial_ldsc.spatial_ldsc_jax import run_spatial_ldsc_jax


def test_cpu_cli_and_python_have_equal_results_and_stage_peaks(tmp_path):
    rng = np.random.default_rng(911)
    cells, genes, snps = 407, 201, 2000
    names = [f"gene{i}" for i in range(genes)]
    snp_names = [f"rs{i}" for i in range(snps)]
    weight = sparse.random(snps, genes, density=0.04, random_state=rng, dtype=np.float32)
    weight = sparse.hstack(
        [weight, sparse.csr_matrix(rng.uniform(0.1, 10, (snps, 1)).astype(np.float32))],
        format="csr",
    )
    weights = ad.AnnData(weight)
    weights.var_names = [*names, "Unmapped"]
    weights.obs_names = snp_names
    weights.write_h5ad(tmp_path / "weight.h5ad")
    marker = ad.AnnData(rng.uniform(0.1, 3, (cells, genes)).astype(np.float32))
    marker.var_names = names
    marker.obs_names = [f"cell{i}" for i in range(cells)]
    marker.write_h5ad(tmp_path / "marker.h5ad")
    wld = tmp_path / "wld"
    wld.mkdir()
    pd.DataFrame({"SNP": snp_names, "L2": rng.uniform(1, 5, snps)}).to_csv(
        wld / "weights.1.l2.ldscore.gz",
        sep="\t",
        index=False,
    )
    gwas = {}
    for name in ("trait_a", "trait_b"):
        path = tmp_path / f"{name}.sumstats.gz"
        pd.DataFrame(
            {
                "SNP": snp_names,
                "N": rng.uniform(10000, 20000, snps),
                "Z": rng.normal(0, 1.5, snps),
            }
        ).to_csv(path, sep="\t", index=False)
        gwas[name] = str(path)
    yaml_path = tmp_path / "gwas.yaml"
    yaml_path.write_text(yaml.safe_dump(gwas))
    config = SpatialLDSCConfig(
        workdir=tmp_path,
        project_name="python",
        use_gpu=False,
        marker_score_h5ad_path=tmp_path / "marker.h5ad",
        snp_gene_weight_adata_path=tmp_path / "weight.h5ad",
        w_ld_dir=wld,
        sumstats_config_file=yaml_path,
        n_blocks=10,
        ldsc_compute_workers=2,
        ldsc_read_workers=2,
    )
    run_spatial_ldsc_jax(config)
    args = ["spatial-ldsc", "--workdir", str(tmp_path), "--project-name", "cli", "--no-gpu"]
    for flag, value in [
        ("marker-score-h5ad-path", tmp_path / "marker.h5ad"),
        ("snp-gene-weight-adata-path", tmp_path / "weight.h5ad"),
        ("w-ld-dir", wld),
        ("sumstats-config-file", yaml_path),
        ("n-blocks", 10),
        ("ldsc-compute-workers", 2),
        ("ldsc-read-workers", 2),
        ("ldsc-gwas-workers", 1),
    ]:
        args.extend(["--" + flag, str(value)])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    for project in ("python", "cli"):
        latest = tmp_path / project / "logs/latest"
        text = (latest / "03_spatial_ldsc.log").read_text()
        assert text.count("Peak memory usage") == 1
        assert "407/407 cells × 2 traits computed" in text
        assert "2/2 chunks complete" in text
        assert "cell-traits/s" in text
        if project == "cli":
            assert "2 pending GWAS traits with 1 reader(s)" in text
        manifest = json.loads((latest / "run_manifest.json").read_text())
        resource = manifest["stages"][0]["resource_usage"]
        assert resource["peak_rss_bytes"] > 0
        assert resource["samples"] >= 2
        assert "memory_at_peak_rss" in resource
        assert "peak_rss_anon_bytes" in resource
        assert "RssAnon=" in text
        assert "RssFile=" in text
    for trait in gwas:
        frames = [
            pd.read_csv(tmp_path / p / "spatial_ldsc" / f"{p}_{trait}.csv.gz")
            for p in ("python", "cli")
        ]
        pd.testing.assert_frame_equal(*frames, check_exact=True)
        assert len(frames[0]) == cells
        assert np.isfinite(frames[0].select_dtypes(include="number")).all().all()
