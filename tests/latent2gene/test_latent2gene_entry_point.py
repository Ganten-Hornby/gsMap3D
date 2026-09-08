from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from gsMap.latent2gene.entry_point import run_latent_to_gene
from gsMap.latent2gene.memmap_io import MemMapDense


def _complete_memmap(path, shape):
    memmap = MemMapDense(
        path,
        shape=shape,
        dtype=np.float16,
        mode="w",
        num_write_workers=0,
    )
    memmap.memmap[:] = 0
    memmap.close()


def test_completed_arrays_repair_missing_metadata_without_recomputation(tmp_path, monkeypatch):
    output_dir = tmp_path / "latent_to_gene"
    output_dir.mkdir()
    paths = SimpleNamespace(
        latent2gene_dir=output_dir,
        concatenated_latent_adata_path=output_dir / "concatenated_latent_adata.h5ad",
        rank_memmap_path=output_dir / "ranks.dat",
        mean_frac_path=output_dir / "mean_frac.parquet",
        marker_scores_memmap_path=output_dir / "marker_scores.dat",
        latent2gene_metadata_path=output_dir / "metadata.yaml",
        sample_h5ad_dict={"section": tmp_path / "section.h5ad"},
        to_dict_with_paths_as_strings=lambda: {"dataset_type": "spatial2D"},
    )
    paths.concatenated_latent_adata_path.touch()
    paths.mean_frac_path.touch()
    _complete_memmap(paths.rank_memmap_path, (3, 2))
    _complete_memmap(paths.marker_scores_memmap_path, (3, 2))

    class UnexpectedCalculator:
        def __init__(self, config):
            del config
            raise AssertionError("completed outputs must not be recomputed")

    monkeypatch.setattr(
        "gsMap.latent2gene.entry_point.RankCalculator",
        UnexpectedCalculator,
    )

    outputs = run_latent_to_gene.__wrapped__(paths)

    assert outputs["rank_memmap"] == str(paths.rank_memmap_path)
    assert outputs["marker_scores"] == str(paths.marker_scores_memmap_path)
    with paths.latent2gene_metadata_path.open(encoding="utf-8") as handle:
        metadata = yaml.safe_load(handle)
    assert metadata["n_cells"] == 3
    assert metadata["n_genes"] == 2
    assert metadata["outputs"] == outputs


def test_completed_rank_is_reused_after_marker_failure(tmp_path, monkeypatch):
    output_dir = tmp_path / "latent_to_gene"
    output_dir.mkdir()
    paths = SimpleNamespace(
        latent2gene_dir=output_dir,
        concatenated_latent_adata_path=output_dir / "concatenated_latent_adata.h5ad",
        rank_memmap_path=output_dir / "ranks.dat",
        mean_frac_path=output_dir / "mean_frac.parquet",
        marker_scores_memmap_path=output_dir / "marker_scores.dat",
        latent2gene_metadata_path=output_dir / "metadata.yaml",
        sample_h5ad_dict={"section": tmp_path / "section.h5ad"},
        annotation=None,
        data_layer="X",
        to_dict_with_paths_as_strings=lambda: {"dataset_type": "spatial2D"},
    )
    paths.concatenated_latent_adata_path.touch()
    paths.mean_frac_path.touch()
    _complete_memmap(paths.rank_memmap_path, (3, 2))

    class UnexpectedRankCalculator:
        def __init__(self, config):
            del config
            raise AssertionError("a completed rank checkpoint must be reused")

    class CompletingMarkerCalculator:
        def __init__(self, config):
            del config

        def calculate_marker_scores(
            self,
            *,
            adata_path,
            rank_memmap_path,
            mean_frac_path,
            output_path,
        ):
            assert Path(adata_path) == paths.concatenated_latent_adata_path
            assert Path(rank_memmap_path) == paths.rank_memmap_path
            assert Path(mean_frac_path) == paths.mean_frac_path
            _complete_memmap(output_path, (3, 2))
            return str(output_path)

    monkeypatch.setattr(
        "gsMap.latent2gene.entry_point.RankCalculator",
        UnexpectedRankCalculator,
    )
    monkeypatch.setattr(
        "gsMap.latent2gene.entry_point.MarkerScoreCalculator",
        CompletingMarkerCalculator,
    )

    outputs = run_latent_to_gene.__wrapped__(paths)

    assert outputs["rank_memmap"] == str(paths.rank_memmap_path)
    assert outputs["marker_scores"] == str(paths.marker_scores_memmap_path)
    assert paths.latent2gene_metadata_path.is_file()
