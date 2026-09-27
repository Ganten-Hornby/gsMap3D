"""Validate page ownership and executable E16.5 performance recipes."""

import csv
import json
import re
import shlex
import subprocess
from pathlib import Path

import pytest
from click import Context
from typer.main import get_command

from gsMap.cli import app
from tests.docs.helpers import cli_settings
from tests.helpers.validate_examples import tutorial_code
from tests.paths import REPO_ROOT as REPO

SOURCE = REPO / "docs/source"


def write_spatial_fixture(path):
    import anndata as ad
    import numpy as np

    data = ad.AnnData(np.ones((3, 2)))
    data.obsm["spatial"] = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    data.layers["count"] = data.X.copy()
    data.obs["annotation"] = ["heart", "heart", "brain"]
    data.write_h5ad(path)


def guide():
    return (SOURCE / "scalability.md").read_text()


def test_scalability_flags_and_table_defaults():
    text = guide()
    options = {}
    for command in get_command(app).commands.values():
        for option in command.get_params(Context(command)):
            for flag in [*option.opts, *option.secondary_opts]:
                options.setdefault(flag, []).append(option)
    for flag in set(re.findall(r"--[a-z][a-z0-9-]+", text)):
        assert flag in options, flag
    for flag, default in re.findall(r"\| `(--[a-z0-9-]+)` \| `(\d+)`", text):
        assert any(option.default == int(default) for option in options[flag]), flag
    assert "--no-gpu` alone does not disable PyTorch" in text
    assert "not which cells" in text


def test_pages_have_distinct_roles():
    text = guide()
    key = (SOURCE / "key_concepts.md").read_text()
    custom = (SOURCE / "advanced_usage.md").read_text()
    assert "```{mermaid}" not in text
    assert key.count("```{mermaid}") == 2
    assert "#### Graph neighbors versus homogeneous neighbors" in key
    assert "{ref}`Key Concepts <pipeline-overview>`" in text
    assert "{doc}`scalability`" in custom
    assert "{doc}`key_concepts`" in custom
    assert "CS8" not in text
    assert "human-gastrulation" not in text
    assert "mouse-e11" not in text
    assert "E16.5" in text
    assert "Spatial2D" in text
    for heading in (
        "Choose CPU or GPU",
        "I/O and fast storage",
        "Tune memory and throughput",
        "Cluster users",
        "Keep large reports responsive",
    ):
        assert f"## {heading}" in text


def test_complete_neighborhoods_and_safe_cluster_advice():
    text = guide()
    assert "complete\nselected neighborhood" in text
    assert "More workers are not always faster" in text
    cluster = text.split("## Cluster users")[1].split("## Keep large reports")[0]
    assert re.findall(r"^(\d)\. ", cluster, re.MULTILINE) == ["1", "2", "3", "4"]
    assert "CPUs assigned to the job" in cluster
    assert "Run shared multi-trait LDSC" in cluster
    assert "Do not launch overlapping jobs" in cluster
    assert "verify complete coverage with no duplicate cells" in cluster
    assert "near-linearly" not in text


def test_documented_windows_preserve_complete_neighborhoods():
    import numpy as np

    from gsMap.latent2gene.window_cache import AdjacentRankWindowIndex, plan_rank_windows

    neighbors = np.array([[0, 100, 900], [1, 900, 100], [2, 800, 900]], dtype=np.int32)
    windows = plan_rank_windows(neighbors, max_cache_rows=5, global_rows=1000)
    assert [(w.start, w.end) for w in windows] == [(0, 2), (2, 3)]
    index = AdjacentRankWindowIndex(5, 1000)
    first = index.map_window(neighbors[:2])
    assert first.rows_to_read.tolist() == [0, 1, 100, 900]
    second = index.map_window(neighbors[2:])
    assert second.rows_to_read.tolist() == [2, 800]
    assert second.retained_rows == 1
    assert second.local_neighbor_indices[0, 2] == first.local_neighbor_indices[0, 2]


def test_overview_connects_neighbors_ranks_and_global_reference():
    text = (SOURCE / "key_concepts.md").read_text()
    diagrams = re.findall(r"```\{mermaid\}\n(.*?)```", text, re.DOTALL)
    assert len(diagrams) == 2
    assert "````{tab} Simplified" in text
    assert "````{tab} Detailed" in text
    assert 'subgraph PIPELINE["Integrated pipeline"]' in diagrams[0]
    assert "fill:#f0edfa" in diagrams[0]
    assert 'A["Annotation · optional"]' in diagrams[1]
    assert "Same-group search" not in diagrams[1]
    assert 'A -.->|"Same-group restriction"| H' in diagrams[1]
    diagram = re.sub(r'\|"[^"]*"\|', "", diagrams[1])
    for edge in (
        "C --> H",
        "N -.-> H",
        "P --> H",
        "A -.-> H",
        "R --> M",
        "F --> M",
        "H --> M",
        "M --> D",
    ):
        assert edge in diagram
    assert "Geometric-mean rank" in diagram
    assert "expression fraction" in diagram
    assert "Niche embedding · optional" in diagram
    assert "not raw expression values" in text


def test_mermaid_theme_is_configured_centrally():
    import runpy

    config = runpy.run_path(str(SOURCE / "conf.py"))
    assert "sphinxcontrib.mermaid" in config["extensions"]
    assert config["mermaid_light_theme"] == "base"
    assert config["mermaid_init_config"]["themeVariables"]["clusterBkg"] == "#f8fafc"
    assert config["mermaid_height"] == "auto"


def test_overview_views_have_independent_inline_sizes():
    text = (SOURCE / "key_concepts.md").read_text()
    css = (SOURCE / "_static/custom.css").read_text()
    assert ":name: pipeline-simple" in text
    assert ":name: pipeline-detailed" in text
    assert "Scroll horizontally" in text
    assert "#pipeline-overview #pipeline-simple > svg" in css
    assert "#pipeline-overview #pipeline-detailed > svg" in css
    assert "height: 560px" in css
    assert "height: 900px" in css
    assert "overflow-x: auto" in css


def test_key_concepts_keeps_practical_inputs_and_current_flags():
    text = (SOURCE / "key_concepts.md").read_text()
    quick = get_command(app).commands["quick-mode"]
    formatter = get_command(app).commands["format-sumstats"]
    flags = {
        flag
        for command in (quick, formatter)
        for option in command.get_params(Context(command))
        for flag in [*option.opts, *option.secondary_opts]
    }
    assert set(re.findall(r"--[a-z][a-z0-9-]+", text)) <= flags
    for heading in (
        "Basic usage",
        "Key Configurations",
        "GWAS Summary Statistics",
        "Formatting your own GWAS",
        "Providing GWAS Data",
    ):
        assert heading in text
    assert "format_sumstats" not in text
    assert "SNP          A1  A2  Z       N" in text
    assert "h5ad_path=[" in text


def test_key_concepts_placeholder_usage_matches_cli_python(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    import gsMap.pipeline
    import gsMap.pipeline.quick_mode

    monkeypatch.chdir(tmp_path)
    for name in ("sample.h5ad", "trait1.sumstats.gz", "resources/snp_gene_weights.h5ad"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    write_spatial_fixture(tmp_path / "sample.h5ad")
    (tmp_path / "resources/w_ld").mkdir()
    text = (SOURCE / "key_concepts.md").read_text()
    shell = re.findall(r"```bash\n(.*?)\n```", text, re.DOTALL)[0]
    python = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)[0]
    assert "E16.5" not in shell + python
    assert "mouse-e16-mosta" not in shell + python
    assert "placeholder paths" in text
    captured = []
    monkeypatch.setattr(gsMap.pipeline, "run_quick_mode", captured.append)
    monkeypatch.setattr(gsMap.pipeline.quick_mode, "run_quick_mode", captured.append)
    exec(compile(python, "Key Concepts Python usage", "exec"), {})
    line = next(
        line for line in shell.replace("\\\n", " ").splitlines() if line.startswith("gsmap ")
    )
    result = CliRunner().invoke(app, shlex.split(line)[1:])
    assert result.exit_code == 0, result.output
    assert len(captured) == 2
    assert all(c.dataset_type == "spatial2D" for c in captured)
    _, explicit = cli_settings(shell)[0]
    for key in explicit:
        left, right = getattr(captured[0], key), getattr(captured[1], key)
        if isinstance(left, list):
            assert [Path(p).resolve() for p in left] == [Path(p).resolve() for p in right]
        elif isinstance(left, Path) or isinstance(right, Path):
            assert Path(left).resolve() == Path(right).resolve()
        else:
            assert left == right, key


def test_key_concepts_formatter_runs_and_preserves_sign(tmp_path, monkeypatch):
    import pandas as pd
    from typer.testing import CliRunner

    monkeypatch.chdir(tmp_path)
    (tmp_path / "my_trait.tsv").write_text(
        "rsid\teffect_allele\tother_allele\tbeta\tse\tn_samples\n"
        "rs12184267\tT\tC\t0.1\t0.05\t10000\n"
        "rs12184277\tG\tA\t-0.2\t0.1\t12000\n"
    )
    text = (SOURCE / "key_concepts.md").read_text()
    commands = [
        c
        for c in re.findall(r"```bash\n(.*?)\n```", text, re.DOTALL)
        if c.startswith("gsmap format-sumstats")
    ]
    assert len(commands) == 2
    for command in commands:
        result = CliRunner().invoke(app, shlex.split(command.replace("\\\n", " "))[1:])
        assert result.exit_code == 0, result.output
    output = pd.read_csv(tmp_path / "my_trait.sumstats.gz", sep=r"\s+")
    assert output["Z"].tolist() == [2.0, -2.0]
    assert output["N"].tolist() == [10000, 12000]
    assert output["A1"].tolist() == ["T", "G"]


def test_key_concepts_inspection_snippets(example_workspace):
    import anndata as ad
    import numpy as np
    import pandas as pd

    data = ad.AnnData(np.ones((3, 2)))
    data.layers["count"] = data.X.copy()
    data.obsm["spatial"] = np.zeros((3, 2))
    data.obs["annotation"] = ["heart", "heart", "brain"]
    data.write_h5ad("example_data/mouse-e16-mosta/ST/E16.5_E1S1.MOSTA.h5ad")
    pd.DataFrame({"SNP": ["rs1"], "Z": [1.0], "N": [1000]}).to_csv(
        "example_data/example-traits/IQ.sumstats.gz", sep="\t", index=False
    )
    text = (SOURCE / "key_concepts.md").read_text()
    snippets = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)[1:]
    assert len(snippets) == 2
    for code in snippets:
        exec(compile(code, "Key Concepts data inspection", "exec"), {})


@pytest.fixture
def example_workspace(tmp_path, monkeypatch):
    """Only fixture files are created; full analysis execution is intercepted."""
    monkeypatch.chdir(tmp_path)
    paths = [
        "example_data/mouse-e16-mosta/ST/E16.5_E1S1.MOSTA.h5ad",
        "resources/homologs/mouse_human_homologs.txt",
        "resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad",
        "results/mouse-e16-mosta/logs/latest/02_latent_to_gene.log",
    ]
    paths.extend(
        f"example_data/example-traits/{trait}.sumstats.gz" for trait in ("IQ", "Height", "MCHC")
    )
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    write_spatial_fixture(tmp_path / "example_data/mouse-e16-mosta/ST/E16.5_E1S1.MOSTA.h5ad")
    (tmp_path / "resources/1000g-hc-grch38-v50-eur/w_ld").mkdir()
    yaml_block = re.findall(r"```yaml\n(.*?)\n```", guide(), re.DOTALL)
    assert len(yaml_block) == 1
    (tmp_path / "example_data/example-traits/gwas.yaml").write_text(yaml_block[0] + "\n")
    # Later recipes resume the original per-sample outputs from preprocessing.
    import yaml

    latent = tmp_path / "results/mouse-e16-mosta/find_latent_representations"
    latent.mkdir()
    (latent / "find_latent_metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "outputs": {
                    "latent_files": {
                        "E16.5_E1S1.MOSTA": str(
                            tmp_path / "example_data/mouse-e16-mosta/ST/E16.5_E1S1.MOSTA.h5ad"
                        )
                    }
                }
            }
        )
    )
    return tmp_path


def normalize(value):
    return value.resolve() if isinstance(value, Path) else value


def test_e16_full_cli_python_recipes(example_workspace, monkeypatch):
    from typer.testing import CliRunner

    import gsMap.pipeline
    import gsMap.pipeline.quick_mode

    text = guide()
    shell = [
        code
        for code in re.findall(r"```bash\n(.*?)\n```", text, re.DOTALL)
        if "gsmap quick-mode" in code
    ]
    python = [
        code
        for code in re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
        if "QuickModeConfig(" in code
    ]
    assert len(shell) == len(python) == 3
    captured = []
    monkeypatch.setattr(gsMap.pipeline, "run_quick_mode", captured.append)
    monkeypatch.setattr(gsMap.pipeline.quick_mode, "run_quick_mode", captured.append)
    seen = []
    for cli, code in zip(shell, python, strict=True):
        assert (
            subprocess.run(["bash", "-n"], input=cli, text=True, capture_output=True).returncode
            == 0
        )
        exec(compile(code, "E16 performance Python recipe", "exec"), {})
        python_config = captured.pop()
        line = next(
            line for line in cli.replace("\\\n", " ").splitlines() if line.startswith("gsmap ")
        )
        result = CliRunner().invoke(app, shlex.split(line)[1:])
        assert result.exit_code == 0, result.output
        cli_config = captured.pop()
        _, explicit = cli_settings(cli)[0]
        for key in explicit:
            left, right = getattr(python_config, key), getattr(cli_config, key)
            if isinstance(left, Path) or isinstance(right, Path):
                assert Path(left).resolve() == Path(right).resolve(), key
            elif isinstance(left, list):
                assert [Path(x).resolve() for x in left] == [Path(x).resolve() for x in right], key
            else:
                assert left == right, key
        assert {k: v.resolve() for k, v in python_config.sumstats_config_dict.items()} == {
            k: v.resolve() for k, v in cli_config.sumstats_config_dict.items()
        }
        assert python_config.dataset_type == "spatial2D"
        assert python_config.annotation == "annotation"
        assert python_config.project_name == "mouse-e16-mosta"
        seen.append((python_config, explicit))

    _, tutorial = tutorial_code(REPO, "mouse-e16-mosta", "cli")
    _, original = cli_settings(tutorial)[0]
    pre_config, pre = seen[0]
    for key, value in pre.items():
        if key != "stop_step":
            assert original[key] == value, key
    assert pre_config.stop_step == "latent2gene"
    assert not {"n_adjacent_slices", "spatial_neighbors", "homogeneous_neighbors"} & pre.keys()
    assert seen[1][0].start_step == seen[1][0].stop_step == "spatial_ldsc"
    assert seen[2][0].start_step == "cauchy"
    assert seen[2][0].downsampling_n_spots_2d == 150000
    for config, _ in seen[1:]:
        assert config.trait_name_list == ["IQ", "Height", "MCHC"]
        assert config.ldsc_share_ld is True
        assert config.ldsc_trait_batch_size == 4
        assert config.spots_per_chunk_quick_mode is None


def test_tuning_snippets_match_cli_options(example_workspace, monkeypatch):
    """Run the short Python edits and corresponding CLI overrides on a real config."""
    from typer.testing import CliRunner

    import gsMap.config
    import gsMap.pipeline.quick_mode

    text = guide()
    pairs = re.findall(
        r"````\{tab\} CLI\n```text\n(.*?)\n```\n````\s+"
        r"````\{tab\} Python\n```python\n(.*?)\n```",
        text,
        re.DOTALL,
    )
    assert len(pairs) == 6
    base = [
        "--h5ad-path",
        "example_data/mouse-e16-mosta/ST/E16.5_E1S1.MOSTA.h5ad",
        "--workdir",
        "results",
        "--project-name",
        "mouse-e16-mosta",
        "--annotation",
        "annotation",
        "--trait-name",
        "IQ",
        "--sumstats-file",
        "example_data/example-traits/IQ.sumstats.gz",
    ]
    captured = []
    monkeypatch.setattr(gsMap.pipeline.quick_mode, "run_quick_mode", captured.append)
    scratch = example_workspace / "fast-ssd/gsmap"
    scratch.mkdir(parents=True)
    for flags, code in pairs:
        flags, code = (
            flags.replace("/fast-ssd/gsmap", str(scratch)),
            code.replace("/fast-ssd/gsmap", str(scratch)),
        )
        config = gsMap.config.QuickModeConfig(
            h5ad_path=["example_data/mouse-e16-mosta/ST/E16.5_E1S1.MOSTA.h5ad"],
            workdir="results",
            project_name="mouse-e16-mosta",
            annotation="annotation",
            trait_name="IQ",
            sumstats_file="example_data/example-traits/IQ.sumstats.gz",
        )
        exec(compile(code, "performance tuning snippet", "exec"), {"config": config})
        result = CliRunner().invoke(app, ["quick-mode", *base, *shlex.split(flags)])
        assert result.exit_code == 0, result.output
        cli_config = captured.pop()
        command = "gsmap quick-mode " + flags.replace("\n", " ")
        for key in cli_settings(command)[0][1]:
            assert normalize(getattr(config, key)) == normalize(getattr(cli_config, key)), key


def test_auxiliary_shell_snippets(example_workspace):
    snippets = [
        code
        for code in re.findall(r"```bash\n(.*?)\n```", guide(), re.DOTALL)
        if "gsmap quick-mode" not in code
    ]
    assert len(snippets) == 2
    for code in snippets:
        result = subprocess.run(["bash", "-c", code], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr


def test_scalability_measurements_and_units():
    """Public curves/tables are backed by complete, finite measured sweeps."""
    import numpy as np

    assets = SOURCE / "_static/benchmarks"
    text = guide()
    with (assets / "cpu_batch_scaling.csv").open() as stream:
        cpu = list(csv.DictReader(stream))
    assert len(cpu) == 30
    for batch in (100, 400, 800, 1600, 3200):
        rows = [row for row in cpu if int(row["batch"]) == batch]
        assert [int(row["cores"]) for row in rows] == [1, 2, 4, 8, 16, 32]
        for row in rows:
            assert int(row["repeats"]) == 3
            assert float(row["max_abs_logp"]) < 0.1
            ct_s = float(row["mean_ct_s"])
            assert np.isfinite(ct_s)
            assert ct_s > 0
            assert ct_s == pytest.approx(4 * float(row["mean_cell_s"]))
    with (assets / "gpu_scaling_measurements.csv").open() as stream:
        gpu = list(csv.DictReader(stream))
    assert len(gpu) == 12
    for model in ("A100", "RTX 2080 Ti"):
        for count in (1, 2, 4):
            rows = [r for r in gpu if r["gpu"] == model and int(r["gpus"]) == count]
            assert {int(r["repeat"]) for r in rows} == {0, 1}
            for row in rows:
                assert float(row["max_abs_logp"]) < 0.1
                assert int(row["cells"]) == 1000037
                assert int(row["traits"]) == 4
                assert float(row["cell_trait_s"]) == pytest.approx(
                    1000037 * 4 / float(row["seconds"])
                )
    # Earlier GPU measurements remain reproducible, but are not mixed into the
    # replacement IQ-only/four-trait curves.
    assert "_static/benchmarks/gpu_scaling.png" not in text
    assert "_static/benchmarks/cpu_batch_scaling.png" not in text
    assert "1600" not in text
    for image in ("cpu_trait_scaling", "gpu_trait_scaling"):
        assert f"_static/benchmarks/{image}.png" in text
        assert (assets / f"{image}.png").stat().st_size > 10000
    assert "Cell-traits/s" in text
    assert "Both tests use the same\n**1,000,037 HESTA CS20 cells**" in text
    assert "auto: CPU=400, GPU=100" in text


def test_single_multi_trait_measurements_and_simple_view():
    import numpy as np
    from docs.tools.plot_trait_scalability import validate_coverage

    assets = SOURCE / "_static/benchmarks"
    with (assets / "trait_scaling_measurements.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    validate_coverage(rows)
    text = guide()
    quick_view = text.split("## Spatial LDSC throughput at a glance")[1].split("## Cluster users")[
        0
    ]
    assert not any(line.startswith("|") for line in quick_view.splitlines())
    assert "default batch 400" in quick_view
    for model in ("CPU", "A100", "V100", "RTX 2080 Ti"):
        cpu = model == "CPU"
        for count in [1, 2, 4, 8, 16, 32] if cpu else [1, 2, 4]:
            for traits in (1, 4):
                selected = [
                    r
                    for r in rows
                    if r["hardware"] == model
                    and int(r["workers"]) == count
                    and int(r["traits"]) == traits
                ]
                for row in selected:
                    cells = 1000037
                    assert int(row["cells"]) == cells
                    assert int(row["batch"]) == (400 if cpu else 100)
                    seconds, ct_s, cell_s, delta = (
                        float(row[k])
                        for k in ("seconds", "cell_trait_s", "cell_s", "max_abs_logp")
                    )
                    assert np.isfinite([seconds, ct_s, cell_s, delta]).all()
                    assert seconds > 0
                    assert 0 <= delta < 0.1
                    assert ct_s == pytest.approx(cells * traits / seconds)
                    assert cell_s == pytest.approx(cells / seconds)
    assert max(float(r["max_abs_logp"]) for r in rows) < 0.05005
    for platform in ("cpu", "gpu"):
        svg = (assets / f"{platform}_trait_scaling.svg").read_text()
        assert "1 trait (IQ)" in svg
        assert "4 traits" in svg
        assert "Cell-traits / second" in svg
        assert "Cells / second" not in svg
        assert "Strong scaling" not in svg
        assert "≈" in svg
        assert "1,000,037 cells" in svg


@pytest.mark.parametrize("platform", ["CPU", "GPU"])
def test_throughput_only_plot_matches_measured_means(platform, tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import numpy as np
    from docs.tools.plot_trait_scalability import plot

    with (SOURCE / "_static/benchmarks/trait_scaling_measurements.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    captured = []
    original_close = plt.close
    monkeypatch.setattr(plt, "close", captured.append)
    plot(rows, platform, tmp_path)
    fig = captured[-1]
    try:
        assert len(fig.axes) == 1
        axis = fig.axes[0]
        cpu = platform == "CPU"
        models = ["CPU"] if cpu else ["RTX 2080 Ti", "V100", "A100"]
        counts = [1, 2, 4, 8, 16, 32] if cpu else [1, 2, 4]
        assert len(axis.lines) == 2 * len(models)
        for line, (model, traits) in zip(
            axis.lines, [(m, t) for m in models for t in (1, 4)], strict=True
        ):
            expected = [
                np.mean(
                    [
                        float(r["cell_trait_s"])
                        for r in rows
                        if r["hardware"] == model
                        and int(r["traits"]) == traits
                        and int(r["workers"]) == n
                    ]
                )
                for n in counts
            ]
            np.testing.assert_array_equal(line.get_xdata(), counts)
            np.testing.assert_allclose(line.get_ydata(), expected)
            assert line.get_linestyle() == ("--" if traits == 1 else "-")
        assert len(axis.texts) == len(axis.lines)
        assert all(t.get_text().startswith("≈") for t in axis.texts)
        assert axis.get_ylim()[0] == 0
        assert axis.get_ylim()[1] > max(max(line.get_ydata()) for line in axis.lines)
    finally:
        original_close(fig)


def test_scalability_plot_axes_include_measured_values(tmp_path):
    import matplotlib.pyplot as plt
    import numpy as np
    from docs.tools.plot_scalability import figure, save, series

    fig, axes = figure("Test", "Four traits", [1, 2, 4], "GPUs")
    rates = np.array([47012.0, 92619.0, 182954.0])
    series(axes, [1, 2, 4], rates, rates * 0.99, rates * 1.01, "A100", "#007f86")
    try:
        # Setting bottom=0 before plotting would disable autoscaling at top=1.
        save(fig, tmp_path / "test_scaling", "Observed measurements")
        assert axes[0].get_ylim()[0] == 0
        assert axes[0].get_ylim()[1] > rates.max()
        assert axes[1].get_ylim()[1] >= 4
    finally:
        plt.close(fig)


def test_single_multi_trait_plot_requires_complete_measured_grid():
    from docs.tools.plot_trait_scalability import validate_coverage

    rows = [
        dict(hardware=model, workers=n, traits=t, repeat=r)
        for model in ("CPU", "A100", "V100", "RTX 2080 Ti")
        for n in ([1, 2, 4, 8, 16, 32] if model == "CPU" else [1, 2, 4])
        for t in (1, 4)
        for r in range(3 if model == "CPU" else 2)
    ]
    validate_coverage(rows)
    with pytest.raises(AssertionError, match="Missing cases"):
        validate_coverage(rows[:-1])
    with pytest.raises(AssertionError, match="Duplicate"):
        validate_coverage([*rows, rows[0]])


@pytest.fixture
def matched_benchmark_summaries(tmp_path):
    """Exercise the importer without access to private GWAS or benchmark data."""
    with (SOURCE / "_static/benchmarks/trait_scaling_measurements.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for model in ("CPU", "A100", "V100", "RTX 2080 Ti"):
        for ntraits in (1, 4):
            cpu = model == "CPU"
            traits = [64] if ntraits == 1 else [64, 92, 2, 59]
            selected = [r for r in rows if r["hardware"] == model and int(r["traits"]) == ntraits]
            summary = dict(complete=True, all_passed=True)
            hashes = {"kernel.py": "test-kernel-hash"}
            if cpu:
                summary.update(
                    cells=1000037,
                    traits=traits,
                    source_commit="test-commit",
                    manifest_sha256="test-manifest",
                    matched_gpu=dict(
                        same_cells_as_gpu=True,
                        cells=1000037,
                        row_offset=0,
                        source_commit="test-commit",
                        kernel_hashes=hashes,
                    ),
                )
            else:
                summary.update(
                    label=f"{model.replace(' ', '')}_t{ntraits}",
                    arguments=dict(
                        cells=1000037, traits=",".join(map(str, traits)), batch=100, row_offset=0
                    ),
                    implementation_commit="test-commit",
                    prepared_sha256="test-manifest",
                    kernel_hashes=hashes,
                )
            cases = []
            for row in selected:
                count = int(row["workers"])
                case = dict(
                    repeat=int(row["repeat"]),
                    seconds=float(row["seconds"]),
                    cell_trait_s=float(row["cell_trait_s"]),
                )
                checks = [
                    dict(passed=True, max_abs_logp=float(row["max_abs_logp"])) for _ in traits
                ]
                if cpu:
                    case.update(
                        batch=400,
                        cores=count,
                        numba_threads=count,
                        affinity=list(range(count)),
                        cells=1000037,
                        traits=ntraits,
                        all_passed=True,
                        checks=checks,
                    )
                else:
                    case.update(
                        method="moments", axis="cells", passed=True, gpus=count, audits=checks
                    )
                cases.append(case)
            summary["runs" if cpu else "cases"] = cases
            directory = "cpu/runs_million" if cpu else "gpu/scaling_results"
            path = tmp_path / directory / f"{model}_t{ntraits}" / "summary.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(summary))
    return tmp_path


def test_importer_accepts_only_matched_complete_million_cell_grid(matched_benchmark_summaries):
    from docs.tools.plot_trait_scalability import measurements

    rows = measurements(matched_benchmark_summaries)
    assert len(rows) == 72
    assert {r["cells"] for r in rows} == {1000037}


@pytest.mark.parametrize(
    "problem", ["incomplete", "unmatched", "offset", "small", "source", "nan"]
)
def test_importer_rejects_invalid_cpu_retests(matched_benchmark_summaries, problem):
    from docs.tools.plot_trait_scalability import measurements

    path = next((matched_benchmark_summaries / "cpu/runs_million").glob("*/summary.json"))
    data = json.loads(path.read_text())
    if problem == "incomplete":
        data["complete"] = False
    elif problem == "unmatched":
        data["matched_gpu"]["same_cells_as_gpu"] = False
    elif problem == "offset":
        data["matched_gpu"]["row_offset"] = 1
    elif problem == "small":
        data["cells"] = 30037
    elif problem == "source":
        data["matched_gpu"]["kernel_hashes"]["kernel.py"] = "different-kernel"
    else:
        data["runs"][0]["checks"][-1]["max_abs_logp"] = float("nan")
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        measurements(matched_benchmark_summaries)
