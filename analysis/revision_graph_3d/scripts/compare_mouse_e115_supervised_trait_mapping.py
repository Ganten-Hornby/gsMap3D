#!/usr/bin/env python3
"""Compare Mouse E11.5 supervised 2D and 3D spatial trait mapping."""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RASTER_DPI = 320
METHOD_2D = "2D"
METHOD_3D = "3D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--project-2d", required=True)
    parser.add_argument("--project-3d", required=True)
    parser.add_argument("--trait-manifest", type=Path, required=True)
    parser.add_argument("--annotation", default="heart_regions")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_traits(path: Path) -> list[str]:
    with path.open() as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"Trait manifest must be a non-empty mapping: {path}")
    return [str(trait) for trait in manifest]


def ldsc_path(workdir: Path, project: str, trait: str) -> Path:
    return workdir / project / "spatial_ldsc" / f"{project}_{trait}.csv.gz"


def cauchy_path(workdir: Path, project: str, trait: str, annotation: str) -> Path:
    return (
        workdir
        / project
        / "cauchy_combination"
        / f"{project}_{trait}.{annotation}.cauchy.csv"
    )


def load_spot_scores(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"spot"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")
    if frame["spot"].isna().any():
        raise ValueError(f"Missing spot IDs in {path}")
    frame["spot"] = frame["spot"].astype(str)
    if frame["spot"].duplicated().any():
        raise ValueError(f"Duplicate spot IDs in {path}")

    if "neg_log10_p" in frame:
        score = pd.to_numeric(frame["neg_log10_p"], errors="raise")
        score_kind = "-log10(p)"
    elif "p" in frame:
        p_values = pd.to_numeric(frame["p"], errors="raise")
        if ((p_values < 0) | (p_values > 1)).any():
            raise ValueError(f"p-values outside [0, 1] in {path}")
        score = -np.log10(np.maximum(p_values.to_numpy(), 1e-300))
        score_kind = "-log10(p)"
    elif "z" in frame:
        score = pd.to_numeric(frame["z"], errors="raise")
        score_kind = "z"
    else:
        raise ValueError(f"Expected neg_log10_p, p, or z in {path}")

    result = pd.DataFrame(
        {
            "spot": frame["spot"].to_numpy(),
            "score": np.asarray(score, dtype=np.float64),
        }
    )
    if not np.isfinite(result["score"]).all():
        raise ValueError(f"Non-finite scores in {path}")
    result.attrs["score_kind"] = score_kind
    return result


def align_spot_scores(two_d: pd.DataFrame, three_d: pd.DataFrame) -> pd.DataFrame:
    if two_d.attrs["score_kind"] != three_d.attrs["score_kind"]:
        raise ValueError(
            "2D and 3D score types differ: "
            f"{two_d.attrs['score_kind']} vs {three_d.attrs['score_kind']}"
        )
    merged = two_d.merge(
        three_d,
        on="spot",
        how="inner",
        suffixes=("_2d", "_3d"),
        validate="one_to_one",
    )
    if len(merged) != len(two_d) or len(merged) != len(three_d):
        raise ValueError(
            "2D and 3D spot sets differ: "
            f"2D={len(two_d):,}, 3D={len(three_d):,}, common={len(merged):,}"
        )
    merged.attrs["score_kind"] = two_d.attrs["score_kind"]
    return merged


def load_region_scores(path: Path, annotation: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"trait", "annotation_name", "annotation", "p_cauchy"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {path}")
    if not (frame["annotation_name"].astype(str) == annotation).all():
        raise ValueError(f"Unexpected annotation name in {path}")
    if frame["annotation"].isna().any():
        raise ValueError(f"Missing region labels in {path}")
    frame["region"] = frame["annotation"].astype(str)
    if frame["region"].duplicated().any():
        raise ValueError(f"Duplicate region labels in {path}")
    p_values = pd.to_numeric(frame["p_cauchy"], errors="raise")
    if ((p_values < 0) | (p_values > 1)).any():
        raise ValueError(f"Cauchy p-values outside [0, 1] in {path}")
    frame["score"] = -np.log10(np.maximum(p_values.to_numpy(), 1e-300))
    return frame[["region", "score", "p_cauchy"]]


def align_region_scores(two_d: pd.DataFrame, three_d: pd.DataFrame) -> pd.DataFrame:
    merged = two_d.merge(
        three_d,
        on="region",
        how="inner",
        suffixes=("_2d", "_3d"),
        validate="one_to_one",
    )
    if len(merged) != len(two_d) or len(merged) != len(three_d):
        raise ValueError(
            "2D and 3D heart-region sets differ: "
            f"2D={len(two_d)}, 3D={len(three_d)}, common={len(merged)}"
        )
    merged["rank_2d"] = merged["score_2d"].rank(method="average", ascending=False)
    merged["rank_3d"] = merged["score_3d"].rank(method="average", ascending=False)
    return merged


def correlation(x: pd.Series, y: pd.Series, method: str) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    result = pearsonr(x, y) if method == "pearson" else spearmanr(x, y)
    return float(result.statistic)


def collect_results(
    workdir: Path,
    project_2d: str,
    project_3d: str,
    traits: list[str],
    annotation: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    spot_frames: list[pd.DataFrame] = []
    spot_metrics: list[dict[str, Any]] = []
    region_frames: list[pd.DataFrame] = []
    region_metrics: list[dict[str, Any]] = []
    inputs: dict[str, Any] = {}

    for trait in traits:
        spot_paths = {
            METHOD_2D: ldsc_path(workdir, project_2d, trait),
            METHOD_3D: ldsc_path(workdir, project_3d, trait),
        }
        region_paths = {
            METHOD_2D: cauchy_path(workdir, project_2d, trait, annotation),
            METHOD_3D: cauchy_path(workdir, project_3d, trait, annotation),
        }
        for path in (*spot_paths.values(), *region_paths.values()):
            if not path.is_file():
                raise FileNotFoundError(path)

        aligned_spots = align_spot_scores(
            load_spot_scores(spot_paths[METHOD_2D]),
            load_spot_scores(spot_paths[METHOD_3D]),
        )
        score_kind = aligned_spots.attrs["score_kind"]
        aligned_spots.insert(0, "trait", trait)
        aligned_spots.insert(1, "score_kind", score_kind)
        spot_frames.append(aligned_spots)
        spot_metrics.append(
            {
                "trait": trait,
                "score_kind": score_kind,
                "n_spots": len(aligned_spots),
                "pearson": correlation(
                    aligned_spots["score_2d"], aligned_spots["score_3d"], "pearson"
                ),
                "spearman": correlation(
                    aligned_spots["score_2d"], aligned_spots["score_3d"], "spearman"
                ),
            }
        )

        aligned_regions = align_region_scores(
            load_region_scores(region_paths[METHOD_2D], annotation),
            load_region_scores(region_paths[METHOD_3D], annotation),
        )
        aligned_regions.insert(0, "trait", trait)
        region_frames.append(aligned_regions)
        region_metrics.append(
            {
                "trait": trait,
                "n_regions": len(aligned_regions),
                "score_spearman": correlation(
                    aligned_regions["score_2d"], aligned_regions["score_3d"], "spearman"
                ),
                "rank_spearman": correlation(
                    aligned_regions["rank_2d"], aligned_regions["rank_3d"], "spearman"
                ),
            }
        )
        inputs[trait] = {
            "spot_2d": str(spot_paths[METHOD_2D].resolve()),
            "spot_3d": str(spot_paths[METHOD_3D].resolve()),
            "region_2d": str(region_paths[METHOD_2D].resolve()),
            "region_3d": str(region_paths[METHOD_3D].resolve()),
            "n_spots": len(aligned_spots),
            "n_regions": len(aligned_regions),
        }

    return (
        pd.concat(spot_frames, ignore_index=True),
        pd.DataFrame(spot_metrics),
        pd.concat(region_frames, ignore_index=True),
        {"inputs": inputs, "region_metrics": region_metrics},
    )


def trait_title(trait: str) -> str:
    return trait.replace("_", " ")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=RASTER_DPI, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", dpi=RASTER_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_spot_scatter(
    spot_data: pd.DataFrame, spot_metrics: pd.DataFrame, output_dir: Path
) -> None:
    traits = spot_metrics["trait"].tolist()
    fig, axes = plt.subplots(3, 3, figsize=(18, 16), constrained_layout=True)
    for ax, trait in zip(axes.flat, traits, strict=True):
        data = spot_data.loc[spot_data["trait"] == trait]
        metric = spot_metrics.loc[spot_metrics["trait"] == trait].iloc[0]
        ax.scatter(
            data["score_2d"],
            data["score_3d"],
            s=1.2,
            alpha=0.10,
            color="#225EA8",
            linewidths=0,
            rasterized=True,
        )
        low = float(min(data["score_2d"].min(), data["score_3d"].min()))
        high = float(max(data["score_2d"].max(), data["score_3d"].max()))
        ax.plot([low, high], [low, high], color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{trait_title(trait)}\n"
            f"Pearson={metric['pearson']:.3f}; Spearman={metric['spearman']:.3f}"
        )
        ax.set_xlabel(f"2D {metric['score_kind']}")
        ax.set_ylabel(f"3D {metric['score_kind']}")
    fig.suptitle(
        "Mouse E11.5 supervised trait mapping: matched per-spot scores\n"
        "Every matched spot is plotted (no downsampling)",
        fontsize=15,
    )
    save_figure(fig, output_dir, "per_trait_per_spot_2d_vs_3d_scatter")


def region_colors(regions: list[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab10")
    return {region: cmap(index % 10) for index, region in enumerate(regions)}


def plot_region_scatter(
    region_data: pd.DataFrame, region_metrics: pd.DataFrame, output_dir: Path
) -> None:
    traits = region_metrics["trait"].tolist()
    regions = sorted(region_data["region"].unique())
    colors = region_colors(regions)
    fig, axes = plt.subplots(3, 3, figsize=(18, 16), constrained_layout=True)
    for ax, trait in zip(axes.flat, traits, strict=True):
        data = region_data.loc[region_data["trait"] == trait]
        metric = region_metrics.loc[region_metrics["trait"] == trait].iloc[0]
        for row in data.itertuples(index=False):
            ax.scatter(
                row.score_2d,
                row.score_3d,
                s=55,
                color=colors[row.region],
                edgecolor="white",
                linewidth=0.5,
            )
        low = float(min(data["score_2d"].min(), data["score_3d"].min()))
        high = float(max(data["score_2d"].max(), data["score_3d"].max()))
        padding = max((high - low) * 0.06, 0.05)
        ax.plot(
            [low - padding, high + padding],
            [low - padding, high + padding],
            color="#333333",
            linewidth=0.8,
            linestyle="--",
        )
        ax.set_xlim(low - padding, high + padding)
        ax.set_ylim(low - padding, high + padding)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(
            f"{trait_title(trait)}\n"
            f"score Spearman={metric['score_spearman']:.3f}; "
            f"rank Spearman={metric['rank_spearman']:.3f}"
        )
        ax.set_xlabel("2D heart-region -log10(Cauchy p)")
        ax.set_ylabel("3D heart-region -log10(Cauchy p)")
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=colors[region],
            label=region,
            markersize=7,
        )
        for region in regions
    ]
    fig.legend(handles=handles, loc="outside right center", frameon=False, title="heart_regions")
    fig.suptitle(
        "Mouse E11.5 supervised trait mapping: heart-region Cauchy scores",
        fontsize=15,
    )
    save_figure(fig, output_dir, "per_trait_heart_region_cauchy_scatter")


def plot_rank_flow(
    region_data: pd.DataFrame, region_metrics: pd.DataFrame, output_dir: Path
) -> None:
    traits = region_metrics["trait"].tolist()
    regions = sorted(region_data["region"].unique())
    colors = region_colors(regions)
    fig, axes = plt.subplots(3, 3, figsize=(16, 15), constrained_layout=True)
    for ax, trait in zip(axes.flat, traits, strict=True):
        data = region_data.loc[region_data["trait"] == trait]
        metric = region_metrics.loc[region_metrics["trait"] == trait].iloc[0]
        for row in data.itertuples(index=False):
            ax.plot(
                [0, 1],
                [row.rank_2d, row.rank_3d],
                color=colors[row.region],
                marker="o",
                linewidth=2,
                markersize=5,
            )
        ax.set_xlim(-0.15, 1.15)
        ax.set_xticks([0, 1], [METHOD_2D, METHOD_3D])
        ax.set_yticks(np.arange(1, len(data) + 1))
        ax.set_ylim(len(data) + 0.4, 0.6)
        ax.set_ylabel("Heart-region rank (1 = strongest)")
        ax.set_title(
            f"{trait_title(trait)}\nrank Spearman={metric['rank_spearman']:.3f}"
        )
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    handles = [
        plt.Line2D([], [], color=colors[region], marker="o", label=region, linewidth=2)
        for region in regions
    ]
    fig.legend(handles=handles, loc="outside right center", frameon=False, title="heart_regions")
    fig.suptitle("Heart-region RankFlow: supervised 2D to supervised 3D", fontsize=15)
    save_figure(fig, output_dir, "per_trait_heart_region_rankflow")


def write_report(
    output_dir: Path,
    spot_metrics: pd.DataFrame,
    region_metrics: pd.DataFrame,
    annotation: str,
) -> None:
    lines = [
        "# Mouse E11.5 supervised 2D versus 3D trait mapping",
        "",
        "## Scope",
        "",
        "Both downstream pipelines start from the existing `heart_regions`-supervised embeddings. "
        "The 2D arm computes marker scores within each section; the 3D arm additionally uses "
        "measured-z adjacent-section physical-window neighbors and hierarchical cross-section "
        "pooling. The comparison therefore measures the complete downstream 2D-versus-3D "
        "pipeline effect, not an isolated graph-only effect.",
        "",
        "Every matched spot and every heart region is included. No metric or figure is downsampled.",
        "",
        "## Per-trait, per-spot agreement",
        "",
        "| Trait | Score | Matched spots | Pearson | Spearman |",
        "|---|---|---:|---:|---:|",
    ]
    for row in spot_metrics.itertuples(index=False):
        lines.append(
            f"| {row.trait} | {row.score_kind} | {row.n_spots:,} | "
            f"{row.pearson:.4f} | {row.spearman:.4f} |"
        )
    lines.extend(
        [
            "",
            f"## Per-trait `{annotation}` Cauchy agreement",
            "",
            "| Trait | Regions | Score Spearman | Rank Spearman |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in region_metrics.itertuples(index=False):
        lines.append(
            f"| {row.trait} | {row.n_regions} | {row.score_spearman:.4f} | "
            f"{row.rank_spearman:.4f} |"
        )
    lines.extend(
        [
            "",
            "Cauchy scores are those emitted by gsMap after its configured within-region "
            "median + 3×IQR upper-outlier removal. Region rank 1 denotes the smallest combined "
            "p-value (largest `-log10(Cauchy p)`). With no ties, score Spearman and rank "
            "Spearman are mathematically equivalent; both are reported to make the ranking "
            "definition explicit.",
            "",
            "## Figures and data",
            "",
            "- [Per-trait per-spot scatter](per_trait_per_spot_2d_vs_3d_scatter.png) "
            "([PDF](per_trait_per_spot_2d_vs_3d_scatter.pdf))",
            "- [Per-trait heart-region Cauchy scatter](per_trait_heart_region_cauchy_scatter.png) "
            "([PDF](per_trait_heart_region_cauchy_scatter.pdf))",
            "- [Heart-region RankFlow](per_trait_heart_region_rankflow.png) "
            "([PDF](per_trait_heart_region_rankflow.pdf))",
            "- `per_trait_per_spot_correlations.tsv`: per-trait correlations.",
            "- `per_trait_per_spot_aligned_scores.parquet`: all matched spot-level scores.",
            "- `per_trait_heart_region_correlations.tsv`: region score/rank correlations.",
            "- `per_trait_heart_region_aligned_scores.tsv`: all region scores and ranks.",
            "",
            f"All PNGs are {RASTER_DPI} DPI; PDFs are included for scale-independent viewing.",
            "",
        ]
    )
    (output_dir / "mouse_e115_supervised_trait_mapping_2d_vs_3d_report.md").write_text(
        "\n".join(lines)
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traits = load_traits(args.trait_manifest)
    if len(traits) != 9:
        raise ValueError(f"Expected 9 heart-related traits, found {len(traits)}")

    spot_data, spot_metrics, region_data, audit = collect_results(
        args.workdir,
        args.project_2d,
        args.project_3d,
        traits,
        args.annotation,
    )
    region_metrics = pd.DataFrame(audit.pop("region_metrics"))

    spot_metrics.to_csv(
        args.output_dir / "per_trait_per_spot_correlations.tsv", sep="\t", index=False
    )
    spot_data.to_parquet(
        args.output_dir / "per_trait_per_spot_aligned_scores.parquet", index=False
    )
    region_metrics.to_csv(
        args.output_dir / "per_trait_heart_region_correlations.tsv", sep="\t", index=False
    )
    region_data.to_csv(
        args.output_dir / "per_trait_heart_region_aligned_scores.tsv", sep="\t", index=False
    )

    plot_spot_scatter(spot_data, spot_metrics, args.output_dir)
    plot_region_scatter(region_data, region_metrics, args.output_dir)
    plot_rank_flow(region_data, region_metrics, args.output_dir)
    write_report(args.output_dir, spot_metrics, region_metrics, args.annotation)

    audit.update(
        {
            "projects": {METHOD_2D: args.project_2d, METHOD_3D: args.project_3d},
            "traits": traits,
            "annotation": args.annotation,
            "n_traits": len(traits),
            "sampling_audit": {
                "metrics_subsampling": False,
                "visualization_subsampling": False,
                "all_matched_spots_plotted": True,
                "all_regions_plotted": True,
            },
            "raster_dpi": RASTER_DPI,
            "interpretation_scope": (
                "Full supervised downstream 2D-versus-3D pipeline effect; not graph-only."
            ),
        }
    )
    (args.output_dir / "comparison_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
