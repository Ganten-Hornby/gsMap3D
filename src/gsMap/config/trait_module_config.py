"""
Configuration for trait module discovery.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import typer

from .base import ConfigWithAutoPaths, ensure_path_exists


@dataclass
class TraitModuleConfig(ConfigWithAutoPaths):
    """Trait Module Discovery Configuration"""

    trait_name: Annotated[str, typer.Option(help="Trait to analyze")]

    ldsc_result_path: Annotated[
        Path | None,
        typer.Option(
            help="Path to a spatial LDSC result CSV/CSV.GZ. Defaults to the project LDSC result.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None

    marker_score_h5ad_path: Annotated[
        Path | None,
        typer.Option(
            help="Optional marker-score h5ad with cells as obs and genes as var. "
            "If omitted, uses latent_to_gene/marker_scores.dat with project metadata.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None

    memmap_tmp_dir: Annotated[
        Path | None,
        typer.Option(
            help="Temporary directory for marker-score memmap access.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None

    pvalue_threshold: Annotated[
        float,
        typer.Option(
            help="Cell-level LDSC p-value threshold used to seed trait modules.",
            min=0.0,
            max=1.0,
        ),
    ] = 0.05

    top_gene_fraction: Annotated[
        float,
        typer.Option(
            help="Fraction of genes retained per cell to represent marker specificity.",
            min=0.001,
            max=1.0,
        ),
    ] = 0.05

    max_marker_genes: Annotated[
        int,
        typer.Option(help="Maximum marker genes retained per cell.", min=5),
    ] = 200

    min_module_cells: Annotated[
        int,
        typer.Option(help="Minimum cells required for an independent module.", min=2),
    ] = 10

    max_modules: Annotated[
        int,
        typer.Option(help="Maximum independent modules to report.", min=1),
    ] = 10

    module_similarity_threshold: Annotated[
        float,
        typer.Option(
            help="Minimum cosine similarity to assign an unclaimed significant cell to a module.",
            min=0.0,
            max=1.0,
        ),
    ] = 0.95

    min_specificity_enrichment: Annotated[
        float,
        typer.Option(
            help=(
                "Minimum mean module specificity divided by background mean specificity "
                "required to accept a candidate module. Set to 0 to report all candidates."
            ),
            min=0.0,
        ),
    ] = 1.5

    specificity_null_iterations: Annotated[
        int,
        typer.Option(
            help=(
                "Number of same-size random background draws used to estimate an "
                "empirical p-value for module specificity. Set to 0 to disable."
            ),
            min=0,
        ),
    ] = 100

    max_specificity_null_pvalue: Annotated[
        float,
        typer.Option(
            help=(
                "Maximum empirical background-null p-value allowed when accepting "
                "a candidate module."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = 0.05

    max_module_profile_similarity: Annotated[
        float,
        typer.Option(
            help=(
                "Maximum cosine similarity to any accepted module profile before "
                "rejecting a candidate as a duplicate mechanism."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = 0.9

    min_module_evidence_score: Annotated[
        float,
        typer.Option(
            help=(
                "Minimum diagnostic evidence score expected for a moderate independent "
                "trait module. This is reported as module_evidence_score/status and "
                "does not gate discovery."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = 0.65

    module_stability_iterations: Annotated[
        int,
        typer.Option(
            help=(
                "Number of split-half resampling iterations used to estimate module "
                "marker-profile stability. Set to 0 to disable."
            ),
            min=0,
        ),
    ] = 100

    module_stability_sample_fraction: Annotated[
        float,
        typer.Option(
            help="Fraction of candidate module cells sampled in each stability iteration.",
            min=0.05,
            max=1.0,
        ),
    ] = 0.5

    require_module_stability: Annotated[
        bool,
        typer.Option(
            "--require-module-stability/--diagnose-module-stability",
            help=(
                "Reject candidate modules whose split-half marker-profile stability "
                "does not pass the configured stability thresholds."
            ),
        ),
    ] = False

    min_module_stability_profile_similarity: Annotated[
        float,
        typer.Option(
            help=(
                "Minimum split-half profile cosine similarity allowed when "
                "--require-module-stability is enabled."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = 0.8

    min_module_stability_assignment_fraction: Annotated[
        float,
        typer.Option(
            help=(
                "Minimum fraction of candidate cells recovered by split-half profiles "
                "when --require-module-stability is enabled."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = 0.8

    max_module_stability_null_pvalue: Annotated[
        float,
        typer.Option(
            help=(
                "Maximum empirical p-value from random equal-size seed-cell groups "
                "allowed when --require-module-stability is enabled."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = 0.05

    run_conditional_ldsc: Annotated[
        bool,
        typer.Option(
            "--run-conditional-ldsc/--skip-conditional-ldsc",
            help=(
                "Run a module-level conditional S-LDSC approximation for each candidate. "
                "The candidate module is tested as one LD-score annotation conditioned on "
                "baseline annotations and previously accepted module annotations."
            ),
        ),
    ] = False

    require_conditional_ldsc: Annotated[
        bool,
        typer.Option(
            "--require-conditional-ldsc/--diagnose-conditional-ldsc",
            help=(
                "Reject candidate modules whose module-level conditional LDSC p-value "
                "does not pass conditional_ldsc_pvalue_threshold. This automatically "
                "enables --run-conditional-ldsc."
            ),
        ),
    ] = False

    conditional_ldsc_pvalue_threshold: Annotated[
        float,
        typer.Option(
            help=(
                "Maximum module-level conditional LDSC p-value allowed when "
                "--require-conditional-ldsc is enabled."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = 0.05

    conditional_ldsc_n_blocks: Annotated[
        int,
        typer.Option(
            help="Number of jackknife blocks for module-level conditional LDSC.",
            min=2,
        ),
    ] = 200

    conditional_chisq_max: Annotated[
        int | None,
        typer.Option(
            help=(
                "Maximum GWAS chi-square value for module-level conditional LDSC. "
                "If omitted, uses the same default rule as spatial-ldsc."
            ),
        ),
    ] = None

    conditional_sumstats_file: Annotated[
        Path | None,
        typer.Option(
            help=(
                "GWAS summary statistics for module-level conditional LDSC. If omitted, "
                "trait-modules tries to read the path from gsmap_web_report/report_meta.json."
            ),
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None

    conditional_snp_gene_weight_adata_path: Annotated[
        Path | None,
        typer.Option(
            help=(
                "SNP-gene weight matrix for module-level conditional LDSC. If omitted, "
                "trait-modules tries to read the path from gsmap_web_report/report_meta.json."
            ),
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None

    conditional_w_ld_dir: Annotated[
        Path | None,
        typer.Option(
            help=(
                "Directory containing LDSC regression weight files for module-level "
                "conditional LDSC. If omitted, trait-modules tries report metadata first, "
                "then project_dir/generate_ldscore/w_ld."
            ),
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None

    conditional_additional_baseline_h5ad_path_list: Annotated[
        list[Path],
        typer.Option(
            help="Additional baseline h5ad files for module-level conditional LDSC.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = field(default_factory=list)

    recompute_background_after_modules: Annotated[
        bool,
        typer.Option(
            "--recompute-background-after-modules/--fixed-background-after-modules",
            help=(
                "After accepting a module, raise the per-gene search background using "
                "that module's marker scores before searching for later modules."
            ),
        ),
    ] = True

    iterative_background_quantile: Annotated[
        float,
        typer.Option(
            help=(
                "Quantile of accepted module-cell marker scores used as the iterative "
                "search background for later module discovery."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = 0.5

    condition_on_claimed_modules: Annotated[
        bool,
        typer.Option(
            "--condition-on-claimed-modules/--no-module-conditioning",
            help=(
                "After accepting a module, project its marker profile out of later "
                "iterations so follow-up modules must use residual marker signal."
            ),
        ),
    ] = True

    background_scope: Annotated[
        Literal["all", "seeds"],
        typer.Option(
            help=(
                "Cells used to estimate marker-score background. 'all' samples from the "
                "full marker-score matrix; 'seeds' uses only significant seed cells."
            ),
            case_sensitive=False,
        ),
    ] = "all"

    background_sample_size: Annotated[
        int,
        typer.Option(
            help="Maximum number of cells sampled to estimate global marker-score background.",
            min=1,
        ),
    ] = 50000

    background_quantile: Annotated[
        float,
        typer.Option(
            help="Background marker-score quantile subtracted from selected cells each iteration.",
            min=0.0,
            max=1.0,
        ),
    ] = 0.5

    random_seed: Annotated[
        int,
        typer.Option(help="Random seed used when sampling cells for background estimation."),
    ] = 0

    min_positive_beta: Annotated[
        bool,
        typer.Option(
            "--min-positive-beta/--allow-negative-beta",
            help="Require positive LDSC beta for module seed cells.",
        ),
    ] = True

    output_dir: Annotated[
        Path | None,
        typer.Option(
            help="Output directory for trait-module CSV files. Defaults to project_dir/trait_modules.",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None

    @property
    @ensure_path_exists
    def trait_module_dir(self) -> Path:
        return self.output_dir or (self.project_dir / "trait_modules")

    @property
    def cell_modules_path(self) -> Path:
        return self.trait_module_dir / f"{self.project_name}_{self.trait_name}.cell_modules.csv"

    @property
    def module_summary_path(self) -> Path:
        return self.trait_module_dir / f"{self.project_name}_{self.trait_name}.module_summary.csv"

    @property
    def module_genes_path(self) -> Path:
        return self.trait_module_dir / f"{self.project_name}_{self.trait_name}.module_genes.csv"

    @property
    def module_trace_path(self) -> Path:
        return self.trait_module_dir / f"{self.project_name}_{self.trait_name}.module_trace.csv"

    @property
    def run_summary_path(self) -> Path:
        return self.trait_module_dir / f"{self.project_name}_{self.trait_name}.run_summary.csv"

    def __post_init__(self):
        super().__post_init__()
        if self.require_conditional_ldsc:
            self.run_conditional_ldsc = True
        if self.ldsc_result_path is None:
            self.ldsc_result_path = self.get_ldsc_result_file(self.trait_name)
        if not self.ldsc_result_path.exists():
            raise FileNotFoundError(f"LDSC result file not found: {self.ldsc_result_path}")
        self.show_config(TraitModuleConfig)
