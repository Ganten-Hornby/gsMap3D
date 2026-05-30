"""
Fast trait-module discovery from per-cell LDSC and marker scores.

The prototype implemented here is an approximation layer for independent cell
states. It does not claim conditional genetic independence. Instead, it asks:
among cells already significant for a trait, which groups retain distinct marker
specificity profiles after removing the cells explained by stronger modules?
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, norm

from gsMap.config import TraitModuleConfig
from gsMap.spatial_ldsc.io import _read_sumstats, _read_w_ld, load_marker_scores_memmap_format

logger = logging.getLogger("gsMap.trait_modules")

MODULE_EVIDENCE_STRONG_THRESHOLD = 0.85

CELL_MODULE_STATUS_ASSIGNED = "assigned"
CELL_MODULE_STATUS_REJECTED = "rejected"
CELL_MODULE_STATUS_UNRESOLVED = "unresolved"


MODULE_SUMMARY_COLUMNS = [
    "trait",
    "module",
    "n_cells",
    "module_signal_fraction",
    "cumulative_signal_fraction",
    "residual_signal_fraction_after_module",
    "module_evidence_score",
    "module_evidence_status",
    "mean_neg_log10_p",
    "max_neg_log10_p",
    "conditional_ldsc_neg_log10_p",
    "conditional_ldsc_p",
    "conditional_ldsc_beta",
    "conditional_ldsc_se",
    "conditional_ldsc_z",
    "stability_profile_similarity",
    "stability_assignment_fraction",
    "stability_null_pvalue",
    "mean_specificity",
    "background_mean_specificity",
    "specificity_enrichment",
    "specificity_null_pvalue",
    "max_profile_similarity",
    "most_similar_module",
]

MODULE_GENE_COLUMNS = ["trait", "module", "rank", "gene", "score"]

MODULE_SPECIFICITY_BASE_COLUMNS = [
    "spot",
    "best_trait_module",
    "best_module_specificity",
]

MODULE_PAIR_COLUMNS = [
    "trait",
    "module_a",
    "module_b",
    "profile_similarity",
    "max_allowed_profile_similarity",
    "profile_distinct",
    "marker_jaccard",
    "marker_gene_universe_size",
    "marker_overlap_null_pvalue",
    "shared_marker_count",
    "module_a_marker_count",
    "module_b_marker_count",
    "shared_top_genes",
    "module_a_only_top_genes",
    "module_b_only_top_genes",
    "module_a_n_cells",
    "module_b_n_cells",
    "module_a_signal_fraction",
    "module_b_signal_fraction",
    "module_a_evidence_status",
    "module_b_evidence_status",
]

MODULE_RUN_COLUMNS = [
    "trait",
    "n_seed_cells",
    "n_modules",
    "n_module_cells",
    "module_cell_fraction",
    "n_assigned_seed_cells",
    "assigned_seed_cell_fraction",
    "n_rejected_seed_cells",
    "rejected_seed_cell_fraction",
    "n_unresolved_seed_cells",
    "unresolved_seed_cell_fraction",
    "top_module_signal_fraction",
    "accepted_signal_fraction",
    "residual_signal_fraction",
    "resolution_status",
]

MODULE_TRACE_COLUMNS = [
    "trait",
    "iteration",
    "candidate_module",
    "decision",
    "reason",
    "candidate_signal_fraction",
    "accepted_signal_fraction",
    "residual_signal_fraction_before",
    "residual_signal_fraction_after",
    "module_evidence_score",
    "module_evidence_status",
    "candidate_n_cells",
    "remaining_cells_before",
    "remaining_cells_after",
    "accepted_module",
    "mean_neg_log10_p",
    "mean_conditional_neg_log10_p",
    "conditional_ldsc_neg_log10_p",
    "conditional_ldsc_p",
    "conditional_ldsc_beta",
    "conditional_ldsc_se",
    "conditional_ldsc_z",
    "stability_profile_similarity",
    "stability_assignment_fraction",
    "stability_null_pvalue",
    "mean_specificity",
    "background_mean_specificity",
    "specificity_enrichment",
    "specificity_null_pvalue",
    "max_profile_similarity",
    "most_similar_module",
]


@dataclass
class ModuleResult:
    module_id: int
    cells: pd.Index
    profile: np.ndarray
    specificity: np.ndarray
    mean_neg_log10_p: float
    background_specificity_mean: float
    specificity_enrichment: float
    specificity_null_pvalue: float
    max_profile_similarity: float
    most_similar_module: int
    module_signal_fraction: float = float("nan")
    cumulative_signal_fraction: float = float("nan")
    residual_signal_fraction_after_module: float = float("nan")
    module_evidence_score: float = float("nan")
    module_evidence_status: str = "unknown"
    conditional_ldsc_beta: float = float("nan")
    conditional_ldsc_se: float = float("nan")
    conditional_ldsc_z: float = float("nan")
    conditional_ldsc_p: float = float("nan")
    conditional_ldsc_neg_log10_p: float = float("nan")
    stability_profile_similarity: float = float("nan")
    stability_assignment_fraction: float = float("nan")
    stability_null_pvalue: float = float("nan")


@dataclass
class ModuleStabilityResult:
    profile_similarity: float = float("nan")
    assignment_fraction: float = float("nan")
    null_pvalue: float = float("nan")

    def as_columns(self) -> dict[str, float]:
        return {
            "stability_profile_similarity": self.profile_similarity,
            "stability_assignment_fraction": self.assignment_fraction,
            "stability_null_pvalue": self.null_pvalue,
        }


@dataclass
class ConditionalLDSCResult:
    beta: float = float("nan")
    se: float = float("nan")
    z: float = float("nan")
    p: float = float("nan")
    neg_log10_p: float = float("nan")

    def as_columns(self) -> dict[str, float]:
        return {
            "conditional_ldsc_neg_log10_p": self.neg_log10_p,
            "conditional_ldsc_p": self.p,
            "conditional_ldsc_beta": self.beta,
            "conditional_ldsc_se": self.se,
            "conditional_ldsc_z": self.z,
        }


@dataclass
class ConditionalLDSCContext:
    snp_gene_weight_matrix: object
    marker_gene_indices: np.ndarray
    baseline_ld: np.ndarray
    baseline_ld_sum: np.ndarray
    chisq: np.ndarray
    sample_size: np.ndarray
    regression_weights: np.ndarray
    nbar: float
    n_blocks: int

    def annotation_from_profile(self, profile: np.ndarray) -> np.ndarray:
        profile = np.maximum(profile[self.marker_gene_indices], 0.0).astype(np.float32)
        annotation = self.snp_gene_weight_matrix @ profile
        return np.asarray(annotation, dtype=np.float32).reshape(-1)

    def fit_profile(
        self, profile: np.ndarray, conditioning_annotations: list[np.ndarray]
    ) -> tuple[ConditionalLDSCResult, np.ndarray]:
        candidate_annotation = self.annotation_from_profile(profile)
        return (
            _fit_conditional_ldsc(
                candidate_annotation=candidate_annotation,
                conditioning_annotations=conditioning_annotations,
                baseline_ld=self.baseline_ld,
                baseline_ld_sum=self.baseline_ld_sum,
                chisq=self.chisq,
                sample_size=self.sample_size,
                regression_weights=self.regression_weights,
                nbar=self.nbar,
                n_blocks=self.n_blocks,
            ),
            candidate_annotation,
        )


def _read_ldsc_result(path) -> pd.DataFrame:
    df = pd.read_csv(path, compression="infer")
    if "spot" not in df.columns:
        raise ValueError(f"LDSC result must contain a 'spot' column: {path}")
    if "p" not in df.columns:
        raise ValueError(f"LDSC result must contain a 'p' column: {path}")
    df["spot"] = df["spot"].astype(str)
    if "neg_log10_p" not in df.columns:
        df["neg_log10_p"] = -np.log10(df["p"].clip(lower=1e-300))
    return df.set_index("spot", drop=False)


def _load_marker_scores(config: TraitModuleConfig):
    if config.marker_score_h5ad_path is not None:
        logger.info(f"Loading marker scores from {config.marker_score_h5ad_path}")
        return ad.read_h5ad(config.marker_score_h5ad_path, backed="r")
    logger.info("Loading marker scores from latent_to_gene memmap outputs")
    return load_marker_scores_memmap_format(config)


def _as_dense_float32(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def _select_seed_cells(ldsc: pd.DataFrame, config: TraitModuleConfig) -> pd.Index:
    mask = ldsc["p"] <= config.pvalue_threshold
    if config.min_positive_beta and "beta" in ldsc.columns:
        mask &= ldsc["beta"] > 0
    seeds = pd.Index(ldsc.index[mask])
    logger.info(
        f"Selected {len(seeds)} seed cells at p <= {config.pvalue_threshold} "
        f"from {len(ldsc)} LDSC rows"
    )
    return seeds


def _load_seed_marker_matrix(
    marker_adata, seed_cells: pd.Index
) -> tuple[pd.Index, list[str], np.ndarray]:
    common_cells = pd.Index(marker_adata.obs_names).intersection(seed_cells)
    if common_cells.empty:
        raise ValueError("No common cells between LDSC results and marker scores.")

    cell_pos = pd.Index(marker_adata.obs_names).get_indexer(common_cells)
    matrix = _as_dense_float32(marker_adata.X[cell_pos, :])
    gene_names = list(marker_adata.var_names)
    logger.info(f"Loaded seed marker matrix: {matrix.shape[0]} cells x {matrix.shape[1]} genes")
    return common_cells, gene_names, matrix


def _load_background_marker_matrix(marker_adata, config: TraitModuleConfig) -> np.ndarray:
    if config.background_scope == "seeds":
        raise ValueError("Seed-scoped background should use the seed marker matrix directly.")

    n_cells = marker_adata.n_obs
    n_background = min(config.background_sample_size, n_cells)
    if n_background == n_cells:
        cell_pos = np.arange(n_cells)
    else:
        rng = np.random.default_rng(config.random_seed)
        cell_pos = np.sort(rng.choice(n_cells, size=n_background, replace=False))

    matrix = _as_dense_float32(marker_adata.X[cell_pos, :])
    logger.info(
        f"Loaded global background marker matrix: {matrix.shape[0]} cells x {matrix.shape[1]} genes"
    )
    return matrix


def _top_specificity_mask(specificity: np.ndarray, top_n: int) -> np.ndarray:
    if top_n >= specificity.size:
        mask = specificity > 0
    else:
        top_idx = np.argpartition(specificity, -top_n)[-top_n:]
        mask = np.zeros(specificity.size, dtype=bool)
        mask[top_idx] = True
        mask &= specificity > 0
    return mask


def _top_positive_sum(matrix: np.ndarray, top_n: int) -> np.ndarray:
    positive = np.maximum(matrix, 0.0)
    if top_n >= positive.shape[1]:
        return positive.sum(axis=1)
    return np.partition(positive, -top_n, axis=1)[:, -top_n:].sum(axis=1)


def _module_from_best_seed(
    residual: np.ndarray,
    remaining_mask: np.ndarray,
    weights: np.ndarray,
    top_n: int,
    similarity_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    remaining_idx = np.flatnonzero(remaining_mask)
    if remaining_idx.size == 0:
        return remaining_idx, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    weighted_signal = _top_positive_sum(residual[remaining_idx], top_n) * weights[remaining_idx]
    seed_local = int(np.argmax(weighted_signal))
    seed_idx = remaining_idx[seed_local]

    specificity = residual[seed_idx].copy()
    marker_mask = _top_specificity_mask(specificity, top_n)
    if not np.any(marker_mask):
        return np.array([], dtype=int), specificity, specificity

    profile = np.zeros_like(specificity, dtype=np.float32)
    profile[marker_mask] = specificity[marker_mask]
    # Assign cells by similarity inside the selected marker-gene subspace. Using
    # all genes here over-penalizes broad marker-score residuals that are
    # unrelated to the candidate module.
    marker_residual = np.maximum(residual[remaining_idx][:, marker_mask], 0.0)
    marker_profile = profile[marker_mask]
    marker_profile_norm = marker_profile / max(float(np.linalg.norm(marker_profile)), 1e-8)
    similarities = _row_normalize(marker_residual) @ marker_profile_norm
    module_idx = remaining_idx[similarities >= similarity_threshold]
    if module_idx.size == 0:
        module_idx = np.array([seed_idx], dtype=int)

    module_profile = np.zeros_like(specificity, dtype=np.float32)
    module_profile[marker_mask] = np.maximum(residual[module_idx][:, marker_mask], 0.0).mean(
        axis=0
    )
    return module_idx, module_profile.astype(np.float32), specificity.astype(np.float32)


def _condition_residual_on_profile(residual: np.ndarray, profile: np.ndarray) -> np.ndarray:
    profile = np.maximum(profile, 0.0).astype(np.float32, copy=False)
    profile_norm_sq = float(np.dot(profile, profile))
    if profile_norm_sq <= 1e-8:
        return residual

    projection_scale = (residual @ profile) / profile_norm_sq
    conditioned = residual - np.outer(np.maximum(projection_scale, 0.0), profile)
    return np.maximum(conditioned, 0.0).astype(np.float32, copy=False)


def _compute_iterative_background(
    marker_matrix: np.ndarray,
    base_background: np.ndarray,
    claimed_module_indices: list[np.ndarray],
    *,
    recompute_background_after_modules: bool,
    iterative_background_quantile: float,
) -> np.ndarray:
    search_background = base_background.copy()
    if recompute_background_after_modules:
        for module_idx in claimed_module_indices:
            module_background = np.quantile(
                marker_matrix[module_idx], iterative_background_quantile, axis=0
            ).astype(np.float32)
            search_background = np.maximum(search_background, module_background)
    return search_background


def _condition_residual_on_profiles(
    residual: np.ndarray, claimed_profiles: list[np.ndarray], *, condition_on_claimed_modules: bool
) -> np.ndarray:
    if condition_on_claimed_modules:
        for profile in claimed_profiles:
            residual = _condition_residual_on_profile(residual, profile)
    return residual


def _compute_search_residual(
    marker_matrix: np.ndarray,
    base_background: np.ndarray,
    claimed_module_indices: list[np.ndarray],
    claimed_profiles: list[np.ndarray],
    *,
    recompute_background_after_modules: bool,
    iterative_background_quantile: float,
    condition_on_claimed_modules: bool,
) -> np.ndarray:
    search_background = _compute_iterative_background(
        marker_matrix,
        base_background,
        claimed_module_indices,
        recompute_background_after_modules=recompute_background_after_modules,
        iterative_background_quantile=iterative_background_quantile,
    )
    search_residual = np.maximum(marker_matrix - search_background, 0.0).astype(
        np.float32, copy=False
    )
    return _condition_residual_on_profiles(
        search_residual,
        claimed_profiles,
        condition_on_claimed_modules=condition_on_claimed_modules,
    )


def _conditional_neg_log10_p_proxy(
    original_neg_log10_p: np.ndarray,
    residual: np.ndarray,
    search_residual: np.ndarray,
    top_n: int,
) -> np.ndarray:
    original_signal = _top_positive_sum(residual, top_n)
    conditional_signal = _top_positive_sum(search_residual, top_n)
    retained_fraction = conditional_signal / np.maximum(original_signal, 1e-8)
    return (original_neg_log10_p * np.clip(retained_fraction, 0.0, 1.0)).astype(
        np.float32, copy=False
    )


def _trace_conditional_ldsc_empty() -> dict[str, float]:
    return ConditionalLDSCResult().as_columns()


def _trace_module_stability_empty() -> dict[str, float]:
    return ModuleStabilityResult().as_columns()


def _trace_module_evidence_empty() -> dict[str, float | str]:
    return {
        "module_evidence_score": float("nan"),
        "module_evidence_status": "not_evaluated",
    }


def _read_report_meta(config: TraitModuleConfig) -> dict:
    report_meta_path = config.report_dir / "report_meta.json"
    if not report_meta_path.exists():
        return {}
    try:
        return json.loads(report_meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid report metadata JSON: {report_meta_path}") from exc


def _resolve_optional_path(value) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.exists() else None


def _resolve_conditional_resources(
    config: TraitModuleConfig,
) -> tuple[Path, Path, Path, list[Path]]:
    report_meta = _read_report_meta(config)
    sumstats_by_trait = report_meta.get("sumstats_config_dict") or {}
    sumstats_from_meta = sumstats_by_trait.get(config.trait_name) or report_meta.get(
        "sumstats_file"
    )

    sumstats_file = config.conditional_sumstats_file or _resolve_optional_path(sumstats_from_meta)
    snp_gene_weight_path = config.conditional_snp_gene_weight_adata_path or _resolve_optional_path(
        report_meta.get("snp_gene_weight_adata_path")
    )
    w_ld_dir = (
        config.conditional_w_ld_dir
        or _resolve_optional_path(report_meta.get("w_ld_dir"))
        or _resolve_optional_path(config.ldscore_save_dir / "w_ld")
    )
    additional_baseline_paths = list(config.conditional_additional_baseline_h5ad_path_list)
    if not additional_baseline_paths:
        additional_baseline_paths = [
            Path(path)
            for path in report_meta.get("additional_baseline_h5ad_path_list", [])
            if _resolve_optional_path(path) is not None
        ]

    missing = []
    if sumstats_file is None:
        missing.append("conditional_sumstats_file")
    if snp_gene_weight_path is None:
        missing.append("conditional_snp_gene_weight_adata_path")
    if w_ld_dir is None:
        missing.append("conditional_w_ld_dir")
    if missing:
        raise ValueError(
            "Module-level conditional LDSC requires "
            + ", ".join(missing)
            + ". Provide the paths explicitly or generate a report_meta.json with those fields."
        )

    return sumstats_file, snp_gene_weight_path, w_ld_dir, additional_baseline_paths


def _as_dense_vector(array) -> np.ndarray:
    if hasattr(array, "toarray"):
        array = array.toarray()
    return np.asarray(array, dtype=np.float32).reshape(-1)


def _prepare_conditional_ldsc_context(
    config: TraitModuleConfig, gene_names: list[str]
) -> ConditionalLDSCContext | None:
    if not config.run_conditional_ldsc:
        return None

    sumstats_file, snp_gene_weight_path, w_ld_dir, additional_baseline_paths = (
        _resolve_conditional_resources(config)
    )
    logger.info(
        "Preparing module-level conditional LDSC resources: "
        f"sumstats={sumstats_file}, snp_gene_weight={snp_gene_weight_path}, w_ld_dir={w_ld_dir}"
    )

    w_ld = _read_w_ld(w_ld_dir).set_index("SNP")
    snp_gene_weight_adata = ad.read_h5ad(snp_gene_weight_path)
    try:
        X = snp_gene_weight_adata.X
        if hasattr(X, "tocsr"):
            X = X.tocsr()

        all_gene = _as_dense_vector(X[:, :-1].sum(axis=1))
        unmapped = _as_dense_vector(X[:, -1])
        baseline_df = pd.DataFrame(
            {
                "base": all_gene + unmapped,
                "all_gene": all_gene,
            },
            index=pd.Index(snp_gene_weight_adata.obs_names, name="SNP"),
        )

        common_snps = baseline_df.index.intersection(w_ld.index)
        baseline_df = baseline_df.loc[common_snps]
        w_ld = w_ld.loc[common_snps]

        for baseline_path in additional_baseline_paths:
            add_adata = ad.read_h5ad(baseline_path)
            try:
                common_snps = common_snps.intersection(add_adata.obs_names)
                baseline_df = baseline_df.loc[common_snps]
                w_ld = w_ld.loc[common_snps]
                add_x = add_adata[common_snps].X
                if hasattr(add_x, "toarray"):
                    add_x = add_x.toarray()
                add_df = pd.DataFrame(add_x, index=common_snps, columns=add_adata.var_names)
                baseline_df = pd.concat([baseline_df, add_df], axis=1)
            finally:
                if getattr(add_adata, "isbacked", False):
                    add_adata.file.close()

        sumstats = _read_sumstats(sumstats_file, alleles=False, dropna=False)
        sumstats.set_index("SNP", inplace=True)
        sumstats = sumstats.astype(np.float32)
        sumstats["chisq"] = sumstats.Z**2
        chisq_max = config.conditional_chisq_max
        if chisq_max is None:
            chisq_max = max(0.001 * float(sumstats.N.max()), 80)
        sumstats = sumstats[sumstats.chisq < chisq_max]
        common_snps = common_snps.intersection(sumstats.index)
        if len(common_snps) < config.conditional_ldsc_n_blocks:
            raise ValueError(
                f"Only {len(common_snps)} common SNPs available for conditional LDSC; "
                f"conditional_ldsc_n_blocks={config.conditional_ldsc_n_blocks}."
            )

        baseline_df = baseline_df.loc[common_snps]
        w_ld = w_ld.loc[common_snps]
        sumstats = sumstats.loc[common_snps]
        snp_positions = snp_gene_weight_adata.obs_names.get_indexer(common_snps)

        marker_genes = pd.Index(gene_names)
        weight_genes = pd.Index(snp_gene_weight_adata.var_names[:-1])
        common_genes = marker_genes.intersection(weight_genes)
        if common_genes.empty:
            raise ValueError("No common genes between marker scores and SNP-gene weight matrix.")

        marker_gene_indices = marker_genes.get_indexer(common_genes)
        weight_gene_indices = weight_genes.get_indexer(common_genes)
        snp_gene_weight_matrix = X[snp_positions, :][:, weight_gene_indices]
        if hasattr(snp_gene_weight_matrix, "tocsr"):
            snp_gene_weight_matrix = snp_gene_weight_matrix.tocsr()

        n_blocks = min(config.conditional_ldsc_n_blocks, len(common_snps))
        return ConditionalLDSCContext(
            snp_gene_weight_matrix=snp_gene_weight_matrix,
            marker_gene_indices=marker_gene_indices,
            baseline_ld=baseline_df.to_numpy(dtype=np.float32),
            baseline_ld_sum=baseline_df.sum(axis=1).to_numpy(dtype=np.float32),
            chisq=sumstats.chisq.to_numpy(dtype=np.float32),
            sample_size=sumstats.N.to_numpy(dtype=np.float32),
            regression_weights=w_ld.LD_weights.to_numpy(dtype=np.float32),
            nbar=float(sumstats.N.mean()),
            n_blocks=n_blocks,
        )
    finally:
        if getattr(snp_gene_weight_adata, "isbacked", False):
            snp_gene_weight_adata.file.close()


def _weighted_ldsc_design(
    candidate_annotation: np.ndarray,
    conditioning_annotations: list[np.ndarray],
    baseline_ld: np.ndarray,
    baseline_ld_sum: np.ndarray,
    chisq: np.ndarray,
    sample_size: np.ndarray,
    regression_weights: np.ndarray,
    nbar: float,
) -> tuple[np.ndarray, np.ndarray]:
    annotation_columns = [candidate_annotation.astype(np.float32)]
    annotation_columns.extend(a.astype(np.float32) for a in conditioning_annotations)
    annotation_columns.append(baseline_ld * sample_size.reshape(-1, 1) / nbar)
    annotation_columns.append(np.ones((len(candidate_annotation), 1), dtype=np.float32))
    design = np.column_stack(annotation_columns).astype(np.float64)

    x_tot = candidate_annotation + baseline_ld_sum
    for annotation in conditioning_annotations:
        x_tot = x_tot + annotation

    hsq = 10000.0 * (float(np.mean(chisq)) - 1.0) / max(float(np.mean(x_tot * sample_size)), 1e-8)
    hsq = float(np.clip(hsq, 0.0, 1.0))
    ld_clip = np.maximum(x_tot, 1.0)
    w_ld_clip = np.maximum(regression_weights, 1.0)
    c = hsq * sample_size / 10000.0
    weights = np.sqrt(1.0 / (2.0 * np.square(1.0 + c * ld_clip) * w_ld_clip))
    weights = weights / max(float(weights.sum()), 1e-8)
    weights = weights.astype(np.float64)

    return design * weights[:, None], chisq.astype(np.float64) * weights


def _fit_conditional_ldsc(
    *,
    candidate_annotation: np.ndarray,
    conditioning_annotations: list[np.ndarray],
    baseline_ld: np.ndarray,
    baseline_ld_sum: np.ndarray,
    chisq: np.ndarray,
    sample_size: np.ndarray,
    regression_weights: np.ndarray,
    nbar: float,
    n_blocks: int,
) -> ConditionalLDSCResult:
    n_snps = len(candidate_annotation)
    if n_snps < n_blocks:
        n_blocks = n_snps
    block_size = n_snps // n_blocks
    n_used = block_size * n_blocks
    if n_used <= 0:
        return ConditionalLDSCResult()

    weighted_x, weighted_y = _weighted_ldsc_design(
        candidate_annotation=candidate_annotation[:n_used],
        conditioning_annotations=[a[:n_used] for a in conditioning_annotations],
        baseline_ld=baseline_ld[:n_used],
        baseline_ld_sum=baseline_ld_sum[:n_used],
        chisq=chisq[:n_used],
        sample_size=sample_size[:n_used],
        regression_weights=regression_weights[:n_used],
        nbar=nbar,
    )
    if np.linalg.matrix_rank(weighted_x) < weighted_x.shape[1]:
        return ConditionalLDSCResult()

    block_x = weighted_x.reshape(n_blocks, block_size, -1)
    block_y = weighted_y.reshape(n_blocks, block_size)
    xtx_blocks = np.einsum("nbp,nbq->npq", block_x, block_x)
    xty_blocks = np.einsum("nbp,nb->np", block_x, block_y)
    xtx_total = xtx_blocks.sum(axis=0)
    xty_total = xty_blocks.sum(axis=0)
    estimate = np.linalg.lstsq(xtx_total, xty_total, rcond=None)[0]

    delete_estimates = []
    for block_idx in range(n_blocks):
        delete_xtx = xtx_total - xtx_blocks[block_idx]
        delete_xty = xty_total - xty_blocks[block_idx]
        if np.linalg.matrix_rank(delete_xtx) < delete_xtx.shape[0]:
            return ConditionalLDSCResult()
        delete_estimates.append(np.linalg.solve(delete_xtx, delete_xty))

    delete_estimates = np.asarray(delete_estimates)
    pseudovalues = n_blocks * estimate - (n_blocks - 1) * delete_estimates
    jackknife_estimate = pseudovalues.mean(axis=0)
    jackknife_cov = np.cov(pseudovalues.T, ddof=1) / n_blocks
    se = float(np.sqrt(np.maximum(np.diag(jackknife_cov)[0], 0.0)))
    beta = float(jackknife_estimate[0] / nbar)
    se = se / nbar
    if not np.isfinite(se) or se <= 0:
        return ConditionalLDSCResult(beta=beta)

    z = beta / se
    p = float(norm.sf(z))
    neg_log10_p = float(-np.log10(max(p, 1e-300)))
    return ConditionalLDSCResult(beta=beta, se=se, z=z, p=p, neg_log10_p=neg_log10_p)


def _module_specificity_scores(matrix: np.ndarray, profile: np.ndarray) -> np.ndarray:
    module_marker_mask = profile > 0
    if not np.any(module_marker_mask):
        return np.zeros(matrix.shape[0], dtype=np.float32)

    normalized = _row_normalize(np.maximum(matrix[:, module_marker_mask], 0.0))
    profile_markers = profile[module_marker_mask]
    profile_norm = profile_markers / max(float(np.linalg.norm(profile_markers)), 1e-8)
    return (normalized @ profile_norm).astype(np.float32)


def _module_specificity_column(module_id: int) -> str:
    return f"module_{module_id}_specificity"


def _module_specificity_table(
    spot_names: pd.Index,
    residual: np.ndarray,
    module_results: list[ModuleResult],
) -> pd.DataFrame:
    columns = MODULE_SPECIFICITY_BASE_COLUMNS + [
        _module_specificity_column(result.module_id) for result in module_results
    ]
    if not module_results:
        return pd.DataFrame(columns=columns)

    module_scores = {
        _module_specificity_column(result.module_id): _module_specificity_scores(
            residual, result.profile
        )
        for result in module_results
    }
    score_matrix = np.column_stack(list(module_scores.values()))
    best_pos = np.argmax(score_matrix, axis=1)
    best_scores = score_matrix[np.arange(score_matrix.shape[0]), best_pos]
    module_ids = np.array([result.module_id for result in module_results], dtype=int)
    best_modules = module_ids[best_pos]
    best_modules = np.where(best_scores > 0.0, best_modules, -1)

    df = pd.DataFrame(
        {
            "spot": spot_names.astype(str),
            "best_trait_module": best_modules,
            "best_module_specificity": best_scores.astype(np.float32),
        }
    )
    for column, values in module_scores.items():
        df[column] = values
    return df[columns]


def _module_specificity_table_from_marker_adata(
    marker_adata,
    base_background: np.ndarray,
    module_results: list[ModuleResult],
    *,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    columns = MODULE_SPECIFICITY_BASE_COLUMNS + [
        _module_specificity_column(result.module_id) for result in module_results
    ]
    if not module_results:
        return pd.DataFrame(columns=columns)

    frames = []
    spot_names = pd.Index(marker_adata.obs_names)
    for start in range(0, marker_adata.n_obs, chunk_size):
        end = min(start + chunk_size, marker_adata.n_obs)
        matrix = _as_dense_float32(marker_adata.X[start:end, :])
        residual = np.maximum(matrix - base_background, 0.0).astype(np.float32, copy=False)
        frames.append(_module_specificity_table(spot_names[start:end], residual, module_results))

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)[columns]


def _profile_cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left = np.maximum(left, 0.0)
    right = np.maximum(right, 0.0)
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _profile_from_indices(residual: np.ndarray, module_idx: np.ndarray, marker_mask: np.ndarray):
    profile = np.zeros(residual.shape[1], dtype=np.float32)
    if module_idx.size == 0 or not np.any(marker_mask):
        return profile
    profile[marker_mask] = np.maximum(residual[module_idx][:, marker_mask], 0.0).mean(axis=0)
    return profile


def _module_assignment_fraction(
    residual: np.ndarray,
    module_idx: np.ndarray,
    profile: np.ndarray,
    similarity_threshold: float,
) -> float:
    if module_idx.size == 0:
        return float("nan")
    specificity = _module_specificity_scores(residual[module_idx], profile)
    return float(np.mean(specificity >= similarity_threshold))


def _module_stability(
    residual: np.ndarray,
    module_idx: np.ndarray,
    candidate_profile: np.ndarray,
    remaining_before_mask: np.ndarray,
    top_n: int,
    similarity_threshold: float,
    n_iterations: int,
    sample_fraction: float,
    rng: np.random.Generator,
) -> ModuleStabilityResult:
    if n_iterations <= 0:
        return ModuleStabilityResult()
    if module_idx.size < 2:
        return ModuleStabilityResult()

    marker_mask = _top_specificity_mask(candidate_profile, top_n)
    if not np.any(marker_mask):
        return ModuleStabilityResult()

    sample_size = max(1, int(np.ceil(module_idx.size * sample_fraction)))
    sample_size = min(sample_size, module_idx.size)

    profile_similarities = []
    assignment_fractions = []
    for _ in range(n_iterations):
        sample_idx = rng.choice(module_idx, size=sample_size, replace=False)
        sampled_profile = _profile_from_indices(residual, sample_idx, marker_mask)
        profile_similarities.append(_profile_cosine_similarity(sampled_profile, candidate_profile))
        assignment_fractions.append(
            _module_assignment_fraction(
                residual, module_idx, sampled_profile, similarity_threshold
            )
        )

    observed_profile_similarity = float(np.mean(profile_similarities))
    observed_assignment_fraction = float(np.mean(assignment_fractions))

    remaining_idx = np.flatnonzero(remaining_before_mask)
    background_idx = np.setdiff1d(remaining_idx, module_idx, assume_unique=False)
    if background_idx.size < sample_size:
        null_pvalue = float("nan")
    else:
        n_extreme = 0
        for _ in range(n_iterations):
            null_idx = rng.choice(background_idx, size=sample_size, replace=False)
            null_profile = _profile_from_indices(residual, null_idx, marker_mask)
            null_similarity = _profile_cosine_similarity(null_profile, candidate_profile)
            if null_similarity >= observed_profile_similarity:
                n_extreme += 1
        null_pvalue = float((n_extreme + 1) / (n_iterations + 1))

    return ModuleStabilityResult(
        profile_similarity=observed_profile_similarity,
        assignment_fraction=observed_assignment_fraction,
        null_pvalue=null_pvalue,
    )


def _specificity_empirical_pvalue(
    observed_specificity: np.ndarray,
    background_specificity: np.ndarray,
    n_iterations: int,
    rng: np.random.Generator,
) -> float:
    if n_iterations <= 0:
        return float("nan")
    if observed_specificity.size == 0 or background_specificity.size == 0:
        return 1.0

    observed_mean = float(observed_specificity.mean())
    sample_size = observed_specificity.size
    replace = background_specificity.size < sample_size
    n_extreme = 0
    for _ in range(n_iterations):
        null_sample = rng.choice(background_specificity, size=sample_size, replace=replace)
        if float(null_sample.mean()) >= observed_mean:
            n_extreme += 1
    return float((n_extreme + 1) / (n_iterations + 1))


def _bounded_metric(value: float, low: float, high: float) -> float | None:
    if not np.isfinite(value):
        return None
    if high <= low:
        return float(value >= high)
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _inverse_bounded_metric(value: float, low: float, high: float) -> float | None:
    if not np.isfinite(value):
        return None
    if high <= low:
        return float(value <= low)
    return float(np.clip((high - value) / (high - low), 0.0, 1.0))


def _module_evidence(
    *,
    decision: str,
    specificity_enrichment: float,
    specificity_null_pvalue: float,
    max_profile_similarity: float,
    stability: ModuleStabilityResult,
    conditional_ldsc: ConditionalLDSCResult,
    config: TraitModuleConfig,
) -> tuple[float, str]:
    """
    Collapse heterogeneous diagnostics into one descriptive confidence score.

    The score is diagnostic and intentionally separate from acceptance gates:
    gates decide whether a module is kept, while this ranks kept or rejected
    candidates by trait specificity, uniqueness, reproducibility, and optional
    conditional genetic support.
    """
    metrics: list[float] = []

    enrichment_score = _bounded_metric(
        specificity_enrichment,
        config.min_specificity_enrichment,
        config.min_specificity_enrichment * 2,
    )
    if enrichment_score is not None:
        metrics.append(enrichment_score)

    if config.specificity_null_iterations > 0:
        null_score = _inverse_bounded_metric(
            specificity_null_pvalue,
            config.max_specificity_null_pvalue,
            1.0,
        )
        if null_score is not None:
            metrics.append(null_score)

    uniqueness_score = _inverse_bounded_metric(
        max_profile_similarity,
        config.max_module_profile_similarity,
        1.0,
    )
    if uniqueness_score is not None:
        metrics.append(uniqueness_score)

    stability_profile_score = _bounded_metric(
        stability.profile_similarity,
        config.min_module_stability_profile_similarity,
        1.0,
    )
    if stability_profile_score is not None:
        metrics.append(stability_profile_score)

    stability_assignment_score = _bounded_metric(
        stability.assignment_fraction,
        config.min_module_stability_assignment_fraction,
        1.0,
    )
    if stability_assignment_score is not None:
        metrics.append(stability_assignment_score)

    stability_null_score = _inverse_bounded_metric(
        stability.null_pvalue,
        config.max_module_stability_null_pvalue,
        1.0,
    )
    if stability_null_score is not None:
        metrics.append(stability_null_score)

    if config.run_conditional_ldsc:
        conditional_score = _inverse_bounded_metric(
            conditional_ldsc.p,
            config.conditional_ldsc_pvalue_threshold,
            1.0,
        )
        if conditional_score is not None:
            metrics.append(conditional_score)

    score = float(np.mean(metrics)) if metrics else float("nan")
    moderate_threshold = config.min_module_evidence_score
    strong_threshold = min(
        1.0,
        max(MODULE_EVIDENCE_STRONG_THRESHOLD, moderate_threshold + (1.0 - moderate_threshold) / 2),
    )
    if decision == "rejected":
        status = "failed_gate"
    elif not np.isfinite(score):
        status = "unknown"
    elif score >= strong_threshold:
        status = "strong"
    elif score >= moderate_threshold:
        status = "moderate"
    else:
        status = "weak"
    return score, status


def _trait_resolution_status(
    *,
    n_modules: int,
    top_module_signal_fraction: float,
    accepted_signal_fraction: float,
    residual_signal_fraction: float,
) -> str:
    if n_modules == 0:
        return "no_modules"
    if accepted_signal_fraction < 0.5:
        return "weakly_resolved"
    if residual_signal_fraction > 0.25:
        return "partially_resolved"
    if n_modules == 1 or top_module_signal_fraction >= 0.75:
        return "single_module_dominant"
    return "multi_module_resolved"


def _profile_similarity_to_accepted(
    profile: np.ndarray, module_results: list[ModuleResult]
) -> tuple[float, int]:
    profile = np.maximum(profile, 0.0)
    profile_norm = float(np.linalg.norm(profile))
    if profile_norm <= 1e-8 or not module_results:
        return 0.0, -1

    max_similarity = 0.0
    most_similar_module = -1
    for result in module_results:
        accepted_profile = np.maximum(result.profile, 0.0)
        accepted_norm = float(np.linalg.norm(accepted_profile))
        if accepted_norm <= 1e-8:
            continue
        similarity = float(np.dot(profile, accepted_profile) / (profile_norm * accepted_norm))
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_module = result.module_id
    return max_similarity, most_similar_module


def _sparse_profile_similarity(
    left_profile: dict[str, float], right_profile: dict[str, float]
) -> float:
    left_norm = float(np.sqrt(sum(value * value for value in left_profile.values())))
    right_norm = float(np.sqrt(sum(value * value for value in right_profile.values())))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return 0.0
    shared_genes = set(left_profile).intersection(right_profile)
    dot = sum(left_profile[gene] * right_profile[gene] for gene in shared_genes)
    return float(dot / (left_norm * right_norm))


def _format_gene_set(genes: list[str]) -> str:
    return ";".join(genes)


def _marker_gene_universe_size(
    profile_by_module: dict[int, np.ndarray | dict[str, float]],
) -> int | None:
    profile_sizes = [
        int(profile.size)
        for profile in profile_by_module.values()
        if isinstance(profile, np.ndarray)
    ]
    if profile_sizes:
        return max(profile_sizes)
    return None


def _marker_overlap_null_pvalue(
    *,
    shared_marker_count: int,
    module_a_marker_count: int,
    module_b_marker_count: int,
    marker_gene_universe_size: int | None,
) -> float:
    if marker_gene_universe_size is None:
        return float("nan")
    if (
        marker_gene_universe_size <= 0
        or module_a_marker_count <= 0
        or module_b_marker_count <= 0
        or module_a_marker_count > marker_gene_universe_size
        or module_b_marker_count > marker_gene_universe_size
    ):
        return float("nan")
    if shared_marker_count <= 0:
        return 1.0
    return float(
        hypergeom.sf(
            shared_marker_count - 1,
            marker_gene_universe_size,
            module_a_marker_count,
            module_b_marker_count,
        )
    )


def _trait_module_pair_rows(
    config: TraitModuleConfig,
    summary_by_module: dict[int, dict],
    gene_order_by_module: dict[int, list[str]],
    profile_by_module: dict[int, np.ndarray | dict[str, float]],
) -> pd.DataFrame:
    modules = sorted(summary_by_module)
    if len(modules) < 2:
        return pd.DataFrame(columns=MODULE_PAIR_COLUMNS)

    marker_gene_universe_size = _marker_gene_universe_size(profile_by_module)
    rows = []
    for module_a_pos, module_a in enumerate(modules[:-1]):
        for module_b in modules[module_a_pos + 1 :]:
            genes_a = gene_order_by_module.get(module_a, [])
            genes_b = gene_order_by_module.get(module_b, [])
            gene_set_a = set(genes_a)
            gene_set_b = set(genes_b)
            shared = gene_set_a.intersection(gene_set_b)
            union = gene_set_a.union(gene_set_b)
            marker_jaccard = len(shared) / len(union) if union else 0.0
            marker_overlap_null_pvalue = _marker_overlap_null_pvalue(
                shared_marker_count=len(shared),
                module_a_marker_count=len(gene_set_a),
                module_b_marker_count=len(gene_set_b),
                marker_gene_universe_size=marker_gene_universe_size,
            )
            profile_a = profile_by_module.get(module_a, {})
            profile_b = profile_by_module.get(module_b, {})
            if isinstance(profile_a, np.ndarray) and isinstance(profile_b, np.ndarray):
                profile_similarity = _profile_cosine_similarity(profile_a, profile_b)
            else:
                profile_similarity = _sparse_profile_similarity(profile_a, profile_b)
            profile_distinct = profile_similarity <= config.max_module_profile_similarity

            module_a_summary = summary_by_module[module_a]
            module_b_summary = summary_by_module[module_b]
            shared_ordered = [gene for gene in genes_a if gene in shared]
            rows.append(
                {
                    "trait": config.trait_name,
                    "module_a": module_a,
                    "module_b": module_b,
                    "profile_similarity": profile_similarity,
                    "max_allowed_profile_similarity": config.max_module_profile_similarity,
                    "profile_distinct": bool(profile_distinct),
                    "marker_jaccard": marker_jaccard,
                    "marker_gene_universe_size": marker_gene_universe_size,
                    "marker_overlap_null_pvalue": marker_overlap_null_pvalue,
                    "shared_marker_count": len(shared),
                    "module_a_marker_count": len(gene_set_a),
                    "module_b_marker_count": len(gene_set_b),
                    "shared_top_genes": _format_gene_set(shared_ordered),
                    "module_a_only_top_genes": _format_gene_set(
                        [gene for gene in genes_a if gene not in gene_set_b]
                    ),
                    "module_b_only_top_genes": _format_gene_set(
                        [gene for gene in genes_b if gene not in gene_set_a]
                    ),
                    "module_a_n_cells": int(module_a_summary.get("n_cells", 0)),
                    "module_b_n_cells": int(module_b_summary.get("n_cells", 0)),
                    "module_a_signal_fraction": float(
                        module_a_summary.get("module_signal_fraction", float("nan"))
                    ),
                    "module_b_signal_fraction": float(
                        module_b_summary.get("module_signal_fraction", float("nan"))
                    ),
                    "module_a_evidence_status": module_a_summary.get(
                        "module_evidence_status", "unknown"
                    ),
                    "module_b_evidence_status": module_b_summary.get(
                        "module_evidence_status", "unknown"
                    ),
                }
            )

    return pd.DataFrame(rows, columns=MODULE_PAIR_COLUMNS)


def _summarize_trait_module_pairs_from_results(
    config: TraitModuleConfig,
    module_results: list[ModuleResult],
    gene_order_by_module: dict[int, list[str]],
) -> pd.DataFrame:
    summary_by_module = {
        result.module_id: {
            "n_cells": len(result.cells),
            "module_signal_fraction": result.module_signal_fraction,
            "module_evidence_status": result.module_evidence_status,
        }
        for result in module_results
    }
    profile_by_module = {result.module_id: result.profile for result in module_results}
    return _trait_module_pair_rows(
        config,
        summary_by_module,
        gene_order_by_module,
        profile_by_module,
    )


def summarize_trait_module_pairs(
    config: TraitModuleConfig,
    summary_df: pd.DataFrame,
    genes_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize pairwise distinctness among accepted trait modules.

    This public fallback uses exported marker genes because full module profiles
    are not available after loading CSV outputs. The discovery writer uses the
    full in-memory profiles through _summarize_trait_module_pairs_from_results.
    """
    if (
        summary_df.empty
        or genes_df.empty
        or "module" not in summary_df
        or "module" not in genes_df
    ):
        return pd.DataFrame(columns=MODULE_PAIR_COLUMNS)

    summary = summary_df.copy()
    summary["module"] = pd.to_numeric(summary["module"], errors="coerce")
    summary = summary.dropna(subset=["module"]).copy()
    summary["module"] = summary["module"].astype(int)
    modules = sorted(summary["module"].unique())
    if len(modules) < 2:
        return pd.DataFrame(columns=MODULE_PAIR_COLUMNS)

    genes = genes_df.copy()
    genes["module"] = pd.to_numeric(genes["module"], errors="coerce")
    genes["rank"] = pd.to_numeric(genes.get("rank"), errors="coerce")
    genes["score"] = pd.to_numeric(genes.get("score"), errors="coerce").fillna(0.0)
    genes = genes.dropna(subset=["module", "rank"]).copy()
    genes["module"] = genes["module"].astype(int)

    summary_by_module = {int(row.module): row._asdict() for row in summary.itertuples(index=False)}
    gene_order_by_module: dict[int, list[str]] = {}
    sparse_profile_by_module: dict[int, dict[str, float]] = {}
    for module_id in modules:
        module_genes = genes.loc[genes["module"] == module_id].sort_values(
            ["rank", "gene"], kind="stable"
        )
        gene_order = [str(gene) for gene in module_genes["gene"].tolist()]
        gene_order_by_module[module_id] = gene_order
        sparse_profile_by_module[module_id] = {
            str(row.gene): max(float(row.score), 0.0)
            for row in module_genes.itertuples(index=False)
        }

    return _trait_module_pair_rows(
        config,
        summary_by_module,
        gene_order_by_module,
        sparse_profile_by_module,
    )


def _discover_trait_modules_with_trace_and_pairs(
    config: TraitModuleConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Discover fast independent trait modules.

    Returns seed-cell assignments, module summaries, module marker genes,
    candidate-level iteration trace diagnostics, pairwise module distinctness
    diagnostics, and per-marker-cell module specificity scores for report
    coloring.
    """
    ldsc = _read_ldsc_result(config.ldsc_result_path)
    seed_cells = _select_seed_cells(ldsc, config)
    if len(seed_cells) < config.min_module_cells:
        raise ValueError(
            f"Only {len(seed_cells)} seed cells passed the threshold; "
            f"min_module_cells={config.min_module_cells}."
        )

    marker_adata = _load_marker_scores(config)
    try:
        cells, gene_names, marker_matrix = _load_seed_marker_matrix(marker_adata, seed_cells)
        if config.background_scope == "all":
            background_matrix = _load_background_marker_matrix(marker_adata, config)
        else:
            background_matrix = marker_matrix
    finally:
        if getattr(marker_adata, "isbacked", False):
            marker_adata.file.close()

    background = np.quantile(background_matrix, config.background_quantile, axis=0).astype(
        np.float32
    )
    residual = np.maximum(marker_matrix - background, 0.0)
    claimed_module_indices: list[np.ndarray] = []
    claimed_profiles: list[np.ndarray] = []
    accepted_module_annotations: list[np.ndarray] = []
    search_residual = _compute_search_residual(
        marker_matrix,
        background,
        claimed_module_indices,
        claimed_profiles,
        recompute_background_after_modules=config.recompute_background_after_modules,
        iterative_background_quantile=config.iterative_background_quantile,
        condition_on_claimed_modules=config.condition_on_claimed_modules,
    )
    null_rng = np.random.default_rng(config.random_seed + 1729)
    n_top = min(
        config.max_marker_genes,
        max(1, int(np.ceil(residual.shape[1] * config.top_gene_fraction))),
    )
    weights = ldsc.loc[cells, "neg_log10_p"].to_numpy(dtype=np.float32)
    conditional_weights = _conditional_neg_log10_p_proxy(weights, residual, search_residual, n_top)
    conditional_ldsc_context = _prepare_conditional_ldsc_context(config, gene_names)
    stability_rng = np.random.default_rng(config.random_seed + 7919)

    remaining_mask = np.ones(len(cells), dtype=bool)
    assigned_module = np.full(len(cells), -1, dtype=int)
    module_specificity = np.zeros(len(cells), dtype=np.float32)
    cell_module_status = np.full(len(cells), CELL_MODULE_STATUS_UNRESOLVED, dtype=object)
    cell_module_reason = np.full(len(cells), "not_selected", dtype=object)
    total_trait_signal = max(float(weights.sum()), 1e-8)
    accepted_signal_fraction = 0.0
    residual_signal_fraction = (
        float(conditional_weights[remaining_mask].sum()) / total_trait_signal
    )
    module_results: list[ModuleResult] = []
    trace_rows = []
    iteration = 0

    while len(module_results) < config.max_modules:
        iteration += 1
        module_id = len(module_results) + 1
        remaining_before = int(remaining_mask.sum())
        residual_signal_fraction_before = residual_signal_fraction
        module_idx, profile, specificity = _module_from_best_seed(
            residual=search_residual,
            remaining_mask=remaining_mask,
            weights=conditional_weights,
            top_n=n_top,
            similarity_threshold=config.module_similarity_threshold,
        )
        if module_idx.size < config.min_module_cells:
            unresolved_idx = np.flatnonzero(remaining_mask)
            cell_module_reason[unresolved_idx] = "min_module_cells"
            mean_neg_log10_p = (
                float(ldsc.loc[cells[module_idx], "neg_log10_p"].mean())
                if module_idx.size
                else float("nan")
            )
            mean_conditional_neg_log10_p = (
                float(conditional_weights[module_idx].mean()) if module_idx.size else float("nan")
            )
            trace_rows.append(
                {
                    "trait": config.trait_name,
                    "iteration": iteration,
                    "candidate_module": module_id,
                    "decision": "stopped",
                    "reason": "min_module_cells",
                    "candidate_signal_fraction": float("nan"),
                    "accepted_signal_fraction": accepted_signal_fraction,
                    "residual_signal_fraction_before": residual_signal_fraction_before,
                    "residual_signal_fraction_after": residual_signal_fraction_before,
                    "candidate_n_cells": int(module_idx.size),
                    "remaining_cells_before": remaining_before,
                    "remaining_cells_after": remaining_before,
                    "accepted_module": -1,
                    **_trace_module_evidence_empty(),
                    "mean_neg_log10_p": mean_neg_log10_p,
                    "mean_conditional_neg_log10_p": mean_conditional_neg_log10_p,
                    "mean_specificity": float("nan"),
                    "background_mean_specificity": float("nan"),
                    "specificity_enrichment": float("nan"),
                    "specificity_null_pvalue": float("nan"),
                    **_trace_conditional_ldsc_empty(),
                    **_trace_module_stability_empty(),
                }
            )
            logger.info(
                f"Stopping at module {module_id}: candidate has {module_idx.size} cells, "
                f"below min_module_cells={config.min_module_cells}."
            )
            break

        module_cells = cells[module_idx]
        search_background = _compute_iterative_background(
            marker_matrix,
            background,
            claimed_module_indices,
            recompute_background_after_modules=config.recompute_background_after_modules,
            iterative_background_quantile=config.iterative_background_quantile,
        )
        search_background_residual = np.maximum(background_matrix - search_background, 0.0).astype(
            np.float32, copy=False
        )
        search_background_residual = _condition_residual_on_profiles(
            search_background_residual,
            claimed_profiles,
            condition_on_claimed_modules=config.condition_on_claimed_modules,
        )
        cell_specificity = _module_specificity_scores(search_residual[module_idx], profile)
        background_specificity = _module_specificity_scores(search_background_residual, profile)
        background_specificity_mean = (
            float(background_specificity.mean()) if background_specificity.size else 0.0
        )
        specificity_enrichment = float(cell_specificity.mean()) / max(
            background_specificity_mean, 1e-8
        )
        specificity_null_pvalue = _specificity_empirical_pvalue(
            cell_specificity,
            background_specificity,
            config.specificity_null_iterations,
            null_rng,
        )
        stability = _module_stability(
            residual=residual,
            module_idx=module_idx,
            candidate_profile=profile,
            remaining_before_mask=remaining_mask.copy(),
            top_n=n_top,
            similarity_threshold=config.module_similarity_threshold,
            n_iterations=config.module_stability_iterations,
            sample_fraction=config.module_stability_sample_fraction,
            rng=stability_rng,
        )
        max_profile_similarity, most_similar_module = _profile_similarity_to_accepted(
            profile, module_results
        )
        conditional_ldsc = ConditionalLDSCResult()
        candidate_annotation = None
        if conditional_ldsc_context is not None:
            conditional_ldsc, candidate_annotation = conditional_ldsc_context.fit_profile(
                profile, accepted_module_annotations
            )
        candidate_signal_fraction = (
            float(conditional_weights[module_idx].sum()) / total_trait_signal
        )
        fails_enrichment = specificity_enrichment < config.min_specificity_enrichment
        fails_null = (
            config.specificity_null_iterations > 0
            and specificity_null_pvalue > config.max_specificity_null_pvalue
        )
        fails_profile_similarity = (
            most_similar_module > 0
            and max_profile_similarity > config.max_module_profile_similarity
        )
        fails_stability = config.require_module_stability and (
            not np.isfinite(stability.profile_similarity)
            or not np.isfinite(stability.assignment_fraction)
            or stability.profile_similarity < config.min_module_stability_profile_similarity
            or stability.assignment_fraction < config.min_module_stability_assignment_fraction
            or (
                np.isfinite(stability.null_pvalue)
                and stability.null_pvalue > config.max_module_stability_null_pvalue
            )
        )
        fails_conditional_ldsc = (
            config.require_conditional_ldsc
            and conditional_ldsc_context is not None
            and (
                not np.isfinite(conditional_ldsc.p)
                or conditional_ldsc.p > config.conditional_ldsc_pvalue_threshold
            )
        )
        candidate_decision = (
            "rejected"
            if (
                fails_enrichment
                or fails_null
                or fails_profile_similarity
                or fails_stability
                or fails_conditional_ldsc
            )
            else "accepted"
        )
        module_evidence_score, module_evidence_status = _module_evidence(
            decision=candidate_decision,
            specificity_enrichment=specificity_enrichment,
            specificity_null_pvalue=specificity_null_pvalue,
            max_profile_similarity=max_profile_similarity,
            stability=stability,
            conditional_ldsc=conditional_ldsc,
            config=config,
        )
        if (
            fails_enrichment
            or fails_null
            or fails_profile_similarity
            or fails_stability
            or fails_conditional_ldsc
        ):
            remaining_mask[module_idx] = False
            residual_signal_fraction_after = (
                float(conditional_weights[remaining_mask].sum()) / total_trait_signal
            )
            rejection_reasons = []
            rejection_messages = []
            if fails_enrichment:
                rejection_reasons.append("min_specificity_enrichment")
                rejection_messages.append(
                    f"specificity_enrichment={specificity_enrichment:.3f} "
                    f"below min_specificity_enrichment={config.min_specificity_enrichment:.3f}"
                )
            if fails_null:
                rejection_reasons.append("specificity_null_pvalue")
                rejection_messages.append(
                    f"specificity_null_pvalue={specificity_null_pvalue:.3f} "
                    f"above max_specificity_null_pvalue={config.max_specificity_null_pvalue:.3f}"
                )
            if fails_profile_similarity:
                rejection_reasons.append("module_profile_similarity")
                rejection_messages.append(
                    f"max_profile_similarity={max_profile_similarity:.3f} to module "
                    f"{most_similar_module} above "
                    f"max_module_profile_similarity={config.max_module_profile_similarity:.3f}"
                )
            if fails_stability:
                rejection_reasons.append("module_stability")
                rejection_messages.append(
                    "module stability below thresholds "
                    f"(profile_similarity={stability.profile_similarity:.3f}, "
                    f"assignment_fraction={stability.assignment_fraction:.3f}, "
                    f"null_pvalue={stability.null_pvalue:.3f})"
                )
            if fails_conditional_ldsc:
                rejection_reasons.append("conditional_ldsc_pvalue")
                if np.isfinite(conditional_ldsc.p):
                    rejection_messages.append(
                        f"conditional_ldsc_p={conditional_ldsc.p:.3g} above "
                        f"conditional_ldsc_pvalue_threshold="
                        f"{config.conditional_ldsc_pvalue_threshold:.3g}"
                    )
                else:
                    rejection_messages.append("conditional_ldsc_p is unavailable")
            rejection_reason = ",".join(rejection_reasons)
            cell_module_status[module_idx] = CELL_MODULE_STATUS_REJECTED
            cell_module_reason[module_idx] = rejection_reason
            trace_rows.append(
                {
                    "trait": config.trait_name,
                    "iteration": iteration,
                    "candidate_module": module_id,
                    "decision": "rejected",
                    "reason": rejection_reason,
                    "candidate_signal_fraction": candidate_signal_fraction,
                    "accepted_signal_fraction": accepted_signal_fraction,
                    "residual_signal_fraction_before": residual_signal_fraction_before,
                    "residual_signal_fraction_after": residual_signal_fraction_after,
                    "candidate_n_cells": int(module_idx.size),
                    "remaining_cells_before": remaining_before,
                    "remaining_cells_after": int(remaining_mask.sum()),
                    "accepted_module": -1,
                    "module_evidence_score": module_evidence_score,
                    "module_evidence_status": module_evidence_status,
                    "mean_neg_log10_p": float(ldsc.loc[module_cells, "neg_log10_p"].mean()),
                    "mean_conditional_neg_log10_p": float(conditional_weights[module_idx].mean()),
                    "mean_specificity": float(cell_specificity.mean()),
                    "background_mean_specificity": background_specificity_mean,
                    "specificity_enrichment": specificity_enrichment,
                    "specificity_null_pvalue": specificity_null_pvalue,
                    **conditional_ldsc.as_columns(),
                    **stability.as_columns(),
                    "max_profile_similarity": max_profile_similarity,
                    "most_similar_module": most_similar_module,
                }
            )
            logger.info(
                f"Rejected module candidate before module {module_id}: "
                f"{'; '.join(rejection_messages)}; "
                f"removed {len(module_cells)} background-like cells from later searches."
            )
            residual_signal_fraction = residual_signal_fraction_after
            if remaining_mask.sum() < config.min_module_cells:
                break
            continue

        assigned_module[module_idx] = module_id
        module_specificity[module_idx] = cell_specificity
        cell_module_status[module_idx] = CELL_MODULE_STATUS_ASSIGNED
        cell_module_reason[module_idx] = "accepted"
        remaining_mask[module_idx] = False
        accepted_signal_fraction += candidate_signal_fraction
        residual_signal_fraction_after = (
            float(conditional_weights[remaining_mask].sum()) / total_trait_signal
        )
        trace_rows.append(
            {
                "trait": config.trait_name,
                "iteration": iteration,
                "candidate_module": module_id,
                "decision": "accepted",
                "reason": "accepted",
                "candidate_signal_fraction": candidate_signal_fraction,
                "accepted_signal_fraction": accepted_signal_fraction,
                "residual_signal_fraction_before": residual_signal_fraction_before,
                "residual_signal_fraction_after": residual_signal_fraction_after,
                "candidate_n_cells": int(module_idx.size),
                "remaining_cells_before": remaining_before,
                "remaining_cells_after": int(remaining_mask.sum()),
                "accepted_module": module_id,
                "module_evidence_score": module_evidence_score,
                "module_evidence_status": module_evidence_status,
                "mean_neg_log10_p": float(ldsc.loc[module_cells, "neg_log10_p"].mean()),
                "mean_conditional_neg_log10_p": float(conditional_weights[module_idx].mean()),
                "mean_specificity": float(cell_specificity.mean()),
                "background_mean_specificity": background_specificity_mean,
                "specificity_enrichment": specificity_enrichment,
                "specificity_null_pvalue": specificity_null_pvalue,
                **conditional_ldsc.as_columns(),
                **stability.as_columns(),
                "max_profile_similarity": max_profile_similarity,
                "most_similar_module": most_similar_module,
            }
        )

        module_results.append(
            ModuleResult(
                module_id=module_id,
                cells=module_cells,
                profile=profile,
                specificity=specificity,
                mean_neg_log10_p=float(ldsc.loc[module_cells, "neg_log10_p"].mean()),
                background_specificity_mean=background_specificity_mean,
                specificity_enrichment=specificity_enrichment,
                specificity_null_pvalue=specificity_null_pvalue,
                max_profile_similarity=max_profile_similarity,
                most_similar_module=most_similar_module,
                module_signal_fraction=candidate_signal_fraction,
                cumulative_signal_fraction=accepted_signal_fraction,
                residual_signal_fraction_after_module=residual_signal_fraction_after,
                module_evidence_score=module_evidence_score,
                module_evidence_status=module_evidence_status,
                conditional_ldsc_beta=conditional_ldsc.beta,
                conditional_ldsc_se=conditional_ldsc.se,
                conditional_ldsc_z=conditional_ldsc.z,
                conditional_ldsc_p=conditional_ldsc.p,
                conditional_ldsc_neg_log10_p=conditional_ldsc.neg_log10_p,
                stability_profile_similarity=stability.profile_similarity,
                stability_assignment_fraction=stability.assignment_fraction,
                stability_null_pvalue=stability.null_pvalue,
            )
        )
        logger.info(
            f"Module {module_id}: {len(module_cells)} cells, "
            f"mean -log10(p)={module_results[-1].mean_neg_log10_p:.3f}, "
            f"specificity_enrichment={specificity_enrichment:.3f}, "
            f"specificity_null_pvalue={specificity_null_pvalue:.3f}"
        )

        claimed_module_indices.append(module_idx.copy())
        claimed_profiles.append(profile.copy())
        if candidate_annotation is not None:
            accepted_module_annotations.append(candidate_annotation)
        if config.recompute_background_after_modules or config.condition_on_claimed_modules:
            search_residual = _compute_search_residual(
                marker_matrix,
                background,
                claimed_module_indices,
                claimed_profiles,
                recompute_background_after_modules=config.recompute_background_after_modules,
                iterative_background_quantile=config.iterative_background_quantile,
                condition_on_claimed_modules=config.condition_on_claimed_modules,
            )
            conditional_weights = _conditional_neg_log10_p_proxy(
                weights, residual, search_residual, n_top
            )
            residual_signal_fraction = (
                float(conditional_weights[remaining_mask].sum()) / total_trait_signal
            )
            if trace_rows:
                trace_rows[-1]["residual_signal_fraction_after"] = residual_signal_fraction
                if module_results:
                    module_results[
                        -1
                    ].residual_signal_fraction_after_module = residual_signal_fraction
            logger.info(f"Updated residual search space after accepting module {module_id}")
        else:
            residual_signal_fraction = residual_signal_fraction_after

        if remaining_mask.sum() < config.min_module_cells:
            break

    cell_df = ldsc.loc[cells].copy()
    cell_df["trait_module"] = assigned_module
    cell_df["module_specificity"] = module_specificity
    cell_df["is_module_cell"] = cell_df["trait_module"] > 0
    cell_df["module_status"] = cell_module_status
    cell_df["module_reason"] = cell_module_reason

    summaries = []
    gene_rows = []
    gene_order_by_module = {}
    for result in module_results:
        marker_mask = _top_specificity_mask(result.profile, n_top)
        marker_idx = np.flatnonzero(marker_mask)
        marker_idx = marker_idx[np.argsort(result.profile[marker_idx])[::-1]]
        gene_order_by_module[result.module_id] = [
            gene_names[gene_idx] for gene_idx in marker_idx[: config.max_marker_genes]
        ]

        summaries.append(
            {
                "trait": config.trait_name,
                "module": result.module_id,
                "n_cells": len(result.cells),
                "module_signal_fraction": result.module_signal_fraction,
                "cumulative_signal_fraction": result.cumulative_signal_fraction,
                "residual_signal_fraction_after_module": (
                    result.residual_signal_fraction_after_module
                ),
                "module_evidence_score": result.module_evidence_score,
                "module_evidence_status": result.module_evidence_status,
                "mean_neg_log10_p": result.mean_neg_log10_p,
                "max_neg_log10_p": float(ldsc.loc[result.cells, "neg_log10_p"].max()),
                "conditional_ldsc_neg_log10_p": result.conditional_ldsc_neg_log10_p,
                "conditional_ldsc_p": result.conditional_ldsc_p,
                "conditional_ldsc_beta": result.conditional_ldsc_beta,
                "conditional_ldsc_se": result.conditional_ldsc_se,
                "conditional_ldsc_z": result.conditional_ldsc_z,
                "stability_profile_similarity": result.stability_profile_similarity,
                "stability_assignment_fraction": result.stability_assignment_fraction,
                "stability_null_pvalue": result.stability_null_pvalue,
                "mean_specificity": float(cell_df.loc[result.cells, "module_specificity"].mean()),
                "background_mean_specificity": result.background_specificity_mean,
                "specificity_enrichment": result.specificity_enrichment,
                "specificity_null_pvalue": result.specificity_null_pvalue,
                "max_profile_similarity": result.max_profile_similarity,
                "most_similar_module": result.most_similar_module,
            }
        )

        for rank, gene_idx in enumerate(marker_idx[: config.max_marker_genes], start=1):
            gene_rows.append(
                {
                    "trait": config.trait_name,
                    "module": result.module_id,
                    "rank": rank,
                    "gene": gene_names[gene_idx],
                    "score": float(result.profile[gene_idx]),
                }
            )

    summary_df = pd.DataFrame(summaries, columns=MODULE_SUMMARY_COLUMNS)
    genes_df = pd.DataFrame(gene_rows, columns=MODULE_GENE_COLUMNS)
    pair_df = _summarize_trait_module_pairs_from_results(
        config,
        module_results,
        gene_order_by_module,
    )
    trace_df = pd.DataFrame(trace_rows, columns=MODULE_TRACE_COLUMNS)

    if module_results:
        marker_adata_for_specificity = _load_marker_scores(config)
        try:
            specificity_df = _module_specificity_table_from_marker_adata(
                marker_adata_for_specificity,
                background,
                module_results,
            )
        finally:
            if getattr(marker_adata_for_specificity, "isbacked", False):
                marker_adata_for_specificity.file.close()
    else:
        specificity_df = pd.DataFrame(columns=MODULE_SPECIFICITY_BASE_COLUMNS)

    return cell_df.reset_index(drop=True), summary_df, genes_df, trace_df, pair_df, specificity_df


def discover_trait_modules_with_trace(
    config: TraitModuleConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Discover fast independent trait modules.

    Returns cell assignments, module summaries, module marker genes, and
    candidate-level iteration trace diagnostics.
    """
    cell_df, summary_df, genes_df, trace_df, _, _ = _discover_trait_modules_with_trace_and_pairs(
        config
    )
    return cell_df, summary_df, genes_df, trace_df


def discover_trait_modules(
    config: TraitModuleConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Discover fast independent trait modules.

    Returns cell assignments, module summaries, and module marker genes.
    """
    cell_df, summary_df, genes_df, _ = discover_trait_modules_with_trace(config)
    return cell_df, summary_df, genes_df


def summarize_trait_module_run(
    config: TraitModuleConfig,
    cell_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    trace_df: pd.DataFrame,
) -> pd.DataFrame:
    n_seed_cells = int(len(cell_df))
    n_modules = int(len(summary_df))
    n_module_cells = int(cell_df["is_module_cell"].sum()) if "is_module_cell" in cell_df else 0
    module_cell_fraction = n_module_cells / max(n_seed_cells, 1)
    if "module_status" in cell_df:
        status_counts = cell_df["module_status"].value_counts()
        n_assigned_seed_cells = int(status_counts.get(CELL_MODULE_STATUS_ASSIGNED, 0))
        n_rejected_seed_cells = int(status_counts.get(CELL_MODULE_STATUS_REJECTED, 0))
        n_unresolved_seed_cells = int(status_counts.get(CELL_MODULE_STATUS_UNRESOLVED, 0))
    else:
        n_assigned_seed_cells = n_module_cells
        n_rejected_seed_cells = 0
        n_unresolved_seed_cells = max(n_seed_cells - n_module_cells, 0)
    seed_denominator = max(n_seed_cells, 1)
    assigned_seed_cell_fraction = n_assigned_seed_cells / seed_denominator
    rejected_seed_cell_fraction = n_rejected_seed_cells / seed_denominator
    unresolved_seed_cell_fraction = n_unresolved_seed_cells / seed_denominator
    top_module_signal_fraction = (
        float(summary_df["module_signal_fraction"].max()) if n_modules else 0.0
    )
    accepted_signal_fraction = (
        float(summary_df["cumulative_signal_fraction"].iloc[-1]) if n_modules else 0.0
    )
    residual_trace = trace_df["residual_signal_fraction_after"].dropna()
    residual_signal_fraction = float(residual_trace.iloc[-1]) if not residual_trace.empty else 1.0
    resolution_status = _trait_resolution_status(
        n_modules=n_modules,
        top_module_signal_fraction=top_module_signal_fraction,
        accepted_signal_fraction=accepted_signal_fraction,
        residual_signal_fraction=residual_signal_fraction,
    )

    return pd.DataFrame(
        [
            {
                "trait": config.trait_name,
                "n_seed_cells": n_seed_cells,
                "n_modules": n_modules,
                "n_module_cells": n_module_cells,
                "module_cell_fraction": module_cell_fraction,
                "n_assigned_seed_cells": n_assigned_seed_cells,
                "assigned_seed_cell_fraction": assigned_seed_cell_fraction,
                "n_rejected_seed_cells": n_rejected_seed_cells,
                "rejected_seed_cell_fraction": rejected_seed_cell_fraction,
                "n_unresolved_seed_cells": n_unresolved_seed_cells,
                "unresolved_seed_cell_fraction": unresolved_seed_cell_fraction,
                "top_module_signal_fraction": top_module_signal_fraction,
                "accepted_signal_fraction": accepted_signal_fraction,
                "residual_signal_fraction": residual_signal_fraction,
                "resolution_status": resolution_status,
            }
        ],
        columns=MODULE_RUN_COLUMNS,
    )


def run_trait_module_discovery(config: TraitModuleConfig):
    cell_df, summary_df, genes_df, trace_df, pair_df, specificity_df = (
        _discover_trait_modules_with_trace_and_pairs(config)
    )
    run_summary_df = summarize_trait_module_run(config, cell_df, summary_df, trace_df)
    config.trait_module_dir.mkdir(parents=True, exist_ok=True)

    cell_df.to_csv(config.cell_modules_path, index=False)
    summary_df.to_csv(config.module_summary_path, index=False)
    genes_df.to_csv(config.module_genes_path, index=False)
    specificity_df.to_csv(config.module_specificity_path, index=False)
    pair_df.to_csv(config.module_pairs_path, index=False)
    trace_df.to_csv(config.module_trace_path, index=False)
    run_summary_df.to_csv(config.run_summary_path, index=False)

    logger.info(f"Saved cell module assignments to {config.cell_modules_path}")
    logger.info(f"Saved module summary to {config.module_summary_path}")
    logger.info(f"Saved module marker genes to {config.module_genes_path}")
    logger.info(f"Saved module specificity scores to {config.module_specificity_path}")
    logger.info(f"Saved module pair diagnostics to {config.module_pairs_path}")
    logger.info(f"Saved module iteration trace to {config.module_trace_path}")
    logger.info(f"Saved trait module run summary to {config.run_summary_path}")
