"""
Fast trait-module discovery from per-cell LDSC and marker scores.

The prototype implemented here is an approximation layer for independent cell
states. It does not claim conditional genetic independence. Instead, it asks:
among cells already significant for a trait, which groups retain distinct marker
specificity profiles after removing the cells explained by stronger modules?
"""

import logging
from dataclasses import dataclass

import anndata as ad
import numpy as np
import pandas as pd

from gsMap.config import TraitModuleConfig
from gsMap.spatial_ldsc.io import load_marker_scores_memmap_format

logger = logging.getLogger("gsMap.trait_modules")


@dataclass
class ModuleResult:
    module_id: int
    cells: pd.Index
    profile: np.ndarray
    specificity: np.ndarray
    mean_neg_log10_p: float


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

    weighted_signal = residual[remaining_idx].sum(axis=1) * weights[remaining_idx]
    seed_local = int(np.argmax(weighted_signal))
    seed_idx = remaining_idx[seed_local]

    specificity = residual[seed_idx].copy()
    marker_mask = _top_specificity_mask(specificity, top_n)
    if not np.any(marker_mask):
        return np.array([], dtype=int), specificity, specificity

    profile = np.zeros_like(specificity, dtype=np.float32)
    profile[marker_mask] = specificity[marker_mask]
    profile_norm = profile / max(float(np.linalg.norm(profile)), 1e-8)

    normalized = _row_normalize(np.maximum(residual[remaining_idx], 0.0))
    similarities = normalized @ profile_norm
    module_idx = remaining_idx[similarities >= similarity_threshold]
    if module_idx.size == 0:
        module_idx = np.array([seed_idx], dtype=int)

    module_profile = np.maximum(residual[module_idx], 0.0).mean(axis=0)
    return module_idx, module_profile.astype(np.float32), specificity.astype(np.float32)


def _condition_residual_on_profile(residual: np.ndarray, profile: np.ndarray) -> np.ndarray:
    profile = np.maximum(profile, 0.0).astype(np.float32, copy=False)
    profile_norm_sq = float(np.dot(profile, profile))
    if profile_norm_sq <= 1e-8:
        return residual

    projection_scale = (residual @ profile) / profile_norm_sq
    conditioned = residual - np.outer(np.maximum(projection_scale, 0.0), profile)
    return np.maximum(conditioned, 0.0).astype(np.float32, copy=False)


def discover_trait_modules(
    config: TraitModuleConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Discover fast independent trait modules.

    Returns cell assignments, module summaries, and module marker genes.
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
    search_residual = residual.copy()
    weights = ldsc.loc[cells, "neg_log10_p"].to_numpy(dtype=np.float32)
    n_top = min(
        config.max_marker_genes,
        max(1, int(np.ceil(residual.shape[1] * config.top_gene_fraction))),
    )

    remaining_mask = np.ones(len(cells), dtype=bool)
    assigned_module = np.full(len(cells), -1, dtype=int)
    module_specificity = np.zeros(len(cells), dtype=np.float32)
    module_results: list[ModuleResult] = []

    for module_id in range(1, config.max_modules + 1):
        module_idx, profile, specificity = _module_from_best_seed(
            residual=search_residual,
            remaining_mask=remaining_mask,
            weights=weights,
            top_n=n_top,
            similarity_threshold=config.module_similarity_threshold,
        )
        if module_idx.size < config.min_module_cells:
            logger.info(
                f"Stopping at module {module_id}: candidate has {module_idx.size} cells, "
                f"below min_module_cells={config.min_module_cells}."
            )
            break

        module_cells = cells[module_idx]
        normalized = _row_normalize(np.maximum(residual[module_idx], 0.0))
        profile_norm = profile / max(float(np.linalg.norm(profile)), 1e-8)
        cell_specificity = normalized @ profile_norm

        assigned_module[module_idx] = module_id
        module_specificity[module_idx] = cell_specificity
        remaining_mask[module_idx] = False

        module_results.append(
            ModuleResult(
                module_id=module_id,
                cells=module_cells,
                profile=profile,
                specificity=specificity,
                mean_neg_log10_p=float(ldsc.loc[module_cells, "neg_log10_p"].mean()),
            )
        )
        logger.info(
            f"Module {module_id}: {len(module_cells)} cells, "
            f"mean -log10(p)={module_results[-1].mean_neg_log10_p:.3f}"
        )

        if config.condition_on_claimed_modules:
            search_residual = _condition_residual_on_profile(search_residual, profile)
            logger.info(f"Conditioned later module search on module {module_id} profile")

        if remaining_mask.sum() < config.min_module_cells:
            break

    cell_df = ldsc.loc[cells].copy()
    cell_df["trait_module"] = assigned_module
    cell_df["module_specificity"] = module_specificity
    cell_df["is_module_cell"] = cell_df["trait_module"] > 0

    summaries = []
    gene_rows = []
    for result in module_results:
        marker_mask = _top_specificity_mask(result.profile, n_top)
        marker_idx = np.flatnonzero(marker_mask)
        marker_idx = marker_idx[np.argsort(result.profile[marker_idx])[::-1]]

        summaries.append(
            {
                "trait": config.trait_name,
                "module": result.module_id,
                "n_cells": len(result.cells),
                "mean_neg_log10_p": result.mean_neg_log10_p,
                "max_neg_log10_p": float(ldsc.loc[result.cells, "neg_log10_p"].max()),
                "mean_specificity": float(cell_df.loc[result.cells, "module_specificity"].mean()),
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

    summary_df = pd.DataFrame(summaries)
    genes_df = pd.DataFrame(gene_rows)
    return cell_df.reset_index(drop=True), summary_df, genes_df


def run_trait_module_discovery(config: TraitModuleConfig):
    cell_df, summary_df, genes_df = discover_trait_modules(config)
    config.trait_module_dir.mkdir(parents=True, exist_ok=True)

    cell_df.to_csv(config.cell_modules_path, index=False)
    summary_df.to_csv(config.module_summary_path, index=False)
    genes_df.to_csv(config.module_genes_path, index=False)

    logger.info(f"Saved cell module assignments to {config.cell_modules_path}")
    logger.info(f"Saved module summary to {config.module_summary_path}")
    logger.info(f"Saved module marker genes to {config.module_genes_path}")
