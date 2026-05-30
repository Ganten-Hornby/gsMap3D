import anndata as ad
import numpy as np
import pandas as pd

from gsMap.config import TraitModuleConfig
from gsMap.trait_modules import (
    ConditionalLDSCResult,
    ModuleResult,
    ModuleStabilityResult,
    _condition_residual_on_profile,
    _conditional_neg_log10_p_proxy,
    _fit_conditional_ldsc,
    _module_evidence,
    _module_from_best_seed,
    _module_stability,
    _summarize_trait_module_pairs_from_results,
    _top_positive_sum,
    _trait_resolution_status,
    discover_trait_modules,
    discover_trait_modules_with_trace,
    run_trait_module_discovery,
    summarize_trait_module_pairs,
    summarize_trait_module_run,
)


def _write_trait_module_inputs(
    tmp_path,
    project_name: str,
    trait_name: str,
    marker_scores: np.ndarray,
    p_values: np.ndarray,
):
    project_dir = tmp_path / project_name
    ldsc_dir = project_dir / "spatial_ldsc"
    ldsc_dir.mkdir(parents=True)

    cells = [f"cell_{i}" for i in range(marker_scores.shape[0])]
    genes = [f"gene_{i}" for i in range(marker_scores.shape[1])]

    marker_adata = ad.AnnData(X=marker_scores.astype(np.float32))
    marker_adata.obs_names = cells
    marker_adata.var_names = genes
    marker_path = tmp_path / f"{project_name}_marker_scores.h5ad"
    marker_adata.write_h5ad(marker_path)

    ldsc = pd.DataFrame(
        {
            "spot": cells,
            "beta": np.ones(marker_scores.shape[0]),
            "se": np.full(marker_scores.shape[0], 0.1),
            "z": np.full(marker_scores.shape[0], 10.0),
            "p": p_values,
        }
    )
    ldsc_path = ldsc_dir / f"{project_name}_{trait_name}.csv.gz"
    ldsc.to_csv(ldsc_path, index=False, compression="gzip")
    return cells, genes, marker_path


def test_discover_trait_modules_recovers_two_marker_programs(tmp_path):
    workdir = tmp_path
    project_name = "module_test"

    n_cells = 60
    n_genes = 40

    rng = np.random.default_rng(42)
    marker_scores = rng.normal(0.05, 0.02, size=(n_cells, n_genes)).astype(np.float32)
    marker_scores[:20, :6] += 2.0
    marker_scores[20:40, 10:16] += 1.8
    marker_scores = np.maximum(marker_scores, 0.0)

    p_values = np.full(n_cells, 0.9)
    p_values[:20] = 1e-8
    p_values[20:40] = 1e-6
    cells, genes, marker_path = _write_trait_module_inputs(
        tmp_path, project_name, "IQ", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=workdir,
        project_name=project_name,
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.15,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=4,
        max_marker_genes=6,
    )

    cell_df, summary_df, genes_df = discover_trait_modules(config)

    assert summary_df["n_cells"].tolist() == [20, 20]
    assert set(cell_df.loc[cell_df["trait_module"] > 0, "spot"]) == set(cells[:40])

    module_1_genes = set(genes_df.loc[genes_df["module"] == 1, "gene"])
    module_2_genes = set(genes_df.loc[genes_df["module"] == 2, "gene"])

    assert module_1_genes <= set(genes[:6])
    assert module_2_genes <= set(genes[10:16])


def test_discover_trait_modules_uses_global_background_for_pure_signal(tmp_path):
    n_cells = 100
    n_genes = 30
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:20, :5] = 2.1

    p_values = np.full(n_cells, 0.9)
    p_values[:20] = 1e-8
    cells, genes, marker_path = _write_trait_module_inputs(
        tmp_path, "pure_module_test", "BMI", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="pure_module_test",
        trait_name="BMI",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.2,
        min_module_cells=5,
        module_similarity_threshold=0.8,
        max_modules=3,
        max_marker_genes=5,
    )

    cell_df, summary_df, genes_df = discover_trait_modules(config)

    assert summary_df["n_cells"].tolist() == [20]
    assert set(cell_df.loc[cell_df["trait_module"] == 1, "spot"]) == set(cells[:20])
    assert set(genes_df["gene"]) == set(genes[:5])


def test_condition_residual_on_profile_removes_claimed_program_only():
    residual = np.array(
        [
            [2.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 3.0],
        ],
        dtype=np.float32,
    )
    profile = np.array([2.0, 2.0, 0.0, 0.0], dtype=np.float32)

    conditioned = _condition_residual_on_profile(residual, profile)

    np.testing.assert_allclose(conditioned[:2], 0.0, atol=1e-6)
    np.testing.assert_allclose(conditioned[2], residual[2], atol=1e-6)


def test_module_similarity_uses_marker_gene_subspace():
    residual = np.zeros((2, 100), dtype=np.float32)
    residual[0, :5] = 2.0
    residual[1, :5] = 2.0
    residual[1, 5:] = 5.0

    module_idx, _, _ = _module_from_best_seed(
        residual=residual,
        remaining_mask=np.array([True, True]),
        weights=np.array([100.0, 1.0], dtype=np.float32),
        top_n=5,
        similarity_threshold=0.9,
    )

    assert module_idx.tolist() == [0, 1]


def test_module_seed_score_uses_concentrated_marker_signal():
    residual = np.zeros((2, 100), dtype=np.float32)
    residual[0, :] = 0.2
    residual[1, :5] = 2.0

    scores = _top_positive_sum(residual, top_n=5)

    assert scores[1] > scores[0]

    module_idx, profile, _ = _module_from_best_seed(
        residual=residual,
        remaining_mask=np.array([True, True]),
        weights=np.ones(2, dtype=np.float32),
        top_n=5,
        similarity_threshold=0.9,
    )

    assert module_idx.tolist() == [0, 1]
    assert set(np.flatnonzero(profile)) == set(range(5))
    np.testing.assert_allclose(profile[:5], 1.1)


def test_conditional_neg_log10_p_proxy_uses_retained_marker_signal():
    original_neg_log10_p = np.array([10.0, 8.0], dtype=np.float32)
    residual = np.array(
        [
            [3.0, 3.0, 0.0, 0.0],
            [2.0, 0.0, 5.0, 0.0],
        ],
        dtype=np.float32,
    )
    search_residual = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 5.0, 0.0],
        ],
        dtype=np.float32,
    )

    conditional = _conditional_neg_log10_p_proxy(
        original_neg_log10_p, residual, search_residual, top_n=2
    )

    assert conditional[0] == 0.0
    np.testing.assert_allclose(conditional[1], 8.0 * 6.0 / 7.0, rtol=1e-6)


def test_conditional_ldsc_fit_recovers_candidate_signal():
    rng = np.random.default_rng(7)
    n_snps = 80
    candidate = rng.uniform(0.2, 2.0, n_snps).astype(np.float32)
    baseline = rng.uniform(0.1, 1.0, (n_snps, 2)).astype(np.float32)
    baseline_sum = baseline.sum(axis=1)
    sample_size = np.full(n_snps, 1000.0, dtype=np.float32)
    regression_weights = np.ones(n_snps, dtype=np.float32)
    chisq = (1.0 + 0.7 * candidate + baseline @ np.array([0.2, -0.1])).astype(np.float32)

    result = _fit_conditional_ldsc(
        candidate_annotation=candidate,
        conditioning_annotations=[],
        baseline_ld=baseline,
        baseline_ld_sum=baseline_sum,
        chisq=chisq,
        sample_size=sample_size,
        regression_weights=regression_weights,
        nbar=1000.0,
        n_blocks=8,
    )

    assert result.beta > 0
    assert result.z > 0
    assert result.p < 0.05
    assert result.neg_log10_p > 1.0


def test_module_stability_detects_reproducible_profile():
    rng = np.random.default_rng(11)
    residual = np.full((30, 20), 0.1, dtype=np.float32)
    residual[:10, :5] = 2.1
    module_idx = np.arange(10)
    profile = np.zeros(20, dtype=np.float32)
    profile[:5] = 2.1
    remaining_mask = np.ones(30, dtype=bool)

    stability = _module_stability(
        residual=residual,
        module_idx=module_idx,
        candidate_profile=profile,
        remaining_before_mask=remaining_mask,
        top_n=5,
        similarity_threshold=0.95,
        n_iterations=50,
        sample_fraction=0.5,
        rng=rng,
    )

    assert stability.profile_similarity > 0.99
    assert stability.assignment_fraction > 0.99
    assert stability.null_pvalue <= 0.05


def test_module_evidence_scores_specific_unique_stable_module(tmp_path):
    ldsc_path = tmp_path / "dummy.csv.gz"
    ldsc_path.write_text("spot,p\n")
    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="evidence_score_test",
        trait_name="IQ",
        ldsc_result_path=ldsc_path,
        min_specificity_enrichment=1.5,
        specificity_null_iterations=100,
        max_specificity_null_pvalue=0.05,
        max_module_profile_similarity=0.9,
        module_stability_iterations=100,
        run_conditional_ldsc=True,
        conditional_ldsc_pvalue_threshold=0.05,
    )

    score, status = _module_evidence(
        decision="accepted",
        specificity_enrichment=3.0,
        specificity_null_pvalue=0.01,
        max_profile_similarity=0.0,
        stability=ModuleStabilityResult(
            profile_similarity=0.98,
            assignment_fraction=0.96,
            null_pvalue=0.02,
        ),
        conditional_ldsc=ConditionalLDSCResult(p=0.01, neg_log10_p=2.0),
        config=config,
    )

    assert score > 0.85
    assert status == "strong"


def test_module_evidence_marks_rejected_candidate_as_failed_gate(tmp_path):
    ldsc_path = tmp_path / "dummy.csv.gz"
    ldsc_path.write_text("spot,p\n")
    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="evidence_failed_test",
        trait_name="IQ",
        ldsc_result_path=ldsc_path,
        min_specificity_enrichment=1.5,
        specificity_null_iterations=0,
        module_stability_iterations=0,
    )

    score, status = _module_evidence(
        decision="rejected",
        specificity_enrichment=0.8,
        specificity_null_pvalue=float("nan"),
        max_profile_similarity=0.95,
        stability=ModuleStabilityResult(),
        conditional_ldsc=ConditionalLDSCResult(),
        config=config,
    )

    assert np.isfinite(score)
    assert status == "failed_gate"


def test_trait_resolution_status_classifies_signal_patterns():
    assert (
        _trait_resolution_status(
            n_modules=0,
            top_module_signal_fraction=0.0,
            accepted_signal_fraction=0.0,
            residual_signal_fraction=1.0,
        )
        == "no_modules"
    )
    assert (
        _trait_resolution_status(
            n_modules=1,
            top_module_signal_fraction=0.8,
            accepted_signal_fraction=0.8,
            residual_signal_fraction=0.1,
        )
        == "single_module_dominant"
    )
    assert (
        _trait_resolution_status(
            n_modules=2,
            top_module_signal_fraction=0.55,
            accepted_signal_fraction=0.85,
            residual_signal_fraction=0.1,
        )
        == "multi_module_resolved"
    )
    assert (
        _trait_resolution_status(
            n_modules=2,
            top_module_signal_fraction=0.4,
            accepted_signal_fraction=0.6,
            residual_signal_fraction=0.35,
        )
        == "partially_resolved"
    )


def test_module_specificity_uses_module_marker_subspace(tmp_path):
    n_cells = 20
    n_genes = 100
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:10, :5] = 2.1
    marker_scores[5:10, 5:] = 1.0

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-8
    _, _, marker_path = _write_trait_module_inputs(
        tmp_path, "specificity_subspace_test", "IQ", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="specificity_subspace_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.05,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=2,
        max_marker_genes=5,
    )

    cell_df, summary_df, _ = discover_trait_modules(config)

    assert summary_df["n_cells"].tolist() == [10]
    assigned = cell_df[cell_df["trait_module"] == 1]
    assert assigned["module_specificity"].min() > 0.99


def test_module_summary_reports_background_specificity_enrichment(tmp_path):
    n_cells = 40
    n_genes = 20
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:10, :5] = 2.1

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-8
    _, _, marker_path = _write_trait_module_inputs(
        tmp_path, "specificity_enrichment_test", "IQ", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="specificity_enrichment_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.25,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=2,
        max_marker_genes=5,
    )

    _, summary_df, _ = discover_trait_modules(config)

    assert {
        "background_mean_specificity",
        "specificity_enrichment",
        "specificity_null_pvalue",
        "max_profile_similarity",
        "most_similar_module",
    } <= set(summary_df.columns)
    assert summary_df.loc[0, "mean_specificity"] > summary_df.loc[0, "background_mean_specificity"]
    assert summary_df.loc[0, "specificity_enrichment"] > 1.0
    assert summary_df.loc[0, "specificity_null_pvalue"] <= 0.05
    assert summary_df.loc[0, "max_profile_similarity"] == 0.0
    assert summary_df.loc[0, "most_similar_module"] == -1


def test_background_like_candidate_is_rejected_before_distinct_module(tmp_path):
    n_cells = 80
    n_genes = 30
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:10, :5] = 2.1
    marker_scores[20:70, :5] = 2.1
    marker_scores[10:20, 10:15] = 2.1

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-10
    p_values[10:20] = 1e-8
    cells, genes, marker_path = _write_trait_module_inputs(
        tmp_path, "enrichment_gate_test", "IQ", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="enrichment_gate_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.2,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=3,
        max_marker_genes=5,
        background_quantile=0.0,
        min_specificity_enrichment=1.5,
        specificity_null_iterations=0,
    )

    cell_df, summary_df, genes_df, trace_df = discover_trait_modules_with_trace(config)

    assert summary_df["n_cells"].tolist() == [10]
    assert set(cell_df.loc[cell_df["trait_module"] == 1, "spot"]) == set(cells[10:20])
    assert set(cell_df.loc[cell_df["trait_module"] == -1, "spot"]) == set(cells[:10])
    assert set(cell_df.loc[cell_df["module_status"] == "rejected", "spot"]) == set(cells[:10])
    assert set(
        cell_df.loc[cell_df["module_reason"] == "min_specificity_enrichment", "spot"]
    ) == set(cells[:10])
    assert set(cell_df.loc[cell_df["module_status"] == "assigned", "spot"]) == set(cells[10:20])
    assert set(genes_df["gene"]) == set(genes[10:15])
    assert summary_df.loc[0, "specificity_enrichment"] > 1.5
    assert trace_df["decision"].tolist() == ["rejected", "accepted"]
    assert trace_df["reason"].tolist() == ["min_specificity_enrichment", "accepted"]
    assert trace_df["candidate_n_cells"].tolist() == [10, 10]
    assert trace_df["remaining_cells_before"].tolist() == [20, 10]
    assert trace_df["remaining_cells_after"].tolist() == [10, 0]
    assert trace_df["accepted_module"].tolist() == [-1, 1]


def test_iterative_background_removes_claimed_program_variants(tmp_path):
    n_cells = 50
    n_genes = 20
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:10, :5] = 3.1
    marker_scores[10:20, [0, 2, 4]] = 3.1
    marker_scores[20:30, 10:15] = 3.1

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-12
    p_values[10:20] = 1e-11
    p_values[20:30] = 1e-8
    cells, genes, marker_path = _write_trait_module_inputs(
        tmp_path, "iterative_background_test", "IQ", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="iterative_background_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.25,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=3,
        max_marker_genes=5,
        min_specificity_enrichment=1.5,
        recompute_background_after_modules=True,
        condition_on_claimed_modules=False,
    )

    cell_df, summary_df, genes_df, trace_df = discover_trait_modules_with_trace(config)

    assert summary_df["n_cells"].tolist() == [10, 10]
    assert set(cell_df.loc[cell_df["trait_module"] == 1, "spot"]) == set(cells[:10])
    assert set(cell_df.loc[cell_df["trait_module"] == 2, "spot"]) == set(cells[20:30])
    assert set(cell_df.loc[cell_df["trait_module"] == -1, "spot"]) == set(cells[10:20])
    assert set(genes_df.loc[genes_df["module"] == 1, "gene"]) == set(genes[:5])
    assert set(genes_df.loc[genes_df["module"] == 2, "gene"]) == set(genes[10:15])
    assert "mean_conditional_neg_log10_p" in trace_df.columns
    accepted_trace = trace_df[trace_df["decision"] == "accepted"]
    assert accepted_trace["mean_conditional_neg_log10_p"].notna().all()
    assert (
        accepted_trace["mean_conditional_neg_log10_p"] <= accepted_trace["mean_neg_log10_p"]
    ).all()
    assert accepted_trace["candidate_signal_fraction"].between(0, 1).all()
    assert accepted_trace["accepted_signal_fraction"].is_monotonic_increasing
    assert accepted_trace["residual_signal_fraction_after"].is_monotonic_decreasing
    assert (
        summary_df["module_signal_fraction"].tolist()
        == accepted_trace["candidate_signal_fraction"].tolist()
    )
    assert (
        summary_df["cumulative_signal_fraction"].tolist()
        == accepted_trace["accepted_signal_fraction"].tolist()
    )


def test_module_specificity_uses_conditioned_residual_for_later_modules(tmp_path):
    n_cells = 30
    n_genes = 12
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:10, :2] = 10.1
    marker_scores[10:15, 0] = 10.1
    marker_scores[10:15, 2:4] = 3.1
    marker_scores[15:20, 1] = 10.1
    marker_scores[15:20, 2:4] = 3.1

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-12
    p_values[10:20] = 1e-8
    cells, _, marker_path = _write_trait_module_inputs(
        tmp_path, "conditioned_specificity_test", "IQ", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="conditioned_specificity_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.2,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=3,
        max_marker_genes=2,
        min_specificity_enrichment=1.0,
        specificity_null_iterations=0,
        recompute_background_after_modules=False,
        condition_on_claimed_modules=True,
    )

    cell_df, summary_df, _, trace_df = discover_trait_modules_with_trace(config)

    assert summary_df["n_cells"].tolist() == [10, 10]
    assert trace_df["decision"].tolist() == ["accepted", "accepted"]
    second_module = cell_df[cell_df["trait_module"] == 2]
    assert set(second_module["spot"]) == set(cells[10:20])
    assert second_module["module_specificity"].min() > 0.999


def test_specificity_null_rejects_common_background_program(tmp_path):
    n_cells = 100
    n_genes = 20
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:90, :5] = 2.1

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-8
    _, _, marker_path = _write_trait_module_inputs(
        tmp_path, "specificity_null_test", "IQ", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="specificity_null_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.25,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=2,
        max_marker_genes=5,
        background_quantile=0.0,
        min_specificity_enrichment=1.0,
        specificity_null_iterations=200,
        max_specificity_null_pvalue=0.05,
    )

    cell_df, summary_df, genes_df, trace_df = discover_trait_modules_with_trace(config)

    assert cell_df["trait_module"].tolist() == [-1] * 10
    assert summary_df.empty
    assert genes_df.empty
    assert trace_df["decision"].tolist() == ["rejected"]
    assert trace_df["reason"].tolist() == ["specificity_null_pvalue"]
    assert trace_df.loc[0, "specificity_null_pvalue"] > 0.05


def test_require_module_stability_rejects_unstable_candidate(tmp_path, monkeypatch):
    n_cells = 30
    n_genes = 20
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:10, :5] = 2.1

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-8
    _, _, marker_path = _write_trait_module_inputs(
        tmp_path, "module_stability_gate_test", "IQ", marker_scores, p_values
    )

    monkeypatch.setattr(
        "gsMap.trait_modules._module_stability",
        lambda **kwargs: ModuleStabilityResult(
            profile_similarity=0.2,
            assignment_fraction=0.3,
            null_pvalue=0.9,
        ),
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="module_stability_gate_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.25,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=2,
        max_marker_genes=5,
        min_specificity_enrichment=0.0,
        specificity_null_iterations=0,
        require_module_stability=True,
    )

    cell_df, summary_df, _, trace_df = discover_trait_modules_with_trace(config)

    assert cell_df["trait_module"].tolist() == [-1] * 10
    assert summary_df.empty
    assert trace_df["decision"].tolist() == ["rejected"]
    assert trace_df["reason"].tolist() == ["module_stability"]
    assert trace_df.loc[0, "stability_profile_similarity"] == 0.2


def test_conditional_ldsc_diagnostics_do_not_gate_by_default(tmp_path, monkeypatch):
    n_cells = 30
    n_genes = 20
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:10, :5] = 2.1

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-8
    _, _, marker_path = _write_trait_module_inputs(
        tmp_path, "conditional_ldsc_gate_test", "IQ", marker_scores, p_values
    )

    class FakeConditionalLDSCContext:
        def fit_profile(self, profile, conditioning_annotations):
            return (
                ConditionalLDSCResult(
                    beta=0.1,
                    se=0.1,
                    z=1.0,
                    p=0.75,
                    neg_log10_p=-np.log10(0.75),
                ),
                np.ones(5, dtype=np.float32),
            )

    monkeypatch.setattr(
        "gsMap.trait_modules._prepare_conditional_ldsc_context",
        lambda config, gene_names: FakeConditionalLDSCContext(),
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="conditional_ldsc_gate_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.25,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=2,
        max_marker_genes=5,
        min_specificity_enrichment=0.0,
        specificity_null_iterations=0,
        run_conditional_ldsc=True,
        conditional_ldsc_pvalue_threshold=0.05,
    )

    cell_df, summary_df, _, trace_df = discover_trait_modules_with_trace(config)

    assert set(cell_df.loc[cell_df["trait_module"] == 1, "spot"]) == {
        f"cell_{i}" for i in range(10)
    }
    assert summary_df["n_cells"].tolist() == [10]
    assert trace_df["decision"].tolist() == ["accepted"]
    assert trace_df["reason"].tolist() == ["accepted"]
    assert trace_df.loc[0, "conditional_ldsc_p"] == 0.75


def test_require_conditional_ldsc_gate_rejects_candidate(tmp_path, monkeypatch):
    n_cells = 30
    n_genes = 20
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:10, :5] = 2.1

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-8
    _, _, marker_path = _write_trait_module_inputs(
        tmp_path, "conditional_ldsc_gate_test", "IQ", marker_scores, p_values
    )

    class FakeConditionalLDSCContext:
        def fit_profile(self, profile, conditioning_annotations):
            return (
                ConditionalLDSCResult(
                    beta=0.1,
                    se=0.1,
                    z=1.0,
                    p=0.75,
                    neg_log10_p=-np.log10(0.75),
                ),
                np.ones(5, dtype=np.float32),
            )

    monkeypatch.setattr(
        "gsMap.trait_modules._prepare_conditional_ldsc_context",
        lambda config, gene_names: FakeConditionalLDSCContext(),
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="conditional_ldsc_gate_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.25,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=2,
        max_marker_genes=5,
        min_specificity_enrichment=0.0,
        specificity_null_iterations=0,
        require_conditional_ldsc=True,
        conditional_ldsc_pvalue_threshold=0.05,
    )

    cell_df, summary_df, _, trace_df = discover_trait_modules_with_trace(config)

    assert cell_df["trait_module"].tolist() == [-1] * 10
    assert summary_df.empty
    assert trace_df["decision"].tolist() == ["rejected"]
    assert trace_df["reason"].tolist() == ["conditional_ldsc_pvalue"]
    assert trace_df.loc[0, "conditional_ldsc_p"] == 0.75


def test_profile_similarity_rejects_duplicate_mechanism(tmp_path):
    n_cells = 50
    n_genes = 20
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:10, :5] = 2.1
    marker_scores[10:20, [0, 1, 2, 3, 5]] = 2.1

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-12
    p_values[10:20] = 1e-11
    cells, _, marker_path = _write_trait_module_inputs(
        tmp_path, "profile_similarity_test", "IQ", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="profile_similarity_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.25,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=3,
        max_marker_genes=5,
        min_specificity_enrichment=1.0,
        specificity_null_iterations=0,
        max_module_profile_similarity=0.75,
        recompute_background_after_modules=False,
        condition_on_claimed_modules=False,
    )

    cell_df, summary_df, _, trace_df = discover_trait_modules_with_trace(config)

    assert summary_df["n_cells"].tolist() == [10]
    assert set(cell_df.loc[cell_df["trait_module"] == 1, "spot"]) == set(cells[:10])
    assert set(cell_df.loc[cell_df["trait_module"] == -1, "spot"]) == set(cells[10:20])
    assert trace_df["decision"].tolist() == ["accepted", "rejected"]
    assert trace_df["reason"].tolist() == ["accepted", "module_profile_similarity"]
    assert trace_df.loc[1, "max_profile_similarity"] > 0.75
    assert trace_df.loc[1, "most_similar_module"] == 1


def test_all_background_like_candidates_keep_empty_output_schema(tmp_path):
    n_cells = 50
    n_genes = 20
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:, :5] = 2.1

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-8
    _, _, marker_path = _write_trait_module_inputs(
        tmp_path, "empty_schema_test", "IQ", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="empty_schema_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.25,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=2,
        max_marker_genes=5,
        background_quantile=0.0,
        min_specificity_enrichment=1.5,
    )

    cell_df, summary_df, genes_df = discover_trait_modules(config)

    assert cell_df["trait_module"].tolist() == [-1] * 10
    assert not cell_df["is_module_cell"].any()
    assert cell_df["module_status"].tolist() == ["unresolved"] * 10
    assert cell_df["module_reason"].tolist() == ["min_module_cells"] * 10
    assert list(summary_df.columns) == [
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
    assert summary_df.empty
    assert list(genes_df.columns) == ["trait", "module", "rank", "gene", "score"]
    assert genes_df.empty


def test_module_trace_reports_stop_condition(tmp_path):
    n_cells = 30
    n_genes = 20
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:10, :5] = 2.1

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-8
    cells, _, marker_path = _write_trait_module_inputs(
        tmp_path, "trace_stop_test", "IQ", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="trace_stop_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.25,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=2,
        max_marker_genes=5,
    )

    cell_df, summary_df, _, trace_df = discover_trait_modules_with_trace(config)

    assert set(cell_df.loc[cell_df["trait_module"] == 1, "spot"]) == set(cells[:10])
    assert summary_df["n_cells"].tolist() == [10]
    assert trace_df["decision"].tolist() == ["accepted"]
    assert trace_df["remaining_cells_after"].tolist() == [0]


def test_summarize_trait_module_run_reports_resolution_status(tmp_path):
    n_cells = 50
    n_genes = 20
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:10, :5] = 3.1
    marker_scores[10:20, 10:15] = 3.1

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-12
    p_values[10:20] = 1e-8
    _, _, marker_path = _write_trait_module_inputs(
        tmp_path, "run_summary_test", "IQ", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="run_summary_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.25,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=3,
        max_marker_genes=5,
    )

    cell_df, summary_df, _, trace_df = discover_trait_modules_with_trace(config)
    run_summary = summarize_trait_module_run(config, cell_df, summary_df, trace_df)

    assert list(run_summary.columns) == [
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
    assert run_summary.loc[0, "n_modules"] == 2
    assert run_summary.loc[0, "n_module_cells"] == 20
    assert run_summary.loc[0, "n_assigned_seed_cells"] == 20
    assert run_summary.loc[0, "assigned_seed_cell_fraction"] == 1.0
    assert run_summary.loc[0, "n_rejected_seed_cells"] == 0
    assert run_summary.loc[0, "n_unresolved_seed_cells"] == 0
    assert (
        run_summary.loc[0, "accepted_signal_fraction"]
        == summary_df.loc[1, "cumulative_signal_fraction"]
    )
    assert run_summary.loc[0, "resolution_status"] in {
        "multi_module_resolved",
        "partially_resolved",
    }


def test_summarize_trait_module_pairs_reports_distinct_marker_programs(tmp_path):
    _, _, marker_path = _write_trait_module_inputs(
        tmp_path,
        "pair_summary_test",
        "IQ",
        np.ones((4, 5), dtype=np.float32),
        np.full(4, 0.9),
    )
    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="pair_summary_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        max_module_profile_similarity=0.9,
    )
    summary_df = pd.DataFrame(
        {
            "trait": ["IQ", "IQ"],
            "module": [1, 2],
            "n_cells": [12, 8],
            "module_signal_fraction": [0.6, 0.25],
            "module_evidence_status": ["strong", "moderate"],
        }
    )
    genes_df = pd.DataFrame(
        {
            "trait": ["IQ"] * 6,
            "module": [1, 1, 1, 2, 2, 2],
            "rank": [1, 2, 3, 1, 2, 3],
            "gene": ["G1", "G2", "G3", "G2", "G4", "G5"],
            "score": [4.0, 2.0, 1.0, 3.0, 3.0, 1.0],
        }
    )

    pair_df = summarize_trait_module_pairs(config, summary_df, genes_df)

    assert list(pair_df.columns) == [
        "trait",
        "module_a",
        "module_b",
        "profile_similarity",
        "max_allowed_profile_similarity",
        "profile_distinct",
        "marker_jaccard",
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
    assert pair_df.loc[0, "module_a"] == 1
    assert pair_df.loc[0, "module_b"] == 2
    assert pair_df.loc[0, "profile_distinct"]
    assert pair_df.loc[0, "shared_marker_count"] == 1
    assert pair_df.loc[0, "marker_jaccard"] == 0.2
    assert pair_df.loc[0, "shared_top_genes"] == "G2"
    assert pair_df.loc[0, "module_a_only_top_genes"] == "G1;G3"
    assert pair_df.loc[0, "module_b_only_top_genes"] == "G4;G5"
    assert pair_df.loc[0, "module_a_signal_fraction"] == 0.6
    assert pair_df.loc[0, "module_b_evidence_status"] == "moderate"


def test_trait_module_pairs_from_results_use_full_profiles(tmp_path):
    _, _, marker_path = _write_trait_module_inputs(
        tmp_path,
        "pair_result_test",
        "IQ",
        np.ones((4, 6), dtype=np.float32),
        np.full(4, 0.9),
    )
    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="pair_result_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        max_module_profile_similarity=0.9,
    )
    module_results = [
        ModuleResult(
            module_id=1,
            cells=pd.Index(["cell_1", "cell_2"]),
            profile=np.array([4.0, 3.0, 0.0, 0.0, 2.0, 0.0], dtype=np.float32),
            specificity=np.zeros(6, dtype=np.float32),
            mean_neg_log10_p=8.0,
            background_specificity_mean=0.1,
            specificity_enrichment=2.0,
            specificity_null_pvalue=0.01,
            max_profile_similarity=0.0,
            most_similar_module=-1,
            module_signal_fraction=0.6,
            module_evidence_status="strong",
        ),
        ModuleResult(
            module_id=2,
            cells=pd.Index(["cell_3", "cell_4"]),
            profile=np.array([0.0, 0.0, 5.0, 4.0, 2.0, 0.0], dtype=np.float32),
            specificity=np.zeros(6, dtype=np.float32),
            mean_neg_log10_p=7.0,
            background_specificity_mean=0.1,
            specificity_enrichment=2.0,
            specificity_null_pvalue=0.01,
            max_profile_similarity=0.0,
            most_similar_module=-1,
            module_signal_fraction=0.25,
            module_evidence_status="moderate",
        ),
    ]
    gene_order_by_module = {
        1: ["G1", "G2"],
        2: ["G3", "G4"],
    }

    pair_df = _summarize_trait_module_pairs_from_results(
        config,
        module_results,
        gene_order_by_module,
    )

    assert len(pair_df) == 1
    assert pair_df.loc[0, "profile_similarity"] > 0.1
    assert pair_df.loc[0, "profile_distinct"]
    assert pair_df.loc[0, "marker_jaccard"] == 0.0
    assert pair_df.loc[0, "module_a_marker_count"] == 2
    assert pair_df.loc[0, "module_b_marker_count"] == 2
    assert pair_df.loc[0, "module_a_signal_fraction"] == 0.6


def test_run_trait_module_discovery_writes_trace_csv(tmp_path):
    n_cells = 30
    n_genes = 20
    marker_scores = np.full((n_cells, n_genes), 0.1, dtype=np.float32)
    marker_scores[:10, :5] = 2.1

    p_values = np.full(n_cells, 0.9)
    p_values[:10] = 1e-8
    _, _, marker_path = _write_trait_module_inputs(
        tmp_path, "trace_write_test", "IQ", marker_scores, p_values
    )

    config = TraitModuleConfig(
        workdir=tmp_path,
        project_name="trace_write_test",
        trait_name="IQ",
        marker_score_h5ad_path=marker_path,
        pvalue_threshold=0.01,
        top_gene_fraction=0.25,
        min_module_cells=5,
        module_similarity_threshold=0.95,
        max_modules=2,
        max_marker_genes=5,
    )

    run_trait_module_discovery(config)

    trace_df = pd.read_csv(config.module_trace_path)
    run_summary_df = pd.read_csv(config.run_summary_path)
    pair_df = pd.read_csv(config.module_pairs_path)
    assert trace_df["decision"].tolist() == ["accepted"]
    assert trace_df["candidate_n_cells"].tolist() == [10]
    assert run_summary_df["n_modules"].tolist() == [1]
    assert run_summary_df["resolution_status"].tolist() == ["single_module_dominant"]
    assert list(pair_df.columns) == [
        "trait",
        "module_a",
        "module_b",
        "profile_similarity",
        "max_allowed_profile_similarity",
        "profile_distinct",
        "marker_jaccard",
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
    assert pair_df.empty
