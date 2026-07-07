import numpy as np
import pandas as pd

from ripple.modules.anchored import AnchoredModuleLibrary, empirical_upper
from ripple.modules.distributed import (
    RippleDConfig,
    assign_pseudo_loci,
    classify_distributed_module,
    contribution_metrics,
    positive_locus_robust_stat,
    prepare_locus_inputs,
    rank_locus_enrichment_stat,
    locus_robust_stat,
    ripple_d_module_tests,
    summarize_module_distribution,
)


def synthetic_scores(values: dict[str, float]) -> pd.DataFrame:
    genes = list(values)
    starts = np.arange(len(genes)) * 1_300_000 + 100_000
    return pd.DataFrame(
        {
            "gene_symbol": genes,
            "assoc_resid_score": [values[gene] for gene in genes],
            "chrom": [1] * len(genes),
            "gene_start": starts,
            "gene_end": starts + 10_000,
            "graph_degree": np.arange(len(genes)) + 1,
            "gene_length": 10_000,
            "n_mapped_snps": 10,
            "local_ld_score": 2.0,
        }
    )


def test_assign_pseudo_loci_merges_overlapping_expanded_intervals():
    scores = pd.DataFrame(
        {
            "gene_symbol": ["A", "B", "C"],
            "assoc_resid_score": [1.0, 2.0, 3.0],
            "chrom": [1, 1, 1],
            "gene_start": [1000, 1800, 10_000],
            "gene_end": [1200, 2000, 10_200],
        }
    )

    out = assign_pseudo_loci(scores, window_bp=500)

    assert out.loc[out["gene_symbol"].eq("A"), "locus_id"].iloc[0] == out.loc[
        out["gene_symbol"].eq("B"), "locus_id"
    ].iloc[0]
    assert out["locus_id"].nunique() == 2


def test_locus_collapse_uses_max_not_sum_for_same_locus():
    scores = pd.DataFrame(
        {
            "gene_symbol": ["A", "B", "C", "D", "E"],
            "assoc_resid_score": [2.0, 2.0, 1.5, 1.5, 1.5],
            "chrom": [1, 1, 1, 1, 1],
            "gene_start": [1000, 1500, 2_000_000, 3_300_000, 4_600_000],
            "gene_end": [1200, 1700, 2_010_000, 3_310_000, 4_610_000],
            "graph_degree": [1, 1, 1, 1, 1],
        }
    )
    config = RippleDConfig(locus_window_bp=500, score_cap=3.0)
    work = assign_pseudo_loci(scores, window_bp=config.locus_window_bp)
    work["ripple_d_capped_score"] = work["assoc_resid_score"]
    work["score_rank_fraction"] = 0.5

    summary = summarize_module_distribution(work, ["A", "B", "C", "D", "E"], config)

    assert summary["n_loci"] == 4
    assert summary["locus_robust_stat"] == locus_robust_stat(np.array([2.0, 1.5, 1.5, 1.5]))
    assert summary["positive_locus_robust_stat"] == positive_locus_robust_stat(np.array([2.0, 1.5, 1.5, 1.5]))


def test_rank_locus_enrichment_stat_is_high_for_top_ranks():
    top_enriched = rank_locus_enrichment_stat(np.array([0.01, 0.02, 0.10]))
    depleted = rank_locus_enrichment_stat(np.array([0.80, 0.90, 0.95]))

    assert top_enriched > depleted
    assert top_enriched > 0.9


def test_contribution_metrics_detect_single_locus_dominance():
    metrics = contribution_metrics(np.array([3.0, 0.2, 0.1]))

    assert metrics["n_effective_loci"] < 2
    assert metrics["top1_locus_contribution"] > 0.85


def test_distributed_gate_classifies_sparse_and_distributed_rows():
    config = RippleDConfig()
    distributed = {
        "locus_robust_empirical_p": 0.02,
        "ripple_d_empirical_p": 0.02,
        "positive_locus_empirical_p": 0.02,
        "module_specific_rank_empirical_p": 0.02,
        "moderate_locus_burden_empirical_p": 0.04,
        "leave_top1_locus_empirical_p": 0.08,
        "raw_gene_empirical_p": 0.01,
        "n_effective_loci": 6.0,
        "top1_locus_contribution": 0.25,
        "top5_locus_contribution": 0.65,
    }
    sparse = dict(distributed)
    sparse["n_effective_loci"] = 1.4
    sparse["top1_locus_contribution"] = 0.82

    assert classify_distributed_module(distributed, config) == "distributed_weak_signal_module_candidate"
    assert classify_distributed_module(sparse, config) == "top_locus_dominant_module"


def test_degraded_null_nan_does_not_block_classifier():
    config = RippleDConfig()
    row = {
        "locus_robust_empirical_p": 0.02,
        "ripple_d_empirical_p": 0.02,
        "positive_locus_empirical_p": 0.02,
        "module_specific_rank_empirical_p": 0.02,
        "moderate_locus_burden_empirical_p": 0.04,
        "leave_top1_locus_empirical_p": 0.08,
        "raw_gene_empirical_p": 0.01,
        "n_effective_loci": 6.0,
        "top1_locus_contribution": 0.25,
        "top5_locus_contribution": 0.65,
        "null_gene_count_match_degraded": np.nan,
    }

    assert classify_distributed_module(row, config) == "distributed_weak_signal_module_candidate"


def test_tiered_classifier_reports_mixed_sparse_distributed_support():
    config = RippleDConfig()
    row = {
        "locus_robust_empirical_p": 0.04,
        "ripple_d_empirical_p": 0.04,
        "positive_locus_empirical_p": 0.04,
        "module_specific_rank_empirical_p": 0.20,
        "moderate_locus_burden_empirical_p": 0.20,
        "leave_top1_locus_empirical_p": 0.20,
        "raw_gene_empirical_p": 0.01,
        "n_effective_loci": 6.0,
        "top1_locus_contribution": 0.30,
        "top5_locus_contribution": 0.80,
    }

    assert classify_distributed_module(row, config) == "mixed_sparse_distributed_candidate"


def test_empirical_upper_plus_one_for_ripple_d_nulls():
    assert empirical_upper(np.array([0.0, 1.0, 2.0]), 1.0) == 0.75


def test_ripple_d_module_tests_rejects_single_top_locus_artifact():
    values = {f"G{i}": 0.0 for i in range(80)}
    for idx in range(80):
        values[f"G{idx}"] = -0.2
    values["G0"] = 12.0
    scores = synthetic_scores(values)
    library = AnchoredModuleLibrary(
        gene_sets={"artifact": {f"G{i}" for i in range(10)}},
        module_source={"artifact": "unit_test"},
        annotation_source_type={"artifact": "internal_support"},
    )

    modules, _, _, _ = ripple_d_module_tests(
        scores,
        library,
        config=RippleDConfig(locus_window_bp=500, degree_bins=4, property_bins=2),
        n_null=30,
        seed=1,
    )

    row = modules.iloc[0]
    assert row["module_status"] != "distributed_weak_signal_module_candidate"
    assert row["n_effective_loci"] < 5


def test_ripple_d_module_tests_detects_distributed_moderate_signal():
    values = {f"G{i}": 0.0 for i in range(120)}
    for idx in range(8):
        values[f"G{idx}"] = 2.5
    scores = synthetic_scores(values)
    library = AnchoredModuleLibrary(
        gene_sets={"distributed": {f"G{i}" for i in range(12)}},
        module_source={"distributed": "unit_test"},
        annotation_source_type={"distributed": "internal_support"},
    )

    modules, _, _, _ = ripple_d_module_tests(
        scores,
        library,
        config=RippleDConfig(locus_window_bp=500, degree_bins=4, property_bins=2),
        n_null=50,
        seed=2,
    )

    row = modules.iloc[0]
    assert row["n_effective_loci"] >= 5
    assert row["top1_locus_contribution"] <= 0.35
    assert row["module_status"] == "distributed_weak_signal_module_candidate"


def test_null_preserves_module_gene_count_within_sampled_locus():
    scores = pd.DataFrame(
        {
            "gene_symbol": ["M1", "B1", "B2", "B3", "B4", "B5", "C1", "C2", "C3", "C4"],
            "assoc_resid_score": [1.5, 0.1, 0.1, 0.1, 30.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            "chrom": [1] * 10,
            "gene_start": [1000, 10_000, 10_500, 11_000, 11_500, 12_000, 100_000, 200_000, 300_000, 400_000],
            "gene_end": [1100, 10_100, 10_600, 11_100, 11_600, 12_100, 100_100, 200_100, 300_100, 400_100],
            "graph_degree": [1] * 10,
            "gene_length": [100] * 10,
            "n_mapped_snps": [10] * 10,
            "local_ld_score": [2.0] * 10,
        }
    )
    library = AnchoredModuleLibrary(
        gene_sets={"single_gene_module": {"M1", "C1", "C2", "C3", "C4"}},
        module_source={"single_gene_module": "unit_test"},
        annotation_source_type={"single_gene_module": "internal_support"},
    )

    modules, _, nulls, _ = ripple_d_module_tests(
        scores,
        library,
        config=RippleDConfig(
            locus_window_bp=200,
            degree_bins=2,
            property_bins=2,
            null_gene_subset_sampling=True,
        ),
        n_null=80,
        seed=101,
        return_null_details=True,
    )

    row = modules.iloc[0]
    locus_null = nulls.loc[
        (nulls["null_type"].eq("locus_matched_competitive_null"))
        & (nulls["statistic_name"].eq("locus_robust_stat"))
    ]
    assert row["n_module_genes_per_locus"].split(",")[0] == "1"
    assert float(locus_null["statistic_value"].max()) < 20.0


def test_moderate_locus_null_uses_configured_thresholds():
    values = {f"G{i}": 0.0 for i in range(80)}
    for idx in range(5):
        values[f"G{idx}"] = 0.75
    scores = synthetic_scores(values)
    library = AnchoredModuleLibrary(
        gene_sets={"low_moderate": {f"G{i}" for i in range(10)}},
        module_source={"low_moderate": "unit_test"},
        annotation_source_type={"low_moderate": "internal_support"},
    )

    modules, _, _, _ = ripple_d_module_tests(
        scores,
        library,
        config=RippleDConfig(
            moderate_score_low=0.5,
            moderate_score_high=1.0,
            locus_window_bp=500,
            degree_bins=4,
            property_bins=2,
        ),
        n_null=30,
        seed=202,
    )

    row = modules.iloc[0]
    assert row["moderate_locus_burden"] >= 5
    assert np.isfinite(row["moderate_locus_burden_empirical_p"])


def test_precomputed_locus_inputs_preserve_module_results():
    values = {f"G{i}": 0.0 for i in range(80)}
    for idx in range(6):
        values[f"G{idx}"] = 2.0
    scores = synthetic_scores(values)
    library = AnchoredModuleLibrary(
        gene_sets={"distributed": {f"G{i}" for i in range(10)}},
        module_source={"distributed": "unit_test"},
        annotation_source_type={"distributed": "internal_support"},
    )
    config = RippleDConfig(locus_window_bp=500, degree_bins=4, property_bins=2)

    direct, _, _, _ = ripple_d_module_tests(scores, library, config=config, n_null=20, seed=303)
    work, background = prepare_locus_inputs(scores, library, config)
    precomputed, _, _, _ = ripple_d_module_tests(
        scores,
        library,
        config=config,
        n_null=20,
        seed=303,
        precomputed_work=work,
        precomputed_locus_background=background,
    )

    columns = [
        "module_status",
        "locus_robust_empirical_p",
        "ripple_d_empirical_p",
        "positive_locus_empirical_p",
        "module_specific_rank_empirical_p",
    ]
    pd.testing.assert_frame_equal(direct[columns], precomputed[columns])


def test_locus_membership_rank_leakage_does_not_drive_module_specific_rank_support():
    scores = pd.DataFrame(
        {
            "gene_symbol": ["A_MODULE_LOW", "B_NONMODULE_TOP", "C1", "C2", "C3", "C4", "C5", "C6"],
            "assoc_resid_score": [0.1, 8.0, 0.2, 0.1, 0.0, -0.1, 0.0, 0.1],
            "chrom": [1] * 8,
            "gene_start": [1000, 1100, 2_000_000, 3_500_000, 5_000_000, 6_500_000, 8_000_000, 9_500_000],
            "gene_end": [1050, 1150, 2_010_000, 3_510_000, 5_010_000, 6_510_000, 8_010_000, 9_510_000],
            "graph_degree": [1] * 8,
            "gene_length": [100] * 8,
            "n_mapped_snps": [10] * 8,
            "local_ld_score": [2.0] * 8,
        }
    )
    library = AnchoredModuleLibrary(
        gene_sets={"rank_leakage": {"A_MODULE_LOW", "C1", "C2", "C3", "C4"}},
        module_source={"rank_leakage": "unit_test"},
        annotation_source_type={"rank_leakage": "internal_support"},
    )

    modules, _, _, _ = ripple_d_module_tests(
        scores,
        library,
        config=RippleDConfig(locus_window_bp=500, degree_bins=2, property_bins=1),
        n_null=30,
        seed=404,
    )

    row = modules.iloc[0]
    assert row["locus_membership_rank_enrichment_stat"] > row["module_specific_rank_enrichment_stat"]
    assert row["module_status"] != "module_specific_rank_supported_module"
    assert row["module_status"] != "distributed_weak_signal_module_candidate"


def test_gene_count_replacement_audit_marks_degraded_null():
    scores = pd.DataFrame(
        {
            "gene_symbol": ["M1", "M2", "M3", "M4", "S1", "S2", "S3", "S4", "S5"],
            "assoc_resid_score": [1.0, 1.1, 1.2, 1.3, 0.0, 0.1, 0.2, 0.0, -0.1],
            "chrom": [1] * 9,
            "gene_start": [1000, 1100, 1200, 1300, 2_000_000, 3_500_000, 5_000_000, 6_500_000, 8_000_000],
            "gene_end": [1050, 1150, 1250, 1350, 2_010_000, 3_510_000, 5_010_000, 6_510_000, 8_010_000],
            "graph_degree": [1] * 9,
            "gene_length": [100] * 9,
            "n_mapped_snps": [10] * 9,
            "local_ld_score": [2.0] * 9,
        }
    )
    library = AnchoredModuleLibrary(
        gene_sets={"multi_gene_locus": {"M1", "M2", "M3", "M4", "S1"}},
        module_source={"multi_gene_locus": "unit_test"},
        annotation_source_type={"multi_gene_locus": "internal_support"},
    )

    modules, _, _, _ = ripple_d_module_tests(
        scores,
        library,
        config=RippleDConfig(locus_window_bp=500, degree_bins=1, property_bins=1),
        n_null=30,
        seed=505,
    )

    row = modules.iloc[0]
    assert row["null_with_replacement_rate"] > 0
    assert row["null_loci_with_insufficient_gene_pool_rate"] > 0
    assert bool(row["null_gene_count_match_degraded"])
    assert row["module_status"] == "null_degraded_unresolved"
