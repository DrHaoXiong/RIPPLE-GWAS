import numpy as np
import pandas as pd

from ripple.modules.anchored import AnchoredModuleLibrary, empirical_upper
from ripple.modules.distributed import (
    RippleDConfig,
    add_module_redundancy_fields,
    add_ripple_d_v16_claim_readiness,
    assign_pseudo_loci,
    classify_distributed_module,
    contribution_metrics,
    external_locus_audit_table,
    positive_locus_robust_stat,
    prepare_locus_inputs,
    rank_locus_enrichment_stat,
    locus_robust_stat,
    ripple_d_module_tests,
    ripple_d_stat,
    ripple_d_v16_stat,
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


def test_v16_stat_adds_top5_penalty_without_changing_v15_stat():
    config = RippleDConfig(top5_contribution_max=0.70)
    scores = np.array([3.0, 3.0, 3.0, 3.0, 3.0, 0.1, 0.1, 0.1])

    assert ripple_d_v16_stat(scores, config) < ripple_d_stat(scores, config)


def test_v16_claim_readiness_downgrades_top_tail_artifact():
    config = RippleDConfig()
    modules = pd.DataFrame(
        [
            {
                "module_name": "strong_artifact",
                "module_status": "distributed_weak_signal_module_candidate",
                "present_genes": "A,B,C,D,E,F,G,H",
                "n_present": 8,
                "n_loci": 8,
                "ripple_d_v16_empirical_p": 0.001,
                "locus_robust_empirical_p": 0.001,
                "module_specific_rank_empirical_p": 0.001,
                "positive_locus_empirical_p": 0.001,
                "leave_top1_locus_empirical_p": 0.01,
                "leave_top3_locus_empirical_p": 0.01,
                "top_locus_conditioned_leave_top1_p": 0.01,
                "top_locus_conditioned_leave_top3_p": 0.01,
                "null_exact_match_rate": 1.0,
                "null_global_fallback_rate": 0.0,
                "null_reuse_fallback_rate": 0.0,
                "null_with_replacement_rate": 0.0,
                "min_match_pool_size": 50,
                "null_loci_with_insufficient_gene_pool_rate": 0.0,
                "ld_block_locus_sensitivity_status": "external_locus_column_used",
                "pseudo_locus_window_stability_status": "passed",
                "fraction_loci_with_uncapped_score_gt_3": 0.75,
                "fraction_positive_signal_from_uncapped_gt_3": 0.80,
                "n_loci_in_genome_top_1pct": 2,
                "top5_locus_contribution": 0.60,
                "n_loci_with_uncapped_score_gt_3": 6,
            }
        ]
    )

    out = add_ripple_d_v16_claim_readiness(modules, config)

    assert out["v16_claim_status"].iloc[0] == "multi_strong_locus_pathway_overlap"
    assert not bool(out["top_tail_pass"].iloc[0])


def test_v16_claim_readiness_requires_full_library_q_values():
    config = RippleDConfig()
    rows = []
    for idx in range(30):
        rows.append(
            {
                "module_name": f"M{idx}",
                "module_status": "distributed_weak_signal_module_candidate",
                "present_genes": f"G{idx},H{idx},I{idx},J{idx},K{idx}",
                "n_present": 5,
                "n_loci": 5,
                "ripple_d_v16_empirical_p": 0.05 if idx == 0 else 0.90,
                "locus_robust_empirical_p": 0.05 if idx == 0 else 0.90,
                "module_specific_rank_empirical_p": 0.05 if idx == 0 else 0.90,
                "positive_locus_empirical_p": 0.05 if idx == 0 else 0.90,
                "leave_top1_locus_empirical_p": 0.01,
                "leave_top3_locus_empirical_p": 0.01,
                "top_locus_conditioned_leave_top1_p": 0.01,
                "top_locus_conditioned_leave_top3_p": 0.01,
                "null_exact_match_rate": 1.0,
                "null_global_fallback_rate": 0.0,
                "null_reuse_fallback_rate": 0.0,
                "null_with_replacement_rate": 0.0,
                "min_match_pool_size": 50,
                "null_loci_with_insufficient_gene_pool_rate": 0.0,
                "ld_block_locus_sensitivity_status": "external_locus_column_used",
                "pseudo_locus_window_stability_status": "passed",
                "fraction_loci_with_uncapped_score_gt_3": 0.0,
                "fraction_positive_signal_from_uncapped_gt_3": 0.0,
                "n_loci_in_genome_top_1pct": 0,
                "top5_locus_contribution": 0.50,
                "n_loci_with_uncapped_score_gt_3": 0,
            }
        )
    modules = pd.DataFrame(rows)

    out = add_ripple_d_v16_claim_readiness(modules, config)

    assert out.loc[out["module_name"].eq("M0"), "ripple_d_q_full_library"].iloc[0] > 0.10
    assert not bool(out.loc[out["module_name"].eq("M0"), "multiplicity_pass"].iloc[0])


def test_v16b_balanced_null_quality_requires_multiplicity_for_high_confidence():
    config = RippleDConfig()
    rows = []
    for idx in range(30):
        rows.append(
            {
                "module_name": f"M{idx}",
                "module_status": "distributed_weak_signal_module_candidate",
                "present_genes": f"G{idx},H{idx},I{idx},J{idx},K{idx}",
                "n_present": 5,
                "n_loci": 5,
                "ripple_d_v16_empirical_p": 0.001 if idx == 0 else 0.90,
                "locus_robust_empirical_p": 0.001 if idx == 0 else 0.90,
                "module_specific_rank_empirical_p": 0.001 if idx == 0 else 0.90,
                "positive_locus_empirical_p": 0.001 if idx == 0 else 0.90,
                "leave_top1_locus_empirical_p": 0.01,
                "leave_top3_locus_empirical_p": 0.01,
                "top_locus_conditioned_leave_top1_p": 0.01,
                "top_locus_conditioned_leave_top3_p": 0.01,
                "null_exact_match_rate": 1.0,
                "null_global_fallback_rate": 0.0,
                "null_reuse_fallback_rate": 0.0,
                "null_with_replacement_rate": 0.0,
                "min_match_pool_size": 12,
                "median_match_pool_size": 25,
                "null_loci_with_insufficient_gene_pool_rate": 0.0,
                "ld_block_locus_sensitivity_status": "external_locus_column_used",
                "pseudo_locus_window_stability_status": "not_tested",
                "fraction_loci_with_uncapped_score_gt_3": 0.0,
                "fraction_positive_signal_from_uncapped_gt_3": 0.0,
                "n_loci_in_genome_top_1pct": 0,
                "top5_locus_contribution": 0.50,
                "n_loci_with_uncapped_score_gt_3": 0,
            }
        )
    modules = pd.DataFrame(rows)

    out = add_ripple_d_v16_claim_readiness(modules, config)
    first = out.loc[out["module_name"].eq("M0")].iloc[0]

    assert not bool(first["null_quality_strict_pass"])
    assert bool(first["null_quality_balanced_pass"])
    assert bool(first["multiplicity_pass"])
    assert first["v16_claim_status"] == "exploratory_locus_distributed_candidate"
    assert first["v16b_claim_status"] == "high_confidence_diagnostic_candidate"


def test_v16b_balanced_high_confidence_still_requires_multiplicity():
    config = RippleDConfig()
    rows = []
    for idx in range(30):
        rows.append(
            {
                "module_name": f"M{idx}",
                "module_status": "distributed_weak_signal_module_candidate",
                "present_genes": f"G{idx},H{idx},I{idx},J{idx},K{idx}",
                "n_present": 5,
                "n_loci": 5,
                "ripple_d_v16_empirical_p": 0.05 if idx == 0 else 0.90,
                "locus_robust_empirical_p": 0.05 if idx == 0 else 0.90,
                "module_specific_rank_empirical_p": 0.05 if idx == 0 else 0.90,
                "positive_locus_empirical_p": 0.05 if idx == 0 else 0.90,
                "leave_top1_locus_empirical_p": 0.01,
                "leave_top3_locus_empirical_p": 0.01,
                "top_locus_conditioned_leave_top1_p": 0.01,
                "top_locus_conditioned_leave_top3_p": 0.01,
                "null_exact_match_rate": 1.0,
                "null_global_fallback_rate": 0.0,
                "null_reuse_fallback_rate": 0.0,
                "null_with_replacement_rate": 0.0,
                "min_match_pool_size": 12,
                "median_match_pool_size": 25,
                "null_loci_with_insufficient_gene_pool_rate": 0.0,
                "ld_block_locus_sensitivity_status": "external_locus_column_used",
                "pseudo_locus_window_stability_status": "not_tested",
                "fraction_loci_with_uncapped_score_gt_3": 0.0,
                "fraction_positive_signal_from_uncapped_gt_3": 0.0,
                "n_loci_in_genome_top_1pct": 0,
                "top5_locus_contribution": 0.50,
                "n_loci_with_uncapped_score_gt_3": 0,
            }
        )
    modules = pd.DataFrame(rows)

    out = add_ripple_d_v16_claim_readiness(modules, config)
    first = out.loc[out["module_name"].eq("M0")].iloc[0]

    assert bool(first["null_quality_balanced_pass"])
    assert not bool(first["multiplicity_pass"])
    assert first["v16b_claim_status"] == "exploratory_locus_distributed_candidate"


def test_external_locus_audit_flags_cross_chrom_and_unmapped_loci():
    work = pd.DataFrame(
        {
            "gene_symbol": ["A", "B", "C"],
            "chrom": [1, 2, 1],
            "gene_start": [100, 200, 500],
            "gene_end": [150, 250, 550],
            "locus_id": ["L1", "L1", "UNMAPPED:C"],
        }
    )

    audit = external_locus_audit_table(work, locus_id_column="ld_block")

    assert int(audit["n_cross_chrom_loci"].iloc[0]) == 1
    assert audit["unmapped_locus_fraction"].iloc[0] > 0
    assert not bool(audit["external_locus_audit_pass"].iloc[0])


def test_redundancy_collapse_marks_overlapping_supportive_module():
    modules = pd.DataFrame(
        [
            {
                "module_name": "A",
                "present_genes": "G1,G2,G3,G4",
                "n_present": 4,
                "ripple_d_v16_empirical_p": 0.001,
                "module_specific_rank_empirical_p": 0.001,
            },
            {
                "module_name": "B",
                "present_genes": "G1,G2,G3,G5",
                "n_present": 4,
                "ripple_d_v16_empirical_p": 0.01,
                "module_specific_rank_empirical_p": 0.01,
            },
        ]
    )

    out = add_module_redundancy_fields(modules)

    b = out.loc[out["module_name"].eq("B")].iloc[0]
    assert b["representative_module_in_cluster"] == "A"
    assert b["redundancy_downgrade_reason"] == "overlapping_supportive_module"
