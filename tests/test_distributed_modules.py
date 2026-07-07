import numpy as np
import pandas as pd

from ripple.modules.anchored import AnchoredModuleLibrary, empirical_upper
from ripple.modules.distributed import (
    RippleDConfig,
    assign_pseudo_loci,
    classify_distributed_module,
    contribution_metrics,
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


def test_contribution_metrics_detect_single_locus_dominance():
    metrics = contribution_metrics(np.array([3.0, 0.2, 0.1]))

    assert metrics["n_effective_loci"] < 2
    assert metrics["top1_locus_contribution"] > 0.85


def test_distributed_gate_classifies_sparse_and_distributed_rows():
    config = RippleDConfig()
    distributed = {
        "locus_robust_empirical_p": 0.02,
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

