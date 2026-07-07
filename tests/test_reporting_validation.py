import pytest

from ripple.diagnostics import build_trait_suitability_diagnostic, render_trait_architecture_markdown
from ripple.validation.spikein import (
    SpikeinConfig,
    build_synthetic_modular_graph,
    run_synthetic_spikein_validation,
)


def test_trait_suitability_flags_broad_signal_without_network_showcase():
    summary = {
        "trait": "TEST_TRAIT",
        "n_lcc_scored_genes": 1000,
        "snp_permutation_null_summary": {"z": 2.6, "delta": 0.03},
        "degree_stratified_null_summary": {"z": 2.1, "delta": 0.02},
        "degree_matched_node_null_summary": {"z": 0.7, "delta": 0.004},
        "degree_preserving_graph_null_summary": {"z": -2.5, "delta": -0.01},
        "graph_coverage_report": {"largest_component_gene_fraction": 0.75},
        "p_clipping_summary": {"n_clipped": 70, "n_total": 1000, "fraction_clipped": 0.07},
        "percolation_architecture": {
            "architecture_class": "broad_genetic_signal_without_degree_aware_graph_aggregation",
            "interpretation": "Broad signal only.",
        },
        "gsp_retained_energy_fraction": 0.02,
    }

    diagnostic = build_trait_suitability_diagnostic(summary)
    summary["trait_suitability"] = diagnostic
    report = render_trait_architecture_markdown(summary)

    assert diagnostic["verdict"] == "broad_signal_but_not_network_showcase"
    assert diagnostic["degree_matched_signal_status"] == "weak_or_absent"
    assert diagnostic["topology_specificity_status"] == "topology_null_sensitive_negative"
    assert diagnostic["p_clipping_status"] == "elevated"
    assert "TEST_TRAIT RIPPLE Architecture Report" in report
    assert "broad_signal_but_not_network_showcase" in report


def test_synthetic_module_spikein_exceeds_dispersed_degree_matched_signal():
    graph = build_synthetic_modular_graph(n_modules=4, module_size=18, seed=37)
    summary, _ = run_synthetic_spikein_validation(
        graph=graph,
        scenarios=(
            SpikeinConfig("dispersed", effect_size=2.0, target_fraction=0.15),
            SpikeinConfig("module", effect_size=2.0, target_fraction=0.15),
        ),
        seed=37,
        n_score_null=30,
        n_degree_stratified_null=30,
        n_degree_matched_node_null=50,
        n_degree_graph_null=20,
        degree_bins=5,
    )
    rows = summary.set_index("scenario")

    assert rows.loc["module", "degree_matched_z"] > rows.loc["dispersed", "degree_matched_z"]
    assert rows.loc["module", "degree_preserving_graph_z"] > rows.loc["dispersed", "degree_preserving_graph_z"]
    assert rows.loc["module", "architecture_class"] in {
        "topology_specific_module_excess",
        "degree_aware_network_aggregation",
        "degree_aware_aggregation_topology_sensitive",
    }
    assert rows.loc["module", "observed_auc"] == pytest.approx(rows.loc["module", "observed_auc"])
