import pandas as pd

from ripple.claims import build_claim_tier_table


def test_claim_tier_table_separates_degree_and_topology_claims():
    diffusion = pd.DataFrame(
        [
            {
                "graph_name": "string",
                "null_type": "degree_stratified",
                "T_max": 0.7,
                "null_mean": 0.5,
                "null_sd": 0.05,
                "z": 4.0,
                "empirical_p": 0.01,
                "passed": True,
            }
        ]
    )
    table = build_claim_tier_table(
        trait="DR",
        graph_name="string",
        observed_percolation_auc=0.12,
        snp_permutation_null={"mean": 0.08, "sd": 0.01, "z": 4.0, "empirical_p_upper": 0.01},
        degree_matched_node_null={"mean": 0.09, "sd": 0.01, "z": 3.0, "empirical_p_upper": 0.02},
        degree_preserving_graph_null={"mean": 0.13, "sd": 0.01, "z": -1.0, "empirical_p_upper": 0.80},
        diffusion_summary=diffusion,
        local_module_summary={"n_calibrated_modules": 2, "n_topology_specific_modules": 0},
    )

    rows = table.set_index("tier")
    assert rows.loc["TIER_1_degree_calibrated_aggregation", "passed"] == True  # noqa: E712
    assert rows.loc["TIER_1_degree_calibrated_aggregation", "claim_label"] == (
        "degree_calibrated_aggregation_detected"
    )
    assert rows.loc["TIER_3_topology_specific_support", "passed"] == False  # noqa: E712
    assert rows.loc["TIER_3_topology_specific_support", "claim_label"] == (
        "topology_specific_support_not_detected"
    )
    assert "TIER_2_graph_domain_aggregation" in set(table["tier"])
    assert rows.loc["TIER_4_local_calibrated_modules", "observed"] == 2.0


def test_claim_tier_table_separates_supportive_and_final_gates():
    table = build_claim_tier_table(
        trait="DR",
        graph_name="string",
        observed_percolation_auc=0.12,
        snp_permutation_null={"mean": 0.08, "sd": 0.01, "z": 2.2, "empirical_p_upper": 0.02},
        degree_matched_node_null={"mean": 0.09, "sd": 0.01, "z": 2.6, "empirical_p_upper": 0.01},
        degree_preserving_graph_null={"mean": 0.13, "sd": 0.01, "z": 1.0, "empirical_p_upper": 0.50},
        diffusion_summary=pd.DataFrame(),
        local_module_summary={"n_calibrated_modules": 0, "n_topology_specific_modules": 0},
    )

    rows = table.set_index("tier")
    assert rows.loc["TIER_0_gene_signal", "supportive_passed"] == True  # noqa: E712
    assert rows.loc["TIER_0_gene_signal", "passed"] == False  # noqa: E712
    assert rows.loc["TIER_1_degree_calibrated_aggregation", "supportive_passed"] == True  # noqa: E712
    assert rows.loc["TIER_1_degree_calibrated_aggregation", "passed"] == True  # noqa: E712
    assert rows.loc["TIER_1_degree_calibrated_aggregation", "final_z_threshold"] == 2.5
    assert rows.loc["TIER_1_degree_calibrated_aggregation", "supportive_z_threshold"] == 2.0
