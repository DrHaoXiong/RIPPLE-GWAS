import networkx as nx
import pandas as pd

from ripple.modules import (
    calibrate_local_modules,
    discover_local_modules,
    load_gene_sets,
    pathway_subgraph_tests,
    selection_aware_module_null,
)
from ripple.validation.spikein import (
    SpikeinConfig,
    build_synthetic_modular_graph,
    run_synthetic_spikein_validation,
)


def module_test_graph():
    graph = nx.Graph()
    graph.add_edges_from(
        [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("D", "E"),
            ("A", "E"),
            ("A", "C"),
            ("F", "G"),
            ("G", "H"),
            ("H", "I"),
            ("I", "J"),
        ]
    )
    return graph


def module_test_scores():
    genes = list("ABCDEFGHIJ")
    return pd.DataFrame(
        {
            "gene_symbol": genes,
            "assoc_resid_score": [10, 9, 8, 7, 6, 1, 0, -1, -2, -3],
            "graph_degree": [3, 2, 3, 2, 2, 1, 2, 2, 2, 1],
            "assoc_p_g": [1e-4] * 10,
        }
    )


def test_discover_local_modules_extracts_top_connected_component():
    graph = module_test_graph()
    scores = module_test_scores()

    modules = discover_local_modules(
        graph,
        scores,
        fractions=[0.5, 1.0],
        min_module_size=5,
        min_subthreshold_genes=3,
        max_modules=5,
    )

    assert len(modules) >= 1
    first = modules.iloc[0]
    assert first["n_genes"] == 5
    assert set(first["module_genes"].split(",")) == set("ABCDE")
    assert first["n_subthreshold_genes"] == 5
    assert "A" in first["core_genes"]


def test_calibrate_local_modules_adds_degree_matched_statistics():
    graph = module_test_graph()
    scores = module_test_scores()
    modules = discover_local_modules(graph, scores, fractions=[0.5], min_module_size=5)

    calibrated, nulls = calibrate_local_modules(
        graph,
        scores,
        modules,
        n_random=20,
        n_degree_matched=20,
        n_degree_graph=5,
        n_selection_aware=20,
        degree_bins=3,
        seed=11,
    )

    assert "degree_matched_p" in calibrated.columns
    assert "selection_aware_score_p" in calibrated.columns
    assert "degree_preserving_graph_p" in calibrated.columns
    assert "module_claim_label" in calibrated.columns
    assert calibrated["degree_matched_p"].between(0, 1).all()
    assert calibrated["selection_aware_score_p"].between(0, 1).all()
    assert set(nulls["null_type"]) >= {
        "random_score",
        "degree_matched_score",
        "selection_aware_max_mean_score",
    }


def test_selection_aware_module_null_repeats_extraction_pipeline():
    graph = module_test_graph()
    scores = module_test_scores()

    nulls = selection_aware_module_null(
        graph,
        scores,
        fractions=[0.5],
        min_module_size=5,
        min_subthreshold_genes=3,
        n_replicates=10,
        degree_bins=3,
        seed=19,
    )

    assert len(nulls) == 10
    assert {"max_mean_score", "max_edge_density", "n_modules"}.issubset(nulls.columns)
    assert nulls["n_modules"].ge(0).all()


def test_pathway_subgraph_tests_handles_missing_genes_and_degree_null():
    graph = module_test_graph()
    scores = module_test_scores()
    gene_sets = {"top_module": {"A", "B", "C", "D", "E", "Z"}}

    table = pathway_subgraph_tests(
        graph,
        scores,
        gene_sets,
        n_random=20,
        n_degree_matched=20,
        degree_bins=3,
        seed=17,
    )

    assert table.loc[table["gene_set"] == "top_module", "n_present"].iloc[0] == 5
    assert table.loc[table["gene_set"] == "top_module", "n_missing"].iloc[0] == 1
    assert table.loc[table["gene_set"] == "top_module", "degree_matched_empirical_p"].iloc[0] <= 1


def test_load_gene_sets_reads_tsv(tmp_path):
    path = tmp_path / "sets.tsv"
    path.write_text("gene_set\tgene_symbol\nset1\tA\nset1\tB\nset2\tC\n", encoding="utf-8")

    gene_sets = load_gene_sets(path, include_default_dr_panel=False)

    assert gene_sets == {"set1": {"A", "B"}, "set2": {"C"}}


def test_synthetic_module_spikein_produces_local_modules():
    graph = build_synthetic_modular_graph(n_modules=4, module_size=18, seed=101)
    summary, _ = run_synthetic_spikein_validation(
        graph=graph,
        scenarios=(
            SpikeinConfig("null", effect_size=0.0, target_fraction=0.15),
            SpikeinConfig("module", effect_size=2.2, target_fraction=0.15),
        ),
        seed=101,
        n_score_null=25,
        n_degree_stratified_null=25,
        n_degree_matched_node_null=40,
        n_degree_graph_null=15,
        n_module_selection_aware_null=25,
        degree_bins=5,
    )
    rows = summary.set_index("scenario")

    assert rows.loc["module", "n_calibrated_modules"] >= rows.loc["null", "n_calibrated_modules"]
    assert rows.loc["module", "degree_matched_z"] > rows.loc["null", "degree_matched_z"]
