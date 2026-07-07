import networkx as nx
import numpy as np
import pandas as pd

from ripple.modules.anchored import (
    AnchoredModuleLibrary,
    anchored_module_tests,
    bh_fdr,
    build_louvain_anchor_library,
    empirical_upper,
    load_anchored_gene_set_library,
)


def anchored_test_graph() -> nx.Graph:
    graph = nx.Graph()
    module = [f"M{i}" for i in range(8)]
    hub_genes = [f"H{i}" for i in range(12)]
    background = [f"B{i}" for i in range(20)]
    graph.add_nodes_from(module + hub_genes + background)
    for idx, left in enumerate(module):
        for right in module[idx + 1 :]:
            graph.add_edge(left, right)
    graph.add_node("HUB")
    for gene in hub_genes + background[:8]:
        graph.add_edge("HUB", gene)
    for idx, gene in enumerate(background):
        graph.add_edge(gene, background[(idx + 1) % len(background)])
    return graph


def anchored_test_scores() -> pd.DataFrame:
    graph = anchored_test_graph()
    genes = sorted(str(node) for node in graph.nodes())
    values = []
    for gene in genes:
        if gene.startswith("M"):
            values.append(4.5)
        elif gene == "HUB" or gene.startswith("H"):
            values.append(2.0)
        else:
            values.append(0.0)
    return pd.DataFrame(
        {
            "gene_symbol": genes,
            "assoc_resid_score": values,
            "graph_degree": [graph.degree(gene) for gene in genes],
            "assoc_p_g": [0.5] * len(genes),
        }
    )


def test_anchored_module_tests_detects_fixed_module_with_familywise_null():
    graph = anchored_test_graph()
    scores = anchored_test_scores()
    library = AnchoredModuleLibrary(
        gene_sets={
            "oracle_module": {f"M{i}" for i in range(8)},
            "hub_background": {"HUB", *{f"H{i}" for i in range(12)}},
        },
        module_source={
            "oracle_module": "unit_test",
            "hub_background": "unit_test",
        },
        annotation_source_type={
            "oracle_module": "independent_external",
            "hub_background": "independent_external",
        },
    )

    modules, nulls, summary = anchored_module_tests(
        graph,
        scores,
        library,
        min_present=5,
        n_degree_matched=50,
        n_score_permutation=50,
        degree_bins=4,
        seed=7,
    )

    top = modules.iloc[0]
    assert top["module_name"] == "oracle_module"
    assert top["degree_matched_empirical_p"] <= 0.1
    assert "library_familywise_p" in modules.columns
    assert set(nulls["null_type"]) >= {
        "degree_matched_node_set",
        "degree_stratified_score_permutation_fixed_module",
        "degree_stratified_score_permutation_library_max",
    }
    assert summary["n_tested_modules"] == 2


def test_anchored_module_tests_marks_low_overlap_modules():
    graph = anchored_test_graph()
    scores = anchored_test_scores()
    library = AnchoredModuleLibrary(
        gene_sets={"missing_module": {"M1", "Z1", "Z2", "Z3"}},
        module_source={"missing_module": "unit_test"},
        annotation_source_type={"missing_module": "independent_external"},
    )

    modules, nulls, summary = anchored_module_tests(
        graph,
        scores,
        library,
        min_present=3,
        n_degree_matched=5,
        n_score_permutation=5,
        degree_bins=3,
        seed=11,
    )

    assert modules.loc[0, "module_status"] == "not_tested_low_overlap"
    assert modules.loc[0, "n_present"] == 1
    assert nulls.empty
    assert summary["n_tested_modules"] == 0


def test_louvain_anchor_library_records_graph_related_source_type():
    graph = anchored_test_graph()

    library = build_louvain_anchor_library(graph, min_size=3, max_size=20, seed=3)

    assert library.gene_sets
    assert set(library.module_source.values()) == {"graph_louvain_community"}
    assert set(library.annotation_source_type.values()) == {"graph_construction_related"}


def test_empirical_upper_and_bh_fdr_are_stable():
    assert empirical_upper(np.array([0.1, 0.2, 0.3]), 0.2) == 0.75

    q = bh_fdr([0.01, 0.04, np.nan, 0.03])

    assert np.isnan(q[2])
    assert q[0] <= q[1]
    assert np.nanmax(q) <= 1.0


def test_load_anchored_gene_set_library_preserves_tsv_metadata(tmp_path):
    path = tmp_path / "external.tsv"
    path.write_text(
        "\n".join(
            [
                "gene_set\tgene_symbol\tsource_database\tannotation_source_type",
                "reactome_ecm\tCOL4A1\tReactome\tindependent_external",
                "reactome_ecm\tCOL4A2\tReactome\tindependent_external",
                "go_angiogenesis\tVEGFA\tGene Ontology\tindependent_external",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    library = load_anchored_gene_set_library(path)

    assert library.gene_sets["reactome_ecm"] == {"COL4A1", "COL4A2"}
    assert library.module_source["reactome_ecm"] == "Reactome"
    assert library.annotation_source_type["go_angiogenesis"] == "independent_external"
