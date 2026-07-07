import networkx as nx
import numpy as np
import pandas as pd

from ripple.nulls import (
    assign_degree_bins,
    degree_matched_node_sample,
    degree_preserving_graph_replicates,
    degree_stratified_permuted_scores,
    graph_component_summary,
    load_degree_preserving_graph_cache,
)


def test_degree_stratified_permuted_scores_preserve_stratum_values():
    table = pd.DataFrame(
        {
            "score": [1.0, 2.0, 10.0, 20.0],
            "degree": [1, 1, 5, 5],
        }
    )

    bins = assign_degree_bins(table["degree"], n_bins=2)
    permuted = degree_stratified_permuted_scores(
        table,
        score_col="score",
        degree_col="degree",
        n_replicates=5,
        seed=7,
        n_bins=2,
    )

    assert bins.tolist() == [0, 0, 1, 1]
    assert permuted.shape == (5, 4)
    for row in permuted:
        assert sorted(row[:2]) == [1.0, 2.0]
        assert sorted(row[2:]) == [10.0, 20.0]


def test_degree_preserving_graph_replicates_preserve_degree_sequence():
    graph = nx.cycle_graph(["A", "B", "C", "D", "E", "F"])
    graph.add_edge("A", "D")
    expected_degrees = sorted(dict(graph.degree()).values())

    replicates = list(
        degree_preserving_graph_replicates(
            graph,
            n_replicates=3,
            seed=11,
            nswap_per_edge=0.5,
        )
    )

    assert len(replicates) == 3
    for null_graph in replicates:
        assert sorted(null_graph.nodes()) == sorted(graph.nodes())
        assert sorted(dict(null_graph.degree()).values()) == expected_degrees
        assert null_graph.number_of_edges() == graph.number_of_edges()
        summary = graph_component_summary(null_graph)
        assert summary["n_nodes"] == graph.number_of_nodes()
        assert np.isfinite(summary["largest_component_fraction"])


def test_degree_preserving_graph_cache_round_trip(tmp_path):
    graph = nx.cycle_graph(["A", "B", "C", "D", "E", "F"])
    graph.add_edge("A", "D")
    cache_path = tmp_path / "graph_nulls.npz"

    generated = list(
        degree_preserving_graph_replicates(
            graph,
            n_replicates=2,
            seed=11,
            nswap_per_edge=0.5,
            cache_path=cache_path,
        )
    )
    cached = list(load_degree_preserving_graph_cache(cache_path, n_replicates=2))

    assert cache_path.exists()
    assert len(generated) == len(cached) == 2
    for generated_graph, cached_graph in zip(generated, cached, strict=True):
        assert sorted(generated_graph.nodes()) == sorted(cached_graph.nodes())
        assert sorted(generated_graph.edges()) == sorted(cached_graph.edges())
        assert sorted(dict(generated_graph.degree()).values()) == sorted(dict(cached_graph.degree()).values())


def test_degree_matched_node_sample_preserves_selected_degree_bins():
    table = pd.DataFrame(
        {
            "gene_symbol": ["A", "B", "C", "D", "E", "F"],
            "graph_degree": [1, 2, 3, 10, 11, 12],
        }
    )
    selected = ["A", "D", "E"]

    sampled = degree_matched_node_sample(table, selected, n_bins=2, seed=17)

    bins = assign_degree_bins(table["graph_degree"], n_bins=2)
    node_to_bin = dict(zip(table["gene_symbol"], bins, strict=True))
    selected_profile = sorted(node_to_bin[node] for node in selected)
    sampled_profile = sorted(node_to_bin[node] for node in sampled)

    assert len(sampled) == len(selected)
    assert len(set(sampled)) == len(sampled)
    assert sampled_profile == selected_profile
