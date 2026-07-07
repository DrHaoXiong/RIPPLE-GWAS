import numpy as np
import pandas as pd
import pytest

from ripple.graph import (
    adjacency_matrix,
    connected_component_table,
    graph_from_edge_list,
    graph_laplacian,
    largest_connected_component,
    preprocess_reference_graph,
)


def test_graph_from_edge_list_filters_edges_and_keeps_max_duplicate_weight():
    edges = pd.DataFrame(
        {
            "node1": ["A", "B", "A", "D", "X", "C"],
            "node2": ["B", "A", "A", "E", "A", "D"],
            "weight": [0.2, 0.9, 1.0, -1.0, 1.0, 0.5],
        }
    )

    graph = graph_from_edge_list(edges, gene_universe=["A", "B", "C", "D", "E", "Z"])

    assert set(graph.nodes()) == {"A", "B", "C", "D", "E", "Z"}
    assert set(graph.edges()) == {("A", "B"), ("C", "D")}
    assert graph["A"]["B"]["weight"] == pytest.approx(0.9)
    assert "X" not in graph


def test_preprocess_reference_graph_reports_lcc_and_edge_coverage():
    edges = pd.DataFrame(
        {
            "node1": ["A", "B", "D", "A", "Y"],
            "node2": ["B", "C", "E", "A", "Z"],
            "weight": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )

    result = preprocess_reference_graph(
        edges,
        gene_universe=["A", "B", "C", "D", "E", "F"],
    )

    assert set(result.full_graph.nodes()) == {"A", "B", "C", "D", "E", "F"}
    assert set(result.largest_component.nodes()) == {"A", "B", "C"}
    assert result.component_table["n_nodes"].tolist() == [3, 2, 1]

    report = result.coverage_report
    assert report.n_input_edges == 5
    assert report.n_input_nodes == 7
    assert report.n_gene_universe == 6
    assert report.n_graph_nodes == 6
    assert report.n_graph_edges == 3
    assert report.n_edge_covered_nodes == 5
    assert report.n_connected_components == 3
    assert report.largest_component_size == 3
    assert report.largest_component_edges == 2
    assert report.n_isolated_nodes == 1
    assert report.graph_covered_gene_fraction == pytest.approx(5 / 6)
    assert report.largest_component_gene_fraction == pytest.approx(3 / 6)


def test_connected_component_table_and_largest_component_on_empty_graph():
    graph = graph_from_edge_list(pd.DataFrame({"node1": [], "node2": [], "weight": []}))

    table = connected_component_table(graph)
    largest = largest_connected_component(graph)

    assert table.empty
    assert list(table.columns) == [
        "component_id",
        "n_nodes",
        "n_edges",
        "is_largest_component",
    ]
    assert largest.number_of_nodes() == 0


def test_adjacency_matrix_uses_requested_node_order_and_symmetrizes():
    edges = pd.DataFrame(
        {
            "node1": ["A", "B"],
            "node2": ["B", "C"],
            "weight": [2.0, 1.0],
        }
    )
    graph = graph_from_edge_list(edges)

    adjacency, nodes = adjacency_matrix(graph, nodes=["C", "B", "A"])

    assert nodes == ("C", "B", "A")
    assert adjacency.toarray().tolist() == [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 2.0],
        [0.0, 2.0, 0.0],
    ]


def test_graph_laplacian_returns_default_normalized_laplacian():
    graph = graph_from_edge_list(pd.DataFrame({"node1": ["A"], "node2": ["B"], "weight": [2.0]}))

    result = graph_laplacian(graph, nodes=["A", "B"])

    assert result.kind == "normalized"
    assert result.nodes == ("A", "B")
    assert result.degree.tolist() == [2.0, 2.0]
    np.testing.assert_allclose(result.laplacian.toarray(), [[1.0, -1.0], [-1.0, 1.0]])


def test_graph_laplacian_supports_unnormalized_random_walk_and_isolates():
    graph = graph_from_edge_list(
        pd.DataFrame({"node1": ["A"], "node2": ["B"], "weight": [2.0]}),
        gene_universe=["A", "B", "C"],
    )

    unnormalized = graph_laplacian(graph, nodes=["A", "B", "C"], kind="unnormalized")
    random_walk = graph_laplacian(graph, nodes=["A", "B", "C"], kind="random_walk")
    normalized = graph_laplacian(graph, nodes=["A", "B", "C"], kind="normalized")

    np.testing.assert_allclose(
        unnormalized.laplacian.toarray(),
        [[2.0, -2.0, 0.0], [-2.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        random_walk.laplacian.toarray(),
        [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        normalized.laplacian.toarray(),
        [[1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
    )


def test_graph_laplacian_rejects_unknown_kind():
    graph = graph_from_edge_list(pd.DataFrame({"node1": ["A"], "node2": ["B"], "weight": [1.0]}))

    with pytest.raises(ValueError, match="kind must be"):
        graph_laplacian(graph, kind="bad")  # type: ignore[arg-type]
