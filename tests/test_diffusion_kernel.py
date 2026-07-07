import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ripple.graph import graph_laplacian
from ripple.graph_diffusion import (
    degree_stratified_diffusion_null,
    heat_kernel_tau_statistics,
    heat_kernel_tau_statistics_matrix,
    observed_diffusion_statistics,
    parse_tau_grid,
)


def diffusion_scores():
    return pd.DataFrame(
        {
            "gene_symbol": ["A", "B", "C", "D"],
            "assoc_resid_score": [3.0, 2.0, 0.0, -1.0],
            "graph_degree": [1, 2, 2, 1],
        }
    )


def test_parse_tau_grid_rejects_nonpositive_values():
    assert parse_tau_grid("0.25,0.5,1") == (0.25, 0.5, 1.0)
    with pytest.raises(ValueError, match="positive"):
        parse_tau_grid("0.5,0")


def test_weighted_and_unweighted_laplacian_differ_when_weights_present():
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=5.0)
    graph.add_edge("B", "C", weight=1.0)
    unweighted = graph_laplacian(graph, nodes=["A", "B", "C"], weight=None)
    weighted = graph_laplacian(graph, nodes=["A", "B", "C"], weight="weight")

    assert not np.allclose(unweighted.laplacian.toarray(), weighted.laplacian.toarray())


def test_heat_kernel_statistic_is_deterministic():
    graph = nx.path_graph(["A", "B", "C", "D"])
    lap = graph_laplacian(graph, nodes=["A", "B", "C", "D"], weight=None)
    first = heat_kernel_tau_statistics(lap.laplacian, [3.0, 2.0, 0.0, 0.0], tau_grid=[0.5, 1.0])
    second = heat_kernel_tau_statistics(lap.laplacian, [3.0, 2.0, 0.0, 0.0], tau_grid=[0.5, 1.0])

    pd.testing.assert_frame_equal(first, second)
    assert first["T_tau"].between(0, 1).all()


def test_heat_kernel_matrix_matches_single_vector_statistics():
    graph = nx.path_graph(["A", "B", "C", "D"])
    lap = graph_laplacian(graph, nodes=["A", "B", "C", "D"], weight=None)
    scores = np.array([[3.0, 2.0, 0.0, 0.0], [0.0, 2.0, 3.0, 1.0]], dtype=float)
    taus = [0.5, 1.0]

    batch = heat_kernel_tau_statistics_matrix(lap.laplacian, scores, tau_grid=taus, batch_size=2)
    first = heat_kernel_tau_statistics(lap.laplacian, scores[0], tau_grid=taus)
    second = heat_kernel_tau_statistics(lap.laplacian, scores[1], tau_grid=taus)

    assert batch.shape == (2, 2)
    np.testing.assert_allclose(batch[0], first["T_tau"].to_numpy(dtype=float))
    np.testing.assert_allclose(batch[1], second["T_tau"].to_numpy(dtype=float))


def test_degree_stratified_diffusion_null_outputs_summary_tau_and_null_tables():
    graph = nx.path_graph(["A", "B", "C", "D"])
    summary, tau, nulls = degree_stratified_diffusion_null(
        graph,
        diffusion_scores(),
        trait="TEST",
        graph_name="path",
        tau_grid=[0.5, 1.0],
        n_replicates=5,
        seed=7,
        n_bins=2,
    )

    assert summary.loc[0, "T_max"] >= 0
    assert summary.loc[0, "n_null"] == 5
    assert set(tau["tau"]) == {0.5, 1.0}
    assert nulls["replicate"].nunique() == 5
    assert {"T_tau", "T_max", "null_type"}.issubset(nulls.columns)


def test_observed_diffusion_uses_weights_only_when_requested():
    graph = nx.Graph()
    graph.add_edge("A", "B", weight=10.0)
    graph.add_edge("B", "C", weight=1.0)
    graph.add_edge("C", "D", weight=1.0)
    unweighted_tau, unweighted_summary = observed_diffusion_statistics(
        graph,
        diffusion_scores(),
        tau_grid=[1.0],
        weighted_laplacian=False,
    )
    weighted_tau, weighted_summary = observed_diffusion_statistics(
        graph,
        diffusion_scores(),
        tau_grid=[1.0],
        weighted_laplacian=True,
    )

    assert unweighted_summary["weighted_laplacian_used"] is False
    assert weighted_summary["weighted_laplacian_used"] is True
    assert unweighted_tau["T_tau"].iloc[0] != pytest.approx(weighted_tau["T_tau"].iloc[0])
