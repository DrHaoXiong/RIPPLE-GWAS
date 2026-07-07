import networkx as nx
import numpy as np
import pandas as pd
import pytest

from ripple.graph import graph_laplacian
from ripple.gsp import band_energy_table, laplacian_eigendecomposition, project_graph_signal
from ripple.percolation import (
    classify_percolation_architecture,
    compute_degree_matched_node_percolation_null,
    percolation_auc,
    percolation_curve,
    prepare_degree_matched_rank_sets,
    rank_nodes_by_score,
    summarize_percolation_null,
)


def test_laplacian_eigendecomposition_and_signal_projection_on_path_graph():
    graph = nx.path_graph(["A", "B", "C"])
    result = graph_laplacian(graph, nodes=["A", "B", "C"])

    decomp = laplacian_eigendecomposition(result.laplacian, nodes=result.nodes, n_components=None)
    signal = project_graph_signal([1.0, 0.0, -1.0], decomp, laplacian=result.laplacian)

    assert decomp.method == "full"
    assert decomp.eigenvectors.shape == (3, 3)
    assert signal.retained_energy_fraction == pytest.approx(1.0)
    assert signal.smoothness > 0

    bands = band_energy_table(signal.eigenvalues, signal.coefficients)
    assert bands["energy"].sum() == pytest.approx(np.sum(signal.energy))


def test_band_energy_clamps_near_zero_negative_eigenvalue():
    bands = band_energy_table([-1e-12, 0.1], [1.0, 2.0])

    assert "unassigned" not in set(bands["band"])
    assert bands.loc[bands["band"] == "low", "energy"].iloc[0] == pytest.approx(5.0)


def test_ranked_percolation_curve_and_auc():
    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("D", "E")])
    scores = pd.DataFrame(
        {
            "gene_symbol": ["A", "B", "C", "D", "E"],
            "assoc_resid_score": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )

    ranking = rank_nodes_by_score(scores)
    curve = percolation_curve(graph, ranking, [0.2, 0.6, 1.0])
    auc = percolation_auc(curve)

    assert ranking["gene_symbol"].tolist() == ["A", "B", "C", "D", "E"]
    assert curve["n_selected"].tolist() == [1, 3, 5]
    assert curve["largest_component_size"].tolist() == [1, 3, 3]
    assert auc > 0


def test_summarize_percolation_null_reports_delta_z_and_empirical_p():
    null_auc = pd.DataFrame({"percolation_auc": [1.0, 2.0, 3.0]})
    summary = summarize_percolation_null(null_auc, observed_auc=4.0)

    assert summary["n_replicates"] == 3
    assert summary["mean"] == pytest.approx(2.0)
    assert summary["sd"] == pytest.approx(1.0)
    assert summary["delta"] == pytest.approx(2.0)
    assert summary["z"] == pytest.approx(2.0)
    assert summary["empirical_p_upper"] == pytest.approx(0.25)


def test_classify_percolation_architecture_degree_aware_topology_sensitive():
    result = classify_percolation_architecture(
        snp_permutation_null={"z": 4.0},
        degree_stratified_null={"z": 2.5},
        degree_matched_node_null={"z": 3.0},
        degree_preserving_graph_null={"z": -3.0},
    )

    assert result["architecture_class"] == "degree_aware_aggregation_topology_sensitive"
    assert result["degree_matched_node_positive"] is True
    assert result["degree_preserving_graph_negative"] is True


def test_degree_matched_node_percolation_null_outputs_auc_and_curves():
    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("E", "F")])
    scores = pd.DataFrame(
        {
            "gene_symbol": ["A", "B", "C", "D", "E", "F"],
            "assoc_resid_score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "graph_degree": [1, 2, 2, 1, 1, 1],
        }
    )
    ranking = rank_nodes_by_score(scores)
    selected_bin_counts, bin_to_nodes, profile = prepare_degree_matched_rank_sets(
        scores,
        ranking,
        [0.5, 1.0],
        n_bins=2,
    )

    auc, curves = compute_degree_matched_node_percolation_null(
        graph,
        selected_bin_counts,
        bin_to_nodes,
        n_replicates=4,
        seed=13,
    )

    assert profile["rank_fraction"].tolist() == [0.5, 1.0]
    assert auc.shape == (4, 2)
    assert curves["replicate"].nunique() == 4
