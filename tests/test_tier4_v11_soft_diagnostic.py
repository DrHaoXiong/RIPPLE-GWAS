import networkx as nx
import numpy as np
import pandas as pd

from ripple.nulls.score_permutation import assign_degree_bins
from scripts.run_tier4_v11_module_definition_diagnostic import (
    SearchContext,
    annulus_module_stat,
    best_soft_annulus_contrast_neighborhood,
    best_soft_connected_adaptive_neighborhood,
    get_neighborhood_layers,
    make_adjacency_index,
    method_nulls,
    precision_recall_jaccard,
)


def synthetic_hub_background_context() -> tuple[SearchContext, np.ndarray, set[int]]:
    graph = nx.Graph()
    module = [f"M{i}" for i in range(10)]
    module_annulus = [f"MA{i}" for i in range(35)]
    hub = "HUB"
    hub_inner = [f"HB{i}" for i in range(25)]
    hub_annulus = [f"HC{i}" for i in range(40)]
    graph.add_nodes_from(module + module_annulus + [hub] + hub_inner + hub_annulus)
    for left_idx, left in enumerate(module):
        for right in module[left_idx + 1 :]:
            graph.add_edge(left, right)
    for idx, gene in enumerate(module_annulus):
        graph.add_edge(module[idx % len(module)], gene)
    for gene in hub_inner:
        graph.add_edge(hub, gene)
    for idx, gene in enumerate(hub_annulus):
        graph.add_edge(hub_inner[idx % len(hub_inner)], gene)

    nodes = np.array(module + module_annulus + [hub] + hub_inner + hub_annulus, dtype=object)
    node_to_idx = {str(node): idx for idx, node in enumerate(nodes)}
    values = np.zeros(len(nodes), dtype=float)
    for gene in module:
        values[node_to_idx[gene]] = 5.0
    values[node_to_idx[hub]] = 8.0
    for gene in hub_inner + hub_annulus:
        values[node_to_idx[gene]] = 8.0

    degrees = np.array([graph.degree(str(node)) for node in nodes], dtype=float)
    bins = assign_degree_bins(pd.Series(degrees), n_bins=4).to_numpy(dtype=int)
    bin_to_indices = {int(bin_id): np.flatnonzero(bins == bin_id) for bin_id in sorted(np.unique(bins))}
    adjacency = make_adjacency_index(graph, nodes, node_to_idx)
    context = SearchContext(
        graph=graph,
        nodes=nodes,
        node_to_idx=node_to_idx,
        degrees=degrees,
        bins=bins,
        bin_to_indices=bin_to_indices,
        adjacency=adjacency,
        communities=(),
        community_ids=(),
        oracle_library=(),
        oracle_library_ids=(),
        neighborhood_cache={},
        layer_cache={},
    )
    oracle = {node_to_idx[gene] for gene in module}
    return context, values, oracle


def test_annulus_contrast_recovers_compact_module_better_than_raw_soft_under_hub_background():
    context, values, oracle = synthetic_hub_background_context()

    raw = best_soft_connected_adaptive_neighborhood(
        values,
        context,
        n_seeds=10,
        radius=2,
        size_grid=(10, 25, 80),
    )
    annulus = best_soft_annulus_contrast_neighborhood(
        values,
        context,
        n_seeds=10,
        inner_radius=1,
        background_radius=2,
        min_background=5,
        size_grid=(10, 25, 80),
        edge_gain_weight=0.10,
        degree_penalty=0.05,
        seed_pool_size=len(values),
    )
    _, _, raw_jaccard = precision_recall_jaccard(raw.node_indices, oracle)
    _, _, annulus_jaccard = precision_recall_jaccard(annulus.node_indices, oracle)

    assert annulus_jaccard > raw_jaccard
    assert annulus_jaccard >= 0.5
    assert annulus.seed.startswith("M")


def test_annulus_contrast_adapts_size_for_small_module():
    context, values, oracle = synthetic_hub_background_context()

    annulus = best_soft_annulus_contrast_neighborhood(
        values,
        context,
        n_seeds=10,
        inner_radius=1,
        background_radius=2,
        min_background=5,
        size_grid=(10, 25, 80),
        edge_gain_weight=0.10,
        degree_penalty=0.05,
        seed_pool_size=len(values),
    )
    _, recall, _ = precision_recall_jaccard(annulus.node_indices, oracle)

    assert len(annulus.node_indices) in {10, 25}
    assert recall >= 0.5


def test_annulus_background_fallback_returns_finite_statistic():
    graph = nx.path_graph([f"N{i}" for i in range(40)])
    nodes = np.array([str(node) for node in graph.nodes()], dtype=object)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    degrees = np.array([graph.degree(node) for node in nodes], dtype=float)
    bins = assign_degree_bins(pd.Series(degrees), n_bins=3).to_numpy(dtype=int)
    bin_to_indices = {int(bin_id): np.flatnonzero(bins == bin_id) for bin_id in sorted(np.unique(bins))}
    adjacency = make_adjacency_index(graph, nodes, node_to_idx)
    context = SearchContext(
        graph=graph,
        nodes=nodes,
        node_to_idx=node_to_idx,
        degrees=degrees,
        bins=bins,
        bin_to_indices=bin_to_indices,
        adjacency=adjacency,
        communities=(),
        community_ids=(),
        oracle_library=(),
        oracle_library_ids=(),
        neighborhood_cache={},
        layer_cache={},
    )
    values = np.linspace(-1.0, 1.0, len(nodes))
    inner, background, source = get_neighborhood_layers(
        context,
        node_to_idx["N0"],
        inner_radius=1,
        background_radius=2,
        min_background=30,
    )
    stat = annulus_module_stat(values, inner[:5], background)

    assert source in {"degree_bin_fallback", "global_fallback"}
    assert len(background) > 0
    assert np.isfinite(stat)


def test_method_nulls_recomputes_annulus_selection_for_each_replicate():
    context, values, _ = synthetic_hub_background_context()
    groups = [np.flatnonzero(context.bins == bin_id) for bin_id in sorted(np.unique(context.bins))]
    nulls = method_nulls(
        values,
        np.zeros_like(values),
        context,
        groups,
        n_replicates=5,
        seed=9,
        n_seeds=10,
        radius=2,
        soft_max_size=80,
        soft_size_grid=(10, 25, 80),
        soft_inner_radius=1,
        soft_background_radius=2,
        soft_annulus_min_background=20,
        soft_edge_gain_weight=0.10,
        soft_degree_penalty=0.05,
        soft_seed_pool_size=len(values),
        diffusion_top_size=80,
        tau_grid=(0.5,),
    )

    assert "soft_annulus_contrast_neighborhood" in nulls
    assert len(nulls["soft_annulus_contrast_neighborhood"]) == 5
    assert np.isfinite(nulls["soft_annulus_contrast_neighborhood"]).all()
