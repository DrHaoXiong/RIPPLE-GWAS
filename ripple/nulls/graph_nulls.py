"""Graph topology nulls."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from ripple.nulls.score_permutation import assign_degree_bins


def degree_preserving_graph_replicates(
    graph: nx.Graph,
    *,
    n_replicates: int,
    seed: int,
    nswap_per_edge: float = 1.0,
    max_tries_per_swap: float = 20.0,
    cache_path: Path | str | None = None,
) -> Iterator[nx.Graph]:
    """Yield graph null replicates from double-edge swaps.

    The node set and degree sequence are preserved. Edge attributes are not
    interpreted by RIPPLE percolation, so the returned graph is unweighted.
    """

    if n_replicates < 0:
        raise ValueError("n_replicates must be nonnegative.")
    if nswap_per_edge < 0:
        raise ValueError("nswap_per_edge must be nonnegative.")
    if max_tries_per_swap < 1:
        raise ValueError("max_tries_per_swap must be at least 1.")

    if cache_path is not None:
        path = Path(cache_path)
        if path.exists():
            yield from load_degree_preserving_graph_cache(path, n_replicates=n_replicates)
            return

    base = nx.Graph()
    base.add_nodes_from(str(node) for node in graph.nodes())
    base.add_edges_from((str(u), str(v)) for u, v in graph.edges() if str(u) != str(v))

    rng = np.random.default_rng(seed)
    n_edges = base.number_of_edges()
    nswap = int(round(n_edges * nswap_per_edge))
    max_tries = max(nswap + 1, int(round(nswap * max_tries_per_swap)))

    generated: list[nx.Graph] = []
    for _ in range(n_replicates):
        null_graph = base.copy()
        if n_edges >= 2 and nswap > 0:
            local_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            nx.double_edge_swap(null_graph, nswap=nswap, max_tries=max_tries, seed=local_seed)
        if cache_path is not None:
            generated.append(null_graph)
        yield null_graph

    if cache_path is not None:
        write_degree_preserving_graph_cache(Path(cache_path), generated, nodes=tuple(base.nodes()))


def write_degree_preserving_graph_cache(path: Path, graphs: Sequence[nx.Graph], *, nodes: Sequence[str]) -> None:
    """Write degree-preserving graph replicates as a compact indexed edge cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    node_array = np.asarray([str(node) for node in nodes], dtype=object)
    node_to_idx = {str(node): idx for idx, node in enumerate(node_array)}
    edge_blocks: list[np.ndarray] = []
    offsets = [0]
    for graph in graphs:
        edges = np.asarray(
            [
                (node_to_idx[str(u)], node_to_idx[str(v)])
                for u, v in graph.edges()
                if str(u) in node_to_idx and str(v) in node_to_idx and str(u) != str(v)
            ],
            dtype=np.int32,
        )
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=np.int32)
        edge_blocks.append(edges)
        offsets.append(offsets[-1] + len(edges))
    all_edges = np.vstack(edge_blocks) if edge_blocks else np.empty((0, 2), dtype=np.int32)
    np.savez_compressed(
        path,
        nodes=node_array,
        edges=all_edges,
        offsets=np.asarray(offsets, dtype=np.int64),
        n_replicates=np.asarray([len(graphs)], dtype=np.int64),
    )


def load_degree_preserving_graph_cache(path: Path, *, n_replicates: int | None = None) -> Iterator[nx.Graph]:
    """Yield degree-preserving graph replicates from an indexed edge cache."""

    with np.load(path, allow_pickle=True) as data:
        nodes = [str(node) for node in data["nodes"]]
        edges = np.asarray(data["edges"], dtype=np.int32)
        offsets = np.asarray(data["offsets"], dtype=np.int64)
    available = max(0, len(offsets) - 1)
    requested = available if n_replicates is None else int(n_replicates)
    if requested > available:
        raise ValueError(f"Graph cache contains {available} replicates, but {requested} were requested.")
    for idx in range(requested):
        start = int(offsets[idx])
        stop = int(offsets[idx + 1])
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from((nodes[int(u)], nodes[int(v)]) for u, v in edges[start:stop])
        yield graph


def graph_component_summary(graph: nx.Graph) -> dict[str, int | float]:
    """Return compact component diagnostics for a graph null replicate."""

    if graph.number_of_nodes() == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "n_components": 0,
            "largest_component_size": 0,
            "largest_component_fraction": 0.0,
        }

    component_sizes = [len(component) for component in nx.connected_components(graph)]
    largest = max(component_sizes) if component_sizes else 0
    return {
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges": int(graph.number_of_edges()),
        "n_components": int(len(component_sizes)),
        "largest_component_size": int(largest),
        "largest_component_fraction": float(largest / graph.number_of_nodes()),
    }


def degree_matched_node_sample(
    table: pd.DataFrame,
    selected_nodes: Sequence[str],
    *,
    node_col: str = "gene_symbol",
    degree_col: str = "graph_degree",
    n_bins: int = 10,
    seed: int | None = None,
) -> tuple[str, ...]:
    """Sample a node set with the same degree-bin profile as selected nodes."""

    missing = [col for col in (node_col, degree_col) if col not in table.columns]
    if missing:
        raise ValueError(f"Missing node sampling columns: {missing}")
    if not selected_nodes:
        return ()

    work = table.loc[:, [node_col, degree_col]].dropna().copy()
    work[node_col] = work[node_col].astype(str)
    if work[node_col].duplicated().any():
        duplicated = work.loc[work[node_col].duplicated(), node_col].iloc[0]
        raise ValueError(f"Node column must be unique; first duplicate: {duplicated}")

    selected = tuple(str(node) for node in selected_nodes)
    available = set(work[node_col])
    missing_selected = [node for node in selected if node not in available]
    if missing_selected:
        raise ValueError(f"Selected nodes absent from table: {missing_selected[:5]}")

    bins = assign_degree_bins(work[degree_col], n_bins=n_bins)
    node_to_bin = dict(zip(work[node_col], bins, strict=True))
    bin_to_nodes = {
        int(bin_id): work.loc[bins == bin_id, node_col].to_numpy(dtype=object)
        for bin_id in sorted(bins.unique())
    }
    selected_bins = pd.Series([node_to_bin[node] for node in selected]).value_counts(sort=False)
    rng = np.random.default_rng(seed)

    sampled: list[str] = []
    for bin_id, count in selected_bins.sort_index().items():
        candidates = bin_to_nodes[int(bin_id)]
        if int(count) > len(candidates):
            raise ValueError(f"Cannot sample {count} nodes from degree bin {bin_id} with {len(candidates)} nodes.")
        sampled.extend(str(node) for node in rng.choice(candidates, size=int(count), replace=False))
    rng.shuffle(sampled)
    return tuple(sampled)
