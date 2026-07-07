"""Graph Laplacian construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import networkx as nx
import numpy as np
from scipy import sparse

LaplacianKind = Literal["normalized", "unnormalized", "random_walk"]


@dataclass(frozen=True)
class LaplacianResult:
    """Laplacian and aligned graph matrices."""

    laplacian: sparse.csr_matrix
    adjacency: sparse.csr_matrix
    degree: np.ndarray
    nodes: tuple[str, ...]
    kind: LaplacianKind


def adjacency_matrix(
    graph: nx.Graph,
    *,
    nodes: Iterable[str] | None = None,
    weight: str | None = "weight",
) -> tuple[sparse.csr_matrix, tuple[str, ...]]:
    """Return a CSR adjacency matrix and node order."""

    node_order = tuple(str(node) for node in (nodes if nodes is not None else sorted(graph.nodes())))
    if not node_order:
        return sparse.csr_matrix((0, 0), dtype=float), node_order
    matrix = nx.to_scipy_sparse_array(
        graph,
        nodelist=list(node_order),
        weight=weight,
        dtype=float,
        format="csr",
    )
    matrix = sparse.csr_matrix(matrix)
    matrix = 0.5 * (matrix + matrix.T)
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    return matrix, node_order


def graph_laplacian(
    graph: nx.Graph,
    *,
    nodes: Iterable[str] | None = None,
    kind: LaplacianKind = "normalized",
    weight: str | None = "weight",
) -> LaplacianResult:
    """Construct an unnormalized, normalized, or random-walk graph Laplacian."""

    if kind not in {"normalized", "unnormalized", "random_walk"}:
        raise ValueError("kind must be 'normalized', 'unnormalized', or 'random_walk'.")

    adjacency, node_order = adjacency_matrix(graph, nodes=nodes, weight=weight)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    degree_matrix = sparse.diags(degree, format="csr")

    if kind == "unnormalized":
        lap = degree_matrix - adjacency
    elif kind == "normalized":
        inv_sqrt_degree = np.zeros_like(degree, dtype=float)
        positive = degree > 0
        inv_sqrt_degree[positive] = 1.0 / np.sqrt(degree[positive])
        d_inv_sqrt = sparse.diags(inv_sqrt_degree, format="csr")
        identity = sparse.eye(adjacency.shape[0], format="csr")
        lap = identity - d_inv_sqrt @ adjacency @ d_inv_sqrt
        if np.any(~positive):
            lap = lap.tolil()
            for idx in np.where(~positive)[0]:
                lap[idx, idx] = 0.0
            lap = lap.tocsr()
    else:
        inv_degree = np.zeros_like(degree, dtype=float)
        positive = degree > 0
        inv_degree[positive] = 1.0 / degree[positive]
        d_inv = sparse.diags(inv_degree, format="csr")
        identity = sparse.eye(adjacency.shape[0], format="csr")
        lap = identity - d_inv @ adjacency
        if np.any(~positive):
            lap = lap.tolil()
            for idx in np.where(~positive)[0]:
                lap[idx, idx] = 0.0
            lap = lap.tocsr()

    return LaplacianResult(
        laplacian=sparse.csr_matrix(lap),
        adjacency=adjacency,
        degree=degree.astype(float),
        nodes=node_order,
        kind=kind,
    )
