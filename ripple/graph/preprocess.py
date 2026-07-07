"""Graph preprocessing for RIPPLE V1 reference graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class GraphCoverageReport:
    """Coverage and component statistics after graph preprocessing."""

    n_input_edges: int
    n_input_nodes: int
    n_gene_universe: int | None
    n_graph_nodes: int
    n_graph_edges: int
    n_edge_covered_nodes: int
    n_connected_components: int
    largest_component_size: int
    largest_component_edges: int
    n_isolated_nodes: int
    graph_covered_gene_fraction: float | None
    largest_component_gene_fraction: float | None


@dataclass(frozen=True)
class PreprocessedGraph:
    """Full graph plus primary largest connected component."""

    full_graph: nx.Graph
    largest_component: nx.Graph
    component_table: pd.DataFrame
    coverage_report: GraphCoverageReport


def _require_columns(table: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [col for col in columns if col not in table.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def graph_from_edge_list(
    edges: pd.DataFrame,
    *,
    node1_col: str = "node1",
    node2_col: str = "node2",
    weight_col: str = "weight",
    gene_universe: Iterable[str] | None = None,
    drop_nonpositive_weights: bool = True,
) -> nx.Graph:
    """Build an undirected weighted NetworkX graph from a canonical edge list."""

    _require_columns(edges, [node1_col, node2_col, weight_col], "edges")
    graph = nx.Graph()
    universe = {str(gene) for gene in gene_universe} if gene_universe is not None else None
    if universe is not None:
        graph.add_nodes_from(sorted(universe))

    for _, row in edges.iterrows():
        node1 = str(row[node1_col])
        node2 = str(row[node2_col])
        weight = float(row[weight_col])
        if node1 == node2:
            continue
        if drop_nonpositive_weights and weight <= 0:
            continue
        if universe is not None and (node1 not in universe or node2 not in universe):
            continue
        if graph.has_edge(node1, node2):
            graph[node1][node2]["weight"] = max(float(graph[node1][node2]["weight"]), weight)
        else:
            graph.add_edge(node1, node2, weight=weight)
    return graph


def connected_component_table(graph: nx.Graph) -> pd.DataFrame:
    """Return one row per connected component, sorted by size descending."""

    rows: list[dict[str, object]] = []
    components = sorted(nx.connected_components(graph), key=lambda nodes: (-len(nodes), sorted(nodes)[0]))
    for idx, nodes in enumerate(components):
        subgraph = graph.subgraph(nodes)
        rows.append(
            {
                "component_id": idx,
                "n_nodes": subgraph.number_of_nodes(),
                "n_edges": subgraph.number_of_edges(),
                "is_largest_component": idx == 0,
            }
        )
    return pd.DataFrame(rows, columns=["component_id", "n_nodes", "n_edges", "is_largest_component"])


def largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component as a copied graph."""

    if graph.number_of_nodes() == 0:
        return graph.copy()
    nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(nodes).copy()


def preprocess_reference_graph(
    edges: pd.DataFrame,
    *,
    gene_universe: Iterable[str] | None = None,
    node1_col: str = "node1",
    node2_col: str = "node2",
    weight_col: str = "weight",
) -> PreprocessedGraph:
    """Build full and primary largest-component graphs with coverage metrics."""

    _require_columns(edges, [node1_col, node2_col, weight_col], "edges")
    universe_tuple = tuple(str(gene) for gene in gene_universe) if gene_universe is not None else None
    graph = graph_from_edge_list(
        edges,
        node1_col=node1_col,
        node2_col=node2_col,
        weight_col=weight_col,
        gene_universe=universe_tuple,
    )
    lcc = largest_connected_component(graph)
    components = connected_component_table(graph)

    n_universe = len(set(universe_tuple)) if universe_tuple is not None else None
    n_edge_covered_nodes = sum(1 for _, degree in graph.degree() if degree > 0)
    graph_covered_fraction = n_edge_covered_nodes / n_universe if n_universe is not None and n_universe > 0 else None
    largest_fraction = (
        lcc.number_of_nodes() / n_universe if n_universe is not None and n_universe > 0 else None
    )
    report = GraphCoverageReport(
        n_input_edges=int(len(edges)),
        n_input_nodes=int(
            len(set(edges[node1_col].astype(str)).union(set(edges[node2_col].astype(str))))
        ),
        n_gene_universe=n_universe,
        n_graph_nodes=int(graph.number_of_nodes()),
        n_graph_edges=int(graph.number_of_edges()),
        n_edge_covered_nodes=int(n_edge_covered_nodes),
        n_connected_components=int(nx.number_connected_components(graph)) if graph.number_of_nodes() else 0,
        largest_component_size=int(lcc.number_of_nodes()),
        largest_component_edges=int(lcc.number_of_edges()),
        n_isolated_nodes=int(nx.number_of_isolates(graph)),
        graph_covered_gene_fraction=graph_covered_fraction,
        largest_component_gene_fraction=largest_fraction,
    )
    return PreprocessedGraph(
        full_graph=graph,
        largest_component=lcc,
        component_table=components,
        coverage_report=report,
    )
