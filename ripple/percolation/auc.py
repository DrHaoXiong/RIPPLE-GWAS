"""Percolation curve and AUC statistics."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ripple.percolation.ranking import rank_fraction_grid, selected_nodes_at_fraction


def component_stats_for_nodes(graph, selected_nodes: Iterable[str]) -> dict[str, int | float]:
    """Compute selected-subgraph component statistics without materializing a subgraph."""

    selected = {str(node) for node in selected_nodes if graph.has_node(str(node))}
    n_selected = len(selected)
    if n_selected == 0:
        return {
            "n_selected": 0,
            "n_components": 0,
            "largest_component_size": 0,
            "largest_component_fraction": 0.0,
            "graph_node_fraction": 0.0,
        }

    visited: set[str] = set()
    largest_size = 0
    n_components = 0
    for start in selected:
        if start in visited:
            continue
        n_components += 1
        stack = [start]
        visited.add(start)
        component_size = 0
        while stack:
            node = stack.pop()
            component_size += 1
            for neighbor in graph.neighbors(node):
                neighbor = str(neighbor)
                if neighbor in selected and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        largest_size = max(largest_size, component_size)

    n_graph_nodes = graph.number_of_nodes()
    return {
        "n_selected": int(n_selected),
        "n_components": int(n_components),
        "largest_component_size": int(largest_size),
        "largest_component_fraction": float(largest_size / n_selected),
        "graph_node_fraction": float(n_selected / n_graph_nodes) if n_graph_nodes else 0.0,
    }


def percolation_curve(
    graph,
    ranking: pd.DataFrame,
    fractions: Iterable[float],
    *,
    node_col: str = "gene_symbol",
) -> pd.DataFrame:
    """Compute largest selected-component statistics across rank fractions."""

    grid = rank_fraction_grid(fractions)
    rows: list[dict[str, object]] = []
    graph_nodes = set(str(node) for node in graph.nodes())
    for fraction in grid:
        selected = [node for node in selected_nodes_at_fraction(ranking, fraction, node_col=node_col) if node in graph_nodes]
        stats = component_stats_for_nodes(graph, selected)
        rows.append(
            {
                "rank_fraction": float(fraction),
                **stats,
            }
        )
    return pd.DataFrame(rows)


def percolation_auc(curve: pd.DataFrame, *, x_col: str = "rank_fraction", y_col: str = "largest_component_fraction") -> float:
    """Compute trapezoidal AUC over a percolation curve."""

    missing = [col for col in (x_col, y_col) if col not in curve.columns]
    if missing:
        raise ValueError(f"Missing curve columns: {missing}")
    if curve.empty:
        return 0.0
    ordered = curve.sort_values(x_col)
    x = ordered[x_col].to_numpy(dtype=float)
    y = ordered[y_col].to_numpy(dtype=float)
    return float(np.trapz(y, x))
