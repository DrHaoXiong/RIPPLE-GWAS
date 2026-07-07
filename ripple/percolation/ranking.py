"""Percolation ranking utilities."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def rank_nodes_by_score(
    table: pd.DataFrame,
    *,
    node_col: str = "gene_symbol",
    score_col: str = "assoc_resid_score",
    descending: bool = True,
) -> pd.DataFrame:
    """Return nodes sorted by score with one-based ranks."""

    missing = [col for col in (node_col, score_col) if col not in table.columns]
    if missing:
        raise ValueError(f"Missing ranking columns: {missing}")
    out = table.loc[:, [node_col, score_col]].dropna().copy()
    out[node_col] = out[node_col].astype(str)
    out[score_col] = pd.to_numeric(out[score_col], errors="raise").astype(float)
    out = out.sort_values([score_col, node_col], ascending=[not descending, True]).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    out["rank_fraction"] = out["rank"] / len(out) if len(out) else pd.Series(dtype=float)
    return out


def selected_nodes_at_fraction(
    ranking: pd.DataFrame,
    fraction: float,
    *,
    node_col: str = "gene_symbol",
) -> tuple[str, ...]:
    """Return top-ranked nodes at a rank fraction threshold."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    if node_col not in ranking.columns:
        raise ValueError(f"Missing node column: {node_col}")
    n = max(1, int(round(float(fraction) * len(ranking))))
    return tuple(ranking.iloc[:n][node_col].astype(str))


def rank_fraction_grid(values: Iterable[float]) -> tuple[float, ...]:
    """Validate and normalize rank-fraction thresholds."""

    grid = tuple(float(value) for value in values)
    if not grid:
        raise ValueError("rank fraction grid must not be empty.")
    if any(value <= 0 or value > 1 for value in grid):
        raise ValueError("rank fractions must be in (0, 1].")
    return tuple(sorted(dict.fromkeys(grid)))
