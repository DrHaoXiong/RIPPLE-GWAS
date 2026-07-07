"""Score permutation nulls."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_degree_bins(degrees: pd.Series, *, n_bins: int = 10) -> pd.Series:
    """Assign degree quantile bins while preserving input row order."""

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    degree_values = pd.to_numeric(degrees, errors="raise")
    if degree_values.empty:
        return pd.Series(dtype=int, index=degrees.index)
    if degree_values.nunique(dropna=True) <= 1 or n_bins == 1:
        return pd.Series(0, index=degrees.index, dtype=int)

    n_effective_bins = min(int(n_bins), int(degree_values.nunique(dropna=True)))
    ranked = degree_values.rank(method="first")
    bins = pd.qcut(ranked, q=n_effective_bins, labels=False, duplicates="drop")
    return bins.astype(int)


def degree_stratified_permuted_scores(
    table: pd.DataFrame,
    *,
    score_col: str,
    degree_col: str,
    n_replicates: int,
    seed: int,
    n_bins: int = 10,
) -> np.ndarray:
    """Permute scores within graph-degree strata.

    The returned matrix has shape ``(n_replicates, n_rows)`` and preserves the
    input table row order.
    """

    if n_replicates < 0:
        raise ValueError("n_replicates must be nonnegative.")
    missing = [col for col in (score_col, degree_col) if col not in table.columns]
    if missing:
        raise ValueError(f"Missing permutation columns: {missing}")

    scores = pd.to_numeric(table[score_col], errors="raise").to_numpy(dtype=float)
    bins = assign_degree_bins(table[degree_col], n_bins=n_bins)
    groups = [np.flatnonzero(bins.to_numpy(dtype=int) == bin_id) for bin_id in sorted(bins.unique())]
    out = np.empty((n_replicates, len(scores)), dtype=float)
    rng = np.random.default_rng(seed)

    for replicate_idx in range(n_replicates):
        permuted = scores.copy()
        for group_idx in groups:
            if group_idx.size > 1:
                permuted[group_idx] = permuted[rng.permutation(group_idx)]
        out[replicate_idx, :] = permuted

    return out
