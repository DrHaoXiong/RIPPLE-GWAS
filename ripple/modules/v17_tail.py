"""Adaptive Monte Carlo tail-refinement utilities for V1.7 procedures."""

from __future__ import annotations

import math

import pandas as pd


def empirical_exceedance_count(empirical_p: object, n_null: int) -> int | None:
    """Recover the integer exceedance count from a plus-one empirical P value."""

    try:
        value = float(empirical_p)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or n_null < 1:
        return None
    raw = value * (n_null + 1) - 1.0
    count = int(round(raw))
    if count < 0 or count > n_null or not math.isclose(raw, count, abs_tol=1e-7):
        return None
    return count


def tail_refinement_targets(
    confirmation: pd.DataFrame,
    *,
    n_null: int,
    max_exceedances: int,
) -> list[str]:
    """Select confirmation rows occupying the predeclared extreme MC tail."""

    if confirmation.empty or "confirm_p" not in confirmation:
        return []
    targets = []
    for row in confirmation[["module_name", "confirm_p"]].itertuples(index=False):
        count = empirical_exceedance_count(row.confirm_p, n_null)
        if count is not None and count <= max_exceedances:
            targets.append(str(row.module_name))
    return targets


def replace_with_tail_refinement(
    initial: pd.DataFrame,
    refined: pd.DataFrame,
    *,
    initial_n_null: int,
    refined_n_null: int,
) -> pd.DataFrame:
    """Replace targeted initial rows with independent higher-resolution estimates."""

    result = initial.copy().set_index("module_name", drop=False)
    result["confirm_initial_p"] = result["confirm_p"]
    result["confirm_initial_exceedances"] = [
        empirical_exceedance_count(value, initial_n_null) for value in result["confirm_p"]
    ]
    result["confirm_tail_refined"] = False
    result["confirm_n_null"] = initial_n_null
    if refined.empty:
        return result.reset_index(drop=True)

    tail = refined.copy().set_index("module_name", drop=False)
    unknown = sorted(set(tail.index) - set(result.index))
    if unknown:
        raise ValueError(f"tail refinement returned unknown modules: {unknown[:3]}")
    for module_name, row in tail.iterrows():
        for column, value in row.items():
            result.at[module_name, column] = value
        result.at[module_name, "confirm_tail_refined"] = True
        result.at[module_name, "confirm_n_null"] = refined_n_null
    return result.reset_index(drop=True)
