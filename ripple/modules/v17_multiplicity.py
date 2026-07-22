"""Auditable V1.7 multiplicity correction for complete testing families."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Return BH-adjusted p values while preserving missing values."""

    p_values_array = np.asarray(p_values, dtype=float)
    q_values = np.full(p_values_array.shape, np.nan, dtype=float)
    finite = np.isfinite(p_values_array)
    if not finite.any():
        return q_values

    observed = p_values_array[finite]
    order = np.argsort(observed, kind="mergesort")
    ranked = observed[order]
    adjusted = np.empty(len(ranked), dtype=float)
    running_minimum = 1.0
    for index in range(len(ranked) - 1, -1, -1):
        candidate = float(ranked[index] * len(ranked) / (index + 1))
        running_minimum = min(running_minimum, candidate)
        adjusted[index] = running_minimum
    restored = np.empty(len(ranked), dtype=float)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    q_values[finite] = restored
    return q_values


def _require_columns(table: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in table]
    if missing:
        raise ValueError(f"Input table is missing required columns: {', '.join(missing)}")


def _resolution_audit(n_null: pd.Series, n_tests: int, q_max: float) -> pd.DataFrame:
    null_counts = pd.to_numeric(n_null, errors="coerce")
    minimum_p = 1.0 / (null_counts + 1.0)
    threshold = q_max / n_tests if n_tests else np.nan
    passes = (
        (minimum_p <= threshold).fillna(False) if n_tests else pd.Series(False, index=n_null.index)
    )
    return pd.DataFrame(
        {
            "minimum_resolvable_empirical_p": minimum_p,
            "resolution_target_p": threshold,
            "empirical_resolution_pass": passes,
        },
        index=n_null.index,
    )


def complete_family_bh(
    table: pd.DataFrame,
    *,
    p_column: str,
    eligible_column: str = "multiplicity_eligible",
    n_null_column: str = "n_null",
    q_max: float = 0.10,
    prefix: str = "v17_complete_family",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply BH across every eligible row and retain a resolution audit.

    Eligibility describes membership in the predeclared testing family. Rows with
    missing or invalid p values remain auditable but are not correction tests.
    """

    _require_columns(table, (p_column, eligible_column, n_null_column))
    if not 0.0 < q_max <= 1.0:
        raise ValueError("q_max must be in (0, 1].")

    out = table.copy()
    eligible = out[eligible_column].fillna(False).astype(bool)
    p_values = pd.to_numeric(out[p_column], errors="coerce")
    tested = eligible & p_values.notna() & np.isfinite(p_values) & p_values.between(0.0, 1.0)
    n_tests = int(tested.sum())
    q_column = f"{prefix}_bh_q"
    out[f"{prefix}_eligible"] = eligible
    out[f"{prefix}_tested"] = tested
    out[f"{prefix}_n_tests"] = n_tests
    out[q_column] = np.nan
    out.loc[tested, q_column] = bh_fdr(p_values.loc[tested].to_numpy(dtype=float))
    out[f"{prefix}_pass"] = (out[q_column] <= q_max).fillna(False)

    audit = _resolution_audit(out[n_null_column], n_tests, q_max)
    out = out.join(audit.add_prefix(f"{prefix}_"))
    reasons: list[str] = []
    for index in out.index:
        row_reasons: list[str] = []
        if not bool(eligible.loc[index]):
            row_reasons.append("not_in_complete_testing_family")
        elif not bool(tested.loc[index]):
            row_reasons.append("missing_or_invalid_p_value")
        if bool(tested.loc[index]) and not bool(
            out.loc[index, f"{prefix}_empirical_resolution_pass"]
        ):
            row_reasons.append("insufficient_empirical_p_resolution")
        if bool(tested.loc[index]) and not bool(out.loc[index, f"{prefix}_pass"]):
            row_reasons.append("bh_not_significant")
        reasons.append(";".join(row_reasons) or "none")
    out[f"{prefix}_downgrade_reason"] = reasons
    return out, {
        "n_input_rows": int(len(out)),
        "n_eligible_rows": int(eligible.sum()),
        "n_correction_tests": n_tests,
        "q_max": float(q_max),
        "resolution_target_p": q_max / n_tests if n_tests else float("nan"),
    }


def partial_conjunction_bh(
    table: pd.DataFrame,
    *,
    p1_column: str,
    p2_column: str,
    eligible_column: str = "multiplicity_eligible",
    n_null_column: str = "n_null",
    q_max: float = 0.10,
    prefix: str = "v17_partial_conjunction",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply BH to the two-cohort partial-conjunction p value ``max(p1, p2)``."""

    _require_columns(table, (p1_column, p2_column, eligible_column, n_null_column))
    out = table.copy()
    p1 = pd.to_numeric(out[p1_column], errors="coerce")
    p2 = pd.to_numeric(out[p2_column], errors="coerce")
    valid_pair = p1.between(0.0, 1.0) & p2.between(0.0, 1.0)
    out[f"{prefix}_p"] = np.maximum(p1, p2).where(valid_pair)
    out, summary = complete_family_bh(
        out,
        p_column=f"{prefix}_p",
        eligible_column=eligible_column,
        n_null_column=n_null_column,
        q_max=q_max,
        prefix=prefix,
    )
    summary["partial_conjunction"] = "max(p1,p2)"
    return out, summary


def independent_screen_confirm_bh(
    table: pd.DataFrame,
    *,
    screen_p_column: str,
    confirm_p_column: str,
    selected_column: str = "screen_selected",
    eligible_column: str = "multiplicity_eligible",
    declared_family_size: int | None = None,
    q_max: float = 0.10,
    prefix: str = "v17_two_stage",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply full-family BH after an independent Monte Carlo screen.

    Modules are screened with one null stream. Confirmation P values must be
    generated from a distinct, independent null stream. Nonselected eligible
    modules receive final P=1. If the registered family also contains modules
    that cannot be tested after score/coverage QC, ``declared_family_size``
    retains them as virtual P=1 hypotheses in the correction denominator.
    """

    _require_columns(table, (screen_p_column, confirm_p_column, selected_column, eligible_column))
    out = table.copy()
    eligible = out[eligible_column].fillna(False).astype(bool)
    selected = out[selected_column].fillna(False).astype(bool) & eligible
    confirm_p = pd.to_numeric(out[confirm_p_column], errors="coerce")
    missing_selected = selected & (~confirm_p.between(0.0, 1.0))
    if missing_selected.any():
        raise ValueError("selected modules require finite confirmation P values")
    final_p = pd.Series(np.nan, index=out.index, dtype=float)
    final_p.loc[eligible & ~selected] = 1.0
    final_p.loc[selected] = confirm_p.loc[selected]
    out[f"{prefix}_final_p"] = final_p
    out[f"{prefix}_screen_selected"] = selected
    out[f"{prefix}_selection_rule"] = (
        "independent_screen_then_confirmation;unselected_final_p_equals_1"
    )
    out[f"{prefix}_bh_q"] = np.nan
    tested = eligible & final_p.notna()
    n_table_tests = int(tested.sum())
    family_size = n_table_tests if declared_family_size is None else int(declared_family_size)
    if family_size < n_table_tests:
        raise ValueError("declared_family_size cannot be smaller than tested table rows")
    tested_values = final_p.loc[tested].to_numpy(dtype=float)
    virtual_untestable = family_size - n_table_tests
    correction_values = np.concatenate([tested_values, np.ones(virtual_untestable, dtype=float)])
    out.loc[tested, f"{prefix}_bh_q"] = bh_fdr(correction_values)[:n_table_tests]
    out[f"{prefix}_pass"] = (out[f"{prefix}_bh_q"] <= q_max).fillna(False)
    return out, {
        "n_eligible": int(eligible.sum()),
        "n_screen_selected": int(selected.sum()),
        "n_table_bh_tests": n_table_tests,
        "n_virtual_untestable_hypotheses": virtual_untestable,
        "n_final_bh_tests": family_size,
        "q_max": float(q_max),
        "selection_requires_independent_confirmation": True,
    }
