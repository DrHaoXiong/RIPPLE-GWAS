"""Mutually exclusive, auditable V1.7 nomination states."""

from __future__ import annotations

import numpy as np
import pandas as pd


NOMINATION_STATES = (
    "D0_negative_or_not_tested",
    "D1_full_library_fdr_candidate",
    "D2_replicated_fixed_panel_candidate",
    "D3_single_cohort_panel_supported",
    "D4_hypothesis_prioritized_pattern",
    "D5_multi_strong_locus_pathway_overlap",
)


def _bool_column(table: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in table:
        return pd.Series(default, index=table.index, dtype=bool)
    return table[name].fillna(default).astype(bool)


def _numeric_column(table: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in table:
        return pd.Series(default, index=table.index, dtype=float)
    return pd.to_numeric(table[name], errors="coerce")


def derive_leave_topk_pass(
    table: pd.DataFrame,
    *,
    supportive_p_max: float,
    p_column: str = "top_conditioned_leave_top1_positive_burden_empirical_p",
) -> pd.Series:
    """Fail closed unless the confirmation-stage leave-top-1 test is supportive."""

    if not 0.0 < supportive_p_max <= 1.0:
        raise ValueError("supportive_p_max must be in (0, 1].")
    confirm_column = f"{p_column}_confirm"
    source = confirm_column if confirm_column in table else p_column
    return _numeric_column(table, source).lt(supportive_p_max).fillna(False)


def nominate_v17(
    table: pd.DataFrame,
    *,
    q_max: float = 0.10,
    full_library_q_column: str = "v17_complete_family_bh_q",
    fixed_panel_q_column: str = "v17_fixed_panel_bh_q",
    replication_q_column: str = "v17_partial_conjunction_bh_q",
) -> pd.DataFrame:
    """Assign one D0-D5 state and preserve every eligibility downgrade reason.

    Required statistical inputs are q-value columns above. Optional Boolean
    inputs are ``tested``, ``selected_from_same_trait``, ``top_tail_pass``,
    ``external_locus_audit_pass``, ``null_quality_pass``,
    ``hypothesis_prioritized_pattern``, and ``multi_strong_locus_pathway_overlap``.
    Weak eligibility additionally requires ``n_loci >= 5``,
    ``n_effective_loci >= 5``, and ``top1_locus_contribution <= .35``. These
    gates classify nominations only; they are never multiplied into p values.
    """

    if not 0.0 < q_max <= 1.0:
        raise ValueError("q_max must be in (0, 1].")
    out = table.copy()
    tested = _bool_column(out, "tested", default=True)
    n_loci_pass = _numeric_column(out, "n_loci").ge(5).fillna(False)
    effective_loci_pass = _numeric_column(out, "n_effective_loci").ge(5).fillna(False)
    top1_pass = _numeric_column(out, "top1_locus_contribution").le(0.35).fillna(False)
    top_tail_pass = _bool_column(out, "top_tail_pass")
    external_pass = _bool_column(out, "external_locus_audit_pass")
    null_pass = _bool_column(out, "null_quality_pass")
    leave_top_pass = _bool_column(out, "leave_topk_pass", default=False)
    weak_eligibility = (
        n_loci_pass & effective_loci_pass & top1_pass & top_tail_pass
        & external_pass & null_pass & leave_top_pass
    )

    full_q_pass = _numeric_column(out, full_library_q_column).le(q_max).fillna(False)
    panel_q_pass = _numeric_column(out, fixed_panel_q_column).le(q_max).fillna(False)
    replication_q_pass = _numeric_column(out, replication_q_column).le(q_max).fillna(False)
    selected_same_trait = _bool_column(out, "selected_from_same_trait")
    stage = out.get("selection_stage", pd.Series("none", index=out.index)).astype(str)
    selected_same_trait = selected_same_trait | stage.eq("selected_from_same_trait")
    discovery_q_valid = _bool_column(out, "q_value_valid_for_discovery", default=True)
    resolution_pass = _bool_column(out, "empirical_resolution_pass", default=True)
    positive_in_both = _bool_column(out, "positive_in_both_cohorts", default=True)
    locus_support_compatible = _bool_column(out, "locus_support_compatible", default=True)
    hypothesis_pattern = _bool_column(out, "hypothesis_prioritized_pattern")
    pathway_overlap = _bool_column(out, "multi_strong_locus_pathway_overlap")

    role = out.get("library_role", pd.Series("", index=out.index)).astype(str)
    is_broad = role.eq("broad_discovery")
    is_fixed_panel = role.eq("fixed_panel")
    is_replication = role.eq("replication")
    d1 = (
        tested & is_broad & full_q_pass & weak_eligibility & ~selected_same_trait
        & discovery_q_valid & resolution_pass
    )
    d2 = (
        tested & is_replication & ~d1 & panel_q_pass & replication_q_pass
        & weak_eligibility & resolution_pass & positive_in_both & locus_support_compatible
    )
    d3 = tested & is_fixed_panel & ~d1 & ~d2 & panel_q_pass & weak_eligibility
    # D5 takes precedence over a nominal D4 pattern: strong-locus dominance is
    # an interpretation boundary, not a route to a weak-signal claim.
    d5 = tested & ~d1 & ~d2 & ~d3 & pathway_overlap
    d4 = tested & ~d1 & ~d2 & ~d3 & ~d5 & hypothesis_pattern
    out["v17_n_loci_pass"] = n_loci_pass
    out["v17_n_effective_loci_pass"] = effective_loci_pass
    out["v17_top1_locus_contribution_pass"] = top1_pass
    out["v17_top_tail_pass"] = top_tail_pass
    out["v17_external_locus_audit_pass"] = external_pass
    out["v17_null_quality_pass"] = null_pass
    out["v17_leave_topk_pass"] = leave_top_pass
    out["v17_empirical_resolution_pass"] = resolution_pass
    out["v17_positive_in_both_cohorts"] = positive_in_both
    out["v17_locus_support_compatible"] = locus_support_compatible
    out["v17_q_value_valid_for_discovery"] = discovery_q_valid
    out["v17_weak_eligibility_pass"] = weak_eligibility
    out["v17_nomination_state"] = np.select(
        (d1, d2, d3, d4, d5), NOMINATION_STATES[1:], default=NOMINATION_STATES[0]
    )

    reasons: list[str] = []
    for index in out.index:
        state = out.loc[index, "v17_nomination_state"]
        failed: list[str] = []
        if not bool(tested.loc[index]):
            failed.append("not_tested")
        if bool(selected_same_trait.loc[index]):
            failed.append("selected_from_same_trait_not_eligible_for_D1")
        for passed, reason in (
            (n_loci_pass.loc[index], "n_loci_lt_5"),
            (effective_loci_pass.loc[index], "n_effective_loci_lt_5"),
            (top1_pass.loc[index], "top1_locus_contribution_gt_0.35"),
            (top_tail_pass.loc[index], "top_tail_failure"),
            (external_pass.loc[index], "external_locus_audit_failure"),
            (null_pass.loc[index], "null_quality_failure"),
            (leave_top_pass.loc[index], "leave_topk_sensitivity_failure"),
            (resolution_pass.loc[index], "insufficient_empirical_p_resolution"),
            (positive_in_both.loc[index], "non_positive_or_opposing_replication_direction"),
            (locus_support_compatible.loc[index], "material_locus_support_contradiction"),
        ):
            if not bool(passed):
                failed.append(reason)
        if state == "D0_negative_or_not_tested" and not failed:
            failed.append("no_qualifying_nomination_evidence")
        reasons.append(";".join(failed) if failed else "none")
    out["v17_nomination_downgrade_reason"] = reasons
    return out
