"""Claim-tier utilities for RIPPLE-GWAS V1 reporting."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ripple.policy import final_z_threshold, supportive_z_threshold

SUPPORTIVE_Z_THRESHOLD = supportive_z_threshold()
FINAL_MANUSCRIPT_Z_THRESHOLD = final_z_threshold()

CLAIM_TIER_INTERPRETATIONS: dict[str, str] = {
    "TIER_0_gene_signal": "Gene-level weak association signal exists beyond SNP/gene-score null.",
    "TIER_1_degree_calibrated_aggregation": (
        "High residualized gene scores aggregate on a graph beyond degree-aware node nulls."
    ),
    "TIER_2_graph_domain_aggregation": (
        "Residualized weak-signal genes show graph-domain organization under diffusion or spectral statistics."
    ),
    "TIER_3_topology_specific_support": (
        "Observed biological graph topology outperforms degree-preserving or claim-specific graph nulls."
    ),
    "TIER_4_local_calibrated_modules": (
        "Local modules pass size-matched, degree-matched, and selection-aware module nulls."
    ),
}


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _summary_p(summary: Mapping[str, Any]) -> float:
    return _finite_float(summary.get("empirical_p", summary.get("empirical_p_upper")))


def _summary_row(
    *,
    trait: str,
    graph_name: str,
    score_type: str,
    statistic: str,
    tier: str,
    observed: float,
    summary: Mapping[str, Any],
    passed: bool,
    supportive_passed: bool,
    claim_label: str,
    final_z_threshold: float,
    supportive_z_threshold: float,
    interpretation: str | None = None,
) -> dict[str, object]:
    return {
        "trait": trait,
        "graph_name": graph_name,
        "score_type": score_type,
        "statistic": statistic,
        "tier": tier,
        "observed": _finite_float(observed),
        "null_mean": _finite_float(summary.get("mean", summary.get("null_mean"))),
        "null_sd": _finite_float(summary.get("sd", summary.get("null_sd"))),
        "z": _finite_float(summary.get("z")),
        "empirical_p": _summary_p(summary),
        "passed": bool(passed),
        "supportive_passed": bool(supportive_passed),
        "final_z_threshold": float(final_z_threshold),
        "supportive_z_threshold": float(supportive_z_threshold),
        "claim_label": claim_label,
        "interpretation": interpretation or CLAIM_TIER_INTERPRETATIONS[tier],
    }


def _passes_z(summary: Mapping[str, Any], *, z_threshold: float) -> bool:
    z = _finite_float(summary.get("z"))
    return bool(np.isfinite(z) and z >= z_threshold)


def build_claim_tier_table(
    *,
    trait: str,
    graph_name: str,
    observed_percolation_auc: float,
    snp_permutation_null: Mapping[str, Any],
    degree_matched_node_null: Mapping[str, Any],
    degree_preserving_graph_null: Mapping[str, Any],
    diffusion_summary: pd.DataFrame | None = None,
    local_module_summary: Mapping[str, Any] | None = None,
    score_type: str = "assoc_resid_score",
    z_threshold: float = FINAL_MANUSCRIPT_Z_THRESHOLD,
    supportive_z_threshold: float = SUPPORTIVE_Z_THRESHOLD,
) -> pd.DataFrame:
    """Build a standardized claim-tier table for one trait-graph analysis."""

    rows: list[dict[str, object]] = []
    gene_passed = _passes_z(snp_permutation_null, z_threshold=z_threshold)
    gene_supportive = _passes_z(snp_permutation_null, z_threshold=supportive_z_threshold)
    rows.append(
        _summary_row(
            trait=trait,
            graph_name=graph_name,
            score_type=score_type,
            statistic="percolation_auc_snp_pipeline_null",
            tier="TIER_0_gene_signal",
            observed=observed_percolation_auc,
            summary=snp_permutation_null,
            passed=gene_passed,
            supportive_passed=gene_supportive,
            final_z_threshold=z_threshold,
            supportive_z_threshold=supportive_z_threshold,
            claim_label="gene_level_weak_signal_detected"
            if gene_passed
            else "gene_level_weak_signal_not_detected",
        )
    )

    degree_passed = _passes_z(degree_matched_node_null, z_threshold=z_threshold)
    degree_supportive = _passes_z(degree_matched_node_null, z_threshold=supportive_z_threshold)
    rows.append(
        _summary_row(
            trait=trait,
            graph_name=graph_name,
            score_type=score_type,
            statistic="degree_calibrated_top_rank_aggregation",
            tier="TIER_1_degree_calibrated_aggregation",
            observed=observed_percolation_auc,
            summary=degree_matched_node_null,
            passed=degree_passed,
            supportive_passed=degree_supportive,
            final_z_threshold=z_threshold,
            supportive_z_threshold=supportive_z_threshold,
            claim_label="degree_calibrated_aggregation_detected"
            if degree_passed
            else "degree_calibrated_aggregation_not_detected",
        )
    )

    graph_passed = _passes_z(degree_preserving_graph_null, z_threshold=z_threshold)
    graph_supportive = _passes_z(degree_preserving_graph_null, z_threshold=supportive_z_threshold)
    rows.append(
        _summary_row(
            trait=trait,
            graph_name=graph_name,
            score_type=score_type,
            statistic="degree_preserving_graph_percolation",
            tier="TIER_3_topology_specific_support",
            observed=observed_percolation_auc,
            summary=degree_preserving_graph_null,
            passed=graph_passed,
            supportive_passed=graph_supportive,
            final_z_threshold=z_threshold,
            supportive_z_threshold=supportive_z_threshold,
            claim_label="topology_specific_support_detected"
            if graph_passed
            else "topology_specific_support_not_detected",
            interpretation=(
                "Topology-specific support is conditional; it is not required for reporting gene-level or "
                "degree-calibrated aggregation."
            ),
        )
    )

    if diffusion_summary is not None and not diffusion_summary.empty:
        for item in diffusion_summary.to_dict(orient="records"):
            passed = _passes_z(item, z_threshold=z_threshold)
            supportive = _passes_z(item, z_threshold=supportive_z_threshold)
            rows.append(
                _summary_row(
                    trait=trait,
                    graph_name=str(item.get("graph_name", graph_name)),
                    score_type=score_type,
                    statistic=f"diffusion_kernel_Tmax_{item.get('null_type', 'null')}",
                    tier="TIER_2_graph_domain_aggregation",
                    observed=_finite_float(item.get("T_max")),
                    summary=item,
                    passed=passed,
                    supportive_passed=supportive,
                    final_z_threshold=z_threshold,
                    supportive_z_threshold=supportive_z_threshold,
                    claim_label="graph_domain_aggregation_detected"
                    if passed
                    else "graph_domain_aggregation_not_detected",
                )
            )

    modules = local_module_summary or {}
    n_modules = int(modules.get("n_calibrated_modules", 0) or 0)
    n_topology = int(modules.get("n_topology_specific_modules", 0) or 0)
    module_passed = n_modules > 0
    rows.append(
        {
            "trait": trait,
            "graph_name": graph_name,
            "score_type": score_type,
            "statistic": "local_module_count",
            "tier": "TIER_4_local_calibrated_modules",
            "observed": float(n_modules),
            "null_mean": float("nan"),
            "null_sd": float("nan"),
            "z": float("nan"),
            "empirical_p": float("nan"),
            "passed": bool(module_passed),
            "supportive_passed": bool(module_passed),
            "final_z_threshold": float("nan"),
            "supportive_z_threshold": float("nan"),
            "claim_label": "local_calibrated_modules_detected"
            if module_passed
            else "local_calibrated_modules_not_detected",
            "interpretation": (
                f"{n_modules} calibrated weak-signal modules and {n_topology} topology-specific modules "
                "were reportable after the global gate."
            ),
        }
    )
    return pd.DataFrame(rows)
