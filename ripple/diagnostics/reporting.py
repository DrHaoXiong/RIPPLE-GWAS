"""Trait suitability diagnostics and architecture report rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _null_summary(summary: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _as_mapping(summary.get(key, {}))


def _z(summary: Mapping[str, Any], key: str) -> float:
    return _finite_float(_null_summary(summary, key).get("z"))


def _delta(summary: Mapping[str, Any], key: str) -> float:
    return _finite_float(_null_summary(summary, key).get("delta"))


def _status_from_z(z_value: float, *, threshold: float, borderline: float) -> str:
    if z_value >= threshold:
        return "positive"
    if z_value >= borderline:
        return "borderline"
    return "weak_or_absent"


def build_trait_suitability_diagnostic(
    summary: Mapping[str, Any],
    *,
    z_threshold: float = 2.0,
    borderline_z: float = 1.5,
    min_lcc_gene_fraction: float = 0.70,
    clipping_warn_fraction: float = 0.05,
    clipping_high_fraction: float = 0.10,
) -> dict[str, Any]:
    """Build a compact, reusable trait suitability diagnostic.

    The diagnostic is intentionally conservative: RIPPLE is considered a network
    showcase only when signal exceeds the degree-matched node null, graph
    coverage is adequate, and P-value clipping is not high.
    """

    snp_z = _z(summary, "snp_permutation_null_summary")
    degree_strat_z = _z(summary, "degree_stratified_null_summary")
    degree_matched_z = _z(summary, "degree_matched_node_null_summary")
    graph_z = _z(summary, "degree_preserving_graph_null_summary")
    architecture = _as_mapping(summary.get("percolation_architecture", {}))
    graph_coverage = _as_mapping(summary.get("graph_coverage_report", {}))
    clipping = _as_mapping(summary.get("p_clipping_summary", {}))

    lcc_gene_fraction = _finite_float(graph_coverage.get("largest_component_gene_fraction"), default=0.0)
    clipped_fraction = _finite_float(clipping.get("fraction_clipped"), default=0.0)

    gene_signal_status = _status_from_z(snp_z, threshold=z_threshold, borderline=borderline_z)
    degree_signal_status = _status_from_z(degree_matched_z, threshold=z_threshold, borderline=borderline_z)
    degree_strat_status = _status_from_z(degree_strat_z, threshold=z_threshold, borderline=borderline_z)
    if graph_z >= z_threshold:
        topology_status = "topology_specific_excess"
    elif graph_z <= -z_threshold:
        topology_status = "topology_null_sensitive_negative"
    else:
        topology_status = "not_topology_specific"

    if lcc_gene_fraction >= min_lcc_gene_fraction:
        coverage_status = "adequate"
    elif lcc_gene_fraction >= 0.50:
        coverage_status = "partial"
    else:
        coverage_status = "low"

    if clipped_fraction >= clipping_high_fraction:
        clipping_status = "high"
    elif clipped_fraction >= clipping_warn_fraction:
        clipping_status = "elevated"
    else:
        clipping_status = "acceptable"

    if degree_signal_status == "positive" and coverage_status == "adequate" and clipping_status != "high":
        verdict = "primary_degree_calibrated_aggregation_ready"
    elif degree_signal_status == "positive":
        verdict = "degree_calibrated_aggregation_with_reporting_cautions"
    elif gene_signal_status == "positive" or degree_strat_status == "positive":
        verdict = "broad_signal_but_not_network_showcase"
    elif degree_signal_status == "borderline" or gene_signal_status == "borderline":
        verdict = "borderline_signal_requires_sensitivity"
    else:
        verdict = "low_signal_or_graph_mismatch"

    limitations: list[str] = []
    recommendations: list[str] = []
    if gene_signal_status in {"weak_or_absent", "borderline"}:
        limitations.append("Gene-level percolation signal is weak before graph-specific interpretation.")
        recommendations.append("Evaluate higher-powered or more specific GWAS summary statistics.")
    if degree_signal_status != "positive":
        limitations.append("Signal does not exceed the degree-matched node null at the frozen threshold.")
        recommendations.append("Run graph sensitivity and local pathway/subgraph diagnostics before claiming modules.")
    if topology_status == "topology_null_sensitive_negative":
        limitations.append("Observed topology underperforms degree-preserving graph nulls.")
        recommendations.append("Report topology-null sensitivity and test tissue- or cell-type-specific graphs.")
    elif topology_status == "not_topology_specific" and degree_signal_status == "positive":
        limitations.append("Degree-calibrated aggregation is present without clear topology-specific support.")
        recommendations.append(
            "Frame the result as degree-calibrated weak-signal aggregation, not topology-specific modules."
        )
    if clipping_status in {"elevated", "high"}:
        limitations.append("P-value clipping is elevated and may concentrate evidence in extreme genes.")
        recommendations.append("Report clipped-gene counts and inspect high-LD or special-region drivers.")
    if coverage_status != "adequate":
        limitations.append("Graph largest-component coverage is below the preferred threshold.")
        recommendations.append("Improve graph coverage or report graph coverage as a limitation.")
    if not recommendations:
        recommendations.append("Suitable for primary architecture reporting under the current calibration.")

    return {
        "verdict": verdict,
        "z_threshold": float(z_threshold),
        "borderline_z": float(borderline_z),
        "architecture_class": architecture.get("architecture_class", ""),
        "gene_level_signal_status": gene_signal_status,
        "degree_stratified_signal_status": degree_strat_status,
        "degree_matched_signal_status": degree_signal_status,
        "topology_specificity_status": topology_status,
        "graph_coverage_status": coverage_status,
        "p_clipping_status": clipping_status,
        "metrics": {
            "snp_null_z": snp_z,
            "snp_null_delta": _delta(summary, "snp_permutation_null_summary"),
            "degree_stratified_z": degree_strat_z,
            "degree_stratified_delta": _delta(summary, "degree_stratified_null_summary"),
            "degree_matched_z": degree_matched_z,
            "degree_matched_delta": _delta(summary, "degree_matched_node_null_summary"),
            "degree_preserving_graph_z": graph_z,
            "degree_preserving_graph_delta": _delta(summary, "degree_preserving_graph_null_summary"),
            "lcc_gene_fraction": lcc_gene_fraction,
            "p_clipped_fraction": clipped_fraction,
            "p_clipped_count": int(clipping.get("n_clipped", 0) or 0),
        },
        "limitations": limitations,
        "recommended_next_steps": list(dict.fromkeys(recommendations)),
    }


def _fmt(value: object, digits: int = 3) -> str:
    numeric = _finite_float(value)
    if np.isfinite(numeric):
        return f"{numeric:.{digits}f}"
    return "NA"


def render_trait_architecture_markdown(summary: Mapping[str, Any]) -> str:
    """Render a generic architecture and suitability report for one trait."""

    trait = str(summary.get("trait", "UNKNOWN_TRAIT"))
    suitability = _as_mapping(summary.get("trait_suitability", {}))
    if not suitability:
        suitability = build_trait_suitability_diagnostic(summary)
    metrics = _as_mapping(suitability.get("metrics", {}))
    architecture = _as_mapping(summary.get("percolation_architecture", {}))
    graph_coverage = _as_mapping(summary.get("graph_coverage_report", {}))
    clipping = _as_mapping(summary.get("p_clipping_summary", {}))
    module_summary = _as_mapping(summary.get("local_module_summary", {}))
    claim_tiers = summary.get("claim_tiers", [])
    diffusion_rows = summary.get("diffusion_kernel_summary", [])
    limitations = suitability.get("limitations", [])
    if not isinstance(limitations, Sequence) or isinstance(limitations, str):
        limitations = []
    recommendations = suitability.get("recommended_next_steps", [])
    if not isinstance(recommendations, Sequence) or isinstance(recommendations, str):
        recommendations = []

    lines = [
        f"# {trait} RIPPLE Architecture Report (Claim-Tier)",
        "",
        "## Suitability Verdict",
        "",
        f"- Verdict: `{suitability.get('verdict', '')}`",
        f"- Calibration class: `{suitability.get('architecture_class', architecture.get('architecture_class', ''))}`",
        f"- Primary percolation label: `{architecture.get('primary_statistic_label', 'degree_calibrated_top_rank_aggregation')}`",
        f"- Interpretation: {architecture.get('interpretation', '')}",
        (
            "- Topology-specific discovery is a conditional claim tier; it is not required for gene-level, "
            "degree-calibrated, or diffusion-space aggregation claims."
        ),
        "",
        "## Claim Tiers",
        "",
    ]
    if isinstance(claim_tiers, Sequence) and not isinstance(claim_tiers, str) and claim_tiers:
        lines.extend(
            [
                "| Tier | Statistic | Z | Empirical P | Passed | Claim |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for item in claim_tiers:
            item_map = _as_mapping(item)
            lines.append(
                "| "
                f"{item_map.get('tier', '')} | {item_map.get('statistic', '')} | "
                f"{_fmt(item_map.get('z'))} | {_fmt(item_map.get('empirical_p'), 4)} | "
                f"{item_map.get('passed', '')} | {item_map.get('claim_label', '')} |"
            )
    else:
        lines.append("Claim-tier table was not available for this run.")
    lines.extend(
        [
            "",
            "## Calibration Metrics",
            "",
            "| Axis | Z | Delta | Status |",
            "|---|---:|---:|---|",
            (
                "| SNP permutation | "
                f"{_fmt(metrics.get('snp_null_z'))} | "
                f"{_fmt(metrics.get('snp_null_delta'), 6)} | "
                f"{suitability.get('gene_level_signal_status', '')} |"
            ),
            (
                "| Degree-stratified score null | "
                f"{_fmt(metrics.get('degree_stratified_z'))} | "
                f"{_fmt(metrics.get('degree_stratified_delta'), 6)} | "
                f"{suitability.get('degree_stratified_signal_status', '')} |"
            ),
            (
                "| Degree-matched node null | "
                f"{_fmt(metrics.get('degree_matched_z'))} | "
                f"{_fmt(metrics.get('degree_matched_delta'), 6)} | "
                f"{suitability.get('degree_matched_signal_status', '')} |"
            ),
            (
                "| Degree-preserving graph null | "
                f"{_fmt(metrics.get('degree_preserving_graph_z'))} | "
                f"{_fmt(metrics.get('degree_preserving_graph_delta'), 6)} | "
                f"{suitability.get('topology_specificity_status', '')} |"
            ),
            "",
            "## Diffusion Kernel",
            "",
        ]
    )
    if isinstance(diffusion_rows, Sequence) and not isinstance(diffusion_rows, str) and diffusion_rows:
        lines.extend(
            [
                "| Null | Score mode | T max | Tau | Z | Empirical P | Weighted L |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for item in diffusion_rows:
            item_map = _as_mapping(item)
            lines.append(
                "| "
                f"{item_map.get('null_type', '')} | {item_map.get('score_mode', '')} | "
                f"{_fmt(item_map.get('T_max'))} | {_fmt(item_map.get('tau_at_max'))} | "
                f"{_fmt(item_map.get('z'))} | {_fmt(item_map.get('empirical_p'), 4)} | "
                f"{item_map.get('weighted_laplacian_used', False)} |"
            )
    else:
        lines.append("Diffusion-kernel statistic was not run for this analysis.")
    lines.extend(
        [
            "",
            "## Technical Checks",
            "",
            f"- LCC gene fraction: {_fmt(graph_coverage.get('largest_component_gene_fraction'))} "
            f"({suitability.get('graph_coverage_status', '')})",
            f"- LCC scored genes: {int(summary.get('n_lcc_scored_genes', 0) or 0):,}",
            f"- P clipping: {int(clipping.get('n_clipped', 0) or 0):,} / "
            f"{int(clipping.get('n_total', 0) or 0):,} "
            f"({_fmt(clipping.get('fraction_clipped'))}; "
            f"{suitability.get('p_clipping_status', '')})",
            f"- GSP retained energy fraction: {_fmt(summary.get('gsp_retained_energy_fraction'), 6)}",
            (
                "- Candidate local modules tested: "
                f"{int(module_summary.get('n_candidate_modules', 0) or 0):,}"
            ),
            (
                "- Calibrated broad components: "
                f"{int(module_summary.get('n_broad_calibrated_components', 0) or 0):,}"
            ),
            (
                "- Calibrated weak-signal modules: "
                f"{int(module_summary.get('n_calibrated_modules', 0) or 0):,}"
            ),
            (
                "- Topology-specific modules: "
                f"{int(module_summary.get('n_topology_specific_modules', 0) or 0):,}"
            ),
            "",
            "## Top Calibrated Modules",
            "",
        ]
    )
    top_modules = module_summary.get("top_modules", [])
    if isinstance(top_modules, Sequence) and not isinstance(top_modules, str) and top_modules:
        lines.extend(
            [
                "| Rank | Module | k | Mean score | Degree P | Selection P | Graph P | Core genes |",
                "|---:|---|---:|---:|---:|---:|---:|---|",
            ]
        )
        for item in top_modules[:5]:
            item_map = _as_mapping(item)
            lines.append(
                "| "
                f"{int(item_map.get('module_rank', 0) or 0)} | "
                f"{item_map.get('module_id', '')} | "
                f"{int(item_map.get('n_genes', 0) or 0)} | "
                f"{_fmt(item_map.get('mean_score'))} | "
                f"{_fmt(item_map.get('degree_matched_p'), 4)} | "
                f"{_fmt(item_map.get('selection_aware_score_p'), 4)} | "
                f"{_fmt(item_map.get('degree_preserving_graph_p'), 4)} | "
                f"{item_map.get('core_genes', '')} |"
            )
    else:
        lines.append("No calibrated local modules are currently reported for this run.")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in limitations) if limitations else lines.append("- None flagged.")
    lines.extend(["", "## Recommended Next Steps", ""])
    lines.extend(f"- {item}" for item in recommendations) if recommendations else lines.append("- None.")
    return "\n".join(lines) + "\n"
