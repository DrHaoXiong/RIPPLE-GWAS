"""Percolation calibration and architecture classification."""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from ripple.nulls.score_permutation import assign_degree_bins
from ripple.percolation.auc import component_stats_for_nodes, percolation_auc
from ripple.percolation.ranking import selected_nodes_at_fraction


def component_row_for_nodes(graph: nx.Graph, selected_nodes: Iterable[str], fraction: float) -> dict[str, object]:
    """Compute percolation component statistics for an explicit node set."""

    stats = component_stats_for_nodes(graph, selected_nodes)
    return {
        "rank_fraction": float(fraction),
        **stats,
    }


def summarize_percolation_null(null_auc: pd.DataFrame, observed_auc: float) -> dict[str, float | int]:
    """Summarize observed percolation AUC against a null AUC table."""

    if null_auc.empty:
        return {
            "n_replicates": 0,
            "mean": float("nan"),
            "sd": float("nan"),
            "delta": float("nan"),
            "z": float("nan"),
            "empirical_p_upper": float("nan"),
        }
    auc = pd.to_numeric(null_auc["percolation_auc"], errors="raise")
    mean = float(auc.mean())
    sd = float(auc.std(ddof=1)) if len(auc) > 1 else float("nan")
    delta = float(observed_auc - mean)
    z = float(delta / sd) if sd > 0 else float("nan")
    empirical_p = float((1 + (auc >= observed_auc).sum()) / (len(auc) + 1))
    return {
        "n_replicates": int(len(auc)),
        "mean": mean,
        "sd": sd,
        "delta": delta,
        "z": z,
        "empirical_p_upper": empirical_p,
    }


def summarize_percolation_threshold_robustness(
    observed_curve: pd.DataFrame,
    null_curve: pd.DataFrame,
    *,
    trait: str,
    graph_name: str,
    null_type: str,
) -> pd.DataFrame:
    """Summarize per-rank-fraction percolation robustness against a curve null."""

    required = {"rank_fraction", "largest_component_fraction"}
    missing_obs = sorted(required - set(observed_curve.columns))
    missing_null = sorted((required | {"replicate"}) - set(null_curve.columns))
    if missing_obs:
        raise ValueError(f"observed_curve is missing columns: {missing_obs}")
    if null_curve.empty:
        return pd.DataFrame(
            columns=[
                "trait",
                "graph_name",
                "null_type",
                "rank_fraction",
                "observed_lcc_fraction",
                "null_mean",
                "null_sd",
                "z",
                "empirical_p",
            ]
        )
    if missing_null:
        raise ValueError(f"null_curve is missing columns: {missing_null}")

    rows: list[dict[str, object]] = []
    for row in observed_curve.itertuples(index=False):
        fraction = float(row.rank_fraction)
        observed = float(row.largest_component_fraction)
        null_values = pd.to_numeric(
            null_curve.loc[null_curve["rank_fraction"].astype(float) == fraction, "largest_component_fraction"],
            errors="raise",
        ).to_numpy(dtype=float)
        if len(null_values) == 0:
            mean = sd = z = empirical_p = float("nan")
        else:
            mean = float(np.mean(null_values))
            sd = float(np.std(null_values, ddof=1)) if len(null_values) > 1 else float("nan")
            z = float((observed - mean) / sd) if sd > 0 else float("nan")
            empirical_p = float((1 + np.count_nonzero(null_values >= observed)) / (len(null_values) + 1))
        rows.append(
            {
                "trait": trait,
                "graph_name": graph_name,
                "null_type": null_type,
                "rank_fraction": fraction,
                "observed_lcc_fraction": observed,
                "null_mean": mean,
                "null_sd": sd,
                "z": z,
                "empirical_p": empirical_p,
            }
        )
    return pd.DataFrame(rows)


def is_positive_null_result(summary: dict[str, float | int], *, z_threshold: float = 2.0) -> bool:
    """Return whether an observed statistic exceeds a null by the configured Z threshold."""

    z = float(summary.get("z", float("nan")))
    return bool(np.isfinite(z) and z >= z_threshold)


def is_negative_null_result(summary: dict[str, float | int], *, z_threshold: float = 2.0) -> bool:
    """Return whether an observed statistic is below a null by the configured Z threshold."""

    z = float(summary.get("z", float("nan")))
    return bool(np.isfinite(z) and z <= -z_threshold)


def classify_percolation_architecture(
    *,
    snp_permutation_null: dict[str, float | int],
    degree_stratified_null: dict[str, float | int],
    degree_matched_node_null: dict[str, float | int],
    degree_preserving_graph_null: dict[str, float | int],
    z_threshold: float = 2.0,
) -> dict[str, object]:
    """Classify graph-coupled GWAS signal architecture from calibrated nulls."""

    snp_positive = is_positive_null_result(snp_permutation_null, z_threshold=z_threshold)
    degree_stratified_positive = is_positive_null_result(degree_stratified_null, z_threshold=z_threshold)
    degree_matched_positive = is_positive_null_result(degree_matched_node_null, z_threshold=z_threshold)
    degree_graph_positive = is_positive_null_result(degree_preserving_graph_null, z_threshold=z_threshold)
    degree_graph_negative = is_negative_null_result(degree_preserving_graph_null, z_threshold=z_threshold)

    if degree_matched_positive and degree_graph_positive:
        architecture_class = "topology_specific_module_excess"
        interpretation = (
            "Signal exceeds degree-matched node nulls on the observed graph and degree-preserving randomized "
            "topologies; this conditionally supports topology-specific graph support."
        )
    elif degree_matched_positive and degree_graph_negative:
        architecture_class = "degree_aware_aggregation_topology_sensitive"
        interpretation = (
            "Signal exceeds degree-matched node nulls on the observed graph but not degree-preserving graph "
            "nulls; this supports degree-calibrated top-rank weak-signal aggregation, while topology-specific "
            "support is not detected."
        )
    elif degree_matched_positive:
        architecture_class = "degree_aware_network_aggregation"
        interpretation = (
            "Signal exceeds degree-matched node nulls on the observed graph, supporting degree-calibrated "
            "top-rank weak-signal aggregation without clear topology-specific support."
        )
    elif snp_positive or degree_stratified_positive:
        architecture_class = "broad_genetic_signal_without_degree_aware_graph_aggregation"
        interpretation = (
            "Signal exceeds SNP or degree-stratified score nulls but not degree-matched node nulls; this "
            "supports broad genetic signal without robust degree-aware graph aggregation."
        )
    else:
        architecture_class = "no_graph_coupled_signal"
        interpretation = "Observed percolation does not exceed the calibrated nulls."

    return {
        "architecture_class": architecture_class,
        "primary_statistic_label": "degree_calibrated_top_rank_aggregation",
        "z_threshold": float(z_threshold),
        "snp_permutation_positive": snp_positive,
        "degree_stratified_positive": degree_stratified_positive,
        "degree_matched_node_positive": degree_matched_positive,
        "degree_preserving_graph_positive": degree_graph_positive,
        "degree_preserving_graph_negative": degree_graph_negative,
        "interpretation": interpretation,
    }


def prepare_degree_matched_rank_sets(
    scores: pd.DataFrame,
    ranking: pd.DataFrame,
    fractions: Iterable[float],
    *,
    node_col: str = "gene_symbol",
    degree_col: str = "graph_degree",
    n_bins: int = 10,
) -> tuple[dict[float, dict[int, int]], dict[int, np.ndarray], pd.DataFrame]:
    """Prepare degree-bin profiles needed for degree-matched node nulls."""

    missing = [col for col in (node_col, degree_col) if col not in scores.columns]
    if missing:
        raise ValueError(f"Missing degree matching columns: {missing}")

    work = scores.loc[:, [node_col, degree_col]].copy()
    work[node_col] = work[node_col].astype(str)
    work[degree_col] = pd.to_numeric(work[degree_col], errors="raise").astype(float)
    work["degree_bin"] = assign_degree_bins(work[degree_col], n_bins=n_bins).to_numpy(dtype=int)
    bin_to_nodes = {
        int(bin_id): group[node_col].to_numpy(dtype=object)
        for bin_id, group in work.groupby("degree_bin", observed=True)
    }
    node_to_bin = dict(zip(work[node_col], work["degree_bin"], strict=True))
    node_to_degree = dict(zip(work[node_col], work[degree_col], strict=True))

    selected_bin_counts: dict[float, dict[int, int]] = {}
    profile_rows: list[dict[str, float | int]] = []
    for fraction in fractions:
        selected = selected_nodes_at_fraction(ranking, float(fraction), node_col=node_col)
        selected_bins = pd.Series([node_to_bin[node] for node in selected]).value_counts(sort=False).sort_index()
        selected_bin_counts[float(fraction)] = {int(bin_id): int(count) for bin_id, count in selected_bins.items()}
        selected_degrees = np.array([node_to_degree[node] for node in selected], dtype=float)
        profile_rows.append(
            {
                "rank_fraction": float(fraction),
                "n_selected": int(len(selected)),
                "selected_mean_degree": float(np.mean(selected_degrees)),
                "selected_median_degree": float(np.median(selected_degrees)),
                "all_nodes_mean_degree": float(work[degree_col].mean()),
                "all_nodes_median_degree": float(work[degree_col].median()),
            }
        )
    return selected_bin_counts, bin_to_nodes, pd.DataFrame(profile_rows)


def compute_degree_matched_node_percolation_null(
    graph: nx.Graph,
    selected_bin_counts: dict[float, dict[int, int]],
    bin_to_nodes: dict[int, np.ndarray],
    *,
    n_replicates: int,
    seed: int,
    progress_interval: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute percolation nulls from degree-matched node sets on the observed graph."""

    if n_replicates < 0:
        raise ValueError("n_replicates must be nonnegative.")

    rng = np.random.default_rng(seed)
    auc_rows: list[dict[str, float | int]] = []
    curve_rows: list[pd.DataFrame] = []
    for idx in range(n_replicates):
        rows: list[dict[str, object]] = []
        for fraction, bin_counts in selected_bin_counts.items():
            sampled: list[str] = []
            for bin_id, count in sorted(bin_counts.items()):
                candidates = bin_to_nodes[int(bin_id)]
                sampled.extend(str(node) for node in rng.choice(candidates, size=count, replace=False))
            rng.shuffle(sampled)
            rows.append(component_row_for_nodes(graph, sampled, fraction))
        curve = pd.DataFrame(rows)
        auc = percolation_auc(curve)
        auc_rows.append({"replicate": idx, "percolation_auc": auc})
        curve["replicate"] = idx
        curve_rows.append(curve)
        if progress_interval > 0 and ((idx + 1) % progress_interval == 0 or idx + 1 == n_replicates):
            print(f"Computed {idx + 1:,}/{n_replicates:,} degree-matched node nulls", flush=True)

    auc_table = pd.DataFrame(auc_rows)
    curve_table = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    return auc_table, curve_table
