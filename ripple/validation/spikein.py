"""Synthetic spike-in validation for RIPPLE percolation calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd

from ripple.defaults import RANK_FRACTION_GRID
from ripple.diagnostics import build_trait_suitability_diagnostic
from ripple.modules import run_local_module_discovery
from ripple.nulls.graph_nulls import degree_preserving_graph_replicates
from ripple.nulls.score_permutation import degree_stratified_permuted_scores
from ripple.percolation import (
    classify_percolation_architecture,
    compute_degree_matched_node_percolation_null,
    percolation_auc,
    percolation_curve,
    prepare_degree_matched_rank_sets,
    rank_nodes_by_score,
    summarize_percolation_null,
)

SpikeinScenario = Literal["null", "dispersed", "degree_biased", "module"]


@dataclass(frozen=True)
class SpikeinConfig:
    """Configuration for one synthetic scenario."""

    scenario: SpikeinScenario
    effect_size: float = 1.8
    target_fraction: float = 0.10


def build_synthetic_modular_graph(
    *,
    n_modules: int = 5,
    module_size: int = 30,
    intra_p: float = 0.18,
    inter_p: float = 0.008,
    seed: int = 20260614,
) -> nx.Graph:
    """Build a connected modular graph with module labels.

    The graph is intentionally small enough for fast validation but has enough
    modular structure for a connected-module spike-in to be distinguishable from
    random or degree-biased signals.
    """

    if n_modules < 2:
        raise ValueError("n_modules must be at least 2.")
    if module_size < 4:
        raise ValueError("module_size must be at least 4.")

    rng = np.random.default_rng(seed)
    graph = nx.Graph()
    modules: list[list[str]] = []
    for module_idx in range(n_modules):
        nodes = [f"M{module_idx}_G{node_idx}" for node_idx in range(module_size)]
        modules.append(nodes)
        graph.add_nodes_from((node, {"module": module_idx}) for node in nodes)
        for node_idx, node in enumerate(nodes):
            graph.add_edge(node, nodes[(node_idx + 1) % module_size])
            graph.add_edge(node, nodes[(node_idx + 2) % module_size])
        for i, left in enumerate(nodes):
            for right in nodes[i + 3 :]:
                if rng.random() < intra_p:
                    graph.add_edge(left, right)

    for module_idx in range(n_modules):
        graph.add_edge(modules[module_idx][0], modules[(module_idx + 1) % n_modules][0])

    all_nodes = [node for module_nodes in modules for node in module_nodes]
    for i, left in enumerate(all_nodes):
        left_module = graph.nodes[left]["module"]
        for right in all_nodes[i + 1 :]:
            if graph.nodes[right]["module"] != left_module and rng.random() < inter_p:
                graph.add_edge(left, right)

    return graph


def _target_count(graph: nx.Graph, target_fraction: float) -> int:
    if not 0 < target_fraction <= 1:
        raise ValueError("target_fraction must be in (0, 1].")
    return max(1, int(round(graph.number_of_nodes() * target_fraction)))


def simulate_spikein_scores(
    graph: nx.Graph,
    *,
    scenario: SpikeinScenario,
    seed: int,
    effect_size: float = 1.8,
    target_fraction: float = 0.10,
    noise_sd: float = 1.0,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Simulate node scores under a synthetic signal architecture."""

    rng = np.random.default_rng(seed)
    nodes = np.array(sorted(str(node) for node in graph.nodes()), dtype=object)
    scores = rng.normal(loc=0.0, scale=noise_sd, size=len(nodes))
    n_target = _target_count(graph, target_fraction)
    target_nodes: tuple[str, ...]

    if scenario == "null":
        target_nodes = ()
    elif scenario == "dispersed":
        target_nodes = tuple(str(node) for node in rng.choice(nodes, size=n_target, replace=False))
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        scores[[node_to_idx[node] for node in target_nodes]] += effect_size
    elif scenario == "degree_biased":
        degree_order = sorted(graph.degree(), key=lambda item: (item[1], str(item[0])), reverse=True)
        target_nodes = tuple(str(node) for node, _ in degree_order[:n_target])
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        scores[[node_to_idx[node] for node in target_nodes]] += effect_size
    elif scenario == "module":
        module_counts = pd.Series(nx.get_node_attributes(graph, "module")).value_counts().sort_index()
        module_id = int(module_counts.index[0])
        module_nodes = sorted(str(node) for node, value in graph.nodes(data="module") if int(value) == module_id)
        target_nodes = tuple(module_nodes[: max(n_target, len(module_nodes))])
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        scores[[node_to_idx[node] for node in target_nodes]] += effect_size
    else:
        raise ValueError(f"Unknown spike-in scenario: {scenario}")

    table = pd.DataFrame(
        {
            "gene_symbol": nodes,
            "assoc_resid_score": scores,
            "graph_degree": [int(graph.degree(str(node))) for node in nodes],
            "is_spikein_target": [str(node) in set(target_nodes) for node in nodes],
        }
    )
    return table, target_nodes


def _score_null_auc_table(
    graph: nx.Graph,
    nodes: np.ndarray,
    *,
    n_replicates: int,
    seed: int,
    fractions: tuple[float, ...],
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for replicate in range(n_replicates):
        scores = pd.DataFrame(
            {
                "gene_symbol": nodes,
                "assoc_resid_score": rng.normal(size=len(nodes)),
            }
        )
        ranking = rank_nodes_by_score(scores)
        curve = percolation_curve(graph, ranking, fractions)
        rows.append({"replicate": replicate, "percolation_auc": percolation_auc(curve)})
    return pd.DataFrame(rows)


def _degree_stratified_auc_table(
    graph: nx.Graph,
    scores: pd.DataFrame,
    *,
    n_replicates: int,
    seed: int,
    n_bins: int,
    fractions: tuple[float, ...],
) -> pd.DataFrame:
    permuted = degree_stratified_permuted_scores(
        scores,
        score_col="assoc_resid_score",
        degree_col="graph_degree",
        n_replicates=n_replicates,
        seed=seed,
        n_bins=n_bins,
    )
    rows: list[dict[str, float | int]] = []
    base = scores.loc[:, ["gene_symbol"]].copy()
    for replicate in range(n_replicates):
        table = base.copy()
        table["assoc_resid_score"] = permuted[replicate, :]
        ranking = rank_nodes_by_score(table)
        curve = percolation_curve(graph, ranking, fractions)
        rows.append({"replicate": replicate, "percolation_auc": percolation_auc(curve)})
    return pd.DataFrame(rows)


def _degree_preserving_auc_table(
    graph: nx.Graph,
    ranking: pd.DataFrame,
    *,
    n_replicates: int,
    seed: int,
    fractions: tuple[float, ...],
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for replicate, null_graph in enumerate(
        degree_preserving_graph_replicates(
            graph,
            n_replicates=n_replicates,
            seed=seed,
            nswap_per_edge=1.0,
            max_tries_per_swap=20.0,
        )
    ):
        curve = percolation_curve(null_graph, ranking, fractions)
        rows.append({"replicate": replicate, "percolation_auc": percolation_auc(curve)})
    return pd.DataFrame(rows)


def run_synthetic_spikein_validation(
    *,
    graph: nx.Graph | None = None,
    scenarios: tuple[SpikeinConfig, ...] = (
        SpikeinConfig("null", effect_size=0.0),
        SpikeinConfig("dispersed", effect_size=1.8),
        SpikeinConfig("degree_biased", effect_size=1.8),
        SpikeinConfig("module", effect_size=1.8),
    ),
    seed: int = 20260614,
    n_score_null: int = 100,
    n_degree_stratified_null: int = 100,
    n_degree_matched_node_null: int = 200,
    n_degree_graph_null: int = 50,
    n_module_selection_aware_null: int = 100,
    degree_bins: int = 10,
    fractions: tuple[float, ...] = tuple(RANK_FRACTION_GRID),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run synthetic spike-in scenarios through RIPPLE percolation calibration."""

    if graph is None:
        graph = build_synthetic_modular_graph(seed=seed)
    graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_label")
    label_map = {node: str(data["original_label"]) for node, data in graph.nodes(data=True)}
    graph = nx.relabel_nodes(graph, label_map, copy=True)
    nodes = np.array(sorted(str(node) for node in graph.nodes()), dtype=object)

    summary_rows: list[dict[str, object]] = []
    detail_tables: dict[str, pd.DataFrame] = {}
    for scenario_idx, scenario in enumerate(scenarios):
        scenario_seed = seed + (scenario_idx + 1) * 1000
        scores, targets = simulate_spikein_scores(
            graph,
            scenario=scenario.scenario,
            seed=scenario_seed,
            effect_size=scenario.effect_size,
            target_fraction=scenario.target_fraction,
        )
        ranking = rank_nodes_by_score(scores)
        observed_curve = percolation_curve(graph, ranking, fractions)
        observed_auc = percolation_auc(observed_curve)
        score_null = _score_null_auc_table(
            graph,
            nodes,
            n_replicates=n_score_null,
            seed=scenario_seed + 11,
            fractions=fractions,
        )
        degree_stratified_null = _degree_stratified_auc_table(
            graph,
            scores,
            n_replicates=n_degree_stratified_null,
            seed=scenario_seed + 22,
            n_bins=degree_bins,
            fractions=fractions,
        )
        selected_bin_counts, bin_to_nodes, _ = prepare_degree_matched_rank_sets(
            scores,
            ranking,
            fractions,
            n_bins=degree_bins,
        )
        degree_matched_null, _ = compute_degree_matched_node_percolation_null(
            graph,
            selected_bin_counts,
            bin_to_nodes,
            n_replicates=n_degree_matched_node_null,
            seed=scenario_seed + 33,
        )
        degree_graph_null = _degree_preserving_auc_table(
            graph,
            ranking,
            n_replicates=n_degree_graph_null,
            seed=scenario_seed + 44,
            fractions=fractions,
        )
        snp_summary = summarize_percolation_null(score_null, observed_auc)
        degree_stratified_summary = summarize_percolation_null(degree_stratified_null, observed_auc)
        degree_matched_summary = summarize_percolation_null(degree_matched_null, observed_auc)
        degree_graph_summary = summarize_percolation_null(degree_graph_null, observed_auc)
        architecture = classify_percolation_architecture(
            snp_permutation_null=snp_summary,
            degree_stratified_null=degree_stratified_summary,
            degree_matched_node_null=degree_matched_summary,
            degree_preserving_graph_null=degree_graph_summary,
        )
        local_modules, local_module_nulls, _ = run_local_module_discovery(
            graph,
            scores,
            gene_sets={},
            seed=scenario_seed + 55,
            min_module_size=5,
            min_subthreshold_genes=0,
            max_modules=10,
            n_module_random=min(100, n_score_null),
            n_module_degree_matched=min(100, n_degree_matched_node_null),
            n_module_degree_graph=min(20, n_degree_graph_null),
            n_module_selection_aware=min(100, n_module_selection_aware_null),
            selection_null_scores=np.random.default_rng(scenario_seed + 66).normal(
                size=(min(100, n_module_selection_aware_null), len(scores))
            ),
            n_pathway_random=0,
            n_pathway_degree_matched=0,
            degree_bins=degree_bins,
        )
        global_module_gate_pass = bool(architecture["degree_matched_node_positive"])
        if not local_modules.empty:
            local_modules = local_modules.copy()
            is_local_component = local_modules["n_genes"].astype(int) < 200
            local_modules["passes_global_module_gate"] = global_module_gate_pass
            local_modules["is_reportable_calibrated_module"] = (
                local_modules["is_calibrated_weak_signal_module"].astype(bool) & is_local_component & global_module_gate_pass
            )
            local_modules["is_reportable_topology_specific_module"] = (
                local_modules["is_topology_specific_module"].astype(bool) & is_local_component & global_module_gate_pass
            )
        n_calibrated_modules = (
            int(local_modules["is_reportable_calibrated_module"].sum()) if not local_modules.empty else 0
        )
        n_topology_modules = (
            int(local_modules["is_reportable_topology_specific_module"].sum()) if not local_modules.empty else 0
        )
        pseudo_trait_summary = {
            "trait": f"synthetic_{scenario.scenario}",
            "n_lcc_scored_genes": graph.number_of_nodes(),
            "graph_coverage_report": {"largest_component_gene_fraction": 1.0},
            "p_clipping_summary": {"n_clipped": 0, "n_total": graph.number_of_nodes(), "fraction_clipped": 0.0},
            "snp_permutation_null_summary": snp_summary,
            "degree_stratified_null_summary": degree_stratified_summary,
            "degree_matched_node_null_summary": degree_matched_summary,
            "degree_preserving_graph_null_summary": degree_graph_summary,
            "percolation_architecture": architecture,
        }
        suitability = build_trait_suitability_diagnostic(pseudo_trait_summary)
        summary_rows.append(
            {
                "scenario": scenario.scenario,
                "effect_size": scenario.effect_size,
                "target_fraction": scenario.target_fraction,
                "n_targets": len(targets),
                "observed_auc": observed_auc,
                "snp_null_z": snp_summary["z"],
                "degree_stratified_z": degree_stratified_summary["z"],
                "degree_matched_z": degree_matched_summary["z"],
                "degree_preserving_graph_z": degree_graph_summary["z"],
                "n_candidate_modules": len(local_modules),
                "n_module_level_calibrated_candidates": int(local_modules["is_calibrated_weak_signal_module"].sum())
                if not local_modules.empty
                else 0,
                "n_broad_calibrated_components": int(
                    (
                        local_modules["is_calibrated_weak_signal_module"].astype(bool)
                        & (local_modules["n_genes"].astype(int) >= 200)
                    ).sum()
                )
                if not local_modules.empty
                else 0,
                "n_calibrated_modules": n_calibrated_modules,
                "n_topology_specific_modules": n_topology_modules,
                "architecture_class": architecture["architecture_class"],
                "suitability_verdict": suitability["verdict"],
            }
        )
        scenario_key = scenario.scenario
        detail_tables[f"{scenario_key}_scores"] = scores
        detail_tables[f"{scenario_key}_observed_curve"] = observed_curve
        detail_tables[f"{scenario_key}_score_null_auc"] = score_null
        detail_tables[f"{scenario_key}_degree_stratified_null_auc"] = degree_stratified_null
        detail_tables[f"{scenario_key}_degree_matched_null_auc"] = degree_matched_null
        detail_tables[f"{scenario_key}_degree_graph_null_auc"] = degree_graph_null
        detail_tables[f"{scenario_key}_local_modules"] = local_modules
        detail_tables[f"{scenario_key}_local_module_nulls"] = local_module_nulls

    return pd.DataFrame(summary_rows), detail_tables
