#!/usr/bin/env python
"""Validate RIPPLE percolation and diffusion graph statistics on synthetic graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.graph_diffusion import degree_stratified_diffusion_null  # noqa: E402
from ripple.nulls.graph_nulls import degree_preserving_graph_replicates  # noqa: E402
from ripple.nulls.score_permutation import assign_degree_bins  # noqa: E402
from ripple.percolation import (  # noqa: E402
    percolation_auc,
    percolation_curve,
    rank_nodes_by_score,
    summarize_percolation_null,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-replicates", type=int, default=25)
    parser.add_argument("--n-null", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260615)
    return parser.parse_args()


def build_validation_graph(*, n_modules: int = 6, module_size: int = 25, seed: int = 1) -> tuple[nx.Graph, list[str]]:
    rng = np.random.default_rng(seed)
    graph = nx.Graph()
    modules: list[list[str]] = []
    for module_idx in range(n_modules):
        nodes = [f"G{module_idx:02d}_{idx:02d}" for idx in range(module_size)]
        modules.append(nodes)
        graph.add_nodes_from(nodes)
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                if rng.random() < 0.18:
                    graph.add_edge(u, v)
    for module_idx in range(n_modules - 1):
        graph.add_edge(modules[module_idx][0], modules[module_idx + 1][0])
        graph.add_edge(modules[module_idx][1], modules[module_idx + 1][1])
    return graph, modules[0]


def build_mismatch_graph(graph: nx.Graph, *, seed: int) -> nx.Graph:
    """Return a degree-preserving but module-mismatched graph."""

    return next(
        degree_preserving_graph_replicates(
            graph,
            n_replicates=1,
            seed=seed,
            nswap_per_edge=5.0,
            max_tries_per_swap=50.0,
        )
    )


def score_table(graph: nx.Graph, scores: np.ndarray) -> pd.DataFrame:
    nodes = list(graph.nodes())
    return pd.DataFrame(
        {
            "gene_symbol": nodes,
            "assoc_resid_score": scores,
            "graph_degree": [graph.degree(node) for node in nodes],
            "assoc_p_g": [1e-4] * len(nodes),
        }
    )


def simulate_scores(graph: nx.Graph, module_nodes: list[str], scenario: str, rng: np.random.Generator) -> np.ndarray:
    nodes = list(graph.nodes())
    degree = np.array([graph.degree(node) for node in nodes], dtype=float)
    degree_z = (degree - degree.mean()) / (degree.std(ddof=1) or 1.0)
    scores = rng.normal(0.0, 1.0, len(nodes))
    if scenario == "degree_biased_null":
        return 1.4 * degree_z + rng.normal(0.0, 1.0, len(nodes))
    if scenario in {"compact_connected_module_spikein", "graph_mismatch_spikein"}:
        target = set(module_nodes[: max(8, len(module_nodes) // 2)])
        scores += np.array([1.8 if node in target else 0.0 for node in nodes])
    elif scenario == "diffuse_pathway_spikein":
        seeds = module_nodes[:4]
        target = set(seeds)
        for seed in seeds:
            target.update(nx.single_source_shortest_path_length(graph, seed, cutoff=2).keys())
        scores += np.array([1.0 if node in target else 0.0 for node in nodes])
    return scores


def percolation_degree_stratified_null(
    graph: nx.Graph,
    scores: pd.DataFrame,
    *,
    n_null: int,
    seed: int,
) -> dict[str, float | int]:
    ranking = rank_nodes_by_score(scores)
    observed_auc = percolation_auc(percolation_curve(graph, ranking, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]))
    values = scores["assoc_resid_score"].to_numpy(dtype=float)
    bins = assign_degree_bins(scores["graph_degree"], n_bins=10).to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for replicate in range(n_null):
        permuted = values.copy()
        for bin_id in sorted(set(int(item) for item in bins)):
            idx = np.flatnonzero(bins == bin_id)
            if idx.size > 1:
                permuted[idx] = permuted[rng.permutation(idx)]
        null_scores = scores.copy()
        null_scores["assoc_resid_score"] = permuted
        null_ranking = rank_nodes_by_score(null_scores)
        null_auc = percolation_auc(
            percolation_curve(graph, null_ranking, [0.01, 0.02, 0.05, 0.10, 0.15, 0.20])
        )
        rows.append({"replicate": replicate, "percolation_auc": null_auc})
    return summarize_percolation_null(pd.DataFrame(rows), observed_auc)


def run_validation(args: argparse.Namespace) -> pd.DataFrame:
    graph, module_nodes = build_validation_graph(seed=args.seed)
    scenarios = (
        "null_scores",
        "degree_biased_null",
        "compact_connected_module_spikein",
        "diffuse_pathway_spikein",
        "graph_mismatch_spikein",
    )
    rng = np.random.default_rng(args.seed)
    detail_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for replicate in range(args.n_replicates):
            scores = score_table(graph, simulate_scores(graph, module_nodes, scenario, rng))
            analysis_graph = build_mismatch_graph(graph, seed=args.seed + replicate + 3000) if scenario == "graph_mismatch_spikein" else graph
            perc = percolation_degree_stratified_null(
                analysis_graph,
                scores,
                n_null=args.n_null,
                seed=args.seed + replicate + 1000,
            )
            diff_summary, _, _ = degree_stratified_diffusion_null(
                analysis_graph,
                scores,
                trait=scenario,
                graph_name="synthetic_modular_graph",
                n_replicates=args.n_null,
                seed=args.seed + replicate + 2000,
                n_bins=10,
            )
            diff_row = diff_summary.iloc[0].to_dict()
            detail_rows.append(
                {
                    "scenario": scenario,
                    "replicate": replicate,
                    "statistic": "degree_stratified_percolation",
                    "z": float(perc["z"]),
                    "empirical_p": float(perc["empirical_p_upper"]),
                    "positive": bool(float(perc["z"]) >= 2.0),
                }
            )
            detail_rows.append(
                {
                    "scenario": scenario,
                    "replicate": replicate,
                    "statistic": "degree_stratified_diffusion_Tmax",
                    "z": float(diff_row["z"]),
                    "empirical_p": float(diff_row["empirical_p"]),
                    "positive": bool(diff_row["passed"]),
                }
            )
    detail = pd.DataFrame(detail_rows)
    summary_rows: list[dict[str, object]] = []
    for (scenario, statistic), group in detail.groupby(["scenario", "statistic"], observed=True):
        is_null = scenario in {"null_scores", "degree_biased_null"}
        rate = float(group["positive"].mean())
        if scenario == "graph_mismatch_spikein":
            passed = bool(rate <= 0.20)
            interpretation = "Wrong graph should not create a stable positive graph-domain claim."
        elif is_null:
            passed = bool(rate <= 0.20)
            interpretation = "Degree-aware calibration should control false positives."
        else:
            passed = bool(rate >= 0.40)
            interpretation = "Spike-in should increase power for at least one graph statistic."
        summary_rows.append(
            {
                "scenario": scenario,
                "n_replicates": int(group["replicate"].nunique()),
                "statistic": statistic,
                "type1_error_or_power": rate,
                "mean_z": float(group["z"].mean()),
                "median_empirical_p": float(group["empirical_p"].median()),
                "passed": passed,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_validation(args)
    path = args.out_dir / "validation_graph_statistics_summary.tsv"
    summary.to_csv(path, sep="\t", index=False)
    print(f"Wrote validation summary to {path}", flush=True)


if __name__ == "__main__":
    main()
