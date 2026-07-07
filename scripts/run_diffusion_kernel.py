#!/usr/bin/env python
"""Run heat-kernel diffusion statistics on existing RIPPLE LCC gene scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ripple.graph import preprocess_reference_graph  # noqa: E402
from ripple.graph_diffusion import (  # noqa: E402
    DEFAULT_TAU_GRID,
    degree_preserving_graph_diffusion_null,
    degree_stratified_diffusion_null,
    parse_tau_grid,
)
from ripple.io.graph import networkx_to_edge_list, read_edge_list, write_edge_list  # noqa: E402
from run_height_mvp import DEFAULT_STRING_INFO, DEFAULT_STRING_LINKS, build_string_graph, write_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--lcc-scores", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--graph-name", default="string_ppi")
    parser.add_argument("--graph-edge-list", type=Path, default=None)
    parser.add_argument("--analysis-graph-edge-cache", type=Path, default=None)
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--diffusion-score-mode", choices=["positive", "absolute", "raw", "rank"], default="positive")
    parser.add_argument("--tau-grid", default=",".join(str(tau) for tau in DEFAULT_TAU_GRID))
    parser.add_argument(
        "--diffusion-null",
        choices=["degree_stratified", "degree_preserving_graph", "both"],
        default="degree_stratified",
    )
    parser.add_argument("--n-diffusion-null", type=int, default=1000)
    parser.add_argument("--diffusion-degree-bins", type=int, default=20)
    parser.add_argument("--diffusion-batch-size", type=int, default=128)
    parser.add_argument("--weighted-laplacian", action="store_true")
    parser.add_argument("--save-null-distributions", action="store_true")
    parser.add_argument("--degree-graph-nswap-per-edge", type=float, default=1.0)
    parser.add_argument("--degree-graph-max-tries-per-swap", type=float, default=20.0)
    parser.add_argument("--degree-graph-cache", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def default_analysis_graph_edges_path(args: argparse.Namespace) -> Path:
    return args.lcc_scores.parent / f"{args.trait}.analysis_graph_edges.tsv.gz"


def load_graph(args: argparse.Namespace, gene_symbols: tuple[str, ...]):
    if args.graph_edge_list is not None:
        edges = read_edge_list(args.graph_edge_list, source=args.graph_name)
        return preprocess_reference_graph(edges.edges, gene_universe=gene_symbols)
    cache_path = args.analysis_graph_edge_cache or default_analysis_graph_edges_path(args)
    if cache_path.exists():
        print(f"Loading analysis graph edge cache: {cache_path}", flush=True)
        edges = read_edge_list(cache_path, sep="\t", source=f"{args.graph_name}_analysis_cache")
        return preprocess_reference_graph(edges.edges, gene_universe=gene_symbols)
    graph_args = argparse.Namespace(
        string_links=args.string_links,
        string_info=args.string_info,
        string_min_score=args.string_min_score,
    )
    _, graph_pre = build_string_graph(graph_args, gene_symbols)
    write_edge_list(
        cache_path,
        networkx_to_edge_list(graph_pre.largest_component),
        source=f"{args.trait}_{args.graph_name}_largest_connected_component",
    )
    print(f"Wrote analysis graph edge cache: {cache_path}", flush=True)
    return graph_pre


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.lcc_scores, sep="\t", compression="infer")
    gene_symbols = tuple(scores["gene_symbol"].dropna().astype(str).unique())
    graph_pre = load_graph(args, gene_symbols)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tau_grid = parse_tau_grid(args.tau_grid)

    summaries: list[pd.DataFrame] = []
    tau_summaries: list[pd.DataFrame] = []
    nulls: list[pd.DataFrame] = []
    if args.diffusion_null in {"degree_stratified", "both"}:
        summary, tau, null = degree_stratified_diffusion_null(
            graph_pre.largest_component,
            scores,
            trait=args.trait,
            graph_name=args.graph_name,
            score_mode=args.diffusion_score_mode,
            tau_grid=tau_grid,
            n_replicates=args.n_diffusion_null,
            seed=args.seed,
            n_bins=args.diffusion_degree_bins,
            weighted_laplacian=args.weighted_laplacian,
            batch_size=args.diffusion_batch_size,
        )
        summaries.append(summary)
        tau_summaries.append(tau)
        nulls.append(null)
    if args.diffusion_null in {"degree_preserving_graph", "both"}:
        summary, tau, null = degree_preserving_graph_diffusion_null(
            graph_pre.largest_component,
            scores,
            trait=args.trait,
            graph_name=args.graph_name,
            score_mode=args.diffusion_score_mode,
            tau_grid=tau_grid,
            n_replicates=args.n_diffusion_null,
            seed=args.seed + 1001,
            nswap_per_edge=args.degree_graph_nswap_per_edge,
            max_tries_per_swap=args.degree_graph_max_tries_per_swap,
            cache_path=args.degree_graph_cache,
        )
        summaries.append(summary)
        tau_summaries.append(tau)
        nulls.append(null)

    summary_table = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    tau_table = pd.concat(tau_summaries, ignore_index=True) if tau_summaries else pd.DataFrame()
    null_table = pd.concat(nulls, ignore_index=True) if nulls else pd.DataFrame()
    write_table(args.out_dir / f"{args.trait}.diffusion_kernel_summary.tsv", summary_table)
    write_table(args.out_dir / f"{args.trait}.diffusion_kernel_tau_stats.tsv", tau_table)
    if args.save_null_distributions:
        write_table(args.out_dir / f"{args.trait}.diffusion_kernel_null_distribution.tsv.gz", null_table)
    print(f"Wrote diffusion outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
