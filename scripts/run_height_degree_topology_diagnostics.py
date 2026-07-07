#!/usr/bin/env python
"""Post-hoc HEIGHT diagnostics for degree and topology effects in percolation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ripple.defaults import RANK_FRACTION_GRID  # noqa: E402
from ripple.nulls import degree_preserving_graph_replicates, graph_component_summary  # noqa: E402
from ripple.percolation import (  # noqa: E402
    classify_percolation_architecture,
    compute_degree_matched_node_percolation_null,
    percolation_auc,
    percolation_curve,
    prepare_degree_matched_rank_sets,
    rank_nodes_by_score,
    summarize_percolation_null,
)
from run_height_mvp import DEFAULT_STRING_INFO, DEFAULT_STRING_LINKS, build_string_graph, write_table  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_ANALYSIS_DIR = PRIVATE_ROOT / "30_analysis" / "height_irn_analysis_ready"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--n-degree-graph-null", type=int, default=100)
    parser.add_argument("--degree-graph-nswap-per-edge", type=float, default=1.0)
    parser.add_argument("--degree-graph-max-tries-per-swap", type=float, default=20.0)
    parser.add_argument("--n-degree-matched-node-null", type=int, default=500)
    parser.add_argument("--degree-matched-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260613)
    return parser.parse_args()


def compute_degree_preserving_graph_null(
    graph: nx.Graph,
    ranking: pd.DataFrame,
    *,
    n_replicates: int,
    seed: int,
    nswap_per_edge: float,
    max_tries_per_swap: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    auc_rows: list[dict[str, float | int]] = []
    curve_rows: list[pd.DataFrame] = []
    diagnostics: list[dict[str, float | int]] = []
    null_graphs = degree_preserving_graph_replicates(
        graph,
        n_replicates=n_replicates,
        seed=seed,
        nswap_per_edge=nswap_per_edge,
        max_tries_per_swap=max_tries_per_swap,
    )
    for idx, null_graph in enumerate(null_graphs):
        curve = percolation_curve(null_graph, ranking, RANK_FRACTION_GRID, node_col="gene_symbol")
        auc = percolation_auc(curve)
        auc_rows.append({"replicate": idx, "percolation_auc": auc})
        curve["replicate"] = idx
        curve_rows.append(curve)
        diagnostics.append({"replicate": idx, **graph_component_summary(null_graph)})
        if (idx + 1) % 10 == 0 or idx + 1 == n_replicates:
            print(f"Computed {idx + 1:,}/{n_replicates:,} degree-preserving graph nulls", flush=True)
    return (
        pd.DataFrame(auc_rows),
        pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame(),
        pd.DataFrame(diagnostics),
    )


def complete_legacy_null_summary(summary: dict[str, object], observed_auc: float) -> dict[str, object]:
    """Backfill Z/delta fields for summaries written before calibration reporting."""

    out = dict(summary)
    if "mean" in out and "delta" not in out:
        out["delta"] = float(observed_auc - float(out["mean"]))
    if "z" not in out and "delta" in out and "sd" in out:
        sd = float(out["sd"])
        out["z"] = float(float(out["delta"]) / sd) if sd > 0 else float("nan")
    return out


def write_report(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# HEIGHT_IRN Degree/Topology Diagnostics",
        "",
        f"- Architecture class: {summary['architecture_class']}",
        f"- Observed percolation AUC: {summary['observed_auc']:.6f}",
        f"- Extended degree-preserving graph null Z: {summary['degree_preserving_graph_null']['z']:.6f}",
        f"- Degree-matched node null Z: {summary['degree_matched_node_null']['z']:.6f}",
        f"- Degree-preserving graph null mean AUC: {summary['degree_preserving_graph_null']['mean']:.6f}",
        f"- Degree-matched node null mean AUC: {summary['degree_matched_node_null']['mean']:.6f}",
        "",
        summary["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tables_dir = args.analysis_dir / "tables"
    reports_dir = args.analysis_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    scores_path = tables_dir / "HEIGHT_IRN.lcc_gene_scores.1000G_LD.residualized.tsv.gz"
    print(f"Loading LCC scores: {scores_path}", flush=True)
    scores = pd.read_csv(scores_path, sep="\t", compression="infer")
    scores["gene_symbol"] = scores["gene_symbol"].astype(str)

    print("Rebuilding STRING graph for scored LCC genes", flush=True)
    graph_args = argparse.Namespace(
        string_links=args.string_links,
        string_info=args.string_info,
        string_min_score=args.string_min_score,
    )
    _, graph_pre = build_string_graph(graph_args, tuple(scores["gene_symbol"]))
    graph = graph_pre.largest_component
    graph_nodes = {str(node) for node in graph.nodes()}
    scores = scores[scores["gene_symbol"].isin(graph_nodes)].copy().sort_values("gene_symbol").reset_index(drop=True)
    degree = dict(graph.degree())
    scores["graph_degree"] = scores["gene_symbol"].map(degree).astype(int)

    ranking = rank_nodes_by_score(scores, node_col="gene_symbol", score_col="assoc_resid_score")
    observed_curve = percolation_curve(graph, ranking, RANK_FRACTION_GRID, node_col="gene_symbol")
    observed_auc = percolation_auc(observed_curve)

    print("Computing extended degree-preserving graph null", flush=True)
    graph_auc, graph_curves, graph_diagnostics = compute_degree_preserving_graph_null(
        graph,
        ranking,
        n_replicates=args.n_degree_graph_null,
        seed=args.seed + 303,
        nswap_per_edge=args.degree_graph_nswap_per_edge,
        max_tries_per_swap=args.degree_graph_max_tries_per_swap,
    )

    print("Computing degree-matched node null on observed STRING topology", flush=True)
    selected_bin_counts, bin_to_nodes, degree_profile = prepare_degree_matched_rank_sets(
        scores,
        ranking,
        RANK_FRACTION_GRID,
        node_col="gene_symbol",
        degree_col="graph_degree",
        n_bins=args.degree_matched_bins,
    )
    matched_auc, matched_curves = compute_degree_matched_node_percolation_null(
        graph,
        selected_bin_counts,
        bin_to_nodes,
        n_replicates=args.n_degree_matched_node_null,
        seed=args.seed + 404,
        progress_interval=100,
    )

    graph_summary = summarize_percolation_null(graph_auc, observed_auc)
    matched_summary = summarize_percolation_null(matched_auc, observed_auc)
    snp_summary = {"z": float("nan")}
    degree_strat_summary = {"z": float("nan")}
    analysis_summary_path = reports_dir / "HEIGHT_IRN.analysis_ready_summary.json"
    if analysis_summary_path.exists():
        analysis_summary = json.loads(analysis_summary_path.read_text(encoding="utf-8"))
        snp_summary = complete_legacy_null_summary(
            analysis_summary.get("snp_permutation_null_summary", snp_summary),
            observed_auc,
        )
        degree_strat_summary = complete_legacy_null_summary(
            analysis_summary.get("degree_stratified_null_summary", degree_strat_summary),
            observed_auc,
        )
    architecture = classify_percolation_architecture(
        snp_permutation_null=snp_summary,
        degree_stratified_null=degree_strat_summary,
        degree_matched_node_null=matched_summary,
        degree_preserving_graph_null=graph_summary,
    )
    interpretation = str(architecture["interpretation"])

    write_table(tables_dir / "HEIGHT_IRN.percolation_auc.1000G_LD.degree_preserving_graph_null.n100.tsv", graph_auc)
    write_table(tables_dir / "HEIGHT_IRN.percolation_curves.1000G_LD.degree_preserving_graph_null.n100.tsv", graph_curves)
    write_table(tables_dir / "HEIGHT_IRN.degree_preserving_graph_null.n100.diagnostics.tsv", graph_diagnostics)
    write_table(tables_dir / "HEIGHT_IRN.percolation_auc.1000G_LD.degree_matched_node_null.tsv", matched_auc)
    write_table(tables_dir / "HEIGHT_IRN.percolation_curves.1000G_LD.degree_matched_node_null.tsv", matched_curves)
    write_table(tables_dir / "HEIGHT_IRN.degree_profile.by_rank_fraction.tsv", degree_profile)

    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "trait": "HEIGHT_IRN",
        "analysis_dir": str(args.analysis_dir),
        "observed_auc": observed_auc,
        "rank_fraction_grid": list(RANK_FRACTION_GRID),
        "n_lcc_scored_genes": int(len(scores)),
        "degree_preserving_graph_null": graph_summary,
        "degree_preserving_graph_null_parameters": {
            "n_replicates": args.n_degree_graph_null,
            "nswap_per_edge": args.degree_graph_nswap_per_edge,
            "max_tries_per_swap": args.degree_graph_max_tries_per_swap,
        },
        "degree_matched_node_null": matched_summary,
        "percolation_architecture": architecture,
        "architecture_class": architecture["architecture_class"],
        "degree_matched_node_null_parameters": {
            "n_replicates": args.n_degree_matched_node_null,
            "degree_bins": args.degree_matched_bins,
        },
        "interpretation": interpretation,
        "outputs": {
            "degree_preserving_graph_auc": str(
                tables_dir / "HEIGHT_IRN.percolation_auc.1000G_LD.degree_preserving_graph_null.n100.tsv"
            ),
            "degree_matched_node_auc": str(
                tables_dir / "HEIGHT_IRN.percolation_auc.1000G_LD.degree_matched_node_null.tsv"
            ),
            "degree_profile": str(tables_dir / "HEIGHT_IRN.degree_profile.by_rank_fraction.tsv"),
        },
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True)
    (reports_dir / "HEIGHT_IRN.degree_topology_diagnostics_summary.json").write_text(summary_text, encoding="utf-8")
    write_report(reports_dir / "HEIGHT_IRN.degree_topology_diagnostics_report.md", summary)
    print(f"Wrote degree/topology diagnostics to {args.analysis_dir}", flush=True)


if __name__ == "__main__":
    main()
