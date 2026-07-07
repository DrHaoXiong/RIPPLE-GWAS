#!/usr/bin/env python
"""Backfill an analysis-ready LCC graph edge cache for existing RIPPLE outputs.

This utility is for performance maintenance. It lets calibration and diffusion
scripts reuse the exact scored largest connected component graph without
re-reading and re-filtering the original reference graph on every run.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ripple.graph import preprocess_reference_graph  # noqa: E402
from ripple.io.graph import networkx_to_edge_list, read_edge_list, write_edge_list  # noqa: E402
from run_height_mvp import DEFAULT_STRING_INFO, DEFAULT_STRING_LINKS, build_string_graph  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_ANALYSIS_DIR = PRIVATE_ROOT / "30_analysis" / "dr_mvp_string_final5000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--trait", default="DR_MVP")
    parser.add_argument("--graph-name", default="string_ppi")
    parser.add_argument("--graph-edge-list", type=Path, default=None)
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--lcc-scores", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def default_lcc_scores_path(args: argparse.Namespace) -> Path:
    return args.analysis_dir / "tables" / f"{args.trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"


def default_out_path(args: argparse.Namespace) -> Path:
    return args.analysis_dir / "tables" / f"{args.trait}.analysis_graph_edges.tsv.gz"


def load_graph(args: argparse.Namespace, gene_symbols: tuple[str, ...]):
    if args.graph_edge_list is not None:
        graph_edges = read_edge_list(args.graph_edge_list, source=args.graph_name)
        return graph_edges, preprocess_reference_graph(graph_edges.edges, gene_universe=gene_symbols)
    graph_args = argparse.Namespace(
        string_links=args.string_links,
        string_info=args.string_info,
        string_min_score=args.string_min_score,
    )
    return build_string_graph(graph_args, gene_symbols)


def main() -> None:
    args = parse_args()
    lcc_scores_path = args.lcc_scores or default_lcc_scores_path(args)
    out_path = args.out or default_out_path(args)
    if out_path.exists() and not args.force:
        raise FileExistsError(f"{out_path} exists. Use --force to overwrite.")

    scores = pd.read_csv(lcc_scores_path, sep="\t", compression="infer")
    gene_symbols = tuple(scores["gene_symbol"].dropna().astype(str).unique())
    graph_edges, graph_pre = load_graph(args, gene_symbols)
    written = write_edge_list(
        out_path,
        networkx_to_edge_list(graph_pre.largest_component),
        source=f"{args.trait}_{args.graph_name}_largest_connected_component",
    )

    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "purpose": "analysis_ready_lcc_graph_edge_cache",
        "trait": args.trait,
        "graph_name": args.graph_name,
        "lcc_scores": str(lcc_scores_path),
        "out": str(out_path),
        "source_graph_edge_list": str(args.graph_edge_list) if args.graph_edge_list is not None else None,
        "source_graph_load_report": asdict(graph_edges.report),
        "coverage_report": asdict(graph_pre.coverage_report),
        "n_scored_genes": int(len(gene_symbols)),
        "n_cached_graph_nodes": int(graph_pre.largest_component.number_of_nodes()),
        "n_cached_graph_edges": int(written.report.n_edges_output),
    }
    report_path = out_path.with_suffix(out_path.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote analysis graph edge cache: {out_path}", flush=True)
    print(f"Wrote cache report: {report_path}", flush=True)


if __name__ == "__main__":
    main()
