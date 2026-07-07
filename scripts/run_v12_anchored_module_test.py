#!/usr/bin/env python
"""Run RIPPLE V1.2 anchored module diagnostics on existing analysis-ready scores."""

from __future__ import annotations

import argparse
import json
import sys
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
from ripple.io.graph import read_edge_list  # noqa: E402
from ripple.modules import (  # noqa: E402
    DEFAULT_DR_GENE_SETS,
    anchored_module_tests,
    build_louvain_anchor_library,
    gene_sets_to_library,
    load_anchored_gene_set_library,
    merge_anchored_libraries,
    render_anchored_module_report,
)
from run_height_mvp import DEFAULT_STRING_INFO, DEFAULT_STRING_LINKS, build_string_graph  # noqa: E402

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_TRAIT = "DR_MVP"
DEFAULT_ANALYSIS_DIR = ANALYSIS_ROOT / "dr_mvp_analysis_ready"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "tier4_v12_anchored_module_test_dr_mvp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", default=DEFAULT_TRAIT)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--graph-name", default="string_ppi")
    parser.add_argument("--graph-edge-list", type=Path, default=None)
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--gene-set-file", type=Path, default=None)
    parser.add_argument("--external-gene-set-source-type", default="independent_external")
    parser.add_argument("--no-default-dr-panel", action="store_true")
    parser.add_argument("--no-louvain-communities", action="store_true")
    parser.add_argument("--louvain-min-size", type=int, default=10)
    parser.add_argument("--louvain-max-size", type=int, default=300)
    parser.add_argument("--louvain-resolution", type=float, default=1.0)
    parser.add_argument("--min-present", type=int, default=5)
    parser.add_argument("--n-degree-matched-null", type=int, default=200)
    parser.add_argument("--n-score-permutation-null", type=int, default=200)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def lcc_scores_path(analysis_dir: Path, trait: str) -> Path:
    return analysis_dir / "tables" / f"{trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "reports").mkdir(parents=True, exist_ok=True)


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, compression="infer")


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


def build_library(args: argparse.Namespace, graph) -> tuple[object, dict[str, int]]:
    libraries = []
    if not args.no_default_dr_panel:
        libraries.append(
            gene_sets_to_library(
                DEFAULT_DR_GENE_SETS,
                module_source="built_in_dr_panel",
                annotation_source_type="internal_support",
                module_category="built_in_dr_panel",
            )
        )
    if args.gene_set_file is not None:
        libraries.append(
            load_anchored_gene_set_library(
                args.gene_set_file,
                default_module_source=f"user_gene_set_file:{args.gene_set_file}",
                default_annotation_source_type=args.external_gene_set_source_type,
            )
        )
    if not args.no_louvain_communities:
        libraries.append(
            build_louvain_anchor_library(
                graph,
                min_size=args.louvain_min_size,
                max_size=args.louvain_max_size,
                resolution=args.louvain_resolution,
                seed=args.seed,
            )
        )
    if not libraries:
        raise ValueError("No anchored module library requested.")
    merged = merge_anchored_libraries(*libraries)
    counts = {
        "n_modules_total": len(merged.gene_sets),
        "n_builtin_dr_panel": 0 if args.no_default_dr_panel else len(DEFAULT_DR_GENE_SETS),
        "n_louvain_modules": sum(
            1 for source in merged.module_source.values() if source == "graph_louvain_community"
        ),
        "n_external_gene_sets": sum(
            1
            for source in merged.module_source.values()
            if source not in {"built_in_dr_panel", "graph_louvain_community"}
        ),
    }
    return merged, counts


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"

    scores_path = lcc_scores_path(args.analysis_dir, args.trait)
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    scores = pd.read_csv(scores_path, sep="\t", compression="infer")
    gene_symbols = tuple(scores["gene_symbol"].dropna().astype(str).unique())

    print(f"Loading graph for {args.trait}", flush=True)
    graph_edges, graph_pre = load_graph(args, gene_symbols)
    graph = graph_pre.largest_component
    write_table(tables_dir / f"{args.trait}.v12_anchored_graph_edges.tsv.gz", graph_edges.edges)

    print("Building anchored module library", flush=True)
    library, library_counts = build_library(args, graph)
    library_table = pd.DataFrame(
        {
            "module_name": list(library.gene_sets),
            "module_source": [library.module_source[name] for name in library.gene_sets],
            "annotation_source_type": [
                library.annotation_source_type[name] for name in library.gene_sets
            ],
            "module_category": [library.module_category.get(name, "unspecified") for name in library.gene_sets],
            "n_query_genes": [len(library.gene_sets[name]) for name in library.gene_sets],
            "query_genes": [",".join(sorted(library.gene_sets[name])) for name in library.gene_sets],
        }
    )
    write_table(tables_dir / f"{args.trait}.v12_anchored_module_library.tsv.gz", library_table)

    print("Running anchored module tests", flush=True)
    modules, nulls, summary = anchored_module_tests(
        graph,
        scores,
        library,
        min_present=args.min_present,
        n_degree_matched=args.n_degree_matched_null,
        n_score_permutation=args.n_score_permutation_null,
        degree_bins=args.degree_bins,
        seed=args.seed,
    )
    write_table(tables_dir / f"{args.trait}.v12_anchored_module_tests.tsv", modules)
    write_table(tables_dir / f"{args.trait}.v12_anchored_module_nulls.tsv.gz", nulls)

    run_summary = {
        **summary,
        **library_counts,
        "trait": args.trait,
        "analysis_dir": str(args.analysis_dir),
        "graph_name": args.graph_name,
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "score_path": str(scores_path),
        "output_dir": str(args.out_dir),
        "seed": int(args.seed),
        "created_utc": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "parameters": {
            "min_present": int(args.min_present),
            "n_degree_matched_null": int(args.n_degree_matched_null),
            "n_score_permutation_null": int(args.n_score_permutation_null),
            "degree_bins": int(args.degree_bins),
            "louvain_min_size": int(args.louvain_min_size),
            "louvain_max_size": int(args.louvain_max_size),
            "louvain_resolution": float(args.louvain_resolution),
        },
    }
    summary_path = reports_dir / f"{args.trait}.v12_anchored_module_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2, sort_keys=True), encoding="utf-8")

    report = render_anchored_module_report(
        trait=args.trait,
        graph_name=args.graph_name,
        modules=modules,
        summary=run_summary,
    )
    (reports_dir / f"{args.trait}.v12_anchored_module_report.md").write_text(
        report + "\n",
        encoding="utf-8",
    )
    print(f"Wrote V1.2 anchored module diagnostics to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
