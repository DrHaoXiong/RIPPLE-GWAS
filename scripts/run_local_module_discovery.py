#!/usr/bin/env python
"""Backfill RIPPLE local module discovery for existing analysis-ready outputs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ripple.diagnostics import build_trait_suitability_diagnostic, render_trait_architecture_markdown  # noqa: E402
from ripple.graph import preprocess_reference_graph  # noqa: E402
from ripple.io.graph import read_edge_list  # noqa: E402
from ripple.modules import (  # noqa: E402
    load_gene_sets,
    render_module_discovery_report,
    run_local_module_discovery,
)
from ripple.percolation import classify_percolation_architecture, summarize_percolation_null  # noqa: E402
from run_height_mvp import DEFAULT_STRING_INFO, DEFAULT_STRING_LINKS, build_string_graph, write_table  # noqa: E402

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"

CURRENT_ANALYSES = {
    "HEIGHT_IRN": ANALYSIS_ROOT / "height_irn_analysis_ready",
    "BMI_IRN": ANALYSIS_ROOT / "bmi_irn_analysis_ready",
    "T2D": ANALYSIS_ROOT / "t2d_analysis_ready",
    "DM_RETINOPATHY_EXMORE": ANALYSIS_ROOT / "dm_retinopathy_exmore_analysis_ready",
    "DM_RETINOPATHY_EXMORE_WITH_MHC": ANALYSIS_ROOT / "dm_retinopathy_exmore_with_mhc_analysis_ready",
    "DM_RETINOPATHY_EXMORE_NO_MHC_NO_APOE": ANALYSIS_ROOT
    / "dm_retinopathy_exmore_no_mhc_no_apoe_analysis_ready",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", default=None)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--all-current-traits", action="store_true")
    parser.add_argument("--graph-name", default="string_ppi")
    parser.add_argument("--graph-edge-list", type=Path, default=None)
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--gene-set-file", type=Path, default=None)
    parser.add_argument("--no-default-dr-panel", action="store_true")
    parser.add_argument("--min-module-size", type=int, default=5)
    parser.add_argument("--min-module-subthreshold-genes", type=int, default=3)
    parser.add_argument("--max-local-modules", type=int, default=20)
    parser.add_argument("--n-module-random-null", type=int, default=200)
    parser.add_argument("--n-module-degree-matched-null", type=int, default=200)
    parser.add_argument("--n-module-degree-graph-null", type=int, default=20)
    parser.add_argument("--n-module-selection-aware-null", type=int, default=200)
    parser.add_argument("--n-pathway-random-null", type=int, default=200)
    parser.add_argument("--n-pathway-degree-matched-null", type=int, default=200)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260614)
    return parser.parse_args()


def lcc_scores_path(analysis_dir: Path, trait: str) -> Path:
    return analysis_dir / "tables" / f"{trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"


def summary_path(analysis_dir: Path, trait: str) -> Path:
    return analysis_dir / "reports" / f"{trait}.analysis_ready_summary.json"


def null_residualized_scores_path(analysis_dir: Path, trait: str) -> Path:
    return analysis_dir / "tables" / f"{trait}.null_residualized_scores.npz"


def load_selection_null_scores(analysis_dir: Path, trait: str, scores: pd.DataFrame) -> np.ndarray | None:
    path = null_residualized_scores_path(analysis_dir, trait)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        null_resid = np.asarray(data["null_resid"], dtype=float)
        gene_symbols = np.asarray(data["gene_symbols"]).astype(str)
    requested = scores["gene_symbol"].astype(str).to_numpy()
    if np.array_equal(gene_symbols, requested):
        return null_resid
    index_by_gene = {gene: idx for idx, gene in enumerate(gene_symbols)}
    missing = [gene for gene in requested if gene not in index_by_gene]
    if missing:
        raise ValueError(f"Null residualized score matrix is missing {len(missing)} score genes.")
    order = np.array([index_by_gene[gene] for gene in requested], dtype=int)
    return null_resid[:, order]


def global_module_gate_pass_from_summary(summary: dict[str, object]) -> bool:
    architecture = summary.get("percolation_architecture", {})
    if isinstance(architecture, dict) and "degree_matched_node_positive" in architecture:
        return bool(architecture["degree_matched_node_positive"])
    null_summary = summary.get("degree_matched_node_null_summary", {})
    if isinstance(null_summary, dict):
        try:
            return float(null_summary.get("z", float("nan"))) >= 2.0
        except (TypeError, ValueError):
            return False
    return False


def _table_path_from_summary(summary: dict[str, object], output_key: str, fallback: Path) -> Path:
    outputs = summary.get("outputs", {})
    if isinstance(outputs, dict) and outputs.get(output_key):
        return Path(str(outputs[output_key]))
    return fallback


def repair_percolation_summaries(summary: dict[str, object], analysis_dir: Path, trait: str) -> None:
    """Backfill z/delta fields in older summaries from stored null AUC tables."""

    try:
        observed_auc = float(summary["percolation_auc_observed"])
    except (KeyError, TypeError, ValueError):
        return
    tables_dir = analysis_dir / "tables"
    specs = {
        "snp_permutation_null_summary": (
            "null_percolation_auc",
            tables_dir / f"{trait}.percolation_auc.1000G_LD.null.tsv",
        ),
        "degree_stratified_null_summary": (
            "degree_stratified_null_auc",
            tables_dir / f"{trait}.percolation_auc.1000G_LD.degree_stratified_null.tsv",
        ),
        "degree_matched_node_null_summary": (
            "degree_matched_node_null_auc",
            tables_dir / f"{trait}.percolation_auc.1000G_LD.degree_matched_node_null.tsv",
        ),
        "degree_preserving_graph_null_summary": (
            "degree_preserving_graph_null_auc",
            tables_dir / f"{trait}.percolation_auc.1000G_LD.degree_preserving_graph_null.tsv",
        ),
    }
    for summary_key, (output_key, fallback) in specs.items():
        current = summary.get(summary_key)
        if isinstance(current, dict) and "z" in current and "delta" in current:
            continue
        table_path = _table_path_from_summary(summary, output_key, fallback)
        if not table_path.exists():
            continue
        null_auc = pd.read_csv(table_path, sep="\t")
        summary[summary_key] = summarize_percolation_null(null_auc, observed_auc)
    architecture = summary.get("percolation_architecture", {})
    if not isinstance(architecture, dict) or not architecture.get("architecture_class"):
        summary["percolation_architecture"] = classify_percolation_architecture(
            snp_permutation_null=summary.get("snp_permutation_null_summary", {}),
            degree_stratified_null=summary.get("degree_stratified_null_summary", {}),
            degree_matched_node_null=summary.get("degree_matched_node_null_summary", {}),
            degree_preserving_graph_null=summary.get("degree_preserving_graph_null_summary", {}),
        )


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


def summarize_modules(graph_name: str, modules: pd.DataFrame) -> dict[str, object]:
    if modules.empty:
        return {
            "graph_name": graph_name,
            "n_candidate_modules": 0,
            "n_calibrated_modules": 0,
            "n_topology_specific_modules": 0,
            "top_modules": [],
        }
    claim_col = (
        "is_reportable_calibrated_module"
        if "is_reportable_calibrated_module" in modules.columns
        else "is_calibrated_weak_signal_module"
    )
    topology_col = (
        "is_reportable_topology_specific_module"
        if "is_reportable_topology_specific_module" in modules.columns
        else "is_topology_specific_module"
    )
    calibrated = modules.loc[modules[claim_col].astype(bool)].copy()
    return {
        "graph_name": graph_name,
        "global_module_gate_pass": bool(modules.get("passes_global_module_gate", pd.Series([True])).iloc[0]),
        "n_candidate_modules": int(len(modules)),
        "n_module_level_calibrated_candidates": int(modules["is_calibrated_weak_signal_module"].sum()),
        "n_broad_calibrated_components": int(
            (modules["is_calibrated_weak_signal_module"].astype(bool) & (modules["n_genes"].astype(int) >= 200)).sum()
        ),
        "n_calibrated_modules": int(modules[claim_col].sum()),
        "n_topology_specific_modules": int(modules[topology_col].sum()),
        "top_modules": calibrated.head(5).to_dict(orient="records") if not calibrated.empty else [],
    }


def run_one(args: argparse.Namespace, trait: str, analysis_dir: Path) -> None:
    tables_dir = analysis_dir / "tables"
    reports_dir = analysis_dir / "reports"
    scores = pd.read_csv(lcc_scores_path(analysis_dir, trait), sep="\t", compression="infer")
    selection_null_scores = load_selection_null_scores(analysis_dir, trait, scores)
    gene_symbols = tuple(scores["gene_symbol"].dropna().astype(str).unique())
    graph_edges, graph_pre = load_graph(args, gene_symbols)
    gene_sets = load_gene_sets(
        args.gene_set_file,
        include_default_dr_panel=not args.no_default_dr_panel,
    )
    modules, module_nulls, pathway = run_local_module_discovery(
        graph_pre.largest_component,
        scores,
        gene_sets=gene_sets,
        seed=args.seed,
        min_module_size=args.min_module_size,
        min_subthreshold_genes=args.min_module_subthreshold_genes,
        max_modules=args.max_local_modules,
        n_module_random=args.n_module_random_null,
        n_module_degree_matched=args.n_module_degree_matched_null,
        n_module_degree_graph=args.n_module_degree_graph_null,
        n_module_selection_aware=args.n_module_selection_aware_null,
        selection_null_scores=selection_null_scores,
        n_pathway_random=args.n_pathway_random_null,
        n_pathway_degree_matched=args.n_pathway_degree_matched_null,
        degree_bins=args.degree_bins,
    )
    summary_file = summary_path(analysis_dir, trait)
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    if summary:
        repair_percolation_summaries(summary, analysis_dir, trait)
    global_module_gate_pass = global_module_gate_pass_from_summary(summary)
    if not modules.empty:
        modules = modules.copy()
        is_local_component = modules["n_genes"].astype(int) < 200
        modules["passes_global_module_gate"] = global_module_gate_pass
        modules["is_reportable_calibrated_module"] = (
            modules["is_calibrated_weak_signal_module"].astype(bool) & is_local_component & global_module_gate_pass
        )
        modules["is_reportable_topology_specific_module"] = (
            modules["is_topology_specific_module"].astype(bool) & is_local_component & global_module_gate_pass
        )
    local_module_path = tables_dir / f"{trait}.local_modules.tsv"
    local_module_null_path = tables_dir / f"{trait}.local_module_nulls.tsv"
    pathway_path = tables_dir / f"{trait}.pathway_subgraph_tests.tsv"
    write_table(local_module_path, modules)
    write_table(local_module_null_path, module_nulls)
    write_table(pathway_path, pathway)

    if summary_file.exists():
        summary["graph_name"] = str(args.graph_name)
        summary["graph_edge_list"] = str(args.graph_edge_list) if args.graph_edge_list is not None else None
        summary["graph_load_report"] = asdict(graph_edges.report)
        summary["n_module_random_null"] = args.n_module_random_null
        summary["n_module_degree_matched_null"] = args.n_module_degree_matched_null
        summary["n_module_degree_graph_null"] = args.n_module_degree_graph_null
        summary["n_module_selection_aware_null"] = args.n_module_selection_aware_null
        summary["local_module_summary"] = summarize_modules(str(args.graph_name), modules)
        summary.setdefault("outputs", {})
        summary["outputs"].update(
            {
                "local_modules": str(local_module_path),
                "local_module_nulls": str(local_module_null_path),
                "pathway_subgraph_tests": str(pathway_path),
            }
        )
        summary["trait_suitability"] = build_trait_suitability_diagnostic(summary)
        summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (reports_dir / f"{trait}.trait_suitability.json").write_text(
            json.dumps(summary["trait_suitability"], indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (reports_dir / f"{trait}.architecture_report.md").write_text(
            render_trait_architecture_markdown(summary),
            encoding="utf-8",
        )
    (reports_dir / f"{trait}.module_discovery_report.md").write_text(
        render_module_discovery_report(
            trait=trait,
            graph_name=str(args.graph_name),
            modules=modules,
            pathway=pathway,
            global_gate_pass=global_module_gate_pass,
        ),
        encoding="utf-8",
    )
    print(f"Wrote local module discovery outputs for {trait} to {analysis_dir}", flush=True)


def main() -> None:
    args = parse_args()
    if args.all_current_traits:
        for trait, analysis_dir in CURRENT_ANALYSES.items():
            run_one(args, trait, analysis_dir)
        return
    if args.trait is None:
        raise ValueError("Use --trait or --all-current-traits.")
    analysis_dir = args.analysis_dir or ANALYSIS_ROOT / f"{args.trait.lower()}_analysis_ready"
    run_one(args, str(args.trait), analysis_dir)


if __name__ == "__main__":
    main()
