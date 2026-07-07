#!/usr/bin/env python
"""Run full reselection null calibration for DR_MVP local modules.

This script does not rerun LD scoring. It uses existing final-scale LCC
residualized gene scores and analysis graphs, then repeats the complete local
module selection procedure under degree-stratified score permutations.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.graph import graph_from_edge_list  # noqa: E402
from ripple.io.graph import read_edge_list  # noqa: E402
from ripple.modules import selection_aware_module_null  # noqa: E402
from ripple.nulls.score_permutation import assign_degree_bins  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "module_reselection_null_v1"
THIS_SCRIPT = Path(__file__).resolve()


@dataclass(frozen=True)
class AnalysisSpec:
    trait: str
    analysis_id: str
    analysis_dir: Path
    graph_edges_path: Path

    @property
    def tables_dir(self) -> Path:
        return self.analysis_dir / "tables"

    @property
    def scores_path(self) -> Path:
        return self.tables_dir / f"{self.trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"

    @property
    def modules_path(self) -> Path:
        return self.tables_dir / f"{self.trait}.local_modules.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trait", action="append", default=[])
    parser.add_argument("--n-reselection-null", type=int, default=5000)
    parser.add_argument("--engine", choices=["fast", "reference"], default="fast")
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--min-module-size", type=int, default=5)
    parser.add_argument("--min-module-subthreshold-genes", type=int, default=3)
    parser.add_argument("--max-local-modules", type=int, default=20)
    parser.add_argument("--broad-component-min-size", type=int, default=200)
    parser.add_argument("--broad-component-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def default_specs() -> list[AnalysisSpec]:
    default_graph = ANALYSIS_ROOT / "dr_mvp_string_final5000" / "tables" / "DR_MVP.analysis_graph_edges.tsv.gz"
    return [
        AnalysisSpec(
            trait="DR_MVP",
            analysis_id="DR_MVP_default_final5000",
            analysis_dir=ANALYSIS_ROOT / "dr_mvp_string_final5000",
            graph_edges_path=default_graph,
        ),
        AnalysisSpec(
            trait="DR_MVP_NO_MHC_NO_APOE",
            analysis_id="DR_MVP_no_MHC_no_APOE_final5000",
            analysis_dir=ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000",
            graph_edges_path=default_graph,
        ),
    ]


def select_specs(args: argparse.Namespace) -> list[AnalysisSpec]:
    specs = default_specs()
    if not args.trait:
        return specs
    requested = set(args.trait)
    selected = [spec for spec in specs if spec.trait in requested or spec.analysis_id in requested]
    known = {spec.trait for spec in specs} | {spec.analysis_id for spec in specs}
    missing = sorted(requested - known)
    if missing:
        raise ValueError(f"Unknown trait or analysis_id: {missing}")
    return selected


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", compression="infer")


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, compression="infer")


def load_analysis_graph(spec: AnalysisSpec, scores: pd.DataFrame):
    edges = read_edge_list(spec.graph_edges_path, sep="\t", source=spec.analysis_id).edges
    gene_universe = scores["gene_symbol"].dropna().astype(str).unique()
    graph = graph_from_edge_list(edges, gene_universe=gene_universe)
    isolates = list(node for node, degree in graph.degree() if degree == 0)
    graph.remove_nodes_from(isolates)
    if graph.number_of_nodes() == 0:
        raise ValueError(f"Analysis graph is empty for {spec.analysis_id}")
    largest = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest).copy()


def empirical_upper(null_values: np.ndarray, observed: float) -> float:
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float((np.count_nonzero(finite >= observed) + 1) / (finite.size + 1))


def z_score(observed: float, null_values: np.ndarray) -> float:
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    sd = float(np.std(finite, ddof=1))
    return float((observed - float(np.mean(finite))) / sd) if sd > 0 else float("nan")


def bool_col(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(False, index=table.index)
    return table[column].astype(str).str.lower().eq("true")


def fast_full_reselection_null(
    graph: nx.Graph,
    scores: pd.DataFrame,
    *,
    n_replicates: int,
    degree_bins: int,
    min_module_size: int,
    min_subthreshold_genes: int,
    max_modules: int,
    broad_component_min_size: int,
    broad_component_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Fast full reselection null retaining only max module statistics per replicate."""

    work = scores.copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str)
    graph_nodes = set(str(node) for node in graph.nodes())
    work = work[work["gene_symbol"].isin(graph_nodes)].reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()
    nodes = work["gene_symbol"].to_numpy(dtype=str)
    score = pd.to_numeric(work["assoc_resid_score"], errors="raise").to_numpy(dtype=float)
    degree = pd.to_numeric(work["graph_degree"], errors="raise")
    if "assoc_p_g" in work.columns:
        subthreshold = pd.to_numeric(work["assoc_p_g"], errors="coerce").fillna(1.0).to_numpy(dtype=float) > 5e-8
    else:
        subthreshold = np.ones(len(work), dtype=bool)
    bins = assign_degree_bins(degree, n_bins=degree_bins).to_numpy(dtype=int)
    groups = [np.flatnonzero(bins == bin_id) for bin_id in sorted(np.unique(bins))]
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    neighbors = [
        np.array([node_to_idx[str(neighbor)] for neighbor in graph.neighbors(node) if str(neighbor) in node_to_idx])
        for node in nodes
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    fractions = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
    n_nodes = len(nodes)
    threshold_items = [(fraction, max(1, int(round(float(fraction) * n_nodes)))) for fraction in fractions]
    max_k = max(k for _, k in threshold_items)
    for replicate in range(n_replicates):
        permuted = score.copy()
        for group_idx in groups:
            if group_idx.size > 1:
                permuted[group_idx] = permuted[rng.permutation(group_idx)]
        order = np.argsort(-permuted, kind="mergesort")
        parent = np.arange(n_nodes, dtype=int)
        comp_size = np.zeros(n_nodes, dtype=int)
        comp_sum = np.zeros(n_nodes, dtype=float)
        comp_subthreshold = np.zeros(n_nodes, dtype=int)
        comp_edges = np.zeros(n_nodes, dtype=int)
        active = np.zeros(n_nodes, dtype=bool)
        candidates: list[tuple[bool, float, int, float]] = []

        def find(node_idx: int) -> int:
            root = node_idx
            while parent[root] != root:
                root = parent[root]
            while parent[node_idx] != node_idx:
                next_idx = parent[node_idx]
                parent[node_idx] = root
                node_idx = next_idx
            return root

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                comp_edges[left_root] += 1
                return
            if comp_size[left_root] < comp_size[right_root]:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root
            comp_size[left_root] += comp_size[right_root]
            comp_sum[left_root] += comp_sum[right_root]
            comp_subthreshold[left_root] += comp_subthreshold[right_root]
            comp_edges[left_root] += comp_edges[right_root] + 1

        threshold_idx = 0
        for added_count, node_idx in enumerate(order[:max_k], start=1):
            active[node_idx] = True
            parent[node_idx] = node_idx
            comp_size[node_idx] = 1
            comp_sum[node_idx] = permuted[node_idx]
            comp_subthreshold[node_idx] = int(subthreshold[node_idx])
            comp_edges[node_idx] = 0
            for neighbor_idx in neighbors[node_idx]:
                if active[neighbor_idx]:
                    union(node_idx, int(neighbor_idx))
            while threshold_idx < len(threshold_items) and added_count == threshold_items[threshold_idx][1]:
                _, k = threshold_items[threshold_idx]
                roots = {find(int(idx)) for idx in order[:k]}
                for root in roots:
                    size = int(comp_size[root])
                    if size < min_module_size:
                        continue
                    if int(comp_subthreshold[root]) < min_subthreshold_genes:
                        continue
                    is_broad = size >= broad_component_min_size or size / k >= broad_component_fraction
                    mean_score = float(comp_sum[root] / size)
                    edge_density = (
                        float(2.0 * comp_edges[root] / (size * (size - 1)))
                        if size >= 2
                        else 0.0
                    )
                    candidates.append((bool(is_broad), mean_score, size, edge_density))
                threshold_idx += 1
        if candidates:
            candidates = sorted(candidates, key=lambda item: (item[0], -item[1], -item[2], -item[3]))[:max_modules]
            max_mean = max(item[1] for item in candidates)
            max_size = max(item[2] for item in candidates)
            max_edge = max(item[3] for item in candidates)
            n_modules = len(candidates)
        else:
            max_mean = float("-inf")
            max_size = 0
            max_edge = 0.0
            n_modules = 0
        rows.append(
            {
                "replicate": int(replicate),
                "null_source": "degree_stratified_score_permutation",
                "n_modules": int(n_modules),
                "max_mean_score": float(max_mean),
                "max_edge_density": float(max_edge),
                "score_statistic_direction": "greater_is_more_extreme",
                "edge_statistic_direction": "greater_is_more_extreme",
                "max_n_genes": int(max_size),
                "engine": "fast_sparse_full_reselection",
            }
        )
        if (replicate + 1) % 500 == 0:
            print(f"  completed {replicate + 1:,}/{n_replicates:,} full reselection nulls", flush=True)
    return pd.DataFrame(rows)


def summarize_reselection(spec: AnalysisSpec, modules: pd.DataFrame, nulls: pd.DataFrame) -> pd.DataFrame:
    score_null = nulls["max_mean_score"].to_numpy(dtype=float)
    edge_null = nulls["max_edge_density"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for row in modules.to_dict(orient="records"):
        observed_mean = float(row.get("mean_score", float("nan")))
        observed_edge = float(row.get("edge_density", float("nan")))
        score_p = empirical_upper(score_null, observed_mean)
        edge_p = empirical_upper(edge_null, observed_edge)
        is_local = int(row.get("n_genes", 0) or 0) < 200
        previous_calibrated = str(row.get("is_reportable_calibrated_module", "")).lower() == "true"
        previous_topology = str(row.get("is_reportable_topology_specific_module", "")).lower() == "true"
        passes_score_fwer = bool(np.isfinite(score_p) and score_p <= 0.05)
        passes_edge_fwer = bool(np.isfinite(edge_p) and edge_p <= 0.05)
        if previous_calibrated and is_local and passes_score_fwer:
            recommended_label = "calibrated_candidate_module"
        elif previous_calibrated and is_local:
            recommended_label = "post_hoc_candidate_module"
        else:
            recommended_label = "exploratory_module"
        rows.append(
            {
                "trait": spec.trait,
                "analysis_id": spec.analysis_id,
                "module_id": row.get("module_id", ""),
                "rank_fraction": row.get("rank_fraction", ""),
                "n_genes": row.get("n_genes", ""),
                "n_edges": row.get("n_edges", ""),
                "edge_density": observed_edge,
                "mean_score": observed_mean,
                "max_score": row.get("max_score", ""),
                "core_genes": row.get("core_genes", ""),
                "module_claim_label_before_reselection": row.get("module_claim_label", ""),
                "is_reportable_calibrated_module_before_reselection": previous_calibrated,
                "is_reportable_topology_specific_module_before_reselection": previous_topology,
                "is_local_component": is_local,
                "full_reselection_null_source": "degree_stratified_score_permutation",
                "n_full_reselection_null": int(len(nulls)),
                "full_reselection_score_statistic_direction": "greater_is_more_extreme",
                "full_reselection_score_null_mean": float(np.mean(score_null)),
                "full_reselection_score_null_sd": float(np.std(score_null, ddof=1)),
                "full_reselection_score_z": z_score(observed_mean, score_null),
                "full_reselection_score_p": score_p,
                "full_reselection_edge_statistic_direction": "greater_is_more_extreme",
                "full_reselection_edge_null_mean": float(np.mean(edge_null)),
                "full_reselection_edge_null_sd": float(np.std(edge_null, ddof=1)),
                "full_reselection_edge_z": z_score(observed_edge, edge_null),
                "full_reselection_edge_p": edge_p,
                "passes_full_reselection_score_fwer": passes_score_fwer,
                "passes_full_reselection_edge_fwer": passes_edge_fwer,
                "recommended_module_claim_after_reselection": recommended_label,
                "source_module_table": str(spec.modules_path),
                "source_score_table": str(spec.scores_path),
                "source_graph_table": str(spec.graph_edges_path),
                "script_path": str(THIS_SCRIPT),
            }
        )
    return pd.DataFrame(rows)


def render_report(summary: pd.DataFrame, manifest: dict[str, object]) -> str:
    lines = [
        "# DR_MVP Local Module Full Reselection Null",
        "",
        f"Created: {manifest['created_utc']}",
        "",
        "This analysis repeats ranking, top-fraction selection, connected-component extraction, module filtering and max-module statistic collection under degree-stratified score permutations.",
        "",
        "| Analysis | Modules | Calibrated after reselection | Post hoc after reselection | Min score P |",
        "|---|---:|---:|---:|---:|",
    ]
    for analysis_id, group in summary.groupby("analysis_id", observed=True):
        labels = group["recommended_module_claim_after_reselection"].astype(str)
        min_p = pd.to_numeric(group["full_reselection_score_p"], errors="coerce").min()
        lines.append(
            "| "
            f"{analysis_id} | {len(group):,} | "
            f"{int((labels == 'calibrated_candidate_module').sum()):,} | "
            f"{int((labels == 'post_hoc_candidate_module').sum()):,} | "
            f"{float(min_p):.4g} |"
        )
    lines.extend(
        [
            "",
            "Interpretation rule:",
            "",
            "- `calibrated_candidate_module` means the module was already reportable and its mean score exceeded the full reselection max-module null at empirical P <= 0.05.",
            "- This null does not upgrade modules to topology-specific status; Tier 3 graph-null evidence remains the required topology-specific gate.",
            "",
        ]
    )
    return "\n".join(lines)


def run_one(args: argparse.Namespace, spec: AnalysisSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Running full reselection null for {spec.analysis_id}", flush=True)
    scores = read_tsv(spec.scores_path)
    modules = read_tsv(spec.modules_path)
    graph = load_analysis_graph(spec, scores)
    if args.engine == "reference":
        nulls = selection_aware_module_null(
            graph,
            scores,
            selection_null_scores=None,
            min_module_size=args.min_module_size,
            min_subthreshold_genes=args.min_module_subthreshold_genes,
            max_modules=args.max_local_modules,
            broad_component_min_size=args.broad_component_min_size,
            broad_component_fraction=args.broad_component_fraction,
            n_replicates=args.n_reselection_null,
            degree_bins=args.degree_bins,
            seed=args.seed + len(spec.analysis_id),
        )
        nulls["engine"] = "reference_networkx_full_reselection"
    else:
        nulls = fast_full_reselection_null(
            graph,
            scores,
            n_replicates=args.n_reselection_null,
            degree_bins=args.degree_bins,
            min_module_size=args.min_module_size,
            min_subthreshold_genes=args.min_module_subthreshold_genes,
            max_modules=args.max_local_modules,
            broad_component_min_size=args.broad_component_min_size,
            broad_component_fraction=args.broad_component_fraction,
            seed=args.seed + len(spec.analysis_id),
        )
    summary = summarize_reselection(spec, modules, nulls)
    print(
        f"Completed {spec.analysis_id}: {len(nulls):,} null replicates, {len(summary):,} observed modules",
        flush=True,
    )
    return summary, nulls.assign(trait=spec.trait, analysis_id=spec.analysis_id)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.out_dir} exists and is not empty. Use --force.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reports").mkdir(parents=True, exist_ok=True)

    specs = select_specs(args)
    summaries: list[pd.DataFrame] = []
    null_tables: list[pd.DataFrame] = []
    for spec in specs:
        summary, nulls = run_one(args, spec)
        write_table(args.out_dir / "tables" / f"{spec.trait}.module_full_reselection_summary.tsv", summary)
        write_table(args.out_dir / "tables" / f"{spec.trait}.module_full_reselection_null.tsv", nulls)
        summaries.append(summary)
        null_tables.append(nulls)

    combined_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    combined_nulls = pd.concat(null_tables, ignore_index=True) if null_tables else pd.DataFrame()
    write_table(args.out_dir / "tables" / "module_full_reselection_summary.all_traits.tsv", combined_summary)
    write_table(args.out_dir / "tables" / "module_full_reselection_null.all_traits.tsv.gz", combined_nulls)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "analysis_id": "module_reselection_null_v1",
        "traits": [spec.trait for spec in specs],
        "n_full_reselection_null": args.n_reselection_null,
        "null_source": "degree_stratified_score_permutation",
        "degree_bins": args.degree_bins,
        "rank_fraction_grid": "0.01,0.02,0.05,0.10,0.15,0.20",
        "min_module_size": args.min_module_size,
        "min_module_subthreshold_genes": args.min_module_subthreshold_genes,
        "max_local_modules": args.max_local_modules,
        "broad_component_min_size": args.broad_component_min_size,
        "seed": args.seed,
        "script_path": str(THIS_SCRIPT),
        "output_summary": str(args.out_dir / "tables" / "module_full_reselection_summary.all_traits.tsv"),
        "output_null": str(args.out_dir / "tables" / "module_full_reselection_null.all_traits.tsv.gz"),
    }
    (args.out_dir / "module_full_reselection_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out_dir / "reports" / "module_full_reselection_report.md").write_text(
        render_report(combined_summary, manifest),
        encoding="utf-8",
    )
    print(f"Wrote full reselection null outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
