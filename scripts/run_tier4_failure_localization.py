#!/usr/bin/env python
"""Localize the failure point of RIPPLE V1 Tier 4 module calibration.

This diagnostic script does not alter the frozen V1 claim policy. It dissects
oracle spike-in failures into score ranking, threshold fragmentation, component
extraction, fixed-module statistic power, null choice and max-gate penalty.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ripple.defaults import RANK_FRACTION_GRID  # noqa: E402
from ripple.modules import discover_local_modules  # noqa: E402
from ripple.nulls.score_permutation import assign_degree_bins  # noqa: E402
from ripple.percolation import rank_nodes_by_score, selected_nodes_at_fraction  # noqa: E402
from run_dr_mvp_module_reselection_null import (  # noqa: E402
    ANALYSIS_ROOT,
    AnalysisSpec,
    empirical_upper,
    fast_full_reselection_null,
    load_analysis_graph,
    read_tsv,
    write_table,
    z_score,
)
from run_tier4_design_defect_audit import (  # noqa: E402
    as_float,
    best_recovered_module,
    fixed_null_for_nodes,
    parse_effect_grid,
    score_spikein,
    split_genes,
)


DEFAULT_OUT_DIR = ANALYSIS_ROOT / "tier4_failure_localization_v1"
DESIGN_AUDIT_DIR = ANALYSIS_ROOT / "tier4_design_defect_audit_v1"
THIS_SCRIPT = Path(__file__).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--oracle-modules", type=Path, default=DESIGN_AUDIT_DIR / "tables" / "a1_oracle_modules.tsv")
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--effect-grid", default="weak:1.0,moderate:2.5,strong:5.0,very_strong:10.0")
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def dr_default_spec() -> AnalysisSpec:
    return AnalysisSpec(
        trait="DR_MVP",
        analysis_id="DR_MVP_default_final5000",
        analysis_dir=ANALYSIS_ROOT / "dr_mvp_string_final5000",
        graph_edges_path=ANALYSIS_ROOT / "dr_mvp_string_final5000" / "tables" / "DR_MVP.analysis_graph_edges.tsv.gz",
    )


def prepare_scores_and_graph() -> tuple[pd.DataFrame, nx.Graph]:
    spec = dr_default_spec()
    scores = read_tsv(spec.scores_path)
    graph = load_analysis_graph(spec, scores)
    degree = dict(graph.degree())
    work = scores.copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str)
    work = work[work["gene_symbol"].isin(graph.nodes())].reset_index(drop=True)
    work["assoc_resid_score"] = pd.to_numeric(work["assoc_resid_score"], errors="raise").astype(float)
    work["graph_degree"] = work["gene_symbol"].map(degree).astype(int)
    return work, graph


def selected_oracle_diagnostics(
    graph: nx.Graph,
    scores: pd.DataFrame,
    oracle_genes: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    ranking = rank_nodes_by_score(scores, node_col="gene_symbol", score_col="assoc_resid_score")
    rows: list[dict[str, object]] = []
    for fraction in RANK_FRACTION_GRID:
        selected = set(selected_nodes_at_fraction(ranking, float(fraction), node_col="gene_symbol"))
        selected_oracle = selected & oracle_genes
        if selected_oracle:
            sub = graph.subgraph(selected_oracle)
            components = list(nx.connected_components(sub))
            largest = max((len(component) for component in components), default=0)
        else:
            components = []
            largest = 0
        rows.append(
            {
                "top_fraction": float(fraction),
                "selected_oracle_count": int(len(selected_oracle)),
                "selected_oracle_recall": float(len(selected_oracle) / len(oracle_genes)) if oracle_genes else 0.0,
                "fragmentation_count": int(len(components)),
                "largest_oracle_fragment": int(largest),
                "largest_oracle_fragment_fraction": float(largest / len(oracle_genes)) if oracle_genes else 0.0,
            }
        )
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["selected_oracle_recall", "largest_oracle_fragment_fraction"],
        ascending=[False, False],
    ).iloc[0].to_dict()
    return table, best


def module_precision_recall(module_genes: set[str], oracle_genes: set[str]) -> tuple[float, float]:
    if not module_genes:
        return 0.0, 0.0
    overlap = len(module_genes & oracle_genes)
    precision = float(overlap / len(module_genes))
    recall = float(overlap / len(oracle_genes)) if oracle_genes else 0.0
    return precision, recall


def fixed_threshold_reselection_null(
    graph: nx.Graph,
    scores: pd.DataFrame,
    *,
    fraction: float,
    n_replicates: int,
    degree_bins: int,
    seed: int,
    mode: str = "degree_stratified",
) -> np.ndarray:
    """Max component mean-score null at one fixed top-rank fraction."""

    work = scores.copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str)
    nodes = work["gene_symbol"].to_numpy(dtype=str)
    scores_arr = pd.to_numeric(work["assoc_resid_score"], errors="raise").to_numpy(dtype=float)
    degrees = pd.to_numeric(work["graph_degree"], errors="raise")
    bins = assign_degree_bins(degrees, n_bins=1 if mode == "random" else degree_bins).to_numpy(dtype=int)
    groups = [np.flatnonzero(bins == bin_id) for bin_id in sorted(np.unique(bins))]
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    n_nodes = len(nodes)
    k = max(1, int(round(float(fraction) * n_nodes)))
    rng = np.random.default_rng(seed)
    max_values = np.empty(n_replicates, dtype=float)
    for replicate in range(n_replicates):
        permuted = scores_arr.copy()
        for group_idx in groups:
            if group_idx.size > 1:
                permuted[group_idx] = permuted[rng.permutation(group_idx)]
        order = np.argsort(-permuted, kind="mergesort")[:k]
        selected = set(str(nodes[idx]) for idx in order)
        subgraph = graph.subgraph(selected)
        values = []
        for component in nx.connected_components(subgraph):
            component_idx = [node_to_idx[str(node)] for node in component]
            if len(component_idx) >= 5:
                values.append(float(np.mean(permuted[component_idx])))
        max_values[replicate] = max(values) if values else float("-inf")
    return max_values


def null_row(
    *,
    scenario_id: str,
    null_type: str,
    observed_stat: float,
    null_values: np.ndarray,
) -> dict[str, object]:
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {
            "scenario_id": scenario_id,
            "null_type": null_type,
            "n_null": 0,
            "observed_stat": observed_stat,
            "null_mean": np.nan,
            "null_sd": np.nan,
            "null_95": np.nan,
            "null_99": np.nan,
            "empirical_p": np.nan,
            "power": False,
            "fwer": np.nan,
            "calibration_status": "not_available",
        }
    p_value = empirical_upper(finite, observed_stat)
    return {
        "scenario_id": scenario_id,
        "null_type": null_type,
        "n_null": int(len(finite)),
        "observed_stat": float(observed_stat),
        "null_mean": float(np.mean(finite)),
        "null_sd": float(np.std(finite, ddof=1)) if len(finite) >= 2 else np.nan,
        "null_95": float(np.quantile(finite, 0.95)),
        "null_99": float(np.quantile(finite, 0.99)),
        "empirical_p": p_value,
        "power": bool(np.isfinite(p_value) and p_value <= 0.05),
        "fwer": np.nan,
        "calibration_status": "diagnostic_empirical_null",
    }


def classify_failure_stage(
    *,
    fixed_oracle_p: float,
    best_recall: float,
    best_jaccard: float,
    fragmentation_count: int,
    passed_module_filter: bool,
    fixed_recovered_p: float,
    random_full_p: float,
    full_p: float,
    restricted_p: float,
) -> str:
    if np.isfinite(fixed_oracle_p) and fixed_oracle_p > 0.05:
        return "statistic_low_power"
    if best_recall < 0.20:
        return "score_not_ranked"
    if fragmentation_count > 1 and best_jaccard < 0.25:
        return "threshold_fragmentation"
    if not passed_module_filter:
        return "module_filter_failure"
    if best_jaccard < 0.25:
        return "component_extraction_failure"
    if np.isfinite(fixed_recovered_p) and fixed_recovered_p > 0.05:
        return "statistic_low_power"
    if np.isfinite(random_full_p) and random_full_p <= 0.05 and np.isfinite(full_p) and full_p > 0.05:
        return "degree_null_overconservative"
    if np.isfinite(restricted_p) and restricted_p <= 0.05 and np.isfinite(full_p) and full_p > 0.05:
        return "max_gate_penalty"
    if np.isfinite(full_p) and full_p > 0.05:
        return "max_gate_penalty"
    return "unclear"


def dominant_penalty_source(
    *,
    fixed_oracle_p: float,
    fixed_recovered_p: float,
    single_threshold_p: float,
    random_full_p: float,
    full_p: float,
) -> str:
    if np.isfinite(fixed_oracle_p) and fixed_oracle_p > 0.05:
        return "fixed_oracle_statistic"
    if np.isfinite(fixed_recovered_p) and fixed_recovered_p > 0.05:
        return "recovered_module_statistic"
    if np.isfinite(single_threshold_p) and single_threshold_p > 0.05:
        return "component_search_or_degree_null"
    if np.isfinite(random_full_p) and random_full_p <= 0.05 and np.isfinite(full_p) and full_p > 0.05:
        return "degree_stratified_null"
    if np.isfinite(full_p) and full_p > 0.05:
        return "full_max_gate"
    return "unclear_or_passed"


def run_localization(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores, graph = prepare_scores_and_graph()
    oracles = read_tsv(args.oracle_modules)
    effect_grid = parse_effect_grid(args.effect_grid)
    decomposition_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    penalty_rows: list[dict[str, object]] = []
    scenario_index = 0
    for oracle in oracles.to_dict(orient="records"):
        oracle_genes = set(split_genes(oracle["oracle_genes"]))
        for effect_label, effect_size in effect_grid.items():
            scenario_index += 1
            if args.max_scenarios is not None and scenario_index > args.max_scenarios:
                break
            scenario_id = f"{oracle['oracle_id']}_{effect_label}"
            print(f"Tier 4 failure localization: {scenario_id}", flush=True)
            spiked = score_spikein(scores, oracle_genes, effect_size)
            threshold_diag, threshold_best = selected_oracle_diagnostics(graph, spiked, oracle_genes)
            modules = discover_local_modules(graph, spiked)
            best = best_recovered_module(modules, oracle_genes)
            best_module_genes = set(split_genes(best.get("best_module_genes", "")))
            precision, recall = module_precision_recall(best_module_genes, oracle_genes)
            passed_filter = bool(not modules.empty and best.get("best_module_id", ""))
            top_fraction = (
                as_float(modules.loc[modules["module_id"].eq(best.get("best_module_id", "")), "rank_fraction"].iloc[0])
                if passed_filter and "module_id" in modules.columns
                else as_float(threshold_best["top_fraction"])
            )
            fixed_oracle, fixed_oracle_nulls = fixed_null_for_nodes(
                sorted(oracle_genes),
                spiked,
                n_null=args.n_null,
                degree_bins=args.degree_bins,
                seed=args.seed + scenario_index * 10 + 1,
            )
            fixed_recovered: dict[str, object] = {}
            if best_module_genes:
                fixed_recovered, fixed_recovered_nulls = fixed_null_for_nodes(
                    sorted(best_module_genes),
                    spiked,
                    n_null=args.n_null,
                    degree_bins=args.degree_bins,
                    seed=args.seed + scenario_index * 10 + 2,
                )
            else:
                fixed_recovered_nulls = pd.DataFrame()
            full_degree = fast_full_reselection_null(
                graph,
                spiked,
                n_replicates=args.n_null,
                degree_bins=args.degree_bins,
                min_module_size=5,
                min_subthreshold_genes=3,
                max_modules=20,
                broad_component_min_size=200,
                broad_component_fraction=0.50,
                seed=args.seed + scenario_index * 10 + 3,
            )
            full_random = fast_full_reselection_null(
                graph,
                spiked,
                n_replicates=args.n_null,
                degree_bins=1,
                min_module_size=5,
                min_subthreshold_genes=3,
                max_modules=20,
                broad_component_min_size=200,
                broad_component_fraction=0.50,
                seed=args.seed + scenario_index * 10 + 4,
            )
            restricted_degree = fixed_threshold_reselection_null(
                graph,
                spiked,
                fraction=top_fraction,
                n_replicates=args.n_null,
                degree_bins=args.degree_bins,
                seed=args.seed + scenario_index * 10 + 5,
                mode="degree_stratified",
            )
            observed_best = as_float(best.get("best_module_mean_score", np.nan))
            fixed_oracle_p = as_float(fixed_oracle.get("degree_matched_fixed_p", np.nan))
            fixed_recovered_p = as_float(fixed_recovered.get("degree_matched_fixed_p", np.nan))
            full_p = empirical_upper(full_degree["max_mean_score"].to_numpy(dtype=float), observed_best)
            random_full_p = empirical_upper(full_random["max_mean_score"].to_numpy(dtype=float), observed_best)
            restricted_p = empirical_upper(restricted_degree, observed_best)
            null_rows.extend(
                [
                    null_row(
                        scenario_id=scenario_id,
                        null_type="fixed_oracle_random_score_permutation",
                        observed_stat=as_float(fixed_oracle["observed_mean_score"]),
                        null_values=fixed_oracle_nulls["size_matched_mean_score"].to_numpy(dtype=float),
                    ),
                    null_row(
                        scenario_id=scenario_id,
                        null_type="degree_matched_node_set",
                        observed_stat=as_float(fixed_oracle["observed_mean_score"]),
                        null_values=fixed_oracle_nulls["degree_matched_mean_score"].to_numpy(dtype=float),
                    ),
                    null_row(
                        scenario_id=scenario_id,
                        null_type="fixed_recovered_degree_matched_node_set",
                        observed_stat=as_float(fixed_recovered.get("observed_mean_score", np.nan)),
                        null_values=fixed_recovered_nulls["degree_matched_mean_score"].to_numpy(dtype=float)
                        if not fixed_recovered_nulls.empty
                        else np.array([]),
                    ),
                    null_row(
                        scenario_id=scenario_id,
                        null_type="restricted_reselection_fixed_threshold",
                        observed_stat=observed_best,
                        null_values=restricted_degree,
                    ),
                    null_row(
                        scenario_id=scenario_id,
                        null_type="random_score_permutation",
                        observed_stat=observed_best,
                        null_values=full_random["max_mean_score"].to_numpy(dtype=float),
                    ),
                    null_row(
                        scenario_id=scenario_id,
                        null_type="full_reselection",
                        observed_stat=observed_best,
                        null_values=full_degree["max_mean_score"].to_numpy(dtype=float),
                    ),
                ]
            )
            failure_stage = classify_failure_stage(
                fixed_oracle_p=fixed_oracle_p,
                best_recall=recall,
                best_jaccard=as_float(best.get("best_jaccard", 0.0)),
                fragmentation_count=int(threshold_best["fragmentation_count"]),
                passed_module_filter=passed_filter,
                fixed_recovered_p=fixed_recovered_p,
                random_full_p=random_full_p,
                full_p=full_p,
                restricted_p=restricted_p,
            )
            selection_penalty_ratio = full_p / fixed_recovered_p if np.isfinite(full_p) and np.isfinite(fixed_recovered_p) and fixed_recovered_p > 0 else np.nan
            decomposition_rows.append(
                {
                    "scenario_id": scenario_id,
                    "graph_id": "STRING_default",
                    "module_size": oracle["target_size"],
                    "effect_size": effect_size,
                    "degree_bin": oracle["degree_stratum"],
                    "architecture": oracle["oracle_type"],
                    "true_module_gene_count": len(oracle_genes),
                    "top_fraction": top_fraction,
                    "extracted_module_count": int(len(modules)),
                    "best_jaccard": best.get("best_jaccard", 0.0),
                    "best_precision": precision,
                    "best_recall": recall,
                    "fragmentation_count": int(threshold_best["fragmentation_count"]),
                    "passed_module_filter": passed_filter,
                    "fixed_oracle_p": fixed_oracle_p,
                    "fixed_recovered_p": fixed_recovered_p,
                    "restricted_reselection_p": restricted_p,
                    "random_full_reselection_p": random_full_p,
                    "full_reselection_p": full_p,
                    "selection_penalty_ratio": selection_penalty_ratio,
                    "failure_stage": failure_stage,
                    "best_module_id": best.get("best_module_id", ""),
                    "best_module_size": best.get("best_module_size", ""),
                    "fixed_oracle_z": fixed_oracle.get("degree_matched_fixed_z", ""),
                    "fixed_recovered_z": fixed_recovered.get("degree_matched_fixed_z", ""),
                    "full_reselection_z": z_score(observed_best, full_degree["max_mean_score"].to_numpy(dtype=float)),
                }
            )
            penalty_rows.append(
                {
                    "scenario_id": scenario_id,
                    "fixed_module_p": fixed_oracle_p,
                    "fixed_extracted_module_p": fixed_recovered_p,
                    "single_threshold_reselection_p": restricted_p,
                    "max_threshold_reselection_p": full_p,
                    "random_full_reselection_p": random_full_p,
                    "full_reselection_p": full_p,
                    "p_ratio_fixed_to_full": full_p / fixed_oracle_p
                    if np.isfinite(full_p) and np.isfinite(fixed_oracle_p) and fixed_oracle_p > 0
                    else np.nan,
                    "p_ratio_single_to_full": full_p / restricted_p
                    if np.isfinite(full_p) and np.isfinite(restricted_p) and restricted_p > 0
                    else np.nan,
                    "dominant_penalty_source": dominant_penalty_source(
                        fixed_oracle_p=fixed_oracle_p,
                        fixed_recovered_p=fixed_recovered_p,
                        single_threshold_p=restricted_p,
                        random_full_p=random_full_p,
                        full_p=full_p,
                    ),
                }
            )
        if args.max_scenarios is not None and scenario_index >= args.max_scenarios:
            break
    return pd.DataFrame(decomposition_rows), pd.DataFrame(null_rows), pd.DataFrame(penalty_rows)


def render_report(decomposition: pd.DataFrame, nulls: pd.DataFrame, penalty: pd.DataFrame, manifest: dict[str, object]) -> str:
    lines = [
        "# RIPPLE Tier 4 Failure Localization v1",
        "",
        f"Created: {manifest['created_utc']}",
        "",
        "This diagnostic localizes the current Tier 4 failure without changing the frozen RIPPLE V1 claim policy.",
        "",
        "## Six Localization Questions",
        "",
    ]
    if decomposition.empty:
        lines.append("No scenarios were evaluated.")
        return "\n".join(lines)
    fixed_oracle_rate = float((pd.to_numeric(decomposition["fixed_oracle_p"], errors="coerce") <= 0.05).mean())
    high_jaccard_rate = float((pd.to_numeric(decomposition["best_jaccard"], errors="coerce") >= 0.5).mean())
    fixed_recovered_rate = float((pd.to_numeric(decomposition["fixed_recovered_p"], errors="coerce") <= 0.05).mean())
    full_rate = float((pd.to_numeric(decomposition["full_reselection_p"], errors="coerce") <= 0.05).mean())
    random_full_rate = float((pd.to_numeric(decomposition["random_full_reselection_p"], errors="coerce") <= 0.05).mean())
    stage_counts = decomposition["failure_stage"].value_counts().reset_index()
    lines.extend(
        [
            f"1. Fixed oracle statistic power: {fixed_oracle_rate:.3f} of scenarios pass degree-matched fixed-module P <= 0.05.",
            f"2. Extraction recovery: {high_jaccard_rate:.3f} of scenarios recover Jaccard >= 0.5.",
            f"3. Fixed recovered-module support: {fixed_recovered_rate:.3f} of scenarios pass fixed recovered-module P <= 0.05.",
            f"4. Full reselection support: {full_rate:.3f} of scenarios pass full reselection P <= 0.05.",
            f"5. Random full-reselection support: {random_full_rate:.3f} of scenarios pass random-score full reselection P <= 0.05.",
            "6. Failure-stage distribution:",
            "",
            "| Failure stage | Scenarios |",
            "|---|---:|",
        ]
    )
    for row in stage_counts.to_dict(orient="records"):
        lines.append(f"| {row['failure_stage']} | {row['count']} |")
    lines.extend(["", "## Null Ablation Summary", ""])
    null_summary = (
        nulls.groupby("null_type", observed=True)
        .agg(n_rows=("scenario_id", "count"), pass_rate=("power", "mean"), median_p=("empirical_p", "median"))
        .reset_index()
    )
    lines.extend(["| Null type | Rows | Pass rate | Median P |", "|---|---:|---:|---:|"])
    for row in null_summary.to_dict(orient="records"):
        lines.append(f"| {row['null_type']} | {int(row['n_rows'])} | {float(row['pass_rate']):.3f} | {float(row['median_p']):.4g} |")
    lines.extend(["", "## Penalty Source Summary", ""])
    penalty_counts = penalty["dominant_penalty_source"].value_counts().reset_index()
    lines.extend(["| Dominant penalty source | Scenarios |", "|---|---:|"])
    for row in penalty_counts.to_dict(orient="records"):
        lines.append(f"| {row['dominant_penalty_source']} | {row['count']} |")
    lines.extend(
        [
            "",
            "## Working Interpretation",
            "",
            "If fixed oracle and fixed recovered-module tests pass but full reselection fails, the dominant issue is selection/max-gate calibration rather than absence of module signal.",
            "If random full reselection passes but degree-stratified full reselection fails, degree-stratified permutation is likely overconservative for the tested signal architecture.",
            "If fixed oracle fails, the current mean-score statistic is low power even when module membership is known.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.out_dir} exists and is not empty. Use --force.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "reports").mkdir(parents=True, exist_ok=True)
    decomposition, nulls, penalty = run_localization(args)
    write_table(args.out_dir / "tables" / "oracle_pipeline_decomposition.tsv", decomposition)
    write_table(args.out_dir / "tables" / "tier4_null_ablation.tsv", nulls)
    write_table(args.out_dir / "tables" / "tier4_selection_penalty_decomposition.tsv", penalty)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "analysis_id": "tier4_failure_localization_v1",
        "oracle_modules": str(args.oracle_modules),
        "n_null": args.n_null,
        "degree_bins": args.degree_bins,
        "effect_grid": args.effect_grid,
        "rank_fraction_grid": list(RANK_FRACTION_GRID),
        "seed": args.seed,
        "script_path": str(THIS_SCRIPT),
        "outputs": {
            "oracle_pipeline_decomposition": str(args.out_dir / "tables" / "oracle_pipeline_decomposition.tsv"),
            "tier4_null_ablation": str(args.out_dir / "tables" / "tier4_null_ablation.tsv"),
            "tier4_selection_penalty_decomposition": str(
                args.out_dir / "tables" / "tier4_selection_penalty_decomposition.tsv"
            ),
        },
    }
    (args.out_dir / "tier4_failure_localization_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out_dir / "reports" / "tier4_failure_localization_report.md").write_text(
        render_report(decomposition, nulls, penalty, manifest),
        encoding="utf-8",
    )
    print(f"Wrote Tier 4 failure-localization outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
