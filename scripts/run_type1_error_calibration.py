#!/usr/bin/env python
"""Run RIPPLE V1 claim-tier Type 1 error calibration."""

from __future__ import annotations

import argparse
import sys
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
from ripple.graph import graph_laplacian, preprocess_reference_graph  # noqa: E402
from ripple.graph_diffusion import heat_kernel_tau_statistics_matrix, parse_tau_grid  # noqa: E402
from ripple.io.graph import networkx_to_edge_list, read_edge_list, write_edge_list  # noqa: E402
from ripple.modules import discover_local_modules  # noqa: E402
from ripple.nulls.graph_nulls import degree_preserving_graph_replicates  # noqa: E402
from ripple.percolation import (  # noqa: E402
    compute_degree_matched_node_percolation_null,
    percolation_auc,
    percolation_curve,
    prepare_degree_matched_rank_sets,
    rank_nodes_by_score,
    summarize_percolation_null,
)
from run_height_mvp import DEFAULT_STRING_INFO, DEFAULT_STRING_LINKS, build_string_graph, write_table  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_ANALYSIS_DIR = PRIVATE_ROOT / "30_analysis" / "dr_mvp_string_final5000"
DEFAULT_OUT_DIR = PRIVATE_ROOT / "30_analysis" / "type1_error_calibration_v1" / "dr_mvp_string_pipeline_null_n100"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--trait", default="DR_MVP")
    parser.add_argument("--graph-name", default="string_ppi")
    parser.add_argument("--graph-edge-list", type=Path, default=None)
    parser.add_argument("--analysis-graph-edge-cache", type=Path, default=None)
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--lcc-scores", type=Path, default=None)
    parser.add_argument("--null-scores", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--outer-n", type=int, default=100)
    parser.add_argument("--calibration-null-n", type=int, default=1000)
    parser.add_argument("--inner-degree-matched-null", type=int, default=500)
    parser.add_argument("--graph-null-replicates", type=int, default=100)
    parser.add_argument("--graph-null-cache", type=Path, default=None)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--diffusion-score-mode", choices=["positive", "raw"], default="positive")
    parser.add_argument("--diffusion-batch-size", type=int, default=128)
    parser.add_argument("--tau-grid", default="0.25,0.5,1.0,2.0,4.0")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--include-degree-biased-null", action="store_true")
    parser.add_argument("--include-technical-confounded-null", action="store_true")
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--module-alpha", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def default_lcc_scores_path(args: argparse.Namespace) -> Path:
    return args.analysis_dir / "tables" / f"{args.trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"


def default_null_scores_path(args: argparse.Namespace) -> Path:
    return args.analysis_dir / "tables" / f"{args.trait}.null_residualized_scores.npz"


def default_analysis_graph_edges_path(args: argparse.Namespace) -> Path:
    return args.analysis_dir / "tables" / f"{args.trait}.analysis_graph_edges.tsv.gz"


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)


def load_null_matrix(path: Path, n_genes: int, *, max_rows: int | None = None) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        candidates: list[tuple[str, np.ndarray]] = []
        for key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim == 2 and n_genes in arr.shape:
                candidates.append((key, arr))
        if not candidates:
            shapes = {key: np.asarray(data[key]).shape for key in data.files}
            raise ValueError(f"No 2D null-score matrix with n_genes={n_genes}. Available shapes: {shapes}")
        key, matrix = candidates[0]
        print(f"Using null-score matrix key={key} shape={matrix.shape}", flush=True)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[1] != n_genes and matrix.shape[0] == n_genes:
        matrix = matrix.T
    if matrix.shape[1] != n_genes:
        raise ValueError(f"Null matrix shape {matrix.shape} cannot align to {n_genes} genes.")
    if max_rows is not None:
        matrix = matrix[:max_rows, :]
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Null matrix contains non-finite values.")
    return matrix


def load_graph(args: argparse.Namespace, gene_symbols: tuple[str, ...]) -> nx.Graph:
    if args.graph_edge_list is not None:
        edges = read_edge_list(args.graph_edge_list, source=args.graph_name)
        return preprocess_reference_graph(edges.edges, gene_universe=gene_symbols).largest_component
    cache_path = args.analysis_graph_edge_cache or default_analysis_graph_edges_path(args)
    if cache_path.exists():
        print(f"Loading analysis graph edge cache: {cache_path}", flush=True)
        edges = read_edge_list(cache_path, sep="\t", source=f"{args.graph_name}_analysis_cache")
        return preprocess_reference_graph(edges.edges, gene_universe=gene_symbols).largest_component
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
    return graph_pre.largest_component


def make_score_table(base: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    out = base.copy()
    out["assoc_resid_score"] = np.asarray(values, dtype=float)
    out["assoc_p_g"] = 1.0
    return out


def auc_for_scores(graph: nx.Graph, scores: pd.DataFrame) -> float:
    ranking = rank_nodes_by_score(scores, node_col="gene_symbol", score_col="assoc_resid_score")
    curve = percolation_curve(graph, ranking, RANK_FRACTION_GRID)
    return percolation_auc(curve)


def leave_one_summary(values: np.ndarray, observed: float, index: int | None = None) -> dict[str, float | int]:
    null = np.asarray(values, dtype=float)
    if index is not None and 0 <= index < len(null):
        null = np.delete(null, index)
    return summarize_percolation_null(pd.DataFrame({"percolation_auc": null}), observed)


def empirical_upper(values: np.ndarray, observed: float, index: int | None = None) -> float:
    null = np.asarray(values, dtype=float)
    null = null[np.isfinite(null)]
    if index is not None and 0 <= index < len(null):
        null = np.delete(null, index)
    if len(null) == 0:
        return float("nan")
    return float((1 + np.count_nonzero(null >= observed)) / (len(null) + 1))


def z_against(values: np.ndarray, observed: float, index: int | None = None) -> float:
    null = np.asarray(values, dtype=float)
    null = null[np.isfinite(null)]
    if index is not None and 0 <= index < len(null):
        null = np.delete(null, index)
    if len(null) < 2:
        return float("nan")
    sd = float(np.std(null, ddof=1))
    return float((observed - float(np.mean(null))) / sd) if sd > 0 else float("nan")


def score_vector(values: np.ndarray, *, mode: str) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if mode == "positive":
        return np.maximum(0.0, x)
    if mode == "raw":
        return x
    raise ValueError(f"Unsupported diffusion score mode: {mode}")


def diffusion_tmax_distribution(
    graph: nx.Graph,
    base_scores: pd.DataFrame,
    matrix: np.ndarray,
    *,
    mode: str,
    tau_grid: tuple[float, ...],
    batch_size: int,
) -> np.ndarray:
    nodes = tuple(base_scores["gene_symbol"].astype(str))
    lap = graph_laplacian(graph, nodes=nodes, kind="normalized", weight=None)
    transformed = np.asarray([score_vector(row, mode=mode) for row in matrix], dtype=float)
    tau_values = heat_kernel_tau_statistics_matrix(
        lap.laplacian,
        transformed,
        tau_grid=tau_grid,
        batch_size=batch_size,
    )
    return np.max(tau_values, axis=1) if tau_values.size else np.array([], dtype=float)


def degree_matched_summary(
    graph: nx.Graph,
    scores: pd.DataFrame,
    observed_auc: float,
    *,
    n_replicates: int,
    degree_bins: int,
    seed: int,
) -> dict[str, float | int]:
    ranking = rank_nodes_by_score(scores, node_col="gene_symbol", score_col="assoc_resid_score")
    selected_counts, bin_to_nodes, _ = prepare_degree_matched_rank_sets(
        scores,
        ranking,
        RANK_FRACTION_GRID,
        n_bins=degree_bins,
    )
    auc_null, _ = compute_degree_matched_node_percolation_null(
        graph,
        selected_counts,
        bin_to_nodes,
        n_replicates=n_replicates,
        seed=seed,
    )
    return summarize_percolation_null(auc_null, observed_auc)


def graph_null_auc_distribution(
    null_graphs: list[nx.Graph],
    ranking: pd.DataFrame,
) -> np.ndarray:
    values = []
    for graph in null_graphs:
        curve = percolation_curve(graph, ranking, RANK_FRACTION_GRID)
        values.append(percolation_auc(curve))
    return np.asarray(values, dtype=float)


def module_selection_rows(
    graph: nx.Graph,
    base_scores: pd.DataFrame,
    matrix: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, vec in enumerate(matrix):
        scores = make_score_table(base_scores, vec)
        modules = discover_local_modules(graph, scores)
        if modules.empty:
            max_mean = float("-inf")
            max_edge = float("-inf")
            n_candidates = 0
        else:
            max_mean = float(pd.to_numeric(modules["mean_score"], errors="raise").max())
            max_edge = float(pd.to_numeric(modules["edge_density"], errors="raise").max())
            n_candidates = int(len(modules))
        rows.append(
            {
                "outer_replicate": int(idx),
                "n_candidate_modules": n_candidates,
                "max_module_mean_score": max_mean,
                "max_module_edge_density": max_edge,
            }
        )
    table = pd.DataFrame(rows)
    mean_scores = table["max_module_mean_score"].replace([-np.inf, np.inf], np.nan).to_numpy(dtype=float)
    edge_scores = table["max_module_edge_density"].replace([-np.inf, np.inf], np.nan).to_numpy(dtype=float)
    table["selection_aware_score_p"] = [
        empirical_upper(mean_scores, float(value), idx) if np.isfinite(float(value)) else float("nan")
        for idx, value in enumerate(table["max_module_mean_score"])
    ]
    table["selection_aware_edge_p"] = [
        empirical_upper(edge_scores, float(value), idx) if np.isfinite(float(value)) else float("nan")
        for idx, value in enumerate(table["max_module_edge_density"])
    ]
    return table


def standardize(series: pd.Series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(x).any():
        return np.zeros(len(x), dtype=float)
    median = float(np.nanmedian(x))
    x = np.where(np.isfinite(x), x, median)
    sd = float(np.std(x, ddof=1))
    return (x - float(np.mean(x))) / sd if sd > 0 else np.zeros(len(x), dtype=float)


def residualize(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    x = np.column_stack([np.ones(len(y)), design])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    sd = float(np.std(resid, ddof=1))
    return resid / sd if sd > 0 else resid


def technical_matrix(base_scores: pd.DataFrame, *, n: int, seed: int) -> np.ndarray:
    covariate_cols = [
        col
        for col in ("log_gene_length", "log_mapped_snp_count", "log_m_eff", "local_ld_score", "mappability")
        if col in base_scores.columns
    ]
    if not covariate_cols:
        return np.empty((0, len(base_scores)), dtype=float)
    design = np.column_stack([standardize(base_scores[col]) for col in covariate_cols])
    signal = design.mean(axis=1)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        y = signal + rng.normal(0.0, 1.0, len(base_scores))
        rows.append(residualize(y, design))
    return np.asarray(rows, dtype=float)


def degree_biased_matrix(base_scores: pd.DataFrame, *, n: int, seed: int) -> np.ndarray:
    degree = standardize(base_scores["graph_degree"])
    rng = np.random.default_rng(seed)
    return np.asarray([1.4 * degree + rng.normal(0.0, 1.0, len(base_scores)) for _ in range(n)], dtype=float)


def run_scenario(
    *,
    scenario: str,
    matrix: np.ndarray,
    graph: nx.Graph,
    graph_nulls: list[nx.Graph],
    base_scores: pd.DataFrame,
    args: argparse.Namespace,
    tau_grid: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_calibration = min(args.calibration_null_n, len(matrix))
    calibration = matrix[:n_calibration, :]
    rng = np.random.default_rng(args.seed + 17)
    outer_n = min(args.outer_n, len(calibration))
    outer_indices = np.sort(rng.choice(np.arange(n_calibration), size=outer_n, replace=False))

    print(f"[{scenario}] Computing pipeline percolation AUC distribution", flush=True)
    auc_distribution = np.asarray([auc_for_scores(graph, make_score_table(base_scores, row)) for row in calibration])

    print(f"[{scenario}] Computing pooled diffusion T_max distribution", flush=True)
    diffusion_distribution = diffusion_tmax_distribution(
        graph,
        base_scores,
        calibration,
        mode=args.diffusion_score_mode,
        tau_grid=tau_grid,
        batch_size=args.diffusion_batch_size,
    )

    print(f"[{scenario}] Computing module selection-aware screen", flush=True)
    module_table = module_selection_rows(graph, base_scores, calibration)

    rows: list[dict[str, object]] = []
    for counter, idx in enumerate(outer_indices, start=1):
        vec = calibration[int(idx)]
        scores = make_score_table(base_scores, vec)
        observed_auc = float(auc_distribution[int(idx)])
        pipeline_summary = leave_one_summary(auc_distribution, observed_auc, int(idx))
        diffusion_z = z_against(diffusion_distribution, float(diffusion_distribution[int(idx)]), int(idx))
        diffusion_p = empirical_upper(diffusion_distribution, float(diffusion_distribution[int(idx)]), int(idx))

        degree_summary = degree_matched_summary(
            graph,
            scores,
            observed_auc,
            n_replicates=args.inner_degree_matched_null,
            degree_bins=args.degree_bins,
            seed=args.seed + 1000 + int(idx),
        )
        ranking = rank_nodes_by_score(scores, node_col="gene_symbol", score_col="assoc_resid_score")
        graph_auc = graph_null_auc_distribution(graph_nulls, ranking)
        graph_summary = summarize_percolation_null(pd.DataFrame({"percolation_auc": graph_auc}), observed_auc)

        module_p = float(module_table.loc[int(idx), "selection_aware_score_p"])
        rows.append(
            {
                "scenario": scenario,
                "outer_replicate": int(idx),
                "observed_auc": observed_auc,
                "tier0_pipeline_z": float(pipeline_summary["z"]),
                "tier0_pipeline_p": float(pipeline_summary["empirical_p_upper"]),
                "tier0_pass_z": bool(float(pipeline_summary["z"]) >= args.z_threshold),
                "tier1_degree_matched_z": float(degree_summary["z"]),
                "tier1_degree_matched_p": float(degree_summary["empirical_p_upper"]),
                "tier1_pass_z": bool(float(degree_summary["z"]) >= args.z_threshold),
                "tier2_diffusion_tmax": float(diffusion_distribution[int(idx)]),
                "tier2_diffusion_z": diffusion_z,
                "tier2_diffusion_p": diffusion_p,
                "tier2_pass_z": bool(diffusion_z >= args.z_threshold),
                "tier3_graph_z": float(graph_summary["z"]),
                "tier3_graph_p": float(graph_summary["empirical_p_upper"]),
                "tier3_pass_z": bool(float(graph_summary["z"]) >= args.z_threshold),
                "tier4_n_candidate_modules": int(module_table.loc[int(idx), "n_candidate_modules"]),
                "tier4_max_module_mean_score": float(module_table.loc[int(idx), "max_module_mean_score"]),
                "tier4_selection_aware_p": module_p,
                "tier4_selection_fwer_pass": bool(np.isfinite(module_p) and module_p <= args.module_alpha),
            }
        )
        if counter % 10 == 0 or counter == len(outer_indices):
            print(f"[{scenario}] Processed {counter:,}/{len(outer_indices):,} outer null replicates", flush=True)
    module_table = module_table.assign(scenario=scenario)
    return pd.DataFrame(rows), module_table


def summarize_claims(claims: pd.DataFrame, *, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tier_cols = {
        "tier0_pipeline": "tier0_pass_z",
        "tier1_degree_matched": "tier1_pass_z",
        "tier2_diffusion": "tier2_pass_z",
        "tier3_graph": "tier3_pass_z",
        "tier4_module_selection_fwer": "tier4_selection_fwer_pass",
    }
    for scenario, group in claims.groupby("scenario", observed=True):
        any_positive = np.zeros(len(group), dtype=bool)
        for tier, col in tier_cols.items():
            values = group[col].astype(bool).to_numpy()
            any_positive |= values
            rows.append(
                {
                    "scenario": scenario,
                    "claim_tier": tier,
                    "n_outer": int(len(group)),
                    "false_positive_count": int(values.sum()),
                    "false_positive_rate": float(values.mean()),
                    "z_threshold": float(args.z_threshold),
                    "module_alpha": float(args.module_alpha),
                    "acceptable_0_10": bool(float(values.mean()) <= 0.10),
                }
            )
        rows.append(
            {
                "scenario": scenario,
                "claim_tier": "any_tier_positive",
                "n_outer": int(len(group)),
                "false_positive_count": int(any_positive.sum()),
                "false_positive_rate": float(any_positive.mean()),
                "z_threshold": float(args.z_threshold),
                "module_alpha": float(args.module_alpha),
                "acceptable_0_10": bool(float(any_positive.mean()) <= 0.10),
            }
        )
    return pd.DataFrame(rows)


def threshold_sensitivity(claims: pd.DataFrame, *, thresholds: tuple[float, ...] = (2.0, 2.25, 2.5, 2.75, 3.0)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tier_z_cols = {
        "tier0_pipeline": "tier0_pipeline_z",
        "tier1_degree_matched": "tier1_degree_matched_z",
        "tier2_diffusion": "tier2_diffusion_z",
        "tier3_graph": "tier3_graph_z",
    }
    for scenario, group in claims.groupby("scenario", observed=True):
        module_pass = group["tier4_selection_fwer_pass"].astype(bool).to_numpy()
        for threshold in thresholds:
            any_positive = module_pass.copy()
            for tier, col in tier_z_cols.items():
                values = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float) >= threshold
                any_positive |= values
                rows.append(
                    {
                        "scenario": scenario,
                        "z_threshold": float(threshold),
                        "claim_tier": tier,
                        "false_positive_count": int(values.sum()),
                        "false_positive_rate": float(values.mean()),
                    }
                )
            rows.append(
                {
                    "scenario": scenario,
                    "z_threshold": float(threshold),
                    "claim_tier": "tier4_module_selection_fwer",
                    "false_positive_count": int(module_pass.sum()),
                    "false_positive_rate": float(module_pass.mean()),
                }
            )
            rows.append(
                {
                    "scenario": scenario,
                    "z_threshold": float(threshold),
                    "claim_tier": "any_tier_positive",
                    "false_positive_count": int(any_positive.sum()),
                    "false_positive_rate": float(any_positive.mean()),
                }
            )
    return pd.DataFrame(rows)


def render_report(args: argparse.Namespace, summary: pd.DataFrame) -> str:
    lines = [
        "# RIPPLE V1 Type 1 Error Calibration Report",
        "",
        f"Trait: `{args.trait}`",
        f"Graph: `{args.graph_name}`",
        f"Analysis directory: `{args.analysis_dir}`",
        f"Analysis graph edge cache: `{args.analysis_graph_edge_cache or default_analysis_graph_edges_path(args)}`",
        f"Outer null replicates: {args.outer_n}",
        f"Calibration null vectors: {args.calibration_null_n}",
        f"Degree-matched inner null per outer replicate: {args.inner_degree_matched_null}",
        f"Degree-preserving graph null ensemble: {args.graph_null_replicates}",
        "",
        "## Summary",
        "",
        "| Scenario | Claim tier | N | FP count | FP rate | Acceptable <=0.10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.scenario} | {row.claim_tier} | {int(row.n_outer)} | "
            f"{int(row.false_positive_count)} | {float(row.false_positive_rate):.3f} | "
            f"{bool(row.acceptable_0_10)} |"
        )
    failed = summary.loc[~summary["acceptable_0_10"].astype(bool)]
    lines.extend(["", "## Interpretation", ""])
    if failed.empty:
        lines.append("No claim tier exceeded the pre-specified 0.10 false-positive alert threshold.")
    else:
        lines.append("The following claim tiers exceeded the 0.10 false-positive alert threshold:")
        for row in failed.itertuples(index=False):
            lines.append(f"- {row.scenario} / {row.claim_tier}: FP rate {float(row.false_positive_rate):.3f}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Tier 2 is calibrated here against pooled LD-aware pipeline-null diffusion T_max values.",
            "- Tier 4 is reported as a selection-aware module FWER screen, not full module-level recalibration.",
            "- If this run is stable, the next manuscript-grade expansion is 500 outer null replicates.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    lcc_path = args.lcc_scores or default_lcc_scores_path(args)
    null_path = args.null_scores or default_null_scores_path(args)

    base_scores = pd.read_csv(lcc_path, sep="\t", compression="infer")
    base_scores["gene_symbol"] = base_scores["gene_symbol"].astype(str)
    gene_symbols = tuple(base_scores["gene_symbol"])
    null_matrix = load_null_matrix(null_path, len(base_scores), max_rows=args.calibration_null_n)
    graph = load_graph(args, gene_symbols)
    missing = [gene for gene in gene_symbols if not graph.has_node(gene)]
    if missing:
        raise ValueError(f"Graph is missing {len(missing)} LCC score genes; first missing: {missing[:5]}")

    print(f"Generating {args.graph_null_replicates} degree-preserving graph nulls", flush=True)
    graph_nulls = list(
        degree_preserving_graph_replicates(
            graph,
            n_replicates=args.graph_null_replicates,
            seed=args.seed + 5000,
            nswap_per_edge=1.0,
            max_tries_per_swap=20.0,
            cache_path=args.graph_null_cache,
        )
    )

    tau_grid = parse_tau_grid(args.tau_grid)
    scenarios: list[tuple[str, np.ndarray]] = [("ld_pipeline_null", null_matrix)]
    if args.include_degree_biased_null:
        scenarios.append(
            (
                "degree_biased_null",
                degree_biased_matrix(base_scores, n=min(args.calibration_null_n, len(null_matrix)), seed=args.seed + 7000),
            )
        )
    if args.include_technical_confounded_null:
        tech = technical_matrix(base_scores, n=min(args.calibration_null_n, len(null_matrix)), seed=args.seed + 9000)
        if len(tech):
            scenarios.append(("technical_confounded_null", tech))
        else:
            print("Skipping technical_confounded_null because no technical covariates were available.", flush=True)

    claim_tables: list[pd.DataFrame] = []
    module_tables: list[pd.DataFrame] = []
    for scenario, matrix in scenarios:
        claims, modules = run_scenario(
            scenario=scenario,
            matrix=matrix,
            graph=graph,
            graph_nulls=graph_nulls,
            base_scores=base_scores,
            args=args,
            tau_grid=tau_grid,
        )
        claim_tables.append(claims)
        module_tables.append(modules)

    all_claims = pd.concat(claim_tables, ignore_index=True)
    all_modules = pd.concat(module_tables, ignore_index=True)
    summary = summarize_claims(all_claims, args=args)
    sensitivity = threshold_sensitivity(all_claims)

    write_table(args.out_dir / "type1_outer_replicate_claims.tsv", all_claims)
    write_table(args.out_dir / "type1_module_selection.tsv", all_modules)
    write_table(args.out_dir / "type1_summary.tsv", summary)
    write_table(args.out_dir / "type1_threshold_sensitivity.tsv", sensitivity)
    (args.out_dir / "type1_error_calibration_report.md").write_text(render_report(args, summary), encoding="utf-8")
    print(f"Wrote Type 1 error calibration outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
