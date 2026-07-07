#!/usr/bin/env python
"""Run network ablation baselines for manuscript-readiness comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.graph import graph_laplacian, graph_from_edge_list  # noqa: E402
from ripple.graph_diffusion import (  # noqa: E402
    DEFAULT_TAU_GRID,
    aligned_score_vector,
    heat_kernel_tau_statistics,
    heat_kernel_tau_statistics_matrix,
    parse_tau_grid,
)
from ripple.io.graph import read_edge_list  # noqa: E402
from ripple.percolation.auc import component_stats_for_nodes, percolation_auc, percolation_curve  # noqa: E402
from ripple.percolation.calibration import summarize_percolation_null  # noqa: E402
from ripple.percolation.ranking import rank_nodes_by_score, selected_nodes_at_fraction  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "external_baselines" / "network_ablation_v1"
DEFAULT_GRID = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)


@dataclass(frozen=True)
class TraitSpec:
    trait: str
    analysis_id: str
    analysis_dir: Path
    graph_edges_override: Path | None = None

    @property
    def tables_dir(self) -> Path:
        return self.analysis_dir / "tables"

    @property
    def scores_path(self) -> Path:
        return self.tables_dir / f"{self.trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"

    @property
    def graph_edges_path(self) -> Path:
        if self.graph_edges_override is not None:
            return self.graph_edges_override
        return self.tables_dir / f"{self.trait}.analysis_graph_edges.tsv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trait", action="append", default=[])
    parser.add_argument("--n-null", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--rank-fractions", default=",".join(str(item) for item in DEFAULT_GRID))
    parser.add_argument("--tau-grid", default=",".join(str(item) for item in DEFAULT_TAU_GRID))
    parser.add_argument("--diffusion-batch-size", type=int, default=128)
    parser.add_argument("--save-null-distributions", action="store_true")
    return parser.parse_args()


def default_specs() -> list[TraitSpec]:
    return [
        TraitSpec(
            trait="DR_MVP",
            analysis_id="DR_MVP_default_final5000",
            analysis_dir=ANALYSIS_ROOT / "dr_mvp_string_final5000",
        ),
        TraitSpec(
            trait="DR_MVP_NO_MHC_NO_APOE",
            analysis_id="DR_MVP_no_MHC_no_APOE_final5000",
            analysis_dir=ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000",
            graph_edges_override=ANALYSIS_ROOT
            / "dr_mvp_string_final5000"
            / "tables"
            / "DR_MVP.analysis_graph_edges.tsv.gz",
        ),
        TraitSpec(
            trait="SCZ",
            analysis_id="SCZ_no_MHC_final5000",
            analysis_dir=ANALYSIS_ROOT / "scz_no_mhc_string_final5000",
        ),
    ]


def select_specs(args: argparse.Namespace) -> list[TraitSpec]:
    specs = default_specs()
    if not args.trait:
        return specs
    requested = set(args.trait)
    selected = [spec for spec in specs if spec.trait in requested or spec.analysis_id in requested]
    missing = sorted(requested - {item.trait for item in selected} - {item.analysis_id for item in selected})
    if missing:
        raise ValueError(f"Unknown requested traits/analysis IDs: {missing}")
    return selected


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, compression="infer")


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", compression="infer")


def parse_float_grid(value: str) -> tuple[float, ...]:
    out = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not out:
        raise ValueError("grid must not be empty")
    return tuple(sorted(dict.fromkeys(out)))


def load_lcc_graph(spec: TraitSpec, scores: pd.DataFrame) -> nx.Graph:
    edge_table = read_edge_list(spec.graph_edges_path, sep="\t", source=f"{spec.analysis_id}_analysis_graph").edges
    gene_universe = scores["gene_symbol"].dropna().astype(str).unique()
    graph = graph_from_edge_list(edge_table, gene_universe=gene_universe)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_nodes() == 0:
        raise ValueError(f"Analysis graph is empty for {spec.analysis_id}")
    largest = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest).copy()


def null_summary(values: np.ndarray, observed: float, *, direction: str = "greater_is_more_extreme") -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"null_mean": np.nan, "null_sd": np.nan, "z": np.nan, "empirical_p": np.nan, "n_null": 0}
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan
    z = float((observed - mean) / sd) if sd > 0 else np.nan
    if direction == "greater_is_more_extreme":
        exceed = int(np.count_nonzero(finite >= observed))
    elif direction == "less_is_more_extreme":
        exceed = int(np.count_nonzero(finite <= observed))
    else:
        exceed = int(np.count_nonzero(np.abs(finite - mean) >= abs(observed - mean)))
    return {
        "null_mean": mean,
        "null_sd": sd,
        "z": z,
        "empirical_p": float((1 + exceed) / (1 + finite.size)),
        "n_null": int(finite.size),
    }


def random_node_connectivity_null(
    graph: nx.Graph,
    ranking: pd.DataFrame,
    fractions: tuple[float, ...],
    *,
    n_null: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_curve = percolation_curve(graph, ranking, fractions)
    selected_counts = {
        fraction: len(selected_nodes_at_fraction(ranking, fraction))
        for fraction in fractions
    }
    nodes = np.asarray(sorted(str(node) for node in graph.nodes()), dtype=object)
    rng = np.random.default_rng(seed)
    auc_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for replicate in range(n_null):
        rows: list[dict[str, Any]] = []
        for fraction, count in selected_counts.items():
            sampled = rng.choice(nodes, size=count, replace=False)
            stats = component_stats_for_nodes(graph, sampled)
            row = {"rank_fraction": fraction, **stats}
            rows.append(row)
            curve_rows.append({"replicate": replicate, **row})
        auc_rows.append({"replicate": replicate, "percolation_auc": percolation_auc(pd.DataFrame(rows))})
    return observed_curve, pd.DataFrame(auc_rows), pd.DataFrame(curve_rows)


def induced_edge_density_curve(graph: nx.Graph, ranking: pd.DataFrame, fractions: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        selected = [node for node in selected_nodes_at_fraction(ranking, fraction) if graph.has_node(node)]
        n = len(selected)
        edges = int(graph.subgraph(selected).number_of_edges()) if n else 0
        possible = n * (n - 1) / 2
        rows.append(
            {
                "rank_fraction": fraction,
                "n_selected": n,
                "induced_edge_count": edges,
                "induced_edge_density": float(edges / possible) if possible > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def random_node_edge_density_null(
    graph: nx.Graph,
    ranking: pd.DataFrame,
    fractions: tuple[float, ...],
    *,
    n_null: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = induced_edge_density_curve(graph, ranking, fractions)
    selected_counts = {
        fraction: len(selected_nodes_at_fraction(ranking, fraction))
        for fraction in fractions
    }
    nodes = np.asarray(sorted(str(node) for node in graph.nodes()), dtype=object)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    for replicate in range(n_null):
        curve_rows: list[dict[str, Any]] = []
        for fraction, count in selected_counts.items():
            sampled = rng.choice(nodes, size=count, replace=False)
            n = len(sampled)
            edges = int(graph.subgraph(sampled).number_of_edges()) if n else 0
            possible = n * (n - 1) / 2
            row = {
                "replicate": replicate,
                "rank_fraction": fraction,
                "n_selected": n,
                "induced_edge_count": edges,
                "induced_edge_density": float(edges / possible) if possible > 0 else 0.0,
            }
            rows.append(row)
            curve_rows.append(row)
        auc_rows.append(
            {
                "replicate": replicate,
                "edge_density_auc": percolation_auc(
                    pd.DataFrame(curve_rows),
                    y_col="induced_edge_density",
                ),
            }
        )
    return observed, pd.DataFrame(rows), pd.DataFrame(auc_rows)


def unstratified_diffusion_null(
    graph: nx.Graph,
    scores: pd.DataFrame,
    tau_grid: tuple[float, ...],
    *,
    n_null: int,
    seed: int,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = tuple(str(node) for node in scores["gene_symbol"].astype(str) if graph.has_node(str(node)))
    lap = graph_laplacian(graph, nodes=nodes, kind="normalized", weight=None)
    observed_s = aligned_score_vector(scores, lap.nodes, mode="positive")
    observed_tau = heat_kernel_tau_statistics(lap.laplacian, observed_s, tau_grid=tau_grid)
    observed_tmax = float(observed_tau["T_tau"].max())
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for start in range(0, n_null, batch_size):
        stop = min(start + batch_size, n_null)
        null_scores = np.vstack([rng.permutation(observed_s) for _ in range(start, stop)])
        tau_values = heat_kernel_tau_statistics_matrix(
            lap.laplacian,
            null_scores,
            tau_grid=tau_grid,
            batch_size=batch_size,
        )
        tmax_values = np.max(tau_values, axis=1)
        for local_idx, replicate in enumerate(range(start, stop)):
            for tau_idx, tau in enumerate(tau_grid):
                rows.append(
                    {
                        "replicate": replicate,
                        "tau": tau,
                        "T_tau": float(tau_values[local_idx, tau_idx]),
                        "T_max": float(tmax_values[local_idx]),
                    }
                )
    nulls = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "T_max": observed_tmax,
                "tau_at_max": float(observed_tau.loc[observed_tau["T_tau"].idxmax(), "tau"]),
                **null_summary(nulls.groupby("replicate", observed=True)["T_max"].first().to_numpy(), observed_tmax),
            }
        ]
    )
    return observed_tau, summary, nulls


def summary_row(
    *,
    spec: TraitSpec,
    baseline_name: str,
    statistic_name: str,
    null_type: str,
    observed_value: float,
    summary: dict[str, Any],
    source_result_path: Path,
    seed: int,
    interpretation: str,
) -> dict[str, Any]:
    return {
        "trait": spec.trait,
        "analysis_id": spec.analysis_id,
        "graph_id": "STRING_default",
        "baseline_name": baseline_name,
        "statistic_name": statistic_name,
        "statistic_direction": "greater_is_more_extreme",
        "null_type": null_type,
        "observed_value": observed_value,
        "null_mean": summary.get("null_mean", summary.get("mean", np.nan)),
        "null_sd": summary.get("null_sd", summary.get("sd", np.nan)),
        "z": summary.get("z", np.nan),
        "empirical_p": summary.get("empirical_p", summary.get("empirical_p_upper", np.nan)),
        "n_null": summary.get("n_null", summary.get("n_replicates", np.nan)),
        "seed": seed,
        "source_result_path": str(source_result_path),
        "script_path": str(Path(__file__).resolve()),
        "timestamp": datetime.now(UTC).isoformat(),
        "interpretation": interpretation,
    }


def existing_percolation_summary(spec: TraitSpec, null_file: str, observed_auc: float) -> dict[str, Any]:
    nulls = read_tsv(spec.tables_dir / null_file)
    return summarize_percolation_null(nulls, observed_auc)


def existing_diffusion_summary(spec: TraitSpec) -> dict[str, Any]:
    table = read_tsv(spec.tables_dir / f"{spec.trait}.diffusion_kernel_summary.tsv")
    if table.empty:
        return {}
    row = table.iloc[0].to_dict()
    return {
        "null_mean": row.get("null_mean", np.nan),
        "null_sd": row.get("null_sd", np.nan),
        "z": row.get("z", np.nan),
        "empirical_p": row.get("empirical_p", np.nan),
        "n_null": row.get("n_null", np.nan),
        "T_max": row.get("T_max", np.nan),
    }


def run_trait(spec: TraitSpec, args: argparse.Namespace, fractions: tuple[float, ...], tau_grid: tuple[float, ...]) -> dict[str, Any]:
    scores = read_tsv(spec.scores_path)
    graph = load_lcc_graph(spec, scores)
    trait_dir = args.out_dir / spec.analysis_id
    scores = scores.loc[scores["gene_symbol"].astype(str).isin(set(map(str, graph.nodes())))].copy()
    ranking = rank_nodes_by_score(scores)

    observed_curve, random_auc, random_curves = random_node_connectivity_null(
        graph,
        ranking,
        fractions,
        n_null=args.n_null,
        seed=args.seed,
    )
    observed_auc = percolation_auc(observed_curve)
    random_summary = summarize_percolation_null(random_auc, observed_auc)

    edge_observed, edge_null_curve, edge_null_auc = random_node_edge_density_null(
        graph,
        ranking,
        fractions,
        n_null=args.n_null,
        seed=args.seed + 101,
    )
    observed_edge_auc = percolation_auc(edge_observed, y_col="induced_edge_density")
    edge_summary = null_summary(edge_null_auc["edge_density_auc"].to_numpy(dtype=float), observed_edge_auc)

    diffusion_observed, diffusion_summary, diffusion_nulls = unstratified_diffusion_null(
        graph,
        scores,
        tau_grid,
        n_null=args.n_null,
        seed=args.seed + 202,
        batch_size=args.diffusion_batch_size,
    )
    diffusion_summary_dict = diffusion_summary.iloc[0].to_dict()

    write_table(trait_dir / "tables" / f"{spec.trait}.naive_random_node_percolation_auc.tsv", random_auc)
    write_table(trait_dir / "tables" / f"{spec.trait}.naive_random_node_percolation_curve_observed.tsv", observed_curve)
    write_table(trait_dir / "tables" / f"{spec.trait}.naive_edge_density_observed.tsv", edge_observed)
    write_table(trait_dir / "tables" / f"{spec.trait}.naive_edge_density_auc.tsv", edge_null_auc)
    write_table(trait_dir / "tables" / f"{spec.trait}.heat_kernel_unstratified_summary.tsv", diffusion_summary)
    write_table(trait_dir / "tables" / f"{spec.trait}.heat_kernel_unstratified_tau_observed.tsv", diffusion_observed)
    if args.save_null_distributions:
        write_table(trait_dir / "tables" / f"{spec.trait}.naive_random_node_percolation_curves.tsv.gz", random_curves)
        write_table(trait_dir / "tables" / f"{spec.trait}.naive_edge_density_curves.tsv.gz", edge_null_curve)
        write_table(trait_dir / "tables" / f"{spec.trait}.heat_kernel_unstratified_nulls.tsv.gz", diffusion_nulls)

    rows = [
        summary_row(
            spec=spec,
            baseline_name="naive_top_gene_ppi_connectivity",
            statistic_name="percolation_auc_random_node_null",
            null_type="random_node_set_unmatched",
            observed_value=observed_auc,
            summary=random_summary,
            source_result_path=trait_dir / "tables" / f"{spec.trait}.naive_random_node_percolation_auc.tsv",
            seed=args.seed,
            interpretation="Naive top-gene PPI connectivity without degree matching.",
        ),
        summary_row(
            spec=spec,
            baseline_name="naive_top_gene_edge_enrichment",
            statistic_name="edge_density_auc_random_node_null",
            null_type="random_node_set_unmatched",
            observed_value=observed_edge_auc,
            summary=edge_summary,
            source_result_path=trait_dir / "tables" / f"{spec.trait}.naive_edge_density_auc.tsv",
            seed=args.seed + 101,
            interpretation="Naive induced-edge density without degree matching.",
        ),
        summary_row(
            spec=spec,
            baseline_name="heat_kernel_propagation_unstratified",
            statistic_name="diffusion_kernel_Tmax_unstratified",
            null_type="unstratified_score_permutation",
            observed_value=float(diffusion_summary_dict["T_max"]),
            summary=diffusion_summary_dict,
            source_result_path=trait_dir / "tables" / f"{spec.trait}.heat_kernel_unstratified_summary.tsv",
            seed=args.seed + 202,
            interpretation="Heat-kernel propagation baseline before degree-stratified calibration.",
        ),
    ]

    snp_summary = existing_percolation_summary(spec, f"{spec.trait}.percolation_auc.1000G_LD.null.tsv", observed_auc)
    rows.append(
        summary_row(
            spec=spec,
            baseline_name="ripple_without_degree_null",
            statistic_name="percolation_auc_snp_pipeline_null",
            null_type="snp_pipeline_null",
            observed_value=observed_auc,
            summary=snp_summary,
            source_result_path=spec.tables_dir / f"{spec.trait}.percolation_auc.1000G_LD.null.tsv",
            seed=args.seed,
            interpretation="RIPPLE percolation interpreted without the degree-matched node null.",
        )
    )
    degree_summary = existing_percolation_summary(
        spec,
        f"{spec.trait}.percolation_auc.1000G_LD.degree_matched_node_null.tsv",
        observed_auc,
    )
    rows.append(
        summary_row(
            spec=spec,
            baseline_name="ripple_full_degree_calibrated",
            statistic_name="percolation_auc_degree_matched_node_null",
            null_type="degree_matched_node_null",
            observed_value=observed_auc,
            summary=degree_summary,
            source_result_path=spec.tables_dir / f"{spec.trait}.percolation_auc.1000G_LD.degree_matched_node_null.tsv",
            seed=args.seed,
            interpretation="Full RIPPLE Tier 1 degree-calibrated top-rank aggregation.",
        )
    )
    degree_diffusion = existing_diffusion_summary(spec)
    rows.append(
        summary_row(
            spec=spec,
            baseline_name="ripple_full_degree_stratified_diffusion",
            statistic_name="diffusion_kernel_Tmax_degree_stratified",
            null_type="degree_stratified_score_permutation",
            observed_value=float(degree_diffusion.get("T_max", np.nan)),
            summary=degree_diffusion,
            source_result_path=spec.tables_dir / f"{spec.trait}.diffusion_kernel_summary.tsv",
            seed=args.seed,
            interpretation="Full RIPPLE Tier 2 graph-domain diffusion aggregation.",
        )
    )
    summary_table = pd.DataFrame(rows)
    write_table(trait_dir / "tables" / f"{spec.trait}.network_ablation_summary.tsv", summary_table)
    return {
        "trait": spec.trait,
        "analysis_id": spec.analysis_id,
        "n_scores": int(len(scores)),
        "n_graph_nodes": int(graph.number_of_nodes()),
        "n_graph_edges": int(graph.number_of_edges()),
        "summary_rows": int(len(summary_table)),
        "summary_path": str(trait_dir / "tables" / f"{spec.trait}.network_ablation_summary.tsv"),
    }


def write_report(out_dir: Path, summary: pd.DataFrame, manifests: list[dict[str, Any]]) -> None:
    lines = [
        "# RIPPLE V1 Network Ablation Baseline",
        "",
        f"Generated UTC: {datetime.now(UTC).isoformat()}",
        "",
        "This baseline separates naive PPI connectivity, heat-kernel propagation before degree calibration, and full RIPPLE degree-calibrated evidence.",
        "",
        "| Trait | Baseline | Statistic | Null | Z | empirical P |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.trait} | {row.baseline_name} | {row.statistic_name} | {row.null_type} | "
            f"{float(row.z):.3f} | {float(row.empirical_p):.4g} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guardrail: naive random-node positives do not support biological module discovery unless the signal remains positive after degree-aware calibration.",
            "",
            "Run manifests:",
        ]
    )
    for item in manifests:
        lines.append(
            f"- `{item['analysis_id']}`: {item['n_scores']} scored LCC genes, "
            f"{item['n_graph_nodes']} graph nodes, {item['n_graph_edges']} edges."
        )
    (out_dir / "network_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.n_null < 1:
        raise ValueError("--n-null must be positive")
    fractions = parse_float_grid(args.rank_fractions)
    tau_grid = parse_tau_grid(args.tau_grid)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifests = [run_trait(spec, args, fractions, tau_grid) for spec in select_specs(args)]
    summary_tables = [read_tsv(Path(item["summary_path"])) for item in manifests]
    summary = pd.concat(summary_tables, ignore_index=True)
    write_table(args.out_dir / "network_ablation_summary.all_traits.tsv", summary)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "baseline": "network_ablation_v1",
        "n_null": int(args.n_null),
        "seed": int(args.seed),
        "rank_fractions": ",".join(str(item) for item in fractions),
        "tau_grid": ",".join(str(item) for item in tau_grid),
        "manifests": manifests,
    }
    (args.out_dir / "network_ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(args.out_dir, summary, manifests)
    print(f"Wrote network ablation outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
