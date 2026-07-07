#!/usr/bin/env python
"""Run MAGMA/PascalX gene scores through the RIPPLE graph/null layer."""

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
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.graph import graph_from_edge_list  # noqa: E402
from ripple.graph_diffusion import DEFAULT_TAU_GRID, degree_stratified_diffusion_null, parse_tau_grid  # noqa: E402
from ripple.io.graph import read_edge_list  # noqa: E402
from ripple.nulls.graph_nulls import degree_preserving_graph_replicates  # noqa: E402
from ripple.percolation import (  # noqa: E402
    compute_degree_matched_node_percolation_null,
    percolation_auc,
    percolation_curve,
    prepare_degree_matched_rank_sets,
    rank_nodes_by_score,
    summarize_percolation_null,
)
from ripple.policy import classify_z_claim, final_z_threshold, load_claim_policy  # noqa: E402
from ripple.signals.residualize import append_residualized_score  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "external_baselines" / "external_score_graph_layer_v1"
MAGMA_DIR = ANALYSIS_ROOT / "external_baselines" / "magma_v1.10"
PASCALX_DIR = ANALYSIS_ROOT / "external_baselines" / "pascalx"
POLICY_PATH = PROJECT_ROOT / "ripple" / "config" / "claim_policy.yaml"
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


@dataclass(frozen=True)
class ScoreSpec:
    score_source: str
    score_file: Path
    p_col: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--trait", action="append", default=[])
    parser.add_argument("--score-source", action="append", choices=["MAGMA", "PascalX"], default=[])
    parser.add_argument("--n-degree-matched-null", type=int, default=5000)
    parser.add_argument("--n-diffusion-null", type=int, default=5000)
    parser.add_argument("--n-degree-graph-null", type=int, default=100)
    parser.add_argument("--rank-fractions", default=",".join(str(item) for item in DEFAULT_GRID))
    parser.add_argument("--tau-grid", default=",".join(str(item) for item in DEFAULT_TAU_GRID))
    parser.add_argument("--degree-bins", type=int, default=20)
    parser.add_argument("--diffusion-batch-size", type=int, default=128)
    parser.add_argument("--degree-graph-nswap-per-edge", type=float, default=0.10)
    parser.add_argument("--degree-graph-max-tries-per-swap", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--save-null-distributions", action="store_true")
    return parser.parse_args()


def default_trait_specs() -> list[TraitSpec]:
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


def default_score_specs() -> list[ScoreSpec]:
    return [
        ScoreSpec(
            score_source="MAGMA",
            score_file=MAGMA_DIR / "magma_gene_results.all_traits.tsv.gz",
            p_col="magma_p",
        ),
        ScoreSpec(
            score_source="PascalX",
            score_file=PASCALX_DIR / "pascalx_gene_results.all_traits.tsv.gz",
            p_col="pascalx_p",
        ),
    ]


def select_trait_specs(args: argparse.Namespace) -> list[TraitSpec]:
    specs = default_trait_specs()
    if not args.trait:
        return specs
    requested = set(args.trait)
    selected = [spec for spec in specs if spec.trait in requested or spec.analysis_id in requested]
    missing = sorted(requested - {item.trait for item in selected} - {item.analysis_id for item in selected})
    if missing:
        raise ValueError(f"Unknown traits/analysis IDs: {missing}")
    return selected


def select_score_specs(args: argparse.Namespace) -> list[ScoreSpec]:
    specs = default_score_specs()
    if not args.score_source:
        return specs
    requested = set(args.score_source)
    return [spec for spec in specs if spec.score_source in requested]


def parse_float_grid(value: str) -> tuple[float, ...]:
    out = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not out:
        raise ValueError("grid must not be empty")
    return tuple(sorted(dict.fromkeys(out)))


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", compression="infer")


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, compression="infer")


def load_lcc_graph(spec: TraitSpec, scores: pd.DataFrame) -> nx.Graph:
    edges = read_edge_list(spec.graph_edges_path, sep="\t", source=f"{spec.analysis_id}_analysis_graph").edges
    gene_universe = scores["gene_symbol"].dropna().astype(str).unique()
    graph = graph_from_edge_list(edges, gene_universe=gene_universe)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_nodes() == 0:
        raise ValueError(f"Analysis graph is empty for {spec.analysis_id}")
    largest = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest).copy()


def external_score_table(spec: TraitSpec, score_spec: ScoreSpec) -> pd.DataFrame:
    base = read_tsv(spec.scores_path)
    external = read_tsv(score_spec.score_file)
    external = external.loc[external["analysis_id"].astype(str) == spec.analysis_id].copy()
    if external.empty:
        raise ValueError(f"No {score_spec.score_source} scores found for {spec.analysis_id}")
    external = external.loc[:, ["gene_symbol", score_spec.p_col]].dropna().drop_duplicates("gene_symbol")
    joined = base.merge(external, on="gene_symbol", how="inner")
    if joined.empty:
        raise ValueError(f"No overlap between {spec.analysis_id} and {score_spec.score_source}")
    p = pd.to_numeric(joined[score_spec.p_col], errors="coerce").clip(lower=1e-300, upper=1 - 1e-16)
    score_col = f"{score_spec.score_source.lower()}_normal_score"
    resid_col = f"{score_spec.score_source.lower()}_resid_score"
    joined[score_col] = stats.norm.isf(p).astype(float)
    joined[f"{score_spec.score_source.lower()}_p"] = p.astype(float)
    residualized, _ = append_residualized_score(
        joined,
        score_col=score_col,
        output_col=resid_col,
        null_scores=None,
    )
    residualized["assoc_resid_score"] = residualized[resid_col].astype(float)
    residualized["assoc_p_g"] = residualized[f"{score_spec.score_source.lower()}_p"].astype(float)
    return residualized


def graph_percolation_null(
    graph: nx.Graph,
    ranking: pd.DataFrame,
    fractions: tuple[float, ...],
    *,
    n_replicates: int,
    seed: int,
    cache_path: Path,
    nswap_per_edge: float,
    max_tries_per_swap: float,
) -> pd.DataFrame:
    rows = []
    for replicate, null_graph in enumerate(
        degree_preserving_graph_replicates(
            graph,
            n_replicates=n_replicates,
            seed=seed,
            nswap_per_edge=nswap_per_edge,
            max_tries_per_swap=max_tries_per_swap,
            cache_path=cache_path,
        )
    ):
        curve = percolation_curve(null_graph, ranking, fractions)
        rows.append({"replicate": replicate, "percolation_auc": percolation_auc(curve)})
    return pd.DataFrame(rows)


def claim_row(
    *,
    trait: str,
    analysis_id: str,
    score_source: str,
    statistic: str,
    tier: str,
    observed: float,
    summary: dict[str, Any],
    threshold: float,
    null_type: str,
    source_path: Path,
    seed: int,
) -> dict[str, Any]:
    z = summary.get("z", np.nan)
    return {
        "trait": trait,
        "analysis_id": analysis_id,
        "graph_id": "STRING_default",
        "score_source": score_source,
        "score_stream": f"{score_source.lower()}_resid_score",
        "statistic_name": statistic,
        "claim_tier": tier,
        "null_type": null_type,
        "statistic_direction": "greater_is_more_extreme",
        "observed_value": observed,
        "null_mean": summary.get("mean", summary.get("null_mean", np.nan)),
        "null_sd": summary.get("sd", summary.get("null_sd", np.nan)),
        "z": z,
        "empirical_p": summary.get("empirical_p", summary.get("empirical_p_upper", np.nan)),
        "n_null": summary.get("n_replicates", summary.get("n_null", np.nan)),
        "threshold": threshold,
        "claim_status": classify_z_claim(z),
        "source_result_path": str(source_path),
        "script_path": str(Path(__file__).resolve()),
        "seed": seed,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def run_one(
    spec: TraitSpec,
    score_spec: ScoreSpec,
    args: argparse.Namespace,
    fractions: tuple[float, ...],
    tau_grid: tuple[float, ...],
    threshold: float,
) -> dict[str, Any]:
    score_table = external_score_table(spec, score_spec)
    graph = load_lcc_graph(spec, score_table)
    score_table = score_table.loc[score_table["gene_symbol"].astype(str).isin(set(map(str, graph.nodes())))].copy()
    ranking = rank_nodes_by_score(score_table)
    observed_curve = percolation_curve(graph, ranking, fractions)
    observed_auc = percolation_auc(observed_curve)

    out_dir = args.out_dir / spec.analysis_id / score_spec.score_source
    tables_dir = out_dir / "tables"
    prefix = f"{spec.trait}.{score_spec.score_source}"
    write_table(tables_dir / f"{prefix}.lcc_external_score_table.tsv.gz", score_table)
    write_table(tables_dir / f"{prefix}.percolation_curve.observed.tsv", observed_curve)

    selected_bin_counts, bin_to_nodes, degree_profile = prepare_degree_matched_rank_sets(
        score_table,
        ranking,
        fractions,
        n_bins=args.degree_bins,
    )
    degree_auc, degree_curves = compute_degree_matched_node_percolation_null(
        graph,
        selected_bin_counts,
        bin_to_nodes,
        n_replicates=args.n_degree_matched_null,
        seed=args.seed,
    )
    degree_summary = summarize_percolation_null(degree_auc, observed_auc)
    write_table(tables_dir / f"{prefix}.percolation_auc.degree_matched_node_null.tsv", degree_auc)
    write_table(tables_dir / f"{prefix}.degree_profile.by_rank_fraction.tsv", degree_profile)
    if args.save_null_distributions:
        write_table(tables_dir / f"{prefix}.percolation_curves.degree_matched_node_null.tsv.gz", degree_curves)

    diffusion_summary, diffusion_tau, diffusion_null = degree_stratified_diffusion_null(
        graph,
        score_table,
        trait=spec.trait,
        graph_name="STRING_default",
        score_mode="positive",
        tau_grid=tau_grid,
        n_replicates=args.n_diffusion_null,
        seed=args.seed + 101,
        n_bins=args.degree_bins,
        score_col="assoc_resid_score",
        batch_size=args.diffusion_batch_size,
    )
    write_table(tables_dir / f"{prefix}.diffusion_kernel_summary.tsv", diffusion_summary)
    write_table(tables_dir / f"{prefix}.diffusion_kernel_tau_stats.tsv", diffusion_tau)
    if args.save_null_distributions:
        write_table(tables_dir / f"{prefix}.diffusion_kernel_null_distribution.tsv.gz", diffusion_null)

    graph_auc = graph_percolation_null(
        graph,
        ranking,
        fractions,
        n_replicates=args.n_degree_graph_null,
        seed=args.seed + 202,
        cache_path=args.out_dir / "graph_null_cache" / f"{spec.analysis_id}.degree_preserving_graph_n{args.n_degree_graph_null}.npz",
        nswap_per_edge=args.degree_graph_nswap_per_edge,
        max_tries_per_swap=args.degree_graph_max_tries_per_swap,
    )
    graph_summary = summarize_percolation_null(graph_auc, observed_auc)
    write_table(tables_dir / f"{prefix}.percolation_auc.degree_preserving_graph_null.tsv", graph_auc)

    rows = [
        claim_row(
            trait=spec.trait,
            analysis_id=spec.analysis_id,
            score_source=score_spec.score_source,
            statistic="degree_calibrated_top_rank_aggregation",
            tier="TIER_1_degree_calibrated_aggregation",
            observed=observed_auc,
            summary=degree_summary,
            threshold=threshold,
            null_type="degree_matched_node_null",
            source_path=tables_dir / f"{prefix}.percolation_auc.degree_matched_node_null.tsv",
            seed=args.seed,
        ),
        claim_row(
            trait=spec.trait,
            analysis_id=spec.analysis_id,
            score_source=score_spec.score_source,
            statistic="degree_preserving_graph_percolation",
            tier="TIER_3_topology_specific_support",
            observed=observed_auc,
            summary=graph_summary,
            threshold=threshold,
            null_type="degree_preserving_graph_null",
            source_path=tables_dir / f"{prefix}.percolation_auc.degree_preserving_graph_null.tsv",
            seed=args.seed + 202,
        ),
    ]
    for item in diffusion_summary.to_dict(orient="records"):
        rows.append(
            claim_row(
                trait=spec.trait,
                analysis_id=spec.analysis_id,
                score_source=score_spec.score_source,
                statistic=f"diffusion_kernel_Tmax_{item.get('null_type', 'null')}",
                tier="TIER_2_graph_domain_aggregation",
                observed=float(item.get("T_max", np.nan)),
                summary=item,
                threshold=threshold,
                null_type=str(item.get("null_type", "degree_stratified")),
                source_path=tables_dir / f"{prefix}.diffusion_kernel_summary.tsv",
                seed=args.seed + 101,
            )
        )
    claim_table = pd.DataFrame(rows)
    write_table(tables_dir / f"{prefix}.external_score_graph_claims.tsv", claim_table)
    return {
        "trait": spec.trait,
        "analysis_id": spec.analysis_id,
        "score_source": score_spec.score_source,
        "n_overlap": int(len(score_table)),
        "observed_auc": float(observed_auc),
        "tier1_z": float(degree_summary["z"]),
        "tier2_z": float(diffusion_summary.iloc[0]["z"]) if not diffusion_summary.empty else float("nan"),
        "tier3_z": float(graph_summary["z"]),
        "claim_path": str(tables_dir / f"{prefix}.external_score_graph_claims.tsv"),
    }


def write_report(out_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# External Gene Scores Through RIPPLE Graph/Null Layer",
        "",
        f"Generated UTC: {datetime.now(UTC).isoformat()}",
        "",
        "MAGMA and PascalX gene-level association scores were transformed, technically residualized, and evaluated with the same RIPPLE graph/null layer.",
        "",
        "| Trait | Score | n overlap | Tier 1 Z | Tier 2 Z | Tier 3 Z |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.trait} | {row.score_source} | {int(row.n_overlap)} | "
            f"{float(row.tier1_z):.3f} | {float(row.tier2_z):.3f} | {float(row.tier3_z):.3f} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guardrail: positive Tier 1/Tier 2 results here support graph-layer robustness, not replacement of established gene-based tests.",
        ]
    )
    (out_dir / "external_score_graph_layer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    policy = load_claim_policy(args.policy)
    threshold = final_z_threshold(policy)
    fractions = parse_float_grid(args.rank_fractions)
    tau_grid = parse_tau_grid(args.tau_grid)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for spec in select_trait_specs(args):
        for score_spec in select_score_specs(args):
            manifests.append(run_one(spec, score_spec, args, fractions, tau_grid, threshold))
    summary = pd.DataFrame(manifests)
    write_table(args.out_dir / "external_score_graph_layer_summary.all_traits.tsv", summary)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "baseline": "external_score_graph_layer_v1",
        "n_degree_matched_null": args.n_degree_matched_null,
        "n_diffusion_null": args.n_diffusion_null,
        "n_degree_graph_null": args.n_degree_graph_null,
        "rank_fractions": ",".join(str(item) for item in fractions),
        "tau_grid": ",".join(str(item) for item in tau_grid),
        "manifests": manifests,
    }
    (args.out_dir / "external_score_graph_layer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(args.out_dir, summary)
    print(f"Wrote external score graph-layer outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
