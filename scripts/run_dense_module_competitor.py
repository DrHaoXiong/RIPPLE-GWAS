#!/usr/bin/env python
"""Run a simplified dense-module-search network competitor baseline."""

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.graph import graph_from_edge_list  # noqa: E402
from ripple.io.graph import read_edge_list  # noqa: E402
from ripple.nulls.graph_nulls import degree_preserving_graph_replicates  # noqa: E402
from ripple.nulls.score_permutation import assign_degree_bins  # noqa: E402
from ripple.policy import load_claim_policy  # noqa: E402
from ripple.percolation.ranking import rank_nodes_by_score  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "external_baselines" / "dense_module_competitor_v1"
POLICY_PATH = PROJECT_ROOT / "ripple" / "config" / "claim_policy.yaml"


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
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--trait", action="append", default=[])
    parser.add_argument("--n-seeds", type=int, default=80)
    parser.add_argument("--min-size", type=int, default=5)
    parser.add_argument("--max-size", type=int, default=40)
    parser.add_argument("--max-modules", type=int, default=20)
    parser.add_argument("--density-weight", type=float, default=1.0)
    parser.add_argument("--jaccard-dedup", type=float, default=0.80)
    parser.add_argument("--n-node-null", type=int, default=500)
    parser.add_argument("--n-degree-graph-null", type=int, default=0)
    parser.add_argument("--degree-bins", type=int, default=20)
    parser.add_argument("--degree-graph-nswap-per-edge", type=float, default=0.10)
    parser.add_argument("--degree-graph-max-tries-per-swap", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260705)
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
        raise ValueError(f"Unknown traits/analysis IDs: {missing}")
    return selected


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", compression="infer")


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, compression="infer")


def load_graph(spec: TraitSpec, scores: pd.DataFrame) -> nx.Graph:
    edge_table = read_edge_list(spec.graph_edges_path, sep="\t", source=f"{spec.analysis_id}_analysis_graph").edges
    gene_universe = scores["gene_symbol"].dropna().astype(str).unique()
    graph = graph_from_edge_list(edge_table, gene_universe=gene_universe)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    if graph.number_of_nodes() == 0:
        raise ValueError(f"Analysis graph is empty for {spec.analysis_id}")
    largest = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest).copy()


def score_maps(scores: pd.DataFrame) -> tuple[dict[str, float], dict[str, int]]:
    score_by_node = dict(zip(scores["gene_symbol"].astype(str), scores["assoc_resid_score"].astype(float), strict=True))
    degree_by_node = dict(zip(scores["gene_symbol"].astype(str), scores["graph_degree"].astype(int), strict=True))
    return score_by_node, degree_by_node


def edge_density(graph: nx.Graph, nodes: tuple[str, ...]) -> float:
    n = len(nodes)
    if n < 2:
        return 0.0
    return float(2.0 * graph.subgraph(nodes).number_of_edges() / (n * (n - 1)))


def module_statistic(
    graph: nx.Graph,
    nodes: tuple[str, ...],
    score_by_node: dict[str, float],
    *,
    density_weight: float,
) -> float:
    positive = np.array([max(0.0, score_by_node[node]) for node in nodes], dtype=float)
    if positive.size == 0:
        return 0.0
    score_term = float(positive.sum() / np.sqrt(positive.size))
    return score_term + density_weight * edge_density(graph, nodes)


def greedy_dense_module(
    graph: nx.Graph,
    seed: str,
    score_by_node: dict[str, float],
    *,
    max_size: int,
    density_weight: float,
) -> tuple[str, ...]:
    module = {str(seed)}
    best = module_statistic(graph, tuple(module), score_by_node, density_weight=density_weight)
    while len(module) < max_size:
        frontier = sorted({str(neighbor) for node in module for neighbor in graph.neighbors(node)} - module)
        if not frontier:
            break
        candidate_scores = []
        for candidate in frontier:
            nodes = tuple(sorted(module | {candidate}))
            stat = module_statistic(graph, nodes, score_by_node, density_weight=density_weight)
            candidate_scores.append((stat, score_by_node.get(candidate, float("-inf")), candidate))
        stat, _, candidate = max(candidate_scores, key=lambda item: (item[0], item[1], item[2]))
        if stat <= best:
            break
        module.add(candidate)
        best = stat
    return tuple(sorted(module))


def deduplicate_modules(
    modules: list[tuple[str, ...]],
    graph: nx.Graph,
    score_by_node: dict[str, float],
    *,
    density_weight: float,
    jaccard_cutoff: float,
    max_modules: int,
) -> list[tuple[str, ...]]:
    ranked = sorted(
        modules,
        key=lambda nodes: (
            module_statistic(graph, nodes, score_by_node, density_weight=density_weight),
            len(nodes),
            nodes,
        ),
        reverse=True,
    )
    kept: list[tuple[str, ...]] = []
    kept_sets: list[set[str]] = []
    for nodes in ranked:
        node_set = set(nodes)
        if any(len(node_set & other) / len(node_set | other) >= jaccard_cutoff for other in kept_sets):
            continue
        kept.append(nodes)
        kept_sets.append(node_set)
        if len(kept) >= max_modules:
            break
    return kept


def discover_modules(graph: nx.Graph, scores: pd.DataFrame, args: argparse.Namespace) -> list[tuple[str, ...]]:
    score_by_node, _ = score_maps(scores)
    ranking = rank_nodes_by_score(scores)
    seeds = [node for node in ranking["gene_symbol"].astype(str).head(args.n_seeds) if graph.has_node(node)]
    modules = []
    for seed in seeds:
        nodes = greedy_dense_module(
            graph,
            seed,
            score_by_node,
            max_size=args.max_size,
            density_weight=args.density_weight,
        )
        if len(nodes) >= args.min_size:
            modules.append(nodes)
    return deduplicate_modules(
        modules,
        graph,
        score_by_node,
        density_weight=args.density_weight,
        jaccard_cutoff=args.jaccard_dedup,
        max_modules=args.max_modules,
    )


def prepare_degree_bins(scores: pd.DataFrame, n_bins: int) -> tuple[dict[str, int], dict[int, np.ndarray]]:
    work = scores.loc[:, ["gene_symbol", "graph_degree"]].copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str)
    bins = assign_degree_bins(work["graph_degree"], n_bins=n_bins).astype(int)
    node_to_bin = dict(zip(work["gene_symbol"], bins, strict=True))
    bin_to_nodes = {
        int(bin_id): work.loc[bins == bin_id, "gene_symbol"].to_numpy(dtype=object)
        for bin_id in sorted(bins.unique())
    }
    return node_to_bin, bin_to_nodes


def sample_degree_matched(
    nodes: tuple[str, ...],
    node_to_bin: dict[str, int],
    bin_to_nodes: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    counts: dict[int, int] = {}
    for node in nodes:
        bin_id = int(node_to_bin[node])
        counts[bin_id] = counts.get(bin_id, 0) + 1
    sampled = []
    for bin_id, count in sorted(counts.items()):
        candidates = np.asarray(bin_to_nodes[bin_id], dtype=object)
        sampled.extend(str(node) for node in rng.choice(candidates, size=count, replace=count > len(candidates)))
    rng.shuffle(sampled)
    return tuple(sampled)


def empirical_upper(null_values: np.ndarray, observed: float) -> float:
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float((1 + np.count_nonzero(finite >= observed)) / (1 + finite.size))


def z_score(null_values: np.ndarray, observed: float) -> float:
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    sd = float(np.std(finite, ddof=1))
    return float((observed - float(np.mean(finite))) / sd) if sd > 0 else float("nan")


def calibrate_module(
    graph: nx.Graph,
    scores: pd.DataFrame,
    nodes: tuple[str, ...],
    score_by_node: dict[str, float],
    node_to_bin: dict[str, int],
    bin_to_nodes: dict[int, np.ndarray],
    *,
    args: argparse.Namespace,
    rng: np.random.Generator,
    graph_nulls: list[nx.Graph],
) -> dict[str, Any]:
    observed_stat = module_statistic(graph, nodes, score_by_node, density_weight=args.density_weight)
    observed_density = edge_density(graph, nodes)
    all_nodes = scores["gene_symbol"].astype(str).to_numpy(dtype=object)
    random_stats = np.empty(args.n_node_null, dtype=float)
    degree_stats = np.empty(args.n_node_null, dtype=float)
    for idx in range(args.n_node_null):
        random_nodes = tuple(str(node) for node in rng.choice(all_nodes, size=len(nodes), replace=False))
        degree_nodes = sample_degree_matched(nodes, node_to_bin, bin_to_nodes, rng)
        random_stats[idx] = module_statistic(graph, random_nodes, score_by_node, density_weight=args.density_weight)
        degree_stats[idx] = module_statistic(graph, degree_nodes, score_by_node, density_weight=args.density_weight)
    graph_density = np.asarray([edge_density(null_graph, nodes) for null_graph in graph_nulls], dtype=float)
    return {
        "module_statistic": observed_stat,
        "edge_density": observed_density,
        "random_node_p": empirical_upper(random_stats, observed_stat),
        "random_node_z": z_score(random_stats, observed_stat),
        "degree_matched_p": empirical_upper(degree_stats, observed_stat),
        "degree_matched_z": z_score(degree_stats, observed_stat),
        "fixed_module_edge_density_graph_p": empirical_upper(graph_density, observed_density),
        "fixed_module_edge_density_graph_z": z_score(graph_density, observed_density),
        "random_null_mean": float(np.mean(random_stats)),
        "random_null_sd": float(np.std(random_stats, ddof=1)),
        "degree_null_mean": float(np.mean(degree_stats)),
        "degree_null_sd": float(np.std(degree_stats, ddof=1)),
        "graph_density_null_mean": float(np.mean(graph_density)) if graph_density.size else float("nan"),
        "graph_density_null_sd": float(np.std(graph_density, ddof=1)) if graph_density.size > 1 else float("nan"),
    }


def run_trait(spec: TraitSpec, args: argparse.Namespace) -> dict[str, Any]:
    scores = read_tsv(spec.scores_path)
    graph = load_graph(spec, scores)
    scores = scores.loc[scores["gene_symbol"].astype(str).isin(set(map(str, graph.nodes())))].copy()
    score_by_node, degree_by_node = score_maps(scores)
    modules = discover_modules(graph, scores, args)
    node_to_bin, bin_to_nodes = prepare_degree_bins(scores, args.degree_bins)
    graph_null_cache = args.out_dir / "graph_null_cache" / f"{spec.analysis_id}.degree_preserving_graph_n{args.n_degree_graph_null}.npz"
    graph_nulls = (
        list(
            degree_preserving_graph_replicates(
                graph,
                n_replicates=args.n_degree_graph_null,
                seed=args.seed + 101,
                nswap_per_edge=args.degree_graph_nswap_per_edge,
                max_tries_per_swap=args.degree_graph_max_tries_per_swap,
                cache_path=graph_null_cache,
            )
        )
        if args.n_degree_graph_null > 0
        else []
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    for idx, nodes in enumerate(modules, start=1):
        calibration = calibrate_module(
            graph,
            scores,
            nodes,
            score_by_node,
            node_to_bin,
            bin_to_nodes,
            args=args,
            rng=rng,
            graph_nulls=graph_nulls,
        )
        degrees = np.array([degree_by_node[node] for node in nodes], dtype=float)
        row = {
            "trait": spec.trait,
            "analysis_id": spec.analysis_id,
            "competitor": "simplified_dense_module_search",
            "module_id": f"{spec.trait}_DMS_{idx:03d}",
            "n_genes": len(nodes),
            "n_edges": graph.subgraph(nodes).number_of_edges(),
            "mean_score": float(np.mean([score_by_node[node] for node in nodes])),
            "max_score": float(np.max([score_by_node[node] for node in nodes])),
            "mean_graph_degree": float(np.mean(degrees)),
            "max_graph_degree": int(np.max(degrees)),
            "module_genes": ",".join(nodes),
            **calibration,
        }
        row["naive_positive"] = bool(row["random_node_p"] <= 0.05)
        row["degree_robust_positive"] = bool(row["degree_matched_p"] <= 0.05)
        row["fixed_edge_density_positive"] = bool(row["fixed_module_edge_density_graph_p"] <= 0.05)
        if row["naive_positive"] and not row["degree_robust_positive"]:
            row["interpretation_label"] = "naive_positive_not_degree_robust"
        elif row["fixed_edge_density_positive"]:
            row["interpretation_label"] = "fixed_module_edge_density_positive_selection_biased"
        elif row["degree_robust_positive"]:
            row["interpretation_label"] = "degree_robust_no_topology_claim"
        else:
            row["interpretation_label"] = "not_positive"
        rows.append(row)
    table = pd.DataFrame(rows)
    out_dir = args.out_dir / spec.analysis_id / "tables"
    write_table(out_dir / f"{spec.trait}.dense_module_competitor_modules.tsv", table)
    return {
        "trait": spec.trait,
        "analysis_id": spec.analysis_id,
        "n_candidate_modules": int(len(table)),
        "n_naive_positive": int(table["naive_positive"].sum()) if not table.empty else 0,
        "n_degree_robust_positive": int(table["degree_robust_positive"].sum()) if not table.empty else 0,
        "n_fixed_edge_density_positive": int(table["fixed_edge_density_positive"].sum()) if not table.empty else 0,
        "module_table": str(out_dir / f"{spec.trait}.dense_module_competitor_modules.tsv"),
    }


def write_report(out_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Simplified Dense Module Search Competitor",
        "",
        f"Generated UTC: {datetime.now(UTC).isoformat()}",
        "",
        "This is a deliberately simple graph/module competitor baseline. It is not reported as dmGWAS/EW-dmGWAS, but it captures the common greedy dense-module pattern for review-facing ablation.",
        "",
        "| Trait | candidate modules | naive positive | degree robust | fixed-module edge-density |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.trait} | {int(row.n_candidate_modules)} | {int(row.n_naive_positive)} | "
            f"{int(row.n_degree_robust_positive)} | {int(row.n_fixed_edge_density_positive)} |"
        )
    lines.extend(
        [
            "",
            "Interpretation guardrail: modules that are positive only against random node sets are not treated as calibrated RIPPLE discoveries.",
            "Fixed-module edge-density checks do not rerun module selection and therefore cannot establish topology-specific discovery.",
        ]
    )
    (out_dir / "dense_module_competitor_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    _ = load_claim_policy(args.policy)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([run_trait(spec, args) for spec in select_specs(args)])
    write_table(args.out_dir / "dense_module_competitor_summary.all_traits.tsv", summary)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "baseline": "simplified_dense_module_search",
        "n_node_null": args.n_node_null,
        "n_degree_graph_null": args.n_degree_graph_null,
        "n_seeds": args.n_seeds,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "density_weight": args.density_weight,
        "manifests": summary.to_dict(orient="records"),
    }
    (args.out_dir / "dense_module_competitor_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(args.out_dir, summary)
    print(f"Wrote dense module competitor outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
