#!/usr/bin/env python
"""Smoke-test RIPPLE V1.2 direction 2/3 candidate module definitions.

Direction 2:
    diffusion-localized neighborhoods selected from the graph around high-score
    seeds. The null repeats seed selection, tau selection and neighborhood
    selection.

Direction 3:
    fixed graph-community and pathway/library anchors. The null repeats the
    max-over-library operation for Louvain communities, GO sets and Reactome
    sets.

This script is diagnostic only. It does not modify the frozen RIPPLE V1 claim
policy and does not upgrade modules to topology-specific discoveries.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.modules.anchored import empirical_upper, z_score  # noqa: E402
from ripple.nulls.score_permutation import degree_stratified_permuted_scores  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_ANCHORED_ROOT = ANALYSIS_ROOT / "tier4_v12_anchored_broad_reactome_go_cross_trait_n500"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "tier4_v12_direction2_3_smoke_v1"
THIS_SCRIPT = Path(__file__).resolve()


@dataclass(frozen=True)
class AnchoredRun:
    analysis_id: str
    trait: str
    result_dir: Path
    summary_path: Path
    module_table_path: Path
    graph_edge_path: Path
    score_path: Path


@dataclass(frozen=True)
class Candidate:
    method: str
    candidate_id: str
    statistic: float
    indices: tuple[int, ...]
    seed_gene: str = ""
    tau: float | None = None
    score_basis: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchored-root", type=Path, default=DEFAULT_ANCHORED_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--analysis-id", action="append", default=[])
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--diffusion-n-seeds", type=int, default=50)
    parser.add_argument("--diffusion-radius", type=int, default=2)
    parser.add_argument("--diffusion-top-size", type=int, default=80)
    parser.add_argument("--tau-grid", default="0.5,1.0,2.0")
    parser.add_argument("--community-min-size", type=int, default=10)
    parser.add_argument("--community-max-size", type=int, default=300)
    parser.add_argument("--community-resolution", type=float, default=1.0)
    parser.add_argument("--min-pathway-present", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "reports").mkdir(parents=True, exist_ok=True)


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, compression="infer")


def parse_tau_grid(value: str) -> tuple[float, ...]:
    taus = tuple(float(part.strip()) for part in str(value).split(",") if part.strip())
    if not taus or any((not np.isfinite(tau)) or tau <= 0.0 for tau in taus):
        raise ValueError("tau-grid must contain positive finite values.")
    return taus


def discover_runs(root: Path, requested: Sequence[str]) -> list[AnchoredRun]:
    requested_set = set(requested)
    runs: list[AnchoredRun] = []
    for result_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if requested_set and result_dir.name not in requested_set:
            continue
        reports = result_dir / "reports"
        tables = result_dir / "tables"
        summary_paths = sorted(reports.glob("*.v12_anchored_module_summary.json"))
        module_paths = sorted(tables.glob("*.v12_anchored_module_tests.tsv"))
        graph_paths = sorted(tables.glob("*.v12_anchored_graph_edges.tsv.gz"))
        if len(summary_paths) != 1 or len(module_paths) != 1 or len(graph_paths) != 1:
            continue
        summary = json.loads(summary_paths[0].read_text(encoding="utf-8"))
        score_path = Path(str(summary.get("score_path", "")))
        if not score_path.exists():
            raise FileNotFoundError(f"Missing score_path for {result_dir}: {score_path}")
        runs.append(
            AnchoredRun(
                analysis_id=result_dir.name,
                trait=str(summary.get("trait", result_dir.name)),
                result_dir=result_dir,
                summary_path=summary_paths[0],
                module_table_path=module_paths[0],
                graph_edge_path=graph_paths[0],
                score_path=score_path,
            )
        )
    if not runs:
        raise FileNotFoundError(f"No requested anchored runs found under {root}")
    return runs


def load_graph(edge_path: Path) -> nx.Graph:
    edges = pd.read_csv(edge_path, sep="\t", compression="infer")
    required = {"node1", "node2"}
    missing = sorted(required - set(edges.columns))
    if missing:
        raise ValueError(f"{edge_path} missing columns: {missing}")
    graph = nx.Graph()
    weight_col = "weight" if "weight" in edges.columns else None
    for row in edges.itertuples(index=False):
        node1 = str(getattr(row, "node1")).upper()
        node2 = str(getattr(row, "node2")).upper()
        if weight_col:
            graph.add_edge(node1, node2, weight=float(getattr(row, weight_col)))
        else:
            graph.add_edge(node1, node2)
    return graph


def load_scores(score_path: Path, graph: nx.Graph) -> pd.DataFrame:
    scores = pd.read_csv(score_path, sep="\t", compression="infer")
    required = {"gene_symbol", "assoc_resid_score"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"{score_path} missing columns: {missing}")
    work = scores.copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str).str.upper()
    work["assoc_resid_score"] = pd.to_numeric(work["assoc_resid_score"], errors="raise").astype(float)
    work = work.drop_duplicates("gene_symbol", keep="first")
    work = work[work["gene_symbol"].isin(set(graph.nodes()))].reset_index(drop=True)
    if work.empty:
        raise ValueError(f"No score genes overlap graph for {score_path}")
    degree = dict(graph.degree())
    work["graph_degree"] = work["gene_symbol"].map(degree).astype(int)
    return work


def split_genes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def pathway_sets(module_table_path: Path, source: str, gene_to_idx: dict[str, int], min_present: int) -> tuple[list[str], list[tuple[int, ...]]]:
    modules = pd.read_csv(module_table_path, sep="\t")
    subset = modules.loc[
        modules["module_status"].ne("not_tested_low_overlap")
        & modules["module_source"].astype(str).eq(source)
    ].copy()
    ids: list[str] = []
    sets: list[tuple[int, ...]] = []
    for row in subset.to_dict(orient="records"):
        indices = tuple(sorted({gene_to_idx[gene] for gene in split_genes(row.get("present_genes", "")) if gene in gene_to_idx}))
        if len(indices) >= min_present:
            ids.append(str(row["module_name"]))
            sets.append(indices)
    return ids, sets


def louvain_sets(
    graph: nx.Graph,
    gene_to_idx: dict[str, int],
    *,
    min_size: int,
    max_size: int,
    resolution: float,
    seed: int,
) -> tuple[list[str], list[tuple[int, ...]]]:
    communities = nx.community.louvain_communities(graph, resolution=float(resolution), seed=int(seed))
    ids: list[str] = []
    sets: list[tuple[int, ...]] = []
    for idx, community in enumerate(communities, start=1):
        indices = tuple(sorted(gene_to_idx[str(node).upper()] for node in community if str(node).upper() in gene_to_idx))
        if min_size <= len(indices) <= max_size:
            ids.append(f"STRING_Louvain_{idx:04d}")
            sets.append(indices)
    return ids, sets


def membership_matrix(n_genes: int, sets: Sequence[Sequence[int]]) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for set_idx, indices in enumerate(sets):
        idx = np.asarray(indices, dtype=int)
        if idx.size == 0:
            continue
        rows.extend(int(item) for item in idx)
        cols.extend([set_idx] * int(idx.size))
        data.extend([1.0 / np.sqrt(float(idx.size))] * int(idx.size))
    return sparse.csr_matrix(
        (np.asarray(data, dtype=float), (np.asarray(rows), np.asarray(cols))),
        shape=(n_genes, len(sets)),
    )


def fixed_set_best(values: np.ndarray, ids: Sequence[str], sets: Sequence[Sequence[int]], method: str) -> Candidate:
    if not sets:
        return Candidate(method, "", float("nan"), ())
    matrix = membership_matrix(len(values), sets)
    stats = np.asarray(values @ matrix, dtype=float)
    best_idx = int(np.argmax(stats))
    return Candidate(
        method=method,
        candidate_id=str(ids[best_idx]),
        statistic=float(stats[best_idx]),
        indices=tuple(int(idx) for idx in sets[best_idx]),
        score_basis="raw_sum_over_sqrt_n",
    )


def fixed_set_nulls(null_scores: np.ndarray, sets: Sequence[Sequence[int]]) -> np.ndarray:
    if not sets:
        return np.array([], dtype=float)
    matrix = membership_matrix(null_scores.shape[1], sets)
    stats = np.asarray(null_scores @ matrix, dtype=float)
    return np.max(stats, axis=1)


def adjacency_index(graph: nx.Graph, nodes: Sequence[str], gene_to_idx: dict[str, int]) -> tuple[np.ndarray, ...]:
    out: list[np.ndarray] = []
    for node in nodes:
        neighbors = sorted(gene_to_idx[str(neighbor).upper()] for neighbor in graph.neighbors(node) if str(neighbor).upper() in gene_to_idx)
        out.append(np.asarray(neighbors, dtype=int))
    return tuple(out)


def neighborhood(
    seed_idx: int,
    adjacency: Sequence[np.ndarray],
    *,
    radius: int,
    cache: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if seed_idx in cache:
        return cache[seed_idx]
    seen = {int(seed_idx)}
    frontier = {int(seed_idx)}
    distances = {int(seed_idx): 0}
    for distance in range(1, radius + 1):
        next_frontier: set[int] = set()
        for idx in frontier:
            for neighbor in adjacency[idx]:
                n_idx = int(neighbor)
                if n_idx not in seen:
                    seen.add(n_idx)
                    distances[n_idx] = distance
                    next_frontier.add(n_idx)
        frontier = next_frontier
        if not frontier:
            break
    ordered = np.asarray(sorted(seen), dtype=int)
    dist = np.asarray([distances[int(idx)] for idx in ordered], dtype=float)
    cache[seed_idx] = (ordered, dist)
    return ordered, dist


def diffusion_best(
    values: np.ndarray,
    nodes: Sequence[str],
    adjacency: Sequence[np.ndarray],
    *,
    n_seeds: int,
    radius: int,
    tau_grid: Sequence[float],
    top_size: int,
    cache: dict[int, tuple[np.ndarray, np.ndarray]],
) -> Candidate:
    order = np.argsort(values, kind="mergesort")[::-1]
    best = Candidate("direction2_diffusion_localized_neighborhood", "", float("-inf"), ())
    for seed_idx in order[: min(n_seeds, len(order))]:
        seed_idx = int(seed_idx)
        neigh, dist = neighborhood(seed_idx, adjacency, radius=radius, cache=cache)
        positive = np.maximum(0.0, values[neigh])
        if positive.size == 0 or np.max(positive) <= 0:
            continue
        for tau in tau_grid:
            weights = np.exp(-dist / float(tau))
            contribution = weights * positive
            n_keep = min(int(top_size), contribution.size)
            if n_keep < contribution.size:
                keep_local = np.argpartition(-contribution, n_keep - 1)[:n_keep]
            else:
                keep_local = np.arange(contribution.size)
            keep_local = keep_local[np.argsort(-contribution[keep_local], kind="mergesort")]
            selected = neigh[keep_local]
            selected_weights = weights[keep_local]
            selected_positive = positive[keep_local]
            denom = float(np.sqrt(np.sum(selected_weights * selected_weights)))
            if denom <= 0:
                continue
            stat = float(np.dot(selected_weights, selected_positive) / denom)
            if stat > best.statistic:
                best = Candidate(
                    method="direction2_diffusion_localized_neighborhood",
                    candidate_id=f"seed={nodes[seed_idx]};radius={radius};tau={float(tau):g};top_size={n_keep}",
                    statistic=stat,
                    indices=tuple(int(idx) for idx in selected),
                    seed_gene=str(nodes[seed_idx]),
                    tau=float(tau),
                    score_basis="positive_part_distance_weighted_top_neighborhood",
                )
    return best


def diffusion_nulls(
    null_scores: np.ndarray,
    nodes: Sequence[str],
    adjacency: Sequence[np.ndarray],
    *,
    n_seeds: int,
    radius: int,
    tau_grid: Sequence[float],
    top_size: int,
) -> np.ndarray:
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    out = np.empty(null_scores.shape[0], dtype=float)
    for idx, row in enumerate(null_scores):
        out[idx] = diffusion_best(
            row,
            nodes,
            adjacency,
            n_seeds=n_seeds,
            radius=radius,
            tau_grid=tau_grid,
            top_size=top_size,
            cache=cache,
        ).statistic
    return out


def finite_summary(null_values: np.ndarray, observed: float) -> dict[str, float | int | str]:
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or not np.isfinite(observed):
        return {
            "n_null": int(finite.size),
            "null_mean": float("nan"),
            "null_sd": float("nan"),
            "z": float("nan"),
            "empirical_p": float("nan"),
            "smoke_status": "not_tested",
        }
    p_value = empirical_upper(finite, observed)
    z = z_score(observed, finite)
    if np.isfinite(p_value) and p_value <= 0.05:
        status = "smoke_positive"
    elif np.isfinite(p_value) and p_value <= 0.10:
        status = "borderline"
    else:
        status = "negative"
    return {
        "n_null": int(finite.size),
        "null_mean": float(np.mean(finite)),
        "null_sd": float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan"),
        "z": z,
        "empirical_p": p_value,
        "smoke_status": status,
    }


def candidate_row(
    *,
    run: AnchoredRun,
    method: str,
    candidate: Candidate,
    null_values: np.ndarray,
    nodes: Sequence[str],
    n_candidates: int,
) -> dict[str, object]:
    summary = finite_summary(null_values, candidate.statistic)
    genes = [str(nodes[idx]) for idx in candidate.indices]
    return {
        "trait": run.trait,
        "analysis_id": run.analysis_id,
        "direction": "direction2" if method.startswith("direction2") else "direction3",
        "method": method,
        "n_candidates": int(n_candidates),
        "candidate_id": candidate.candidate_id,
        "candidate_seed": candidate.seed_gene,
        "tau": float(candidate.tau) if candidate.tau is not None else np.nan,
        "candidate_size": int(len(candidate.indices)),
        "observed_stat": float(candidate.statistic),
        **summary,
        "candidate_score_basis": candidate.score_basis,
        "candidate_genes": ",".join(genes[:300]) + (",..." if len(genes) > 300 else ""),
        "source_result_path": str(run.result_dir),
        "script_path": str(THIS_SCRIPT),
        "timestamp": datetime.now(UTC).isoformat(),
    }


def run_one(run: AnchoredRun, args: argparse.Namespace, *, run_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    graph = load_graph(run.graph_edge_path)
    scores = load_scores(run.score_path, graph)
    nodes = tuple(scores["gene_symbol"].astype(str))
    gene_to_idx = {gene: idx for idx, gene in enumerate(nodes)}
    values = scores["assoc_resid_score"].to_numpy(dtype=float)
    tau_grid = parse_tau_grid(args.tau_grid)
    null_scores = degree_stratified_permuted_scores(
        scores,
        score_col="assoc_resid_score",
        degree_col="graph_degree",
        n_replicates=args.n_null,
        seed=int(args.seed + run_index * 100_003),
        n_bins=args.degree_bins,
    )

    adjacency = adjacency_index(graph, nodes, gene_to_idx)
    diffusion_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    diffusion_candidate = diffusion_best(
        values,
        nodes,
        adjacency,
        n_seeds=args.diffusion_n_seeds,
        radius=args.diffusion_radius,
        tau_grid=tau_grid,
        top_size=args.diffusion_top_size,
        cache=diffusion_cache,
    )
    diffusion_null = diffusion_nulls(
        null_scores,
        nodes,
        adjacency,
        n_seeds=args.diffusion_n_seeds,
        radius=args.diffusion_radius,
        tau_grid=tau_grid,
        top_size=args.diffusion_top_size,
    )

    community_ids, community_sets = louvain_sets(
        graph,
        gene_to_idx,
        min_size=args.community_min_size,
        max_size=args.community_max_size,
        resolution=args.community_resolution,
        seed=int(args.seed),
    )
    go_ids, go_sets = pathway_sets(run.module_table_path, "Gene Ontology", gene_to_idx, args.min_pathway_present)
    reactome_ids, reactome_sets = pathway_sets(run.module_table_path, "Reactome", gene_to_idx, args.min_pathway_present)
    method_specs = [
        (
            "direction2_diffusion_localized_neighborhood",
            diffusion_candidate,
            diffusion_null,
            args.diffusion_n_seeds * len(tau_grid),
        ),
        (
            "direction3_louvain_community_anchor",
            fixed_set_best(values, community_ids, community_sets, "direction3_louvain_community_anchor"),
            fixed_set_nulls(null_scores, community_sets),
            len(community_sets),
        ),
        (
            "direction3_go_pathway_anchor",
            fixed_set_best(values, go_ids, go_sets, "direction3_go_pathway_anchor"),
            fixed_set_nulls(null_scores, go_sets),
            len(go_sets),
        ),
        (
            "direction3_reactome_pathway_anchor",
            fixed_set_best(values, reactome_ids, reactome_sets, "direction3_reactome_pathway_anchor"),
            fixed_set_nulls(null_scores, reactome_sets),
            len(reactome_sets),
        ),
    ]

    rows = [
        candidate_row(
            run=run,
            method=method,
            candidate=candidate,
            null_values=null_values,
            nodes=nodes,
            n_candidates=n_candidates,
        )
        for method, candidate, null_values, n_candidates in method_specs
    ]
    null_rows: list[dict[str, object]] = []
    for method, _, null_values, _ in method_specs:
        for replicate, value in enumerate(null_values):
            null_rows.append(
                {
                    "trait": run.trait,
                    "analysis_id": run.analysis_id,
                    "method": method,
                    "replicate": int(replicate),
                    "null_type": "degree_stratified_score_permutation_reselect_or_max",
                    "statistic_name": "method_max_statistic",
                    "statistic_value": float(value),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(null_rows)


def render_report(summary: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        "# RIPPLE V1.2 Direction 2/3 Smoke Diagnostic",
        "",
        f"Created: {datetime.now(UTC).isoformat()}",
        "",
        f"Null replicates per trait: {int(args.n_null)}",
        "",
        "This is a small-scale diagnostic. Positive rows indicate candidates worth further validation, "
        "not manuscript-ready claims.",
        "",
        "| Trait | Method | Status | Z | empirical P | Best candidate | Size |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['trait']} | {row['method']} | {row['smoke_status']} | "
            f"{float(row['z']):.3f} | {float(row['empirical_p']):.4g} | "
            f"{row['candidate_id']} | {int(row['candidate_size'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- Direction 2 is worth pursuing only if diffusion-localized neighborhoods are not broadly positive across unrelated traits and produce interpretable candidates.",
            "- Direction 3 is worth pursuing if fixed community/pathway anchors add information beyond the already validated broad anchored layer.",
            "- These smoke outputs do not alter V1/V1.2 claim tiers.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    runs = discover_runs(args.anchored_root, args.analysis_id)
    summary_tables: list[pd.DataFrame] = []
    null_tables: list[pd.DataFrame] = []
    for run_index, run in enumerate(runs):
        print(f"Running direction 2/3 smoke for {run.analysis_id}", flush=True)
        summary, nulls = run_one(run, args, run_index=run_index)
        write_table(args.out_dir / "tables" / f"{run.analysis_id}.direction23_smoke_summary.tsv", summary)
        write_table(args.out_dir / "tables" / f"{run.analysis_id}.direction23_smoke_nulls.tsv.gz", nulls)
        summary_tables.append(summary)
        null_tables.append(nulls)
    combined_summary = pd.concat(summary_tables, ignore_index=True)
    combined_nulls = pd.concat(null_tables, ignore_index=True)
    write_table(args.out_dir / "tables" / "direction23_smoke_summary.all_traits.tsv", combined_summary)
    write_table(args.out_dir / "tables" / "direction23_smoke_nulls.all_traits.tsv.gz", combined_nulls)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "analysis_id": "tier4_v12_direction2_3_smoke_v1",
        "anchored_root": str(args.anchored_root),
        "n_null": int(args.n_null),
        "degree_bins": int(args.degree_bins),
        "diffusion_n_seeds": int(args.diffusion_n_seeds),
        "diffusion_radius": int(args.diffusion_radius),
        "diffusion_top_size": int(args.diffusion_top_size),
        "tau_grid": args.tau_grid,
        "community_min_size": int(args.community_min_size),
        "community_max_size": int(args.community_max_size),
        "community_resolution": float(args.community_resolution),
        "min_pathway_present": int(args.min_pathway_present),
        "script_path": str(THIS_SCRIPT),
    }
    (args.out_dir / "reports" / "direction23_smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "reports" / "direction23_smoke_report.md").write_text(
        render_report(combined_summary, args) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote direction 2/3 smoke diagnostics to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
