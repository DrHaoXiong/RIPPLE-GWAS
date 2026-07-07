#!/usr/bin/env python
"""Diagnose RIPPLE V1 Tier 4 module-layer design failures.

This is a diagnostic audit, not a claim-upgrade script. It asks whether the
frozen Tier 4 full-reselection gate is low-power or whether extracted modules
also fail fixed-module evidence.

Phase A2:
    fixed-module size-matched and degree-matched nulls for existing local
    modules, joined to full-reselection and graph-null evidence.

Phase A1:
    score-level oracle compact module spike-ins on the default STRING LCC.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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

from ripple.modules import discover_local_modules  # noqa: E402
from ripple.nulls.score_permutation import assign_degree_bins  # noqa: E402
from run_cross_trait_module_reselection_null import CROSS_TRAIT_SPECS  # noqa: E402
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


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "tier4_design_defect_audit_v1"
DR_MODULE_RESELECTION_DIR = ANALYSIS_ROOT / "module_reselection_null_v1"
CROSS_TRAIT_RESELECTION_DIR = ANALYSIS_ROOT / "module_reselection_null_cross_trait_v1"
THIS_SCRIPT = Path(__file__).resolve()


@dataclass(frozen=True)
class AuditSpec:
    trait: str
    analysis_id: str
    analysis_dir: Path
    graph_edges_path: Path
    full_reselection_summary: Path

    @property
    def module_spec(self) -> AnalysisSpec:
        return AnalysisSpec(
            trait=self.trait,
            analysis_id=self.analysis_id,
            analysis_dir=self.analysis_dir,
            graph_edges_path=self.graph_edges_path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--phase", choices=["all", "a2", "a1"], default="all")
    parser.add_argument("--n-fixed-null", type=int, default=5000)
    parser.add_argument("--n-oracle-null", type=int, default=500)
    parser.add_argument("--oracle-replicates", type=int, default=1)
    parser.add_argument(
        "--effect-grid",
        default="weak:1.0,moderate:2.5,strong:5.0",
        help="Comma-separated label:value additive residual-score effects for oracle spike-ins.",
    )
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def audit_specs() -> list[AuditSpec]:
    dr_graph = ANALYSIS_ROOT / "dr_mvp_string_final5000" / "tables" / "DR_MVP.analysis_graph_edges.tsv.gz"
    dr_full = DR_MODULE_RESELECTION_DIR / "tables" / "module_full_reselection_summary.all_traits.tsv"
    cross_full = CROSS_TRAIT_RESELECTION_DIR / "tables" / "module_full_reselection_summary.all_traits.tsv"
    specs = [
        AuditSpec(
            trait="DR_MVP",
            analysis_id="DR_MVP_default_final5000",
            analysis_dir=ANALYSIS_ROOT / "dr_mvp_string_final5000",
            graph_edges_path=dr_graph,
            full_reselection_summary=dr_full,
        ),
        AuditSpec(
            trait="DR_MVP_NO_MHC_NO_APOE",
            analysis_id="DR_MVP_no_MHC_no_APOE_final5000",
            analysis_dir=ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000",
            graph_edges_path=dr_graph,
            full_reselection_summary=dr_full,
        ),
    ]
    for spec in CROSS_TRAIT_SPECS.values():
        specs.append(
            AuditSpec(
                trait=spec.trait,
                analysis_id=spec.analysis_id,
                analysis_dir=spec.analysis_dir,
                graph_edges_path=spec.graph_edges_path,
                full_reselection_summary=cross_full,
            )
        )
    return specs


def prepare_scores_and_graph(spec: AuditSpec) -> tuple[pd.DataFrame, nx.Graph]:
    module_spec = spec.module_spec
    scores = read_tsv(module_spec.scores_path)
    graph = load_analysis_graph(module_spec, scores)
    graph_degree = dict(graph.degree())
    scores = scores.copy()
    scores["gene_symbol"] = scores["gene_symbol"].astype(str)
    scores = scores[scores["gene_symbol"].isin(graph.nodes())].reset_index(drop=True)
    scores["assoc_resid_score"] = pd.to_numeric(scores["assoc_resid_score"], errors="raise").astype(float)
    scores["graph_degree"] = scores["gene_symbol"].map(graph_degree).astype(int)
    return scores, graph


def split_genes(value: object) -> list[str]:
    return [gene.strip() for gene in str(value).split(",") if gene.strip()]


def fixed_null_for_nodes(
    nodes: list[str],
    scores: pd.DataFrame,
    *,
    n_null: int,
    degree_bins: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    work = scores.copy()
    work["degree_bin"] = assign_degree_bins(work["graph_degree"], n_bins=degree_bins).astype(int)
    score_by_gene = dict(zip(work["gene_symbol"], work["assoc_resid_score"], strict=True))
    bin_by_gene = dict(zip(work["gene_symbol"], work["degree_bin"], strict=True))
    present = [node for node in nodes if node in score_by_gene]
    if not present:
        raise ValueError("No module nodes are present in score universe.")
    observed = float(np.mean([score_by_gene[node] for node in present]))
    universe = work["gene_symbol"].to_numpy(dtype=object)
    bins = sorted(work["degree_bin"].unique())
    bin_to_genes = {
        int(bin_id): work.loc[work["degree_bin"].eq(bin_id), "gene_symbol"].to_numpy(dtype=object)
        for bin_id in bins
    }
    profile: dict[int, int] = {}
    for node in present:
        bin_id = int(bin_by_gene[node])
        profile[bin_id] = profile.get(bin_id, 0) + 1

    size_null = np.empty(n_null, dtype=float)
    degree_null = np.empty(n_null, dtype=float)
    n_nodes = len(present)
    for replicate in range(n_null):
        size_sample = rng.choice(universe, size=n_nodes, replace=False)
        size_null[replicate] = float(np.mean([score_by_gene[str(node)] for node in size_sample]))
        degree_sample: list[str] = []
        for bin_id, count in sorted(profile.items()):
            candidates = bin_to_genes[int(bin_id)]
            degree_sample.extend(str(node) for node in rng.choice(candidates, size=int(count), replace=False))
        degree_null[replicate] = float(np.mean([score_by_gene[node] for node in degree_sample]))

    summary = {
        "observed_mean_score": observed,
        "fixed_module_size_matched_null_mean": float(size_null.mean()),
        "fixed_module_size_matched_null_sd": float(size_null.std(ddof=1)),
        "fixed_module_p": empirical_upper(size_null, observed),
        "fixed_module_z": z_score(observed, size_null),
        "degree_matched_fixed_null_mean": float(degree_null.mean()),
        "degree_matched_fixed_null_sd": float(degree_null.std(ddof=1)),
        "degree_matched_fixed_p": empirical_upper(degree_null, observed),
        "degree_matched_fixed_z": z_score(observed, degree_null),
        "n_fixed_null": int(n_null),
    }
    null_table = pd.DataFrame(
        {
            "replicate": np.arange(n_null, dtype=int),
            "size_matched_mean_score": size_null,
            "degree_matched_mean_score": degree_null,
        }
    )
    return summary, null_table


def load_full_reselection_summaries() -> pd.DataFrame:
    tables = []
    for path in [
        DR_MODULE_RESELECTION_DIR / "tables" / "module_full_reselection_summary.all_traits.tsv",
        CROSS_TRAIT_RESELECTION_DIR / "tables" / "module_full_reselection_summary.all_traits.tsv",
    ]:
        if path.exists():
            tables.append(read_tsv(path))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def run_a2_fixed_vs_full(args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = load_full_reselection_summaries()
    full_keyed = (
        full.set_index(["analysis_id", "module_id"], drop=False)
        if not full.empty and {"analysis_id", "module_id"}.issubset(full.columns)
        else pd.DataFrame()
    )
    rows: list[dict[str, object]] = []
    null_rows: list[pd.DataFrame] = []
    for spec_idx, spec in enumerate(audit_specs()):
        module_spec = spec.module_spec
        if not module_spec.modules_path.exists():
            continue
        print(f"A2 fixed-vs-full audit: {spec.analysis_id}", flush=True)
        scores, graph = prepare_scores_and_graph(spec)
        modules = read_tsv(module_spec.modules_path)
        for module_idx, module in enumerate(modules.to_dict(orient="records")):
            module_id = str(module.get("module_id", ""))
            nodes = split_genes(module.get("module_genes", ""))
            fixed, null_table = fixed_null_for_nodes(
                nodes,
                scores,
                n_null=args.n_fixed_null,
                degree_bins=args.degree_bins,
                seed=args.seed + 1000 * spec_idx + module_idx,
            )
            full_row: dict[str, object] = {}
            if not full_keyed.empty and (spec.analysis_id, module_id) in full_keyed.index:
                selected = full_keyed.loc[(spec.analysis_id, module_id)]
                if isinstance(selected, pd.DataFrame):
                    selected = selected.iloc[0]
                full_row = selected.to_dict()
            full_p = as_float(full_row.get("full_reselection_score_p", np.nan))
            degree_p = as_float(fixed["degree_matched_fixed_p"])
            selection_penalty_ratio = full_p / degree_p if np.isfinite(full_p) and np.isfinite(degree_p) and degree_p > 0 else np.nan
            delta_log10p = -np.log10(degree_p) + np.log10(full_p) if np.isfinite(selection_penalty_ratio) else np.nan
            graph_p = as_float(module.get("degree_preserving_graph_p", np.nan))
            fixed_support = bool(np.isfinite(degree_p) and degree_p <= 0.05)
            full_support = bool(np.isfinite(full_p) and full_p <= 0.05)
            if fixed_support and not full_support:
                pattern = "fixed_support_lost_after_selection_fwer"
            elif fixed_support and full_support:
                pattern = "selection_calibrated_candidate"
            elif not fixed_support and full_support:
                pattern = "selection_support_without_fixed_support_check"
            else:
                pattern = "no_fixed_or_selection_support"
            row = {
                "trait": spec.trait,
                "analysis_id": spec.analysis_id,
                "module_id": module_id,
                "module_rank": module.get("module_rank", ""),
                "n_genes": module.get("n_genes", ""),
                "edge_density": module.get("edge_density", ""),
                "mean_score": module.get("mean_score", ""),
                "module_claim_label_before_full_reselection": module.get("module_claim_label", ""),
                "is_reportable_calibrated_module_before_full_reselection": module.get(
                    "is_reportable_calibrated_module", ""
                ),
                "full_reselection_p": full_p,
                "full_reselection_z": full_row.get("full_reselection_score_z", ""),
                "degree_preserving_graph_p": graph_p,
                "selection_penalty_ratio": selection_penalty_ratio,
                "delta_log10p_fixed_to_full": delta_log10p,
                "fixed_vs_full_pattern": pattern,
                "source_module_table": str(module_spec.modules_path),
                "source_full_reselection_table": str(spec.full_reselection_summary),
                "script_path": str(THIS_SCRIPT),
            }
            row.update(fixed)
            rows.append(row)
            null_table.insert(0, "module_id", module_id)
            null_table.insert(0, "analysis_id", spec.analysis_id)
            null_table.insert(0, "trait", spec.trait)
            null_rows.append(null_table)
    summary = pd.DataFrame(rows)
    nulls = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    write_table(out_dir / "tables" / "a2_fixed_vs_full_reselection_gap.tsv", summary)
    write_table(out_dir / "tables" / "a2_fixed_module_nulls.tsv.gz", nulls)
    return summary, nulls


def as_float(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def compact_subgraph(
    graph: nx.Graph,
    seed: str,
    size: int,
    degree: dict[str, int],
    *,
    stratum: str,
) -> tuple[str, ...]:
    selected = {seed}
    frontier = set(str(node) for node in graph.neighbors(seed))
    seed_degree = degree[seed]
    while len(selected) < size and frontier:
        if stratum == "hub_adjacent":
            next_node = max(frontier, key=lambda node: (degree[node], -abs(degree[node] - seed_degree), node))
        else:
            next_node = min(frontier, key=lambda node: (abs(degree[node] - seed_degree), degree[node], node))
        frontier.remove(next_node)
        selected.add(next_node)
        frontier.update(str(node) for node in graph.neighbors(next_node) if str(node) not in selected)
    if len(selected) < size:
        raise ValueError(f"Could not grow compact subgraph from {seed} to size {size}")
    return tuple(sorted(selected))


def candidate_seeds(graph: nx.Graph, stratum: str) -> list[str]:
    degree = dict(graph.degree())
    table = pd.DataFrame({"gene": list(degree), "degree": list(degree.values())})
    if stratum == "low_degree":
        low, high = table["degree"].quantile([0.10, 0.35])
    elif stratum == "medium_degree":
        low, high = table["degree"].quantile([0.45, 0.60])
    elif stratum == "hub_adjacent":
        low, high = table["degree"].quantile([0.90, 1.00])
    else:
        raise ValueError(f"Unknown degree stratum: {stratum}")
    subset = table[(table["degree"] >= low) & (table["degree"] <= high)].copy()
    subset = subset.sort_values(["degree", "gene"], ascending=[stratum == "low_degree", True])
    return subset["gene"].astype(str).tolist()


def select_compact_oracle_modules(
    graph: nx.Graph,
    *,
    sizes: tuple[int, ...],
    strata: tuple[str, ...],
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    degree = {str(node): int(value) for node, value in graph.degree()}
    rows: list[dict[str, object]] = []
    for stratum in strata:
        seeds = candidate_seeds(graph, stratum)
        rng.shuffle(seeds)
        used: set[str] = set()
        for size in sizes:
            found = 0
            for seed_node in seeds:
                if seed_node in used:
                    continue
                try:
                    nodes = compact_subgraph(graph, seed_node, size, degree, stratum=stratum)
                except ValueError:
                    continue
                found += 1
                used.update(nodes)
                sub = graph.subgraph(nodes)
                rows.append(
                    {
                        "oracle_id": f"oracle_{stratum}_{size}_{found}",
                        "oracle_type": "single_compact",
                        "degree_stratum": stratum,
                        "target_size": size,
                        "oracle_replicate": found,
                        "oracle_genes": ",".join(nodes),
                        "oracle_n_genes": len(nodes),
                        "oracle_n_edges": int(sub.number_of_edges()),
                        "oracle_edge_density": float(2 * sub.number_of_edges() / (len(nodes) * (len(nodes) - 1))),
                        "oracle_mean_degree": float(np.mean([degree[node] for node in nodes])),
                    }
                )
                if found >= replicates:
                    break
            if found < replicates:
                raise RuntimeError(f"Only found {found} oracle modules for {stratum}, size={size}")
    return pd.DataFrame(rows)


def select_multi_and_diffuse_oracles(graph: nx.Graph, compact_oracles: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    degree = {str(node): int(value) for node, value in graph.degree()}
    rows: list[dict[str, object]] = []
    small = compact_oracles[compact_oracles["target_size"].astype(int).eq(10)].copy()
    if len(small) >= 3:
        chosen = small.groupby("degree_stratum", observed=True).head(1)
        if len(chosen) >= 3:
            nodes = sorted(set().union(*(set(split_genes(value)) for value in chosen["oracle_genes"])))
            sub = graph.subgraph(nodes)
            rows.append(
                {
                    "oracle_id": "oracle_multi_small_mixed_30",
                    "oracle_type": "multi_small_module",
                    "degree_stratum": "mixed",
                    "target_size": len(nodes),
                    "oracle_replicate": 1,
                    "oracle_genes": ",".join(nodes),
                    "oracle_n_genes": len(nodes),
                    "oracle_n_edges": int(sub.number_of_edges()),
                    "oracle_edge_density": float(2 * sub.number_of_edges() / (len(nodes) * (len(nodes) - 1))),
                    "oracle_mean_degree": float(np.mean([degree[node] for node in nodes])),
                }
            )
    candidates = candidate_seeds(graph, "medium_degree")
    rng.shuffle(candidates)
    diffuse: list[str] = []
    diffuse_set: set[str] = set()
    for node in candidates:
        if node in diffuse_set:
            continue
        if any(graph.has_edge(node, selected) for selected in diffuse):
            continue
        diffuse.append(node)
        diffuse_set.add(node)
        if len(diffuse) >= 100:
            break
    if len(diffuse) < 100:
        for node in candidates:
            if node not in diffuse_set:
                diffuse.append(node)
                diffuse_set.add(node)
            if len(diffuse) >= 100:
                break
    nodes = sorted(diffuse[:100])
    sub = graph.subgraph(nodes)
    rows.append(
        {
            "oracle_id": "oracle_diffuse_pathway_like_100",
            "oracle_type": "diffuse_pathway_like",
            "degree_stratum": "medium_degree",
            "target_size": len(nodes),
            "oracle_replicate": 1,
            "oracle_genes": ",".join(nodes),
            "oracle_n_genes": len(nodes),
            "oracle_n_edges": int(sub.number_of_edges()),
            "oracle_edge_density": float(2 * sub.number_of_edges() / (len(nodes) * (len(nodes) - 1))),
            "oracle_mean_degree": float(np.mean([degree[node] for node in nodes])),
        }
    )
    return pd.DataFrame(rows)


def score_spikein(scores: pd.DataFrame, genes: set[str], effect: float) -> pd.DataFrame:
    out = scores.copy()
    mask = out["gene_symbol"].astype(str).isin(genes)
    out.loc[mask, "assoc_resid_score"] = pd.to_numeric(out.loc[mask, "assoc_resid_score"], errors="raise") + effect
    return out


def best_recovered_module(modules: pd.DataFrame, oracle_genes: set[str]) -> dict[str, object]:
    if modules.empty:
        return {
            "best_module_id": "",
            "best_jaccard": 0.0,
            "best_overlap": 0,
            "best_module_size": 0,
            "best_module_mean_score": float("nan"),
            "best_module_edge_density": float("nan"),
            "core_gene_recurrence": 0.0,
            "best_module_genes": "",
        }
    best: dict[str, object] | None = None
    for row in modules.to_dict(orient="records"):
        genes = set(split_genes(row.get("module_genes", "")))
        score = jaccard(genes, oracle_genes)
        if best is None or score > float(best["best_jaccard"]):
            core = set(split_genes(row.get("core_genes", "")))
            best = {
                "best_module_id": row.get("module_id", ""),
                "best_jaccard": score,
                "best_overlap": len(genes & oracle_genes),
                "best_module_size": len(genes),
                "best_module_mean_score": row.get("mean_score", float("nan")),
                "best_module_edge_density": row.get("edge_density", float("nan")),
                "core_gene_recurrence": float(len(core & oracle_genes) / len(core)) if core else 0.0,
                "best_module_genes": ",".join(sorted(genes)),
            }
    return best or {}


def run_a1_oracle_spikein(args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = AuditSpec(
        trait="DR_MVP",
        analysis_id="DR_MVP_default_final5000",
        analysis_dir=ANALYSIS_ROOT / "dr_mvp_string_final5000",
        graph_edges_path=ANALYSIS_ROOT / "dr_mvp_string_final5000" / "tables" / "DR_MVP.analysis_graph_edges.tsv.gz",
        full_reselection_summary=DR_MODULE_RESELECTION_DIR / "tables" / "module_full_reselection_summary.all_traits.tsv",
    )
    scores, graph = prepare_scores_and_graph(spec)
    compact = select_compact_oracle_modules(
        graph,
        sizes=(10, 25, 50, 100),
        strata=("low_degree", "medium_degree", "hub_adjacent"),
        replicates=args.oracle_replicates,
        seed=args.seed + 5000,
    )
    extra = select_multi_and_diffuse_oracles(graph, compact, seed=args.seed + 6000)
    oracles = pd.concat([compact, extra], ignore_index=True)
    effect_grid = parse_effect_grid(args.effect_grid)
    scenario_rows: list[dict[str, object]] = []
    module_rows: list[pd.DataFrame] = []
    for oracle_idx, oracle in enumerate(oracles.to_dict(orient="records")):
        oracle_genes = set(split_genes(oracle["oracle_genes"]))
        for effect_label, effect_value in effect_grid.items():
            scenario_id = f"{oracle['oracle_id']}_{effect_label}"
            print(f"A1 oracle spike-in: {scenario_id}", flush=True)
            spiked = score_spikein(scores, oracle_genes, effect_value)
            modules = discover_local_modules(
                graph,
                spiked,
                min_module_size=5,
                min_subthreshold_genes=3,
                max_modules=20,
                broad_component_min_size=200,
                broad_component_fraction=0.50,
            )
            nulls = fast_full_reselection_null(
                graph,
                spiked,
                n_replicates=args.n_oracle_null,
                degree_bins=args.degree_bins,
                min_module_size=5,
                min_subthreshold_genes=3,
                max_modules=20,
                broad_component_min_size=200,
                broad_component_fraction=0.50,
                seed=args.seed + 10000 + oracle_idx * 100 + int(effect_value * 10),
            )
            best = best_recovered_module(modules, oracle_genes)
            observed_mean = as_float(best.get("best_module_mean_score", np.nan))
            full_p = empirical_upper(nulls["max_mean_score"].to_numpy(dtype=float), observed_mean)
            full_z = z_score(observed_mean, nulls["max_mean_score"].to_numpy(dtype=float))
            oracle_mean_degree = as_float(oracle.get("oracle_mean_degree", np.nan))
            best_genes = set(split_genes(best.get("best_module_genes", "")))
            graph_degree = dict(graph.degree())
            best_mean_degree = (
                float(np.mean([graph_degree[gene] for gene in best_genes])) if best_genes else float("nan")
            )
            scenario_rows.append(
                {
                    "scenario_id": scenario_id,
                    "oracle_id": oracle["oracle_id"],
                    "oracle_type": oracle["oracle_type"],
                    "degree_stratum": oracle["degree_stratum"],
                    "target_size": oracle["target_size"],
                    "effect_label": effect_label,
                    "effect_additive_z": effect_value,
                    "n_oracle_null": args.n_oracle_null,
                    "full_reselection_empirical_p": full_p,
                    "full_reselection_z": full_z,
                    "recovery_jaccard": best.get("best_jaccard", 0.0),
                    "oracle_overlap": best.get("best_overlap", 0),
                    "core_gene_recurrence": best.get("core_gene_recurrence", 0.0),
                    "best_module_id": best.get("best_module_id", ""),
                    "best_module_size": best.get("best_module_size", 0),
                    "best_module_mean_score": observed_mean,
                    "best_module_edge_density": best.get("best_module_edge_density", ""),
                    "module_size_bias": (as_float(best.get("best_module_size", np.nan)) / as_float(oracle["target_size"]))
                    if as_float(oracle["target_size"]) > 0
                    else float("nan"),
                    "oracle_mean_degree": oracle_mean_degree,
                    "best_module_mean_degree": best_mean_degree,
                    "degree_bias": best_mean_degree - oracle_mean_degree
                    if np.isfinite(best_mean_degree) and np.isfinite(oracle_mean_degree)
                    else float("nan"),
                    "selection_calibrated_at_0p05": bool(np.isfinite(full_p) and full_p <= 0.05),
                    "recovered_jaccard_ge_0p25": bool(as_float(best.get("best_jaccard", 0.0)) >= 0.25),
                    "recovered_jaccard_ge_0p50": bool(as_float(best.get("best_jaccard", 0.0)) >= 0.50),
                    "script_path": str(THIS_SCRIPT),
                }
            )
            if not modules.empty:
                tagged = modules.copy()
                tagged.insert(0, "scenario_id", scenario_id)
                tagged.insert(1, "oracle_id", oracle["oracle_id"])
                module_rows.append(tagged)
    scenarios = pd.DataFrame(scenario_rows)
    observed_modules = pd.concat(module_rows, ignore_index=True) if module_rows else pd.DataFrame()
    power = summarize_oracle_power(scenarios)
    write_table(out_dir / "tables" / "a1_oracle_modules.tsv", oracles)
    write_table(out_dir / "tables" / "a1_oracle_spikein_scenarios.tsv", scenarios)
    write_table(out_dir / "tables" / "a1_oracle_spikein_observed_modules.tsv", observed_modules)
    write_table(out_dir / "tables" / "a1_oracle_spikein_power_curve.tsv", power)
    return scenarios, power, oracles


def summarize_oracle_power(scenarios: pd.DataFrame) -> pd.DataFrame:
    if scenarios.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["oracle_type", "target_size", "degree_stratum", "effect_label", "effect_additive_z"]
    for key, group in scenarios.groupby(group_cols, observed=True, dropna=False):
        oracle_type, target_size, degree_stratum, effect_label, effect_value = key
        rows.append(
            {
                "oracle_type": oracle_type,
                "target_size": target_size,
                "degree_stratum": degree_stratum,
                "effect_label": effect_label,
                "effect_additive_z": effect_value,
                "n_scenarios": int(len(group)),
                "selection_calibrated_power": float(group["selection_calibrated_at_0p05"].mean()),
                "recovery_jaccard_ge_0p25_rate": float(group["recovered_jaccard_ge_0p25"].mean()),
                "recovery_jaccard_ge_0p50_rate": float(group["recovered_jaccard_ge_0p50"].mean()),
                "median_full_reselection_p": float(pd.to_numeric(group["full_reselection_empirical_p"]).median()),
                "median_recovery_jaccard": float(pd.to_numeric(group["recovery_jaccard"]).median()),
                "median_module_size_bias": float(pd.to_numeric(group["module_size_bias"]).median()),
                "median_degree_bias": float(pd.to_numeric(group["degree_bias"]).median()),
            }
        )
    return pd.DataFrame(rows)


def parse_effect_grid(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in str(value).split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(f"Effect grid item must be label:value, got {item!r}")
        label, raw = item.split(":", 1)
        out[label.strip()] = float(raw)
    if not out:
        raise ValueError("Effect grid cannot be empty.")
    return out


def render_report(
    *,
    args: argparse.Namespace,
    a2: pd.DataFrame,
    a1: pd.DataFrame,
    power: pd.DataFrame,
    manifest: dict[str, object],
) -> str:
    lines = [
        "# RIPPLE Tier 4 Design-Defect Audit v1",
        "",
        f"Created: {manifest['created_utc']}",
        "",
        "This diagnostic audit does not alter frozen RIPPLE V1 claim policy. It separates fixed-module support from full-reselection selection-aware evidence and tests oracle compact-module power.",
        "",
        "## A2 Fixed-Module vs Full-Reselection Gap",
        "",
    ]
    if a2.empty:
        lines.append("A2 was not run or produced no rows.")
    else:
        counts = a2.groupby(["analysis_id", "fixed_vs_full_pattern"], observed=True).size().reset_index(name="n_modules")
        lines.extend(["| Analysis | Pattern | Modules |", "|---|---|---:|"])
        for row in counts.to_dict(orient="records"):
            lines.append(f"| {row['analysis_id']} | {row['fixed_vs_full_pattern']} | {row['n_modules']} |")
        fixed_hits = int((pd.to_numeric(a2["degree_matched_fixed_p"], errors="coerce") <= 0.05).sum())
        full_hits = int((pd.to_numeric(a2["full_reselection_p"], errors="coerce") <= 0.05).sum())
        lines.extend(
            [
                "",
                f"- Fixed degree-matched support modules: {fixed_hits}",
                f"- Full-reselection support modules: {full_hits}",
                "",
            ]
        )
    lines.extend(["## A1 Oracle Compact Module Spike-In", ""])
    if a1.empty:
        lines.append("A1 was not run or produced no rows.")
    else:
        power_by_effect = (
            a1.groupby("effect_label", observed=True)
            .agg(
                n_scenarios=("scenario_id", "count"),
                selection_calibrated_power=("selection_calibrated_at_0p05", "mean"),
                median_recovery_jaccard=("recovery_jaccard", "median"),
                median_full_reselection_p=("full_reselection_empirical_p", "median"),
            )
            .reset_index()
        )
        lines.extend(["| Effect | Scenarios | Selection-calibrated power | Median Jaccard | Median full P |", "|---|---:|---:|---:|---:|"])
        for row in power_by_effect.to_dict(orient="records"):
            lines.append(
                f"| {row['effect_label']} | {int(row['n_scenarios'])} | "
                f"{float(row['selection_calibrated_power']):.3f} | "
                f"{float(row['median_recovery_jaccard']):.3f} | "
                f"{float(row['median_full_reselection_p']):.4g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- If fixed-module P is small but full-reselection P is large, the module may have weak fixed-candidate support but does not survive search-wide selection-aware calibration.",
            "- If oracle compact modules with moderate or strong effects fail, current Tier 4 likely has low power or a statistic/extraction mismatch.",
            "- If oracle compact modules pass but real traits fail, the real signal is likely diffuse, graph-mismatched, underpowered, or not represented as compact STRING modules.",
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

    a2 = pd.DataFrame()
    a2_nulls = pd.DataFrame()
    a1 = pd.DataFrame()
    power = pd.DataFrame()
    oracles = pd.DataFrame()
    if args.phase in {"all", "a2"}:
        a2, a2_nulls = run_a2_fixed_vs_full(args, args.out_dir)
    if args.phase in {"all", "a1"}:
        a1, power, oracles = run_a1_oracle_spikein(args, args.out_dir)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "analysis_id": "tier4_design_defect_audit_v1",
        "phase": args.phase,
        "n_fixed_null": args.n_fixed_null,
        "n_oracle_null": args.n_oracle_null,
        "oracle_replicates": args.oracle_replicates,
        "effect_grid": args.effect_grid,
        "degree_bins": args.degree_bins,
        "seed": args.seed,
        "script_path": str(THIS_SCRIPT),
        "outputs": {
            "a2_fixed_vs_full_reselection_gap": str(args.out_dir / "tables" / "a2_fixed_vs_full_reselection_gap.tsv"),
            "a2_fixed_module_nulls": str(args.out_dir / "tables" / "a2_fixed_module_nulls.tsv.gz"),
            "a1_oracle_spikein_scenarios": str(args.out_dir / "tables" / "a1_oracle_spikein_scenarios.tsv"),
            "a1_oracle_spikein_power_curve": str(args.out_dir / "tables" / "a1_oracle_spikein_power_curve.tsv"),
        },
    }
    (args.out_dir / "tier4_design_defect_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out_dir / "reports" / "tier4_design_defect_audit_report.md").write_text(
        render_report(args=args, a2=a2, a1=a1, power=power, manifest=manifest),
        encoding="utf-8",
    )
    print(f"Wrote Tier 4 design-defect audit outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
