"""Anchored module tests for RIPPLE V1.2 diagnostics.

Anchored module testing keeps the candidate module library fixed before
calibration. This is intentionally separate from V1 de novo local component
discovery: it tests whether predefined communities or gene sets carry
degree-calibrated weak-signal burden without claiming topology-specific
module discovery.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from ripple.nulls.score_permutation import assign_degree_bins, degree_stratified_permuted_scores
from ripple.percolation import rank_nodes_by_score


@dataclass(frozen=True)
class AnchoredModuleLibrary:
    """Fixed anchored module library plus provenance metadata."""

    gene_sets: dict[str, set[str]]
    module_source: dict[str, str]
    annotation_source_type: dict[str, str]
    module_category: dict[str, str] = field(default_factory=dict)


def empirical_upper(null_values: np.ndarray, observed: float) -> float:
    """Empirical upper-tail P using the RIPPLE null-count correction."""

    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((np.count_nonzero(finite >= observed) + 1) / (finite.size + 1))


def empirical_lower(null_values: np.ndarray, observed: float) -> float:
    """Empirical lower-tail P using the RIPPLE null-count correction."""

    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or not np.isfinite(observed):
        return float("nan")
    return float((np.count_nonzero(finite <= observed) + 1) / (finite.size + 1))


def z_score(observed: float, null_values: np.ndarray) -> float:
    finite = np.asarray(null_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2 or not np.isfinite(observed):
        return float("nan")
    sd = float(np.std(finite, ddof=1))
    return float((observed - float(np.mean(finite))) / sd) if sd > 0 else float("nan")


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted P values, preserving NaNs."""

    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    if not finite_mask.any():
        return q
    finite_p = p[finite_mask]
    order = np.argsort(finite_p, kind="mergesort")
    ranked = finite_p[order]
    m = len(ranked)
    adjusted = np.empty(m, dtype=float)
    running = 1.0
    for idx in range(m - 1, -1, -1):
        running = min(running, float(ranked[idx] * m / (idx + 1)))
        adjusted[idx] = running
    restored = np.empty(m, dtype=float)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    q[finite_mask] = restored
    return q


def sqrt_n_mean_stat(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values) * np.sqrt(values.size))


def top_fraction_mean(values: np.ndarray, *, fraction: float = 0.20) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    n_top = max(1, int(np.ceil(values.size * float(fraction))))
    top = np.partition(values, values.size - n_top)[-n_top:]
    return float(np.mean(top))


def edge_density(graph: nx.Graph, nodes: Sequence[str]) -> float:
    if len(nodes) < 2:
        return 0.0
    subgraph = graph.subgraph(nodes)
    return float(2 * subgraph.number_of_edges() / (len(nodes) * (len(nodes) - 1)))


def largest_component_size(graph: nx.Graph, nodes: Sequence[str]) -> int:
    selected = [str(node) for node in nodes if graph.has_node(str(node))]
    if not selected:
        return 0
    subgraph = graph.subgraph(selected)
    return int(max((len(component) for component in nx.connected_components(subgraph)), default=0))


def prepare_anchored_background(graph: nx.Graph, scores: pd.DataFrame) -> tuple[nx.Graph, pd.DataFrame]:
    """Return uppercase graph and analysis-eligible score table."""

    required = {"gene_symbol", "assoc_resid_score"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"scores is missing required columns: {missing}")

    graph_upper = nx.relabel_nodes(graph, {node: str(node).upper() for node in graph.nodes()}, copy=True)
    work = scores.copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str).str.upper()
    work["assoc_resid_score"] = pd.to_numeric(work["assoc_resid_score"], errors="raise").astype(float)
    work = work.drop_duplicates("gene_symbol", keep="first")
    work = work[work["gene_symbol"].isin(set(graph_upper.nodes()))].reset_index(drop=True)
    if work.empty:
        raise ValueError("No score genes overlap the graph.")
    if "graph_degree" not in work.columns:
        work["graph_degree"] = [int(graph_upper.degree(str(gene))) for gene in work["gene_symbol"]]
    work["graph_degree"] = pd.to_numeric(work["graph_degree"], errors="raise").astype(int)
    return graph_upper, work


def build_louvain_anchor_library(
    graph: nx.Graph,
    *,
    min_size: int = 10,
    max_size: int = 300,
    resolution: float = 1.0,
    seed: int = 20260712,
    prefix: str = "STRING_Louvain",
) -> AnchoredModuleLibrary:
    """Build graph-community anchors from Louvain communities."""

    raw = nx.community.louvain_communities(graph, resolution=float(resolution), seed=int(seed))
    gene_sets: dict[str, set[str]] = {}
    module_source: dict[str, str] = {}
    annotation_source_type: dict[str, str] = {}
    module_category: dict[str, str] = {}
    for idx, community in enumerate(raw, start=1):
        genes = {str(gene).upper() for gene in community}
        if min_size <= len(genes) <= max_size:
            name = f"{prefix}_{idx:04d}"
            gene_sets[name] = genes
            module_source[name] = "graph_louvain_community"
            annotation_source_type[name] = "graph_construction_related"
            module_category[name] = "graph_community"
    return AnchoredModuleLibrary(gene_sets, module_source, annotation_source_type, module_category)


def merge_anchored_libraries(*libraries: AnchoredModuleLibrary) -> AnchoredModuleLibrary:
    gene_sets: dict[str, set[str]] = {}
    module_source: dict[str, str] = {}
    annotation_source_type: dict[str, str] = {}
    module_category: dict[str, str] = {}
    for library in libraries:
        for name, genes in library.gene_sets.items():
            out_name = str(name)
            suffix = 2
            while out_name in gene_sets:
                out_name = f"{name}__{suffix}"
                suffix += 1
            gene_sets[out_name] = {str(gene).upper() for gene in genes}
            module_source[out_name] = library.module_source.get(name, "unspecified")
            annotation_source_type[out_name] = library.annotation_source_type.get(name, "internal_support")
            module_category[out_name] = library.module_category.get(name, "unspecified")
    return AnchoredModuleLibrary(gene_sets, module_source, annotation_source_type, module_category)


def gene_sets_to_library(
    gene_sets: Mapping[str, set[str]],
    *,
    module_source: str,
    annotation_source_type: str,
    module_category: str = "unspecified",
) -> AnchoredModuleLibrary:
    return AnchoredModuleLibrary(
        gene_sets={str(name): {str(gene).upper() for gene in genes} for name, genes in gene_sets.items()},
        module_source={str(name): str(module_source) for name in gene_sets},
        annotation_source_type={str(name): str(annotation_source_type) for name in gene_sets},
        module_category={str(name): str(module_category) for name in gene_sets},
    )


def load_anchored_gene_set_library(
    path: str | Path,
    *,
    default_module_source: str | None = None,
    default_annotation_source_type: str = "independent_external",
) -> AnchoredModuleLibrary:
    """Load a TSV/GMT anchored gene-set library with optional per-set metadata."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".gmt":
        gene_sets: dict[str, set[str]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = [part.strip() for part in line.split("\t") if part.strip()]
            if len(parts) >= 3:
                gene_sets[parts[0]] = {gene.upper() for gene in parts[2:]}
        return gene_sets_to_library(
            gene_sets,
            module_source=default_module_source or f"gmt_file:{path}",
            annotation_source_type=default_annotation_source_type,
        )

    table = pd.read_csv(path, sep="\t", compression="infer")
    if not {"gene_set", "gene_symbol"}.issubset(table.columns):
        raise ValueError("Anchored gene-set TSV must contain gene_set and gene_symbol columns.")

    gene_sets: dict[str, set[str]] = {}
    module_source: dict[str, str] = {}
    annotation_source_type: dict[str, str] = {}
    module_category: dict[str, str] = {}
    for name, group in table.groupby("gene_set", observed=True):
        key = str(name)
        gene_sets[key] = {str(gene).upper() for gene in group["gene_symbol"].dropna()}
        if "module_source" in group.columns and group["module_source"].notna().any():
            module_source[key] = str(group["module_source"].dropna().iloc[0])
        elif "source_database" in group.columns and group["source_database"].notna().any():
            module_source[key] = str(group["source_database"].dropna().iloc[0])
        else:
            module_source[key] = default_module_source or f"tsv_file:{path}"
        if "annotation_source_type" in group.columns and group["annotation_source_type"].notna().any():
            annotation_source_type[key] = str(group["annotation_source_type"].dropna().iloc[0])
        else:
            annotation_source_type[key] = default_annotation_source_type
        if "category" in group.columns and group["category"].notna().any():
            module_category[key] = str(group["category"].dropna().iloc[0])
        else:
            module_category[key] = "unspecified"
    return AnchoredModuleLibrary(gene_sets, module_source, annotation_source_type, module_category)


def _degree_profile(table: pd.DataFrame, genes: Sequence[str], *, n_bins: int) -> dict[int, int]:
    bins = assign_degree_bins(table["graph_degree"], n_bins=n_bins).to_numpy(dtype=int)
    node_to_bin = dict(zip(table["gene_symbol"].astype(str), bins, strict=True))
    profile: dict[int, int] = {}
    for gene in genes:
        bin_id = int(node_to_bin[str(gene)])
        profile[bin_id] = profile.get(bin_id, 0) + 1
    return profile


def _sample_degree_matched_indices(
    bin_to_indices: Mapping[int, np.ndarray],
    profile: Mapping[int, int],
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    sampled: list[int] = []
    for bin_id, count in sorted(profile.items()):
        candidates = np.asarray(bin_to_indices[int(bin_id)], dtype=int)
        replace = int(count) > len(candidates)
        sampled.extend(int(idx) for idx in rng.choice(candidates, size=int(count), replace=replace))
    rng.shuffle(sampled)
    return np.asarray(sampled, dtype=int)


def _fixed_module_score_permutation_null(
    score_null_matrix: np.ndarray,
    module_indices: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    fixed: dict[str, np.ndarray] = {
        module_id: np.empty(score_null_matrix.shape[0], dtype=float) for module_id in module_indices
    }
    family_max = np.full(score_null_matrix.shape[0], -np.inf, dtype=float)
    for replicate_idx in range(score_null_matrix.shape[0]):
        row = score_null_matrix[replicate_idx, :]
        replicate_stats: list[float] = []
        for module_id, indices in module_indices.items():
            stat = sqrt_n_mean_stat(row[indices])
            fixed[module_id][replicate_idx] = stat
            replicate_stats.append(stat)
        if replicate_stats:
            family_max[replicate_idx] = float(np.nanmax(replicate_stats))
    return fixed, family_max


def anchored_module_tests(
    graph: nx.Graph,
    scores: pd.DataFrame,
    library: AnchoredModuleLibrary,
    *,
    min_present: int = 5,
    n_degree_matched: int = 200,
    n_score_permutation: int = 200,
    degree_bins: int = 10,
    seed: int = 20260712,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Test fixed anchored modules against degree and module-library nulls."""

    if not library.gene_sets:
        return pd.DataFrame(), pd.DataFrame(), {"n_input_modules": 0, "n_tested_modules": 0}

    graph_upper, work = prepare_anchored_background(graph, scores)
    rng = np.random.default_rng(seed)
    values = work["assoc_resid_score"].to_numpy(dtype=float)
    genes = work["gene_symbol"].astype(str).to_numpy(dtype=object)
    gene_to_idx = {str(gene): idx for idx, gene in enumerate(genes)}
    background_genes = set(str(gene) for gene in genes)
    rank_table = rank_nodes_by_score(work, node_col="gene_symbol", score_col="assoc_resid_score")
    rank_fraction_by_gene = dict(
        zip(rank_table["gene_symbol"].astype(str), rank_table["rank_fraction"].astype(float), strict=True)
    )
    bins = assign_degree_bins(work["graph_degree"], n_bins=degree_bins).to_numpy(dtype=int)
    bin_to_indices = {int(bin_id): np.flatnonzero(bins == bin_id) for bin_id in sorted(np.unique(bins))}

    module_indices: dict[str, np.ndarray] = {}
    module_gene_lists: dict[str, tuple[str, ...]] = {}
    dropped_gene_lists: dict[str, tuple[str, ...]] = {}
    preliminary_rows: list[dict[str, object]] = []
    for module_idx, (name, raw_genes) in enumerate(library.gene_sets.items(), start=1):
        query = {str(gene).upper() for gene in raw_genes}
        present = tuple(sorted(query & background_genes))
        dropped = tuple(sorted(query - background_genes))
        module_id = f"A{module_idx:04d}"
        if len(present) < min_present:
            preliminary_rows.append(
                {
                    "module_id": module_id,
                    "module_name": str(name),
                    "module_source": library.module_source.get(name, "unspecified"),
                    "annotation_source_type": library.annotation_source_type.get(name, "internal_support"),
                    "module_category": library.module_category.get(name, "unspecified"),
                    "n_query_genes": int(len(query)),
                    "n_present": int(len(present)),
                    "n_missing": int(len(dropped)),
                    "background_size": int(len(background_genes)),
                    "present_genes": ",".join(present),
                    "dropped_genes": ",".join(dropped),
                    "module_status": "not_tested_low_overlap",
                }
            )
            continue
        indices = np.asarray([gene_to_idx[gene] for gene in present], dtype=int)
        module_indices[module_id] = indices
        module_gene_lists[module_id] = present
        dropped_gene_lists[module_id] = dropped
        preliminary_rows.append(
            {
                "module_id": module_id,
                "module_name": str(name),
                "module_source": library.module_source.get(name, "unspecified"),
                "annotation_source_type": library.annotation_source_type.get(name, "internal_support"),
                "module_category": library.module_category.get(name, "unspecified"),
                "n_query_genes": int(len(query)),
                "n_present": int(len(present)),
                "n_missing": int(len(dropped)),
                "background_size": int(len(background_genes)),
                "present_genes": ",".join(present),
                "dropped_genes": ",".join(dropped),
                "module_status": "tested",
            }
        )

    if not module_indices:
        return (
            pd.DataFrame(preliminary_rows),
            pd.DataFrame(),
            {
                "n_input_modules": int(len(library.gene_sets)),
                "n_tested_modules": 0,
                "background_size": int(len(background_genes)),
            },
        )

    score_null_matrix = degree_stratified_permuted_scores(
        work,
        score_col="assoc_resid_score",
        degree_col="graph_degree",
        n_replicates=n_score_permutation,
        seed=seed + 1009,
        n_bins=degree_bins,
    )
    fixed_score_nulls, family_max_null = _fixed_module_score_permutation_null(
        score_null_matrix,
        module_indices,
    )

    rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    prelim_by_id = {str(row["module_id"]): row for row in preliminary_rows}
    for module_id, indices in module_indices.items():
        present = module_gene_lists[module_id]
        observed_values = values[indices]
        observed_stat = sqrt_n_mean_stat(observed_values)
        observed_mean = float(np.mean(observed_values))
        observed_top20 = top_fraction_mean(observed_values, fraction=0.20)
        observed_rank_mean = float(np.mean([rank_fraction_by_gene[gene] for gene in present]))
        subgraph = graph_upper.subgraph(present)
        lcc_size = largest_component_size(graph_upper, present)
        degree_profile = _degree_profile(work, present, n_bins=degree_bins)
        degree_null = np.empty(n_degree_matched, dtype=float)
        degree_mean_null = np.empty(n_degree_matched, dtype=float)
        degree_top20_null = np.empty(n_degree_matched, dtype=float)
        rank_null = np.empty(n_degree_matched, dtype=float)
        for replicate_idx in range(n_degree_matched):
            sampled = _sample_degree_matched_indices(bin_to_indices, degree_profile, rng=rng)
            sampled_values = values[sampled]
            degree_null[replicate_idx] = sqrt_n_mean_stat(sampled_values)
            degree_mean_null[replicate_idx] = float(np.mean(sampled_values))
            degree_top20_null[replicate_idx] = top_fraction_mean(sampled_values, fraction=0.20)
            sampled_genes = [str(genes[idx]) for idx in sampled]
            rank_null[replicate_idx] = float(np.mean([rank_fraction_by_gene[gene] for gene in sampled_genes]))
        fixed_null = fixed_score_nulls[module_id]
        row = {
            **prelim_by_id[module_id],
            "module_status": "negative",
            "statistic_name": "sqrt_n_mean_residualized_score",
            "statistic_direction": "greater_is_more_extreme",
            "observed_value": observed_stat,
            "mean_score": observed_mean,
            "top20_mean_score": observed_top20,
            "mean_rank_fraction": observed_rank_mean,
            "degree_profile": json.dumps({str(k): int(v) for k, v in degree_profile.items()}),
            "mean_graph_degree": float(np.mean(work.loc[indices, "graph_degree"].to_numpy(dtype=float))),
            "induced_edges": int(subgraph.number_of_edges()),
            "edge_density": edge_density(graph_upper, present),
            "largest_component_size": int(lcc_size),
            "largest_component_fraction": float(lcc_size / len(present)),
            "degree_matched_null_mean": float(np.mean(degree_null)),
            "degree_matched_null_sd": float(np.std(degree_null, ddof=1))
            if len(degree_null) >= 2
            else float("nan"),
            "degree_matched_z": z_score(observed_stat, degree_null),
            "degree_matched_empirical_p": empirical_upper(degree_null, observed_stat),
            "degree_matched_mean_score_z": z_score(observed_mean, degree_mean_null),
            "degree_matched_top20_z": z_score(observed_top20, degree_top20_null),
            "rank_enrichment_z": -z_score(observed_rank_mean, rank_null),
            "rank_empirical_p_lower": empirical_lower(rank_null, observed_rank_mean),
            "fixed_score_permutation_null_mean": float(np.mean(fixed_null)),
            "fixed_score_permutation_null_sd": float(np.std(fixed_null, ddof=1))
            if len(fixed_null) >= 2
            else float("nan"),
            "fixed_score_permutation_z": z_score(observed_stat, fixed_null),
            "fixed_score_permutation_p": empirical_upper(fixed_null, observed_stat),
            "library_familywise_p": empirical_upper(family_max_null, observed_stat),
            "n_degree_matched_null": int(n_degree_matched),
            "n_score_permutation_null": int(n_score_permutation),
            "candidate_score_basis": "fixed_anchored_library;sqrt_n_mean_residualized_score",
            "module_label": "no_anchored_module_support",
        }
        rows.append(row)
        for replicate_idx, value in enumerate(degree_null):
            null_rows.append(
                {
                    "module_id": module_id,
                    "module_name": row["module_name"],
                    "replicate": int(replicate_idx),
                    "null_type": "degree_matched_node_set",
                    "statistic_name": "sqrt_n_mean_residualized_score",
                    "statistic_value": float(value),
                }
            )
        for replicate_idx, value in enumerate(fixed_null):
            null_rows.append(
                {
                    "module_id": module_id,
                    "module_name": row["module_name"],
                    "replicate": int(replicate_idx),
                    "null_type": "degree_stratified_score_permutation_fixed_module",
                    "statistic_name": "sqrt_n_mean_residualized_score",
                    "statistic_value": float(value),
                }
            )
    for replicate_idx, value in enumerate(family_max_null):
        null_rows.append(
            {
                "module_id": "ALL_ANCHORED_MODULES",
                "module_name": "max_over_fixed_anchored_library",
                "replicate": int(replicate_idx),
                "null_type": "degree_stratified_score_permutation_library_max",
                "statistic_name": "max_sqrt_n_mean_residualized_score",
                "statistic_value": float(value),
            }
        )

    out = pd.DataFrame(rows)
    out["degree_matched_fdr"] = bh_fdr(out["degree_matched_empirical_p"].to_numpy(dtype=float))
    out["fixed_score_permutation_fdr"] = bh_fdr(out["fixed_score_permutation_p"].to_numpy(dtype=float))
    out["library_familywise_fdr"] = bh_fdr(out["library_familywise_p"].to_numpy(dtype=float))
    out.loc[out["degree_matched_empirical_p"].le(0.05), "module_status"] = "fixed_degree_supported"
    out.loc[
        out["degree_matched_empirical_p"].le(0.05) & out["library_familywise_p"].le(0.05),
        "module_status",
    ] = "anchored_familywise_supported"
    out.loc[out["module_status"].eq("fixed_degree_supported"), "module_label"] = "fixed_module_supported"
    out.loc[
        out["module_status"].eq("anchored_familywise_supported"),
        "module_label",
    ] = "anchored_library_calibrated_module"
    out = out.sort_values(
        ["library_familywise_p", "degree_matched_empirical_p", "observed_value"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    out["anchored_module_rank"] = np.arange(1, len(out) + 1)

    not_tested = pd.DataFrame([row for row in preliminary_rows if row["module_status"] == "not_tested_low_overlap"])
    if not not_tested.empty:
        out = pd.concat([out, not_tested], ignore_index=True, sort=False)

    summary = {
        "n_input_modules": int(len(library.gene_sets)),
        "n_tested_modules": int(len(module_indices)),
        "n_not_tested_low_overlap": int(len(not_tested)),
        "background_size": int(len(background_genes)),
        "n_fixed_degree_supported": int(out["module_status"].eq("fixed_degree_supported").sum())
        if "module_status" in out
        else 0,
        "n_anchored_familywise_supported": int(
            out["module_status"].eq("anchored_familywise_supported").sum()
        )
        if "module_status" in out
        else 0,
        "n_degree_matched_null": int(n_degree_matched),
        "n_score_permutation_null": int(n_score_permutation),
        "degree_bins": int(degree_bins),
        "statistic_name": "sqrt_n_mean_residualized_score",
        "claim_boundary": (
            "Anchored familywise support is a V1.2 diagnostic over a fixed module library; "
            "it does not imply de novo topology-specific module discovery."
        ),
    }
    return out, pd.DataFrame(null_rows), summary


def render_anchored_module_report(
    *,
    trait: str,
    graph_name: str,
    modules: pd.DataFrame,
    summary: Mapping[str, object],
) -> str:
    lines = [
        f"# RIPPLE V1.2 Anchored Module Test: {trait}",
        "",
        f"Graph: `{graph_name}`",
        "",
        "This is a V1.2 diagnostic. It tests a fixed anchored module library and does not "
        "upgrade RIPPLE V1 Tier 4 or make topology-specific discovery claims.",
        "",
        "## Summary",
        "",
        f"- Input anchored modules: {int(summary.get('n_input_modules', 0)):,}",
        f"- Tested modules: {int(summary.get('n_tested_modules', 0)):,}",
        f"- Low-overlap modules not tested: {int(summary.get('n_not_tested_low_overlap', 0)):,}",
        f"- Analysis background genes: {int(summary.get('background_size', 0)):,}",
        f"- Fixed degree-supported modules: {int(summary.get('n_fixed_degree_supported', 0)):,}",
        f"- Anchored library familywise-supported modules: "
        f"{int(summary.get('n_anchored_familywise_supported', 0)):,}",
        "",
        "## Top Anchored Modules",
        "",
        "| Rank | Module | Category | Source | Source type | n | stat | degree P | family P | status |",
        "|---:|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    tested = modules.loc[modules.get("module_status", pd.Series(dtype=str)).ne("not_tested_low_overlap")].copy()
    if tested.empty:
        lines.append("| NA | No tested modules |  |  | 0 | NA | NA | NA | NA |")
    else:
        for row in tested.head(20).itertuples(index=False):
            lines.append(
                f"| {int(row.anchored_module_rank)} | {row.module_name} | "
                f"{getattr(row, 'module_category', '')} | {row.module_source} | "
                f"{row.annotation_source_type} | {int(row.n_present)} | "
                f"{float(row.observed_value):.3f} | {float(row.degree_matched_empirical_p):.4g} | "
                f"{float(row.library_familywise_p):.4g} | {row.module_status} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- `fixed_degree_supported`: the fixed module exceeds degree-matched random gene sets.",
            "- `anchored_familywise_supported`: the module also exceeds the max-over-library score-permutation null.",
            "- These labels support anchored graph-domain diagnostics only; they are not de novo module discovery claims.",
            "- `graph_construction_related` annotations should not be described as independent biological validation.",
            "",
        ]
    )
    return "\n".join(lines)
