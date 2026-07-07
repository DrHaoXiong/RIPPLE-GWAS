"""Local weak-signal module discovery and calibration."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from ripple.defaults import RANK_FRACTION_GRID
from ripple.nulls.graph_nulls import degree_preserving_graph_replicates
from ripple.nulls.score_permutation import assign_degree_bins, degree_stratified_permuted_scores
from ripple.percolation import rank_nodes_by_score, selected_nodes_at_fraction


DEFAULT_DR_GENE_SETS: dict[str, set[str]] = {
    "complement": {
        "C1QA",
        "C1QB",
        "C1QC",
        "C1R",
        "C1S",
        "C2",
        "C3",
        "C4A",
        "C4B",
        "C5",
        "C6",
        "C7",
        "C8A",
        "C8B",
        "C8G",
        "C9",
        "CFB",
        "CFH",
        "CFI",
        "CD46",
        "CD55",
        "CD59",
        "SERPING1",
        "MASP1",
        "MASP2",
    },
    "vegf_angiogenesis": {
        "VEGFA",
        "VEGFB",
        "VEGFC",
        "VEGFD",
        "KDR",
        "FLT1",
        "FLT4",
        "PGF",
        "HIF1A",
        "ANGPT1",
        "ANGPT2",
        "TEK",
        "TIE1",
        "NRP1",
        "NRP2",
        "DLL4",
        "NOTCH1",
        "PDGFB",
        "PDGFRB",
        "ESM1",
    },
    "endothelial_vascular": {
        "PECAM1",
        "VWF",
        "CDH5",
        "ENG",
        "ESAM",
        "ICAM1",
        "VCAM1",
        "SELE",
        "FLT1",
        "KDR",
        "TEK",
        "ROBO4",
        "CLDN5",
        "PLVAP",
        "EMCN",
        "MCAM",
        "EDN1",
        "NOS3",
        "SOX17",
        "ERG",
    },
    "ecm_basement_membrane": {
        "COL1A1",
        "COL1A2",
        "COL4A1",
        "COL4A2",
        "COL4A3",
        "COL4A4",
        "COL4A5",
        "LAMA1",
        "LAMA2",
        "LAMA3",
        "LAMA4",
        "LAMA5",
        "LAMB1",
        "LAMB2",
        "LAMC1",
        "FN1",
        "ELN",
        "MMP2",
        "MMP9",
        "TIMP1",
        "TIMP2",
        "LOX",
        "SPARC",
        "THBS1",
        "ITGA5",
        "ITGB1",
    },
}


def _require_columns(table: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [col for col in columns if col not in table.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _empirical_upper(null_values: np.ndarray, observed: float) -> float:
    if len(null_values) == 0:
        return float("nan")
    return float((np.count_nonzero(null_values >= observed) + 1) / (len(null_values) + 1))


def _empirical_lower(null_values: np.ndarray, observed: float) -> float:
    if len(null_values) == 0:
        return float("nan")
    return float((np.count_nonzero(null_values <= observed) + 1) / (len(null_values) + 1))


def _z_score(observed: float, null_values: np.ndarray) -> float:
    finite_nulls = np.asarray(null_values, dtype=float)
    finite_nulls = finite_nulls[np.isfinite(finite_nulls)]
    if len(finite_nulls) < 2:
        return float("nan")
    sd = float(np.std(finite_nulls, ddof=1))
    return float((observed - float(np.mean(finite_nulls))) / sd) if sd > 0 else float("nan")


def _finite_mean(null_values: np.ndarray) -> float:
    finite_nulls = np.asarray(null_values, dtype=float)
    finite_nulls = finite_nulls[np.isfinite(finite_nulls)]
    return float(np.mean(finite_nulls)) if len(finite_nulls) else float("nan")


def _edge_density(graph: nx.Graph, nodes: Sequence[str]) -> float:
    n_nodes = len(nodes)
    if n_nodes < 2:
        return 0.0
    subgraph = graph.subgraph(nodes)
    return float(2 * subgraph.number_of_edges() / (n_nodes * (n_nodes - 1)))


def _component_lcc_fraction(graph: nx.Graph, nodes: Sequence[str]) -> float:
    selected = [node for node in nodes if graph.has_node(node)]
    if not selected:
        return 0.0
    subgraph = graph.subgraph(selected)
    if subgraph.number_of_nodes() == 0:
        return 0.0
    largest = max((len(component) for component in nx.connected_components(subgraph)), default=0)
    return float(largest / len(selected))


def _degree_bin_profile(table: pd.DataFrame, nodes: Sequence[str], *, n_bins: int) -> dict[int, int]:
    bins = assign_degree_bins(table["graph_degree"], n_bins=n_bins)
    node_to_bin = dict(zip(table["gene_symbol"].astype(str), bins.astype(int), strict=True))
    profile: dict[int, int] = {}
    for node in nodes:
        bin_id = int(node_to_bin[str(node)])
        profile[bin_id] = profile.get(bin_id, 0) + 1
    return profile


def _sample_degree_matched_nodes(
    bin_to_nodes: Mapping[int, np.ndarray],
    profile: Mapping[int, int],
    *,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    sampled: list[str] = []
    for bin_id, count in sorted(profile.items()):
        candidates = np.asarray(bin_to_nodes[int(bin_id)], dtype=object)
        replace = int(count) > len(candidates)
        sampled.extend(str(node) for node in rng.choice(candidates, size=int(count), replace=replace))
    rng.shuffle(sampled)
    return tuple(sampled)


def _score_maps(scores: pd.DataFrame) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    score_by_node = dict(
        zip(scores["gene_symbol"].astype(str), scores["assoc_resid_score"].astype(float), strict=True)
    )
    p_by_node = (
        dict(zip(scores["gene_symbol"].astype(str), scores["assoc_p_g"].astype(float), strict=True))
        if "assoc_p_g" in scores.columns
        else {}
    )
    degree_by_node = dict(zip(scores["gene_symbol"].astype(str), scores["graph_degree"].astype(int), strict=True))
    return score_by_node, p_by_node, degree_by_node


def _module_core_genes(
    graph: nx.Graph,
    nodes: Sequence[str],
    score_by_node: Mapping[str, float],
    *,
    top_fraction: float = 0.20,
) -> tuple[str, ...]:
    if not nodes:
        return ()
    subgraph = graph.subgraph(nodes)
    internal_degree = dict(subgraph.degree())
    degree_cutoff = float(np.median([internal_degree[node] for node in nodes]))
    n_top = max(1, int(np.ceil(len(nodes) * top_fraction)))
    score_sorted = sorted(nodes, key=lambda node: (score_by_node[node], internal_degree[node], node), reverse=True)
    score_top = set(score_sorted[:n_top])
    core = [
        node
        for node in score_sorted
        if node in score_top and float(internal_degree[node]) >= degree_cutoff
    ]
    return tuple(core or score_sorted[:1])


def _module_row(
    *,
    graph: nx.Graph,
    module_id: str,
    rank_fraction: float,
    component_rank: int,
    nodes: Sequence[str],
    all_selected_nodes: set[str],
    score_by_node: Mapping[str, float],
    p_by_node: Mapping[str, float],
    degree_by_node: Mapping[str, int],
    gwas_p_threshold: float,
    broad_component_min_size: int,
    broad_component_fraction: float,
) -> dict[str, object]:
    subgraph = graph.subgraph(nodes)
    scores = np.array([score_by_node[node] for node in nodes], dtype=float)
    degrees = np.array([degree_by_node[node] for node in nodes], dtype=float)
    subthreshold = [node for node in nodes if not p_by_node or p_by_node.get(node, 1.0) > gwas_p_threshold]
    is_largest_selected_component = len(nodes) == max(
        (len(component) for component in nx.connected_components(graph.subgraph(all_selected_nodes))),
        default=0,
    )
    is_broad = len(nodes) >= broad_component_min_size or (
        len(all_selected_nodes) > 0 and len(nodes) / len(all_selected_nodes) >= broad_component_fraction
    )
    core_genes = _module_core_genes(graph, nodes, score_by_node)
    return {
        "module_id": module_id,
        "rank_fraction": float(rank_fraction),
        "component_rank": int(component_rank),
        "n_genes": int(len(nodes)),
        "n_edges": int(subgraph.number_of_edges()),
        "edge_density": _edge_density(graph, nodes),
        "is_largest_selected_component": bool(is_largest_selected_component),
        "is_broad_component": bool(is_broad),
        "mean_score": float(scores.mean()),
        "max_score": float(scores.max()),
        "median_score": float(np.median(scores)),
        "mean_graph_degree": float(degrees.mean()),
        "max_graph_degree": int(degrees.max()),
        "n_subthreshold_genes": int(len(subthreshold)),
        "subthreshold_gene_fraction": float(len(subthreshold) / len(nodes)),
        "core_genes": ",".join(core_genes),
        "module_genes": ",".join(nodes),
    }


def discover_local_modules(
    graph: nx.Graph,
    scores: pd.DataFrame,
    *,
    fractions: Iterable[float] = RANK_FRACTION_GRID,
    min_module_size: int = 5,
    min_subthreshold_genes: int = 3,
    max_modules: int = 20,
    gwas_p_threshold: float = 5e-8,
    broad_component_min_size: int = 200,
    broad_component_fraction: float = 0.50,
) -> pd.DataFrame:
    """Extract top-ranked induced connected components as candidate weak-signal modules."""

    _require_columns(scores, ["gene_symbol", "assoc_resid_score", "graph_degree"], "scores")
    if graph.number_of_nodes() == 0 or scores.empty:
        return pd.DataFrame()
    work = scores.copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str)
    work["assoc_resid_score"] = pd.to_numeric(work["assoc_resid_score"], errors="raise").astype(float)
    work["graph_degree"] = pd.to_numeric(work["graph_degree"], errors="raise").astype(int)
    ranking = rank_nodes_by_score(work, node_col="gene_symbol", score_col="assoc_resid_score")
    score_by_node, p_by_node, degree_by_node = _score_maps(work)
    rows: list[dict[str, object]] = []
    seen_gene_sets: set[tuple[str, ...]] = set()
    module_idx = 0
    for fraction in fractions:
        selected = set(selected_nodes_at_fraction(ranking, float(fraction), node_col="gene_symbol"))
        selected = {node for node in selected if graph.has_node(node)}
        if not selected:
            continue
        subgraph = graph.subgraph(selected)
        components = sorted(nx.connected_components(subgraph), key=lambda nodes: (-len(nodes), sorted(nodes)[0]))
        for component_rank, component_nodes in enumerate(components, start=1):
            nodes = tuple(sorted(str(node) for node in component_nodes))
            if len(nodes) < min_module_size:
                continue
            if p_by_node:
                n_subthreshold = sum(1 for node in nodes if p_by_node.get(node, 1.0) > gwas_p_threshold)
                if n_subthreshold < min_subthreshold_genes:
                    continue
            gene_set_key = tuple(nodes)
            if gene_set_key in seen_gene_sets:
                continue
            seen_gene_sets.add(gene_set_key)
            module_idx += 1
            rows.append(
                _module_row(
                    graph=graph,
                    module_id=f"M{module_idx:04d}",
                    rank_fraction=float(fraction),
                    component_rank=component_rank,
                    nodes=nodes,
                    all_selected_nodes=selected,
                    score_by_node=score_by_node,
                    p_by_node=p_by_node,
                    degree_by_node=degree_by_node,
                    gwas_p_threshold=gwas_p_threshold,
                    broad_component_min_size=broad_component_min_size,
                    broad_component_fraction=broad_component_fraction,
                )
            )
    if not rows:
        return pd.DataFrame()
    modules = pd.DataFrame(rows)
    modules = modules.sort_values(
        ["is_broad_component", "mean_score", "n_genes", "edge_density"],
        ascending=[True, False, False, False],
    ).head(max_modules)
    modules = modules.reset_index(drop=True)
    modules["module_rank"] = np.arange(1, len(modules) + 1)
    return modules


def selection_aware_module_null(
    graph: nx.Graph,
    scores: pd.DataFrame,
    *,
    selection_null_scores: np.ndarray | None = None,
    fractions: Iterable[float] = RANK_FRACTION_GRID,
    min_module_size: int = 5,
    min_subthreshold_genes: int = 3,
    max_modules: int = 20,
    gwas_p_threshold: float = 5e-8,
    broad_component_min_size: int = 200,
    broad_component_fraction: float = 0.50,
    n_replicates: int = 200,
    degree_bins: int = 10,
    seed: int = 20260614,
) -> pd.DataFrame:
    """Repeat local module extraction on score-null replicates.

    This calibrates the full "select top-ranked connected components, then test"
    procedure rather than only testing a module after it has already been chosen.
    Pipeline null score matrices are preferred; degree-stratified score
    permutation is used as a fallback when those null scores are unavailable.
    """

    if n_replicates < 0:
        raise ValueError("n_replicates must be nonnegative.")
    _require_columns(scores, ["gene_symbol", "assoc_resid_score", "graph_degree"], "scores")
    if n_replicates == 0:
        return pd.DataFrame()

    work = scores.copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str)
    work["assoc_resid_score"] = pd.to_numeric(work["assoc_resid_score"], errors="raise").astype(float)
    work["graph_degree"] = pd.to_numeric(work["graph_degree"], errors="raise").astype(int)
    graph_mask = work["gene_symbol"].isin(set(str(node) for node in graph.nodes())).to_numpy()
    work = work.loc[graph_mask].reset_index(drop=True)
    if work.empty:
        return pd.DataFrame()

    null_source = "degree_stratified_score_permutation"
    if selection_null_scores is not None:
        null_matrix = np.asarray(selection_null_scores, dtype=float)
        if null_matrix.ndim != 2:
            raise ValueError("selection_null_scores must be a two-dimensional matrix.")
        if null_matrix.shape[1] != len(scores):
            raise ValueError("selection_null_scores column count must match the score table row count.")
        null_matrix = null_matrix[:, graph_mask]
        null_matrix = null_matrix[: min(n_replicates, null_matrix.shape[0]), :]
        null_source = "pipeline_null_scores"
    else:
        null_matrix = degree_stratified_permuted_scores(
            work,
            score_col="assoc_resid_score",
            degree_col="graph_degree",
            n_replicates=n_replicates,
            seed=seed,
            n_bins=degree_bins,
        )
    rows: list[dict[str, object]] = []
    for replicate in range(null_matrix.shape[0]):
        null_scores = work.copy()
        null_scores["assoc_resid_score"] = null_matrix[replicate, :]
        null_modules = discover_local_modules(
            graph,
            null_scores,
            fractions=fractions,
            min_module_size=min_module_size,
            min_subthreshold_genes=min_subthreshold_genes,
            max_modules=max_modules,
            gwas_p_threshold=gwas_p_threshold,
            broad_component_min_size=broad_component_min_size,
            broad_component_fraction=broad_component_fraction,
        )
        if null_modules.empty:
            rows.append(
                {
                    "replicate": int(replicate),
                    "null_source": null_source,
                    "n_modules": 0,
                    "max_mean_score": float("-inf"),
                    "max_edge_density": 0.0,
                    "max_n_genes": 0,
                }
            )
            continue
        rows.append(
            {
                "replicate": int(replicate),
                "null_source": null_source,
                "n_modules": int(len(null_modules)),
                "max_mean_score": float(null_modules["mean_score"].max()),
                "max_edge_density": float(null_modules["edge_density"].max()),
                "max_n_genes": int(null_modules["n_genes"].max()),
            }
        )
    return pd.DataFrame(rows)


def _prepare_degree_bins(scores: pd.DataFrame, *, n_bins: int) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    work = scores.loc[:, ["gene_symbol", "graph_degree"]].copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str)
    work["graph_degree"] = pd.to_numeric(work["graph_degree"], errors="raise").astype(float)
    work["degree_bin"] = assign_degree_bins(work["graph_degree"], n_bins=n_bins).astype(int)
    bin_to_nodes = {
        int(bin_id): group["gene_symbol"].to_numpy(dtype=object)
        for bin_id, group in work.groupby("degree_bin", observed=True)
    }
    return bin_to_nodes, work


def calibrate_local_modules(
    graph: nx.Graph,
    scores: pd.DataFrame,
    modules: pd.DataFrame,
    *,
    fractions: Iterable[float] = RANK_FRACTION_GRID,
    min_module_size: int = 5,
    min_subthreshold_genes: int = 3,
    max_modules: int = 20,
    gwas_p_threshold: float = 5e-8,
    broad_component_min_size: int = 200,
    broad_component_fraction: float = 0.50,
    n_random: int = 200,
    n_degree_matched: int = 200,
    n_degree_graph: int = 20,
    n_selection_aware: int = 200,
    selection_null_scores: np.ndarray | None = None,
    degree_bins: int = 10,
    seed: int = 20260614,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate local modules against random, degree-matched, and graph nulls."""

    if modules.empty:
        return modules.copy(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    work = scores.copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str)
    work["assoc_resid_score"] = pd.to_numeric(work["assoc_resid_score"], errors="raise").astype(float)
    work["graph_degree"] = pd.to_numeric(work["graph_degree"], errors="raise").astype(int)
    score_by_node, _, _ = _score_maps(work)
    universe = np.array([node for node in work["gene_symbol"] if graph.has_node(node)], dtype=object)
    bin_to_nodes, degree_table = _prepare_degree_bins(work[work["gene_symbol"].isin(universe)], n_bins=degree_bins)
    selection_nulls = selection_aware_module_null(
        graph,
        work,
        selection_null_scores=selection_null_scores,
        fractions=fractions,
        min_module_size=min_module_size,
        min_subthreshold_genes=min_subthreshold_genes,
        max_modules=max_modules,
        gwas_p_threshold=gwas_p_threshold,
        broad_component_min_size=broad_component_min_size,
        broad_component_fraction=broad_component_fraction,
        n_replicates=n_selection_aware,
        degree_bins=degree_bins,
        seed=seed + 2001,
    )
    selection_score_max = (
        selection_nulls["max_mean_score"].to_numpy(dtype=float) if not selection_nulls.empty else np.array([], dtype=float)
    )
    selection_edge_max = (
        selection_nulls["max_edge_density"].to_numpy(dtype=float)
        if not selection_nulls.empty
        else np.array([], dtype=float)
    )

    graph_null_edge_density: dict[str, list[float]] = {str(row.module_id): [] for row in modules.itertuples()}
    graph_null_lcc_fraction: dict[str, list[float]] = {str(row.module_id): [] for row in modules.itertuples()}
    module_node_map = {
        str(row.module_id): tuple(str(node) for node in str(row.module_genes).split(",") if node)
        for row in modules.itertuples()
    }
    for null_graph in degree_preserving_graph_replicates(
        graph,
        n_replicates=n_degree_graph,
        seed=seed + 1001,
        nswap_per_edge=1.0,
        max_tries_per_swap=20.0,
    ):
        for module_id, nodes in module_node_map.items():
            graph_null_edge_density[module_id].append(_edge_density(null_graph, nodes))
            graph_null_lcc_fraction[module_id].append(_component_lcc_fraction(null_graph, nodes))

    calibrated_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    for row in modules.itertuples(index=False):
        module_id = str(row.module_id)
        nodes = module_node_map[module_id]
        n_nodes = len(nodes)
        observed_mean = float(row.mean_score)
        observed_edge_density = float(row.edge_density)
        observed_lcc_fraction = _component_lcc_fraction(graph, nodes)
        random_means = np.empty(n_random, dtype=float)
        degree_means = np.empty(n_degree_matched, dtype=float)
        degree_edge_density = np.empty(n_degree_matched, dtype=float)
        profile = _degree_bin_profile(degree_table, nodes, n_bins=degree_bins)
        for idx in range(n_random):
            sampled = tuple(str(node) for node in rng.choice(universe, size=n_nodes, replace=False))
            random_means[idx] = float(np.mean([score_by_node[node] for node in sampled]))
        for idx in range(n_degree_matched):
            sampled = _sample_degree_matched_nodes(bin_to_nodes, profile, rng=rng)
            degree_means[idx] = float(np.mean([score_by_node[node] for node in sampled]))
            degree_edge_density[idx] = _edge_density(graph, sampled)
        graph_edge_null = np.asarray(graph_null_edge_density[module_id], dtype=float)
        graph_lcc_null = np.asarray(graph_null_lcc_fraction[module_id], dtype=float)
        random_p = _empirical_upper(random_means, observed_mean)
        degree_matched_p = _empirical_upper(degree_means, observed_mean)
        graph_p = _empirical_upper(graph_edge_null, observed_edge_density)
        selection_score_p = _empirical_upper(selection_score_max, observed_mean)
        selection_edge_p = _empirical_upper(selection_edge_max, observed_edge_density)
        selection_available = bool(len(selection_score_max) > 0)
        is_calibrated = bool(
            degree_matched_p <= 0.05 and selection_available and selection_score_p <= 0.05
        )
        is_topology_specific = bool(is_calibrated and graph_p <= 0.05 and selection_edge_p <= 0.05)
        if is_topology_specific:
            module_claim_label = "topology_specific_module"
            interpretation_warning = ""
        elif is_calibrated:
            module_claim_label = "calibrated_weak_signal_module"
            interpretation_warning = ""
        else:
            module_claim_label = "exploratory_module"
            interpretation_warning = (
                "WARNING: module selection-aware null not available; module should be interpreted as exploratory."
                if not selection_available
                else ""
            )
        calibrated = dict(row._asdict())
        calibrated.update(
            {
                "random_score_mean_null": float(random_means.mean()),
                "random_score_z": _z_score(observed_mean, random_means),
                "empirical_p": random_p,
                "degree_matched_score_mean_null": float(degree_means.mean()),
                "degree_matched_score_z": _z_score(observed_mean, degree_means),
                "degree_matched_p": degree_matched_p,
                "selection_aware_score_max_null_mean": _finite_mean(selection_score_max),
                "selection_aware_score_z": _z_score(observed_mean, selection_score_max),
                "selection_aware_score_p": selection_score_p,
                "degree_matched_edge_density_mean_null": float(degree_edge_density.mean()),
                "degree_matched_edge_density_z": _z_score(observed_edge_density, degree_edge_density),
                "degree_matched_edge_density_p": _empirical_upper(degree_edge_density, observed_edge_density),
                "degree_preserving_edge_density_mean_null": float(graph_edge_null.mean())
                if len(graph_edge_null)
                else float("nan"),
                "edge_density_z": _z_score(observed_edge_density, graph_edge_null),
                "degree_preserving_graph_p": graph_p,
                "selection_aware_edge_density_max_null_mean": _finite_mean(selection_edge_max),
                "selection_aware_edge_density_z": _z_score(observed_edge_density, selection_edge_max),
                "selection_aware_edge_density_p": selection_edge_p,
                "observed_lcc_fraction": observed_lcc_fraction,
                "degree_preserving_lcc_fraction_mean_null": float(graph_lcc_null.mean())
                if len(graph_lcc_null)
                else float("nan"),
                "degree_preserving_lcc_fraction_p": _empirical_upper(graph_lcc_null, observed_lcc_fraction),
                "is_calibrated_weak_signal_module": is_calibrated,
                "is_topology_specific_module": is_topology_specific,
                "selection_aware_null_available": selection_available,
                "module_claim_label": module_claim_label,
                "module_interpretation_warning": interpretation_warning,
            }
        )
        calibrated_rows.append(calibrated)
        null_rows.extend(
            {
                "module_id": module_id,
                "null_type": "random_score",
                "replicate": int(idx),
                "value": float(value),
            }
            for idx, value in enumerate(random_means)
        )
        null_rows.extend(
            {
                "module_id": module_id,
                "null_type": "degree_matched_score",
                "replicate": int(idx),
                "value": float(value),
            }
            for idx, value in enumerate(degree_means)
        )
        null_rows.extend(
            {
                "module_id": module_id,
                "null_type": "degree_preserving_edge_density",
                "replicate": int(idx),
                "value": float(value),
            }
            for idx, value in enumerate(graph_edge_null)
        )
    null_rows.extend(
        {
            "module_id": "ALL_MODULE_SELECTION",
            "null_type": "selection_aware_max_mean_score",
            "replicate": int(row.replicate),
            "value": float(row.max_mean_score),
            "null_source": str(row.null_source),
        }
        for row in selection_nulls.itertuples(index=False)
    )
    null_rows.extend(
        {
            "module_id": "ALL_MODULE_SELECTION",
            "null_type": "selection_aware_max_edge_density",
            "replicate": int(row.replicate),
            "value": float(row.max_edge_density),
            "null_source": str(row.null_source),
        }
        for row in selection_nulls.itertuples(index=False)
    )
    out = pd.DataFrame(calibrated_rows)
    out = out.sort_values(
        [
            "is_calibrated_weak_signal_module",
            "selection_aware_score_p",
            "degree_matched_p",
            "empirical_p",
            "mean_score",
        ],
        ascending=[False, True, True, True, False],
    ).reset_index(drop=True)
    out["module_rank"] = np.arange(1, len(out) + 1)
    return out, pd.DataFrame(null_rows)


def load_gene_sets(path: str | Path | None = None, *, include_default_dr_panel: bool = True) -> dict[str, set[str]]:
    """Load gene sets from TSV or GMT and optionally include the default DR panel."""

    gene_sets: dict[str, set[str]] = {}
    if include_default_dr_panel:
        gene_sets.update({name: set(genes) for name, genes in DEFAULT_DR_GENE_SETS.items()})
    if path is None:
        return gene_sets
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".gmt":
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = [part.strip() for part in line.split("\t") if part.strip()]
            if len(parts) >= 3:
                gene_sets[parts[0]] = {gene.upper() for gene in parts[2:]}
        return gene_sets

    table = pd.read_csv(path, sep="\t", compression="infer")
    if {"gene_set", "gene_symbol"}.issubset(table.columns):
        for name, group in table.groupby("gene_set", observed=True):
            gene_sets[str(name)] = {str(gene).upper() for gene in group["gene_symbol"].dropna()}
        return gene_sets
    if {"set_name", "gene"}.issubset(table.columns):
        for name, group in table.groupby("set_name", observed=True):
            gene_sets[str(name)] = {str(gene).upper() for gene in group["gene"].dropna()}
        return gene_sets
    raise ValueError("Gene-set TSV must contain gene_set/gene_symbol or set_name/gene columns.")


def pathway_subgraph_tests(
    graph: nx.Graph,
    scores: pd.DataFrame,
    gene_sets: Mapping[str, set[str]],
    *,
    n_random: int = 200,
    n_degree_matched: int = 200,
    degree_bins: int = 10,
    seed: int = 20260614,
) -> pd.DataFrame:
    """Run local pathway/subgraph diagnostics with random and degree-matched nulls."""

    if not gene_sets:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    work = scores.copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str).str.upper()
    work["assoc_resid_score"] = pd.to_numeric(work["assoc_resid_score"], errors="raise").astype(float)
    work["graph_degree"] = pd.to_numeric(work["graph_degree"], errors="raise").astype(int)
    graph_upper = nx.relabel_nodes(graph, {node: str(node).upper() for node in graph.nodes()}, copy=True)
    work = work[work["gene_symbol"].isin(set(graph_upper.nodes()))].reset_index(drop=True)
    score_by_node = dict(zip(work["gene_symbol"], work["assoc_resid_score"], strict=True))
    rank_table = rank_nodes_by_score(work, node_col="gene_symbol", score_col="assoc_resid_score")
    rank_by_node = dict(zip(rank_table["gene_symbol"], rank_table["rank_fraction"], strict=True))
    universe = work["gene_symbol"].to_numpy(dtype=object)
    bin_to_nodes, degree_table = _prepare_degree_bins(work, n_bins=degree_bins)
    rows: list[dict[str, object]] = []
    for name, raw_genes in gene_sets.items():
        genes = {str(gene).upper() for gene in raw_genes}
        present = tuple(sorted(genes & set(universe)))
        if len(present) < 3:
            rows.append(
                {
                    "gene_set": str(name),
                    "n_query_genes": int(len(genes)),
                    "n_present": int(len(present)),
                    "n_missing": int(len(genes) - len(present)),
                    "mean_score": float("nan"),
                    "mean_score_z": float("nan"),
                    "mean_score_empirical_p": float("nan"),
                    "rank_enrichment_z": float("nan"),
                    "rank_enrichment_p": float("nan"),
                    "induced_edges": 0,
                    "largest_component_size": 0,
                    "largest_component_fraction": 0.0,
                    "degree_matched_empirical_p": float("nan"),
                    "present_genes": ",".join(present),
                }
            )
            continue
        observed_mean = float(np.mean([score_by_node[node] for node in present]))
        observed_rank_mean = float(np.mean([rank_by_node[node] for node in present]))
        subgraph = graph_upper.subgraph(present)
        component_sizes = [len(component) for component in nx.connected_components(subgraph)]
        largest_size = max(component_sizes, default=0)
        random_mean = np.empty(n_random, dtype=float)
        random_rank = np.empty(n_random, dtype=float)
        degree_mean = np.empty(n_degree_matched, dtype=float)
        profile = _degree_bin_profile(degree_table, present, n_bins=degree_bins)
        for idx in range(n_random):
            sampled = tuple(str(node) for node in rng.choice(universe, size=len(present), replace=False))
            random_mean[idx] = float(np.mean([score_by_node[node] for node in sampled]))
            random_rank[idx] = float(np.mean([rank_by_node[node] for node in sampled]))
        for idx in range(n_degree_matched):
            sampled = _sample_degree_matched_nodes(bin_to_nodes, profile, rng=rng)
            degree_mean[idx] = float(np.mean([score_by_node[node] for node in sampled]))
        rows.append(
            {
                "gene_set": str(name),
                "n_query_genes": int(len(genes)),
                "n_present": int(len(present)),
                "n_missing": int(len(genes) - len(present)),
                "mean_score": observed_mean,
                "mean_score_z": _z_score(observed_mean, random_mean),
                "mean_score_empirical_p": _empirical_upper(random_mean, observed_mean),
                "rank_enrichment_z": -_z_score(observed_rank_mean, random_rank),
                "rank_enrichment_p": _empirical_lower(random_rank, observed_rank_mean),
                "induced_edges": int(subgraph.number_of_edges()),
                "edge_density": _edge_density(graph_upper, present),
                "largest_component_size": int(largest_size),
                "largest_component_fraction": float(largest_size / len(present)),
                "degree_matched_mean_score_z": _z_score(observed_mean, degree_mean),
                "degree_matched_empirical_p": _empirical_upper(degree_mean, observed_mean),
                "present_genes": ",".join(present),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["degree_matched_empirical_p", "mean_score_empirical_p", "mean_score_z"],
        ascending=[True, True, False],
    )


def run_local_module_discovery(
    graph: nx.Graph,
    scores: pd.DataFrame,
    *,
    gene_sets: Mapping[str, set[str]] | None = None,
    fractions: Iterable[float] = RANK_FRACTION_GRID,
    seed: int = 20260614,
    min_module_size: int = 5,
    min_subthreshold_genes: int = 3,
    max_modules: int = 20,
    n_module_random: int = 200,
    n_module_degree_matched: int = 200,
    n_module_degree_graph: int = 20,
    n_module_selection_aware: int = 200,
    selection_null_scores: np.ndarray | None = None,
    n_pathway_random: int = 200,
    n_pathway_degree_matched: int = 200,
    degree_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run extraction, calibration, and pathway/subgraph diagnostics."""

    modules = discover_local_modules(
        graph,
        scores,
        fractions=fractions,
        min_module_size=min_module_size,
        min_subthreshold_genes=min_subthreshold_genes,
        max_modules=max_modules,
    )
    modules, module_nulls = calibrate_local_modules(
        graph,
        scores,
        modules,
        fractions=fractions,
        min_module_size=min_module_size,
        min_subthreshold_genes=min_subthreshold_genes,
        max_modules=max_modules,
        n_random=n_module_random,
        n_degree_matched=n_module_degree_matched,
        n_degree_graph=n_module_degree_graph,
        n_selection_aware=n_module_selection_aware,
        selection_null_scores=selection_null_scores,
        degree_bins=degree_bins,
        seed=seed,
    )
    selected_gene_sets = load_gene_sets() if gene_sets is None else gene_sets
    pathway = pathway_subgraph_tests(
        graph,
        scores,
        selected_gene_sets,
        n_random=n_pathway_random,
        n_degree_matched=n_pathway_degree_matched,
        degree_bins=degree_bins,
        seed=seed + 17,
    )
    return modules, module_nulls, pathway


def render_module_discovery_report(
    *,
    trait: str,
    graph_name: str,
    modules: pd.DataFrame,
    pathway: pd.DataFrame,
    global_gate_pass: bool = True,
) -> str:
    """Render a concise local module discovery report."""

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
    n_module_level = (
        int(modules.get("is_calibrated_weak_signal_module", pd.Series(dtype=bool)).sum()) if not modules.empty else 0
    )
    n_broad_calibrated = (
        int(
            (
                modules.get("is_calibrated_weak_signal_module", pd.Series(dtype=bool)).astype(bool)
                & (modules.get("n_genes", pd.Series(dtype=int)).astype(int) >= 200)
            ).sum()
        )
        if not modules.empty
        else 0
    )
    n_reportable = int(modules.get(claim_col, pd.Series(dtype=bool)).sum()) if not modules.empty else 0
    n_topology = int(modules.get(topology_col, pd.Series(dtype=bool)).sum()) if not modules.empty else 0
    lines = [
        f"# {trait} RIPPLE Local Module Discovery",
        "",
        f"- Graph: `{graph_name}`",
        f"- Candidate modules tested: {len(modules):,}",
        f"- Global gate for reportable module claims: {'pass' if global_gate_pass else 'fail'}",
        f"- Module-level calibrated candidates: {n_module_level:,}",
        f"- Calibrated broad components: {n_broad_calibrated:,}",
        f"- Reportable calibrated weak-signal modules: {n_reportable:,}",
        f"- Topology-specific modules: {n_topology:,}",
        "",
        "## Top Calibrated Modules",
        "",
    ]
    if modules.empty:
        lines.append("No candidate modules passed extraction filters.")
    else:
        display_modules = modules.loc[modules[claim_col].astype(bool)].copy()
        if display_modules.empty:
            if global_gate_pass:
                lines.append("No candidate modules passed module-level calibration; top exploratory candidates are shown.")
            else:
                lines.append(
                    "Global calibration gate failed; candidate modules are exploratory and top candidates are shown."
                )
            display_modules = modules.head(10)
        lines.extend(
            [
                "| Rank | Module | Claim | k | Mean score | Degree P | Selection P | Graph P | Selection edge P | Core genes |",
                "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in display_modules.head(10).itertuples(index=False):
            lines.append(
                "| "
                f"{int(row.module_rank)} | {row.module_id} | {getattr(row, 'module_claim_label', '')} | "
                f"{int(row.n_genes)} | "
                f"{float(row.mean_score):.3f} | {float(row.degree_matched_p):.4f} | "
                f"{float(row.selection_aware_score_p):.4f} | "
                f"{float(row.degree_preserving_graph_p):.4f} | "
                f"{float(row.selection_aware_edge_density_p):.4f} | {row.core_genes} |"
            )
    lines.extend(["", "## Pathway/Subgraph Tests", ""])
    if pathway.empty:
        lines.append("No pathway or gene-set tests were available.")
    else:
        lines.extend(
            [
                "| Gene set | n | Mean score Z | Degree-matched P | LCC fraction |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in pathway.head(10).itertuples(index=False):
            lines.append(
                "| "
                f"{row.gene_set} | {int(row.n_present)} | {float(row.mean_score_z):.3f} | "
                f"{float(row.degree_matched_empirical_p):.4f} | "
                f"{float(row.largest_component_fraction):.3f} |"
            )
    return "\n".join(lines) + "\n"
