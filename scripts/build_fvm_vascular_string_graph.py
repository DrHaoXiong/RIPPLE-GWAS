#!/usr/bin/env python
"""Build FVM vascular-supported STRING graphs and marker gene sets.

This is a labeled reference-graph construction script for RIPPLE graph
sensitivity. It uses PRISMA FVM scRNA data to estimate pathologic vascular gene
support from PDR endothelial, pericyte/SMC, and fibroblast cells.

Outputs include:
- a weighted STRING graph that preserves all STRING topology;
- a high-support topology graph retaining the strongest FVM vascular edges;
- FVM-derived marker gene sets for pathway/subgraph testing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.io.graph import read_string_gene_graph  # noqa: E402
from ripple.modules.discovery import DEFAULT_DR_GENE_SETS  # noqa: E402
from run_height_mvp import DEFAULT_STRING_INFO, DEFAULT_STRING_LINKS, write_table  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_FVM_H5AD = (
    PRIVATE_ROOT
    / "20_processed_data"
    / "reference_expression"
    / "prisma_dr_scrna"
    / "h5ad"
    / "GSE165784_Annotated.h5ad"
)
DEFAULT_OUT_DIR = PRIVATE_ROOT / "20_processed_data" / "reference_graphs" / "fvm_vascular_string"
DEFAULT_TARGET_CELL_TYPES = ("Endothelial", "Pericyte_SMC", "Fibroblast")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fvm-h5ad", type=Path, default=DEFAULT_FVM_H5AD)
    parser.add_argument("--matrix-key", choices=["X", "raw.X"], default="raw.X")
    parser.add_argument("--condition", default="PDR")
    parser.add_argument("--target-cell-types", nargs="+", default=list(DEFAULT_TARGET_CELL_TYPES))
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--edge-min-weight-factor", type=float, default=0.25)
    parser.add_argument("--high-support-retain-fraction", type=float, default=0.75)
    parser.add_argument("--top-marker-genes", type=int, default=300)
    parser.add_argument("--min-marker-detected-fraction", type=float, default=0.05)
    parser.add_argument("--min-marker-log2fc", type=float, default=0.25)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    for child in ("tables", "reports"):
        (path / child).mkdir(parents=True, exist_ok=True)


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def read_obs_column(obs: h5py.Group, column: str) -> np.ndarray:
    if column not in obs:
        raise ValueError(f"obs is missing required column: {column}")
    obj = obs[column]
    if isinstance(obj, h5py.Dataset):
        return np.array([_decode(value) for value in obj[:]], dtype=object)
    if isinstance(obj, h5py.Group) and {"codes", "categories"}.issubset(obj.keys()):
        categories = [_decode(value) for value in obj["categories"][:]]
        codes = obj["codes"][:]
        return np.array(
            [categories[int(code)] if 0 <= int(code) < len(categories) else None for code in codes],
            dtype=object,
        )
    raise TypeError(f"Unsupported obs column encoding for {column}: {type(obj)}")


def matrix_group(file_handle: h5py.File, matrix_key: str) -> tuple[h5py.Group | h5py.Dataset, h5py.Group]:
    if matrix_key == "X":
        return file_handle["X"], file_handle["var"]
    return file_handle["raw"]["X"], file_handle["raw"]["var"]


def matrix_shape(matrix: h5py.Group | h5py.Dataset) -> tuple[int, int]:
    if isinstance(matrix, h5py.Dataset):
        return tuple(int(value) for value in matrix.shape)
    shape = matrix.attrs.get("shape")
    if shape is None:
        raise ValueError("Sparse h5ad matrix is missing shape metadata.")
    return tuple(int(value) for value in shape)


def read_sparse_or_dense(matrix: h5py.Group | h5py.Dataset) -> sparse.spmatrix | np.ndarray:
    if isinstance(matrix, h5py.Dataset):
        return np.asarray(matrix)
    if not {"data", "indices", "indptr"}.issubset(matrix.keys()):
        raise ValueError("Unsupported sparse h5ad matrix group.")
    shape = matrix_shape(matrix)
    data = matrix["data"][:]
    indices = matrix["indices"][:]
    indptr = matrix["indptr"][:]
    encoding = matrix.attrs.get("encoding-type", b"csr_matrix")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8", errors="replace")
    if encoding == "csc_matrix":
        return sparse.csc_matrix((data, indices, indptr), shape=shape).tocsr()
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def read_h5ad_inputs(path: Path, matrix_key: str) -> tuple[sparse.spmatrix | np.ndarray, list[str], pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        matrix_obj, var_group = matrix_group(handle, matrix_key)
        shape = matrix_shape(matrix_obj)
        matrix = read_sparse_or_dense(matrix_obj)
        if "_index" not in var_group:
            raise ValueError(f"{matrix_key} var table is missing _index.")
        genes = [_decode(value) for value in var_group["_index"][:]]
        obs = pd.DataFrame(
            {
                "Cell_Type": read_obs_column(handle["obs"], "Cell_Type"),
                "Condition": read_obs_column(handle["obs"], "Condition"),
                "Sample_ID": read_obs_column(handle["obs"], "Sample_ID"),
            }
        )
    if shape[1] != len(genes):
        raise ValueError("Gene count does not match h5ad matrix shape.")
    return matrix, genes, obs


def subset_stats(matrix: sparse.spmatrix | np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    n_cells = int(mask.sum())
    if n_cells == 0:
        n_genes = matrix.shape[1]
        return np.zeros(n_genes), np.zeros(n_genes), 0
    sub = matrix[mask, :]
    if sparse.issparse(sub):
        mean = np.asarray(sub.mean(axis=0)).ravel().astype(float)
        detected_fraction = np.asarray(sub.getnnz(axis=0)).ravel().astype(float) / float(n_cells)
    else:
        arr = np.asarray(sub)
        mean = arr.mean(axis=0).astype(float)
        detected_fraction = np.count_nonzero(arr, axis=0).astype(float) / float(n_cells)
    return mean, detected_fraction, n_cells


def percentile_rank(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def compute_gene_support(
    matrix: sparse.spmatrix | np.ndarray,
    genes: list[str],
    obs: pd.DataFrame,
    *,
    condition: str,
    target_cell_types: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    condition_mask = np.ones(len(obs), dtype=bool) if condition.lower() == "all" else (obs["Condition"].astype(str) == condition).to_numpy()
    selected_obs = obs.loc[condition_mask].reset_index(drop=True)
    if selected_obs.empty:
        raise ValueError(f"No FVM cells match condition={condition!r}.")

    metrics_by_group: list[pd.DataFrame] = []
    group_counts: dict[str, int] = {}
    matrix_condition = matrix[condition_mask, :]
    union_mask = selected_obs["Cell_Type"].isin(target_cell_types).to_numpy()
    groups = {cell_type: selected_obs["Cell_Type"].astype(str).eq(cell_type).to_numpy() for cell_type in target_cell_types}
    groups["vascular_union"] = union_mask

    for group_name, target_mask in groups.items():
        background_mask = ~target_mask
        target_mean, target_detected, n_target = subset_stats(matrix_condition, target_mask)
        background_mean, background_detected, n_background = subset_stats(matrix_condition, background_mask)
        log2fc = np.log2((target_mean + 1e-6) / (background_mean + 1e-6))
        positive_log2fc = np.maximum(log2fc, 0.0)
        expression_rank = percentile_rank(target_detected * np.log1p(target_mean))
        specificity_rank = percentile_rank(positive_log2fc)
        marker_score = 0.7 * expression_rank + 0.3 * specificity_rank
        group_counts[group_name] = int(n_target)
        metrics_by_group.append(
            pd.DataFrame(
                {
                    "gene_symbol": genes,
                    "support_group": group_name,
                    "n_target_cells": int(n_target),
                    "n_background_cells": int(n_background),
                    "target_mean": target_mean,
                    "background_mean": background_mean,
                    "target_detected_fraction": target_detected,
                    "background_detected_fraction": background_detected,
                    "log2fc_target_vs_background": log2fc,
                    "marker_score": marker_score,
                }
            )
        )

    long_metrics = pd.concat(metrics_by_group, ignore_index=True)
    union = long_metrics.loc[long_metrics["support_group"] == "vascular_union"].copy()
    union = union.rename(
        columns={
            "target_mean": "vascular_target_mean",
            "background_mean": "vascular_background_mean",
            "target_detected_fraction": "vascular_detected_fraction",
            "background_detected_fraction": "nonvascular_detected_fraction",
            "log2fc_target_vs_background": "vascular_log2fc",
            "marker_score": "vascular_support_score",
        }
    )
    gene_support = union.loc[
        :,
        [
            "gene_symbol",
            "vascular_target_mean",
            "vascular_background_mean",
            "vascular_detected_fraction",
            "nonvascular_detected_fraction",
            "vascular_log2fc",
            "vascular_support_score",
        ],
    ].copy()
    report = {
        "condition": condition,
        "n_condition_cells": int(len(selected_obs)),
        "target_cell_types": target_cell_types,
        "cell_type_counts_in_condition": dict(Counter(selected_obs["Cell_Type"].astype(str))),
        "target_group_counts": group_counts,
    }
    return gene_support, long_metrics, report


def select_marker_gene_sets(
    long_metrics: pd.DataFrame,
    *,
    top_n: int,
    min_detected_fraction: float,
    min_log2fc: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_name, group in long_metrics.groupby("support_group", observed=True):
        candidates = group.loc[
            (group["target_detected_fraction"] >= min_detected_fraction)
            & (group["log2fc_target_vs_background"] >= min_log2fc)
        ].copy()
        candidates = candidates.sort_values(
            ["marker_score", "target_detected_fraction", "log2fc_target_vs_background"],
            ascending=False,
        ).head(top_n)
        set_name = f"fvm_{str(group_name).lower()}_markers"
        for rank, row in enumerate(candidates.itertuples(index=False), start=1):
            rows.append(
                {
                    "gene_set": set_name,
                    "gene_symbol": str(row.gene_symbol).upper(),
                    "rank": rank,
                    "support_group": str(group_name),
                    "marker_score": float(row.marker_score),
                    "target_detected_fraction": float(row.target_detected_fraction),
                    "log2fc_target_vs_background": float(row.log2fc_target_vs_background),
                }
            )

    union = long_metrics.loc[long_metrics["support_group"] == "vascular_union"].copy()
    ecm_genes = {gene.upper() for gene in DEFAULT_DR_GENE_SETS["ecm_basement_membrane"]}
    ecm = union.loc[
        union["gene_symbol"].astype(str).str.upper().isin(ecm_genes)
        & (union["target_detected_fraction"] > 0)
    ].sort_values(["marker_score", "target_detected_fraction"], ascending=False)
    for rank, row in enumerate(ecm.itertuples(index=False), start=1):
        rows.append(
            {
                "gene_set": "fvm_ecm_basement_membrane_supported",
                "gene_symbol": str(row.gene_symbol).upper(),
                "rank": rank,
                "support_group": "vascular_union",
                "marker_score": float(row.marker_score),
                "target_detected_fraction": float(row.target_detected_fraction),
                "log2fc_target_vs_background": float(row.log2fc_target_vs_background),
            }
        )
    return pd.DataFrame(rows)


def build_graphs(
    gene_support: pd.DataFrame,
    *,
    string_links: Path,
    string_info: Path,
    string_min_score: int,
    edge_min_weight_factor: float,
    high_support_retain_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not 0 <= edge_min_weight_factor <= 1:
        raise ValueError("--edge-min-weight-factor must be in [0, 1].")
    if not 0 < high_support_retain_fraction <= 1:
        raise ValueError("--high-support-retain-fraction must be in (0, 1].")

    string_graph = read_string_gene_graph(string_links, string_info, min_score=string_min_score)
    edges = string_graph.edges.copy()
    support = gene_support.set_index("gene_symbol")["vascular_support_score"].astype(float).to_dict()
    edges["node1_support"] = edges["node1"].map(support).fillna(0.0).astype(float)
    edges["node2_support"] = edges["node2"].map(support).fillna(0.0).astype(float)
    edges["edge_vascular_support"] = np.sqrt(edges["node1_support"] * edges["node2_support"])
    edges["string_weight"] = pd.to_numeric(edges["weight"], errors="raise").astype(float)
    edges["weight"] = edges["string_weight"] * (
        float(edge_min_weight_factor) + (1.0 - float(edge_min_weight_factor)) * edges["edge_vascular_support"]
    )
    weighted_edges = edges.loc[:, ["node1", "node2", "weight", "string_weight", "edge_vascular_support"]].copy()

    positive_support = edges.loc[edges["edge_vascular_support"] > 0, "edge_vascular_support"]
    threshold = (
        float(positive_support.quantile(1.0 - high_support_retain_fraction))
        if len(positive_support)
        else float("inf")
    )
    high_support = edges.loc[
        (edges["edge_vascular_support"] > 0) & (edges["edge_vascular_support"] >= threshold)
    ].copy()
    high_support_edges = high_support.loc[:, ["node1", "node2", "weight", "string_weight", "edge_vascular_support"]].copy()

    all_nodes = set(edges["node1"]).union(set(edges["node2"]))
    high_nodes = set(high_support_edges["node1"]).union(set(high_support_edges["node2"])) if not high_support_edges.empty else set()
    report = {
        "string_min_score": int(string_min_score),
        "string_graph_load_report": asdict(string_graph.report),
        "edge_min_weight_factor": float(edge_min_weight_factor),
        "high_support_retain_fraction": float(high_support_retain_fraction),
        "high_support_retain_fraction_denominator": "positive_edge_vascular_support_edges",
        "high_support_edge_threshold": threshold,
        "n_weighted_edges": int(len(weighted_edges)),
        "n_weighted_nodes": int(len(all_nodes)),
        "n_high_support_edges": int(len(high_support_edges)),
        "n_high_support_nodes": int(len(high_nodes)),
        "high_support_edge_fraction": float(len(high_support_edges) / len(edges)) if len(edges) else 0.0,
        "high_support_positive_edge_fraction": float(len(high_support_edges) / len(positive_support))
        if len(positive_support)
        else 0.0,
        "high_support_node_fraction": float(len(high_nodes) / len(all_nodes)) if all_nodes else 0.0,
        "n_edges_with_nonzero_support": int((edges["edge_vascular_support"] > 0).sum()),
    }
    return weighted_edges, high_support_edges, report


def render_report(summary: dict[str, object]) -> str:
    graph = summary["graph_report"]
    support = summary["support_report"]
    return "\n".join(
        [
            "# FVM Vascular STRING Graph Construction",
            "",
            "This is a RIPPLE reference-graph sensitivity artifact derived from PRISMA FVM scRNA data.",
            "",
            "## Inputs",
            "",
            f"- FVM h5ad: `{summary['source_h5ad']}`",
            f"- Matrix key: `{summary['matrix_key']}`",
            f"- Condition: `{support['condition']}`",
            f"- Target cell types: `{', '.join(support['target_cell_types'])}`",
            f"- STRING min score: `{graph['string_min_score']}`",
            "",
            "## Cell Counts",
            "",
            f"- Condition cells: {support['n_condition_cells']:,}",
            f"- Endothelial cells: {support['target_group_counts'].get('Endothelial', 0):,}",
            f"- Pericyte_SMC cells: {support['target_group_counts'].get('Pericyte_SMC', 0):,}",
            f"- Fibroblast cells: {support['target_group_counts'].get('Fibroblast', 0):,}",
            f"- Vascular union cells: {support['target_group_counts'].get('vascular_union', 0):,}",
            "",
            "## Graph Outputs",
            "",
            f"- Weighted graph edges: {graph['n_weighted_edges']:,}",
            f"- Weighted graph nodes: {graph['n_weighted_nodes']:,}",
            f"- High-support graph edges: {graph['n_high_support_edges']:,}",
            f"- High-support graph nodes: {graph['n_high_support_nodes']:,}",
            f"- High-support edge fraction: {graph['high_support_edge_fraction']:.3f}",
            f"- High-support node fraction: {graph['high_support_node_fraction']:.3f}",
            "",
            "## Interpretation",
            "",
            "The weighted graph preserves STRING topology and changes only edge weights.",
            "Current RIPPLE percolation uses topology, so topology-specific claims should be tested with the high-support topology graph.",
            "The marker gene-set TSV can be passed to `run_trait_ld_analysis.py --gene-set-file` for FVM endothelial/mural/ECM subgraph diagnostics.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"

    matrix, genes, obs = read_h5ad_inputs(args.fvm_h5ad, args.matrix_key)
    gene_support, long_metrics, support_report = compute_gene_support(
        matrix,
        genes,
        obs,
        condition=args.condition,
        target_cell_types=[str(cell_type) for cell_type in args.target_cell_types],
    )
    marker_sets = select_marker_gene_sets(
        long_metrics,
        top_n=args.top_marker_genes,
        min_detected_fraction=args.min_marker_detected_fraction,
        min_log2fc=args.min_marker_log2fc,
    )
    weighted_edges, high_support_edges, graph_report = build_graphs(
        gene_support,
        string_links=args.string_links,
        string_info=args.string_info,
        string_min_score=args.string_min_score,
        edge_min_weight_factor=args.edge_min_weight_factor,
        high_support_retain_fraction=args.high_support_retain_fraction,
    )

    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source_h5ad": str(args.fvm_h5ad),
        "matrix_key": args.matrix_key,
        "support_report": support_report,
        "graph_report": graph_report,
        "marker_parameters": {
            "top_marker_genes": int(args.top_marker_genes),
            "min_marker_detected_fraction": float(args.min_marker_detected_fraction),
            "min_marker_log2fc": float(args.min_marker_log2fc),
        },
        "outputs": {
            "gene_support": str(tables_dir / "fvm_vascular_gene_support.tsv.gz"),
            "long_support_metrics": str(tables_dir / "fvm_vascular_support_by_group.tsv.gz"),
            "marker_gene_sets": str(tables_dir / "fvm_vascular_markers.gene_sets.tsv"),
            "weighted_edges": str(tables_dir / "fvm_vascular_weighted_string.edges.tsv.gz"),
            "high_support_edges": str(tables_dir / "fvm_vascular_high_support_string.edges.tsv.gz"),
            "summary": str(reports_dir / "fvm_vascular_string.summary.json"),
            "report": str(reports_dir / "fvm_vascular_string.report.md"),
        },
    }

    write_table(tables_dir / "fvm_vascular_gene_support.tsv.gz", gene_support)
    write_table(tables_dir / "fvm_vascular_support_by_group.tsv.gz", long_metrics)
    write_table(tables_dir / "fvm_vascular_weighted_string.edges.tsv.gz", weighted_edges)
    write_table(tables_dir / "fvm_vascular_high_support_string.edges.tsv.gz", high_support_edges)
    marker_sets.to_csv(tables_dir / "fvm_vascular_markers.gene_sets.tsv", sep="\t", index=False)
    (reports_dir / "fvm_vascular_string.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (reports_dir / "fvm_vascular_string.report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
