#!/usr/bin/env python
"""Build a retina-expression-filtered STRING graph for RIPPLE sensitivity.

This is a labeled reference-graph construction script, not a GWAS analysis
script. It extracts retina-expressed genes from a processed h5ad file, filters
STRING gene-level physical edges to those genes, and writes a canonical
`node1,node2,weight` edge list for `run_trait_ld_analysis.py --graph-edge-list`.
"""

from __future__ import annotations

import argparse
import json
import sys
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
from run_height_mvp import DEFAULT_STRING_INFO, DEFAULT_STRING_LINKS, write_table  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_RETINA_H5AD = (
    PRIVATE_ROOT
    / "20_processed_data"
    / "reference_expression"
    / "prisma_dr_scrna"
    / "h5ad"
    / "GSE137537_clean.h5ad"
)
DEFAULT_OUT_DIR = PRIVATE_ROOT / "20_processed_data" / "reference_graphs" / "retina_string_filtered"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retina-h5ad", type=Path, default=DEFAULT_RETINA_H5AD)
    parser.add_argument("--matrix-key", choices=["X", "raw.X"], default="X")
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-detected-cells", type=int, default=20)
    parser.add_argument("--min-detected-fraction", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    for child in ("tables", "reports"):
        (path / child).mkdir(parents=True, exist_ok=True)


def _decode(values: np.ndarray) -> list[str]:
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="replace"))
        else:
            out.append(str(value))
    return out


def _matrix_group(file_handle: h5py.File, matrix_key: str) -> tuple[h5py.Group | h5py.Dataset, h5py.Group]:
    if matrix_key == "X":
        return file_handle["X"], file_handle["var"]
    return file_handle["raw"]["X"], file_handle["raw"]["var"]


def _matrix_shape(matrix: h5py.Group | h5py.Dataset) -> tuple[int, int]:
    if isinstance(matrix, h5py.Dataset):
        return tuple(int(x) for x in matrix.shape)
    shape = matrix.attrs.get("shape")
    if shape is None:
        raise ValueError("Sparse h5ad matrix is missing shape metadata.")
    return tuple(int(x) for x in shape)


def _read_sparse_or_dense(matrix: h5py.Group | h5py.Dataset) -> sparse.spmatrix | np.ndarray:
    if isinstance(matrix, h5py.Dataset):
        return np.asarray(matrix)
    required = {"data", "indices", "indptr"}
    if not required.issubset(matrix.keys()):
        raise ValueError(f"Unsupported h5ad matrix group; missing {sorted(required - set(matrix.keys()))}")
    shape = _matrix_shape(matrix)
    data = matrix["data"][:]
    indices = matrix["indices"][:]
    indptr = matrix["indptr"][:]
    encoding = matrix.attrs.get("encoding-type", b"csr_matrix")
    if isinstance(encoding, bytes):
        encoding = encoding.decode("utf-8", errors="replace")
    if encoding == "csc_matrix":
        return sparse.csc_matrix((data, indices, indptr), shape=shape)
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def extract_expression_universe(
    h5ad_path: Path,
    *,
    matrix_key: str,
    min_detected_cells: int,
    min_detected_fraction: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return per-gene retina detection metrics and an extraction report."""

    if not h5ad_path.exists():
        raise FileNotFoundError(h5ad_path)
    with h5py.File(h5ad_path, "r") as handle:
        matrix_obj, var_group = _matrix_group(handle, matrix_key)
        matrix = _read_sparse_or_dense(matrix_obj)
        shape = _matrix_shape(matrix_obj)
        if "_index" not in var_group:
            raise ValueError(f"{matrix_key} var table is missing _index.")
        genes = _decode(var_group["_index"][:])

    if len(genes) != shape[1]:
        raise ValueError(f"Gene count mismatch: {len(genes)} var names for matrix shape {shape}.")
    n_cells = int(shape[0])
    threshold = max(int(min_detected_cells), int(np.ceil(float(min_detected_fraction) * n_cells)))
    if sparse.issparse(matrix):
        detected = np.asarray(matrix.getnnz(axis=0)).ravel().astype(int)
    else:
        detected = np.count_nonzero(np.asarray(matrix), axis=0).astype(int)

    expression = pd.DataFrame(
        {
            "gene_symbol": genes,
            "n_detected_cells": detected,
            "detected_fraction": detected / float(n_cells),
        }
    )
    expression["is_retina_expressed"] = expression["n_detected_cells"] >= threshold
    expression = expression.sort_values(["is_retina_expressed", "n_detected_cells", "gene_symbol"], ascending=[False, False, True])
    report = {
        "source_h5ad": str(h5ad_path),
        "matrix_key": matrix_key,
        "n_cells": n_cells,
        "n_genes": int(len(expression)),
        "min_detected_cells": int(min_detected_cells),
        "min_detected_fraction": float(min_detected_fraction),
        "effective_min_detected_cells": int(threshold),
        "n_retina_expressed_genes": int(expression["is_retina_expressed"].sum()),
    }
    return expression.reset_index(drop=True), report


def build_retina_string_graph(
    expression: pd.DataFrame,
    *,
    string_links: Path,
    string_info: Path,
    string_min_score: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    string_graph = read_string_gene_graph(string_links, string_info, min_score=string_min_score)
    expressed = set(expression.loc[expression["is_retina_expressed"], "gene_symbol"].astype(str))
    edges = string_graph.edges.copy()
    before_nodes = set(edges["node1"]).union(set(edges["node2"])) if not edges.empty else set()
    filtered = edges.loc[edges["node1"].isin(expressed) & edges["node2"].isin(expressed)].copy()
    after_nodes = set(filtered["node1"]).union(set(filtered["node2"])) if not filtered.empty else set()
    report = {
        "graph_name": "retina_string_filtered",
        "string_min_score": int(string_min_score),
        "string_graph_load_report": asdict(string_graph.report),
        "n_string_nodes_before_filter": int(len(before_nodes)),
        "n_string_edges_before_filter": int(len(edges)),
        "n_retina_expressed_genes_in_string": int(len(before_nodes & expressed)),
        "n_retina_filtered_nodes": int(len(after_nodes)),
        "n_retina_filtered_edges": int(len(filtered)),
        "retina_expressed_gene_fraction_in_string": float(len(before_nodes & expressed) / len(before_nodes)) if before_nodes else 0.0,
        "retina_filtered_edge_fraction": float(len(filtered) / len(edges)) if len(edges) else 0.0,
    }
    return filtered.loc[:, ["node1", "node2", "weight"]].sort_values(["node1", "node2"]).reset_index(drop=True), report


def render_report(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# retina_string_filtered Graph Construction",
            "",
            "This is a RIPPLE reference-graph sensitivity artifact. It filters STRING physical edges to genes expressed in the retina reference h5ad.",
            "",
            "## Inputs",
            "",
            f"- Retina h5ad: `{summary['expression_report']['source_h5ad']}`",
            f"- Matrix key: `{summary['expression_report']['matrix_key']}`",
            f"- STRING min score: `{summary['graph_report']['string_min_score']}`",
            "",
            "## Expression Universe",
            "",
            f"- Cells: {summary['expression_report']['n_cells']:,}",
            f"- Genes tested: {summary['expression_report']['n_genes']:,}",
            f"- Effective detection threshold: {summary['expression_report']['effective_min_detected_cells']:,} cells",
            f"- Retina-expressed genes: {summary['expression_report']['n_retina_expressed_genes']:,}",
            "",
            "## Filtered Graph",
            "",
            f"- STRING nodes before filtering: {summary['graph_report']['n_string_nodes_before_filter']:,}",
            f"- STRING edges before filtering: {summary['graph_report']['n_string_edges_before_filter']:,}",
            f"- Retina-expressed STRING nodes: {summary['graph_report']['n_retina_expressed_genes_in_string']:,}",
            f"- Filtered graph nodes: {summary['graph_report']['n_retina_filtered_nodes']:,}",
            f"- Filtered graph edges: {summary['graph_report']['n_retina_filtered_edges']:,}",
            f"- Edge retention fraction: {summary['graph_report']['retina_filtered_edge_fraction']:.4f}",
            "",
            "## Intended Downstream Use",
            "",
            "Use this edge list with `run_trait_ld_analysis.py --graph-name retina_string_filtered --graph-edge-list <edge-list>`.",
            "The key diagnostic is whether DR_MVP degree-preserving graph Z improves relative to default STRING.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"

    expression, expression_report = extract_expression_universe(
        args.retina_h5ad,
        matrix_key=args.matrix_key,
        min_detected_cells=args.min_detected_cells,
        min_detected_fraction=args.min_detected_fraction,
    )
    edge_list, graph_report = build_retina_string_graph(
        expression,
        string_links=args.string_links,
        string_info=args.string_info,
        string_min_score=args.string_min_score,
    )
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "expression_report": expression_report,
        "graph_report": graph_report,
        "outputs": {
            "expression_universe": str(tables_dir / "retina_expression_universe.tsv.gz"),
            "edge_list": str(tables_dir / "retina_string_filtered.edges.tsv.gz"),
            "summary": str(reports_dir / "retina_string_filtered.summary.json"),
            "report": str(reports_dir / "retina_string_filtered.report.md"),
        },
    }

    write_table(tables_dir / "retina_expression_universe.tsv.gz", expression)
    write_table(tables_dir / "retina_string_filtered.edges.tsv.gz", edge_list)
    (reports_dir / "retina_string_filtered.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (reports_dir / "retina_string_filtered.report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
