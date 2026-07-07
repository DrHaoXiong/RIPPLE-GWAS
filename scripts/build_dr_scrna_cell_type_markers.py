#!/usr/bin/env python
"""Build DR-context scRNA cell-type marker sets for RIPPLE annotation.

The output is a fixed contextual annotation resource. It is not genetic
evidence, does not encode DR-vs-control differential expression by default,
and must not be described as DR-specific cell-type validation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_H5AD_DIR = PRIVATE_ROOT / "20_processed_data" / "reference_expression" / "prisma_dr_scrna" / "h5ad"
DEFAULT_OUT_DIR = PRIVATE_ROOT / "20_processed_data" / "reference_expression" / "dr_cell_type_specificity_v0_1"


DATASETS = {
    "GSE137537_retina": {
        "path": "GSE137537_clean.h5ad",
        "title": "Human retina single-cell atlas",
        "geo_title": "Single-cell Transcriptomic Atlas of the Human Retina Identifies Cell Types Associated with Age-Related Macular Degeneration [Microfluidics]",
        "geo_summary": "Human retina single-cell atlas covering major retinal cell types from macula and peripheral retina.",
        "pubmed_id": "31653841",
        "tissue_context": "retina",
        "condition_column": "",
        "condition_scopes": ["all"],
    },
    "GSE165784_fvm": {
        "path": "GSE165784_Annotated.h5ad",
        "title": "PDR/PVR fibrovascular membrane single-cell data",
        "geo_title": "Single-cell RNA-sequencing reveals the heterogeneity of microglia in fibrous membrane derived from proliferative diabetic retinopathy and proliferative vitreoretinopathy",
        "geo_summary": "Fibrovascular membrane single-cell atlas from PDR and PVR samples with microglia, immune and stromal populations.",
        "pubmed_id": "35061025",
        "tissue_context": "fibrovascular_membrane",
        "condition_column": "Condition",
        "condition_scopes": ["PDR", "all"],
    },
    "GSE248284_pbmc": {
        "path": "GSE248284_Annotated.h5ad",
        "title": "T1D DR/NDR PBMC single-cell data",
        "geo_title": "A single cell atlas of circulating immune cells involved in diabetic retinopathy",
        "geo_summary": "PBMC scRNA-seq from six type 1 diabetes patients, including three DR and three NDR samples.",
        "pubmed_id": "38327792",
        "tissue_context": "PBMC",
        "condition_column": "Condition",
        "condition_scopes": ["DR", "all"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad-dir", type=Path, default=DEFAULT_H5AD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--matrix-key", choices=["raw.X", "X"], default="raw.X")
    parser.add_argument("--top-marker-genes", type=int, default=200)
    parser.add_argument("--min-cells", type=int, default=50)
    parser.add_argument("--min-detected-fraction", type=float, default=0.05)
    parser.add_argument("--min-log2fc", type=float, default=0.25)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def read_obs_column(obs: h5py.Group, column: str) -> np.ndarray:
    obj = obs[column]
    if isinstance(obj, h5py.Dataset):
        return np.array([decode(value) for value in obj[:]], dtype=object)
    if isinstance(obj, h5py.Group) and {"codes", "categories"}.issubset(obj.keys()):
        categories = [decode(value) for value in obj["categories"][:]]
        codes = obj["codes"][:]
        return np.array(
            [categories[int(code)] if 0 <= int(code) < len(categories) else "" for code in codes],
            dtype=object,
        )
    raise TypeError(f"Unsupported obs column encoding for {column}: {type(obj)}")


def matrix_group(handle: h5py.File, matrix_key: str) -> tuple[h5py.Group | h5py.Dataset, h5py.Group]:
    if matrix_key == "X":
        return handle["X"], handle["var"]
    return handle["raw"]["X"], handle["raw"]["var"]


def matrix_shape(matrix: h5py.Group | h5py.Dataset) -> tuple[int, int]:
    if isinstance(matrix, h5py.Dataset):
        return tuple(int(value) for value in matrix.shape)
    shape = matrix.attrs.get("shape")
    if shape is None:
        raise ValueError("Sparse matrix group is missing shape metadata.")
    return tuple(int(value) for value in shape)


def read_matrix(matrix: h5py.Group | h5py.Dataset) -> sparse.csr_matrix | np.ndarray:
    if isinstance(matrix, h5py.Dataset):
        return np.asarray(matrix)
    required = {"data", "indices", "indptr"}
    if not required.issubset(matrix.keys()):
        raise ValueError("Unsupported sparse matrix group.")
    shape = matrix_shape(matrix)
    data = matrix["data"][:]
    indices = matrix["indices"][:]
    indptr = matrix["indptr"][:]
    encoding = matrix.attrs.get("encoding-type", "csr_matrix")
    encoding = decode(encoding)
    if encoding == "csc_matrix":
        return sparse.csc_matrix((data, indices, indptr), shape=shape).tocsr()
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def read_h5ad(path: Path, matrix_key: str) -> tuple[sparse.csr_matrix | np.ndarray, list[str], pd.DataFrame]:
    with h5py.File(path, "r") as handle:
        matrix_obj, var_group = matrix_group(handle, matrix_key)
        matrix = read_matrix(matrix_obj)
        genes = [decode(value).upper() for value in var_group["_index"][:]]
        obs_group = handle["obs"]
        obs_cols = list(obs_group.keys())
        obs_data: dict[str, np.ndarray] = {}
        for column in ["Cell_Type", "Condition", "Sample_ID", "individual", "tissue", "Labels"]:
            if column in obs_cols:
                obs_data[column] = read_obs_column(obs_group, column)
        obs = pd.DataFrame(obs_data)
    if matrix.shape[1] != len(genes):
        raise ValueError(f"Matrix gene dimension does not match var names for {path}.")
    return matrix, genes, obs


def subset_stats(matrix: sparse.spmatrix | np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    n_cells = int(mask.sum())
    n_genes = int(matrix.shape[1])
    if n_cells == 0:
        return np.zeros(n_genes), np.zeros(n_genes), 0
    sub = matrix[mask, :]
    if sparse.issparse(sub):
        mean = np.asarray(sub.mean(axis=0)).ravel().astype(float)
        detected = np.asarray(sub.getnnz(axis=0)).ravel().astype(float) / float(n_cells)
    else:
        arr = np.asarray(sub)
        mean = arr.mean(axis=0).astype(float)
        detected = np.count_nonzero(arr, axis=0).astype(float) / float(n_cells)
    return mean, detected, n_cells


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def marker_metrics_for_scope(
    matrix: sparse.csr_matrix | np.ndarray,
    genes: list[str],
    obs: pd.DataFrame,
    *,
    dataset_id: str,
    title: str,
    tissue_context: str,
    condition_scope: str,
    condition_column: str,
    min_cells: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if condition_scope == "all" or not condition_column:
        scope_mask = np.ones(obs.shape[0], dtype=bool)
    else:
        scope_mask = obs[condition_column].astype(str).eq(condition_scope).to_numpy()
    scoped_obs = obs.loc[scope_mask].reset_index(drop=True)
    scoped_matrix = matrix[scope_mask, :]
    if scoped_obs.empty:
        raise ValueError(f"No cells for {dataset_id} condition_scope={condition_scope}.")
    cell_types = sorted(scoped_obs["Cell_Type"].astype(str).unique())
    rows: list[pd.DataFrame] = []
    counts = Counter(scoped_obs["Cell_Type"].astype(str))
    for cell_type in cell_types:
        target_mask = scoped_obs["Cell_Type"].astype(str).eq(cell_type).to_numpy()
        if int(target_mask.sum()) < min_cells:
            continue
        background_mask = ~target_mask
        target_mean, target_detected, n_target = subset_stats(scoped_matrix, target_mask)
        background_mean, background_detected, n_background = subset_stats(scoped_matrix, background_mask)
        log2fc = np.log2((target_mean + 1e-6) / (background_mean + 1e-6))
        specificity_rank = percentile_rank(np.maximum(log2fc, 0.0))
        expression_rank = percentile_rank(target_detected * np.log1p(target_mean))
        marker_score = 0.7 * expression_rank + 0.3 * specificity_rank
        rows.append(
            pd.DataFrame(
                {
                    "dataset_id": dataset_id,
                    "dataset_title": title,
                    "tissue_context": tissue_context,
                    "condition_scope": condition_scope,
                    "cell_type": cell_type,
                    "gene_symbol": genes,
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
    metrics = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    report = {
        "dataset_id": dataset_id,
        "condition_scope": condition_scope,
        "n_scope_cells": int(scoped_obs.shape[0]),
        "n_cell_types_total": int(len(cell_types)),
        "n_cell_types_with_min_cells": int(metrics["cell_type"].nunique()) if not metrics.empty else 0,
        "cell_type_counts": dict(counts),
    }
    return metrics, report


def select_markers(
    metrics: pd.DataFrame,
    *,
    top_marker_genes: int,
    min_detected_fraction: float,
    min_log2fc: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.DataFrame] = []
    gene_set_rows: list[dict[str, object]] = []
    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    grouped = metrics.groupby(["dataset_id", "tissue_context", "condition_scope", "cell_type"], observed=True)
    for (dataset_id, tissue_context, condition_scope, cell_type), group in grouped:
        eligible = group.loc[
            (group["target_detected_fraction"] >= min_detected_fraction)
            & (group["log2fc_target_vs_background"] >= min_log2fc)
        ].copy()
        eligible = eligible.sort_values(
            ["marker_score", "target_detected_fraction", "log2fc_target_vs_background"],
            ascending=False,
        ).head(top_marker_genes)
        if eligible.empty:
            continue
        marker_set_name = sanitize_marker_set_name(dataset_id, condition_scope, cell_type)
        eligible["marker_set"] = marker_set_name
        eligible["marker_rank"] = np.arange(1, len(eligible) + 1)
        selected_rows.append(eligible)
        for row in eligible.to_dict(orient="records"):
            gene_set_rows.append(
                {
                    "gene_set": marker_set_name,
                    "gene_symbol": row["gene_symbol"],
                    "dataset_id": dataset_id,
                    "tissue_context": tissue_context,
                    "condition_scope": condition_scope,
                    "cell_type": cell_type,
                    "marker_rank": int(row["marker_rank"]),
                    "marker_score": float(row["marker_score"]),
                    "n_target_cells": int(row["n_target_cells"]),
                    "n_background_cells": int(row["n_background_cells"]),
                    "target_detected_fraction": float(row["target_detected_fraction"]),
                    "log2fc_target_vs_background": float(row["log2fc_target_vs_background"]),
                    "annotation_source_type": "single_cell_context_support",
                    "marker_qc_status": "low_cell_support" if int(row["n_target_cells"]) < 100 else "standard_cell_support",
                    "claim_boundary": "contextual_expression_support_not_dr_specific_genetic_validation",
                }
            )
    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    return selected, pd.DataFrame(gene_set_rows)


def sanitize_marker_set_name(dataset_id: str, condition_scope: str, cell_type: str) -> str:
    text = f"{dataset_id}_{condition_scope}_{cell_type}".upper()
    return "".join(char if char.isalnum() else "_" for char in text).strip("_")


def main() -> None:
    args = parse_args()
    for subdir in ["tables", "reports"]:
        (args.out_dir / subdir).mkdir(parents=True, exist_ok=True)
    metrics_all: list[pd.DataFrame] = []
    reports: list[dict[str, object]] = []
    dataset_inventory: list[dict[str, object]] = []
    for dataset_id, spec in DATASETS.items():
        h5ad_path = args.h5ad_dir / str(spec["path"])
        matrix, genes, obs = read_h5ad(h5ad_path, args.matrix_key)
        dataset_inventory.append(
            {
                "dataset_id": dataset_id,
                "dataset_title": spec["title"],
                "geo_title": spec["geo_title"],
                "geo_summary": spec["geo_summary"],
                "pubmed_id": spec["pubmed_id"],
                "tissue_context": spec["tissue_context"],
                "h5ad_path": str(h5ad_path),
                "matrix_key": args.matrix_key,
                "n_cells": int(obs.shape[0]),
                "n_genes": int(len(genes)),
                "cell_type_column": "Cell_Type",
                "condition_column": spec["condition_column"],
                "condition_scopes": ",".join(spec["condition_scopes"]),
                "source_accession": dataset_id.split("_")[0],
                "annotation_source_type": "single_cell_context_support",
            }
        )
        for condition_scope in spec["condition_scopes"]:
            metrics, report = marker_metrics_for_scope(
                matrix,
                genes,
                obs,
                dataset_id=dataset_id,
                title=str(spec["title"]),
                tissue_context=str(spec["tissue_context"]),
                condition_scope=condition_scope,
                condition_column=str(spec["condition_column"]),
                min_cells=args.min_cells,
            )
            if not metrics.empty:
                metrics_all.append(metrics)
            reports.append(report)
    all_metrics = pd.concat(metrics_all, ignore_index=True) if metrics_all else pd.DataFrame()
    selected, gene_sets = select_markers(
        all_metrics,
        top_marker_genes=args.top_marker_genes,
        min_detected_fraction=args.min_detected_fraction,
        min_log2fc=args.min_log2fc,
    )
    marker_summary = (
        gene_sets.groupby(["gene_set", "dataset_id", "tissue_context", "condition_scope", "cell_type"], observed=True)
        .agg(
            n_marker_genes=("gene_symbol", "nunique"),
            n_target_cells=("n_target_cells", "max"),
            n_background_cells=("n_background_cells", "max"),
            marker_qc_status=("marker_qc_status", "first"),
        )
        .reset_index()
        if not gene_sets.empty
        else pd.DataFrame()
    )
    write_table(args.out_dir / "tables" / "dr_scrna_dataset_inventory.tsv", pd.DataFrame(dataset_inventory))
    write_table(args.out_dir / "tables" / "dr_scrna_cell_type_marker_metrics.tsv.gz", all_metrics)
    write_table(args.out_dir / "tables" / "dr_scrna_cell_type_markers.tsv", selected)
    write_table(args.out_dir / "tables" / "dr_scrna_cell_type_marker_gene_sets.tsv", gene_sets)
    write_table(args.out_dir / "tables" / "dr_scrna_cell_type_marker_summary.tsv", marker_summary)
    manifest = {
        "created_utc": now_utc(),
        "script_path": str(Path(__file__).resolve()),
        "out_dir": str(args.out_dir),
        "matrix_key": args.matrix_key,
        "top_marker_genes": int(args.top_marker_genes),
        "min_cells": int(args.min_cells),
        "min_detected_fraction": float(args.min_detected_fraction),
        "min_log2fc": float(args.min_log2fc),
        "n_marker_sets": int(marker_summary.shape[0]) if not marker_summary.empty else 0,
        "n_marker_gene_rows": int(gene_sets.shape[0]) if not gene_sets.empty else 0,
        "reports": reports,
        "claim_boundary": "single-cell markers provide contextual expression support only; they are not DR-specific genetic validation",
    }
    (args.out_dir / "reports" / "dr_scrna_cell_type_marker_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# DR scRNA cell-type marker resource v0.1",
        "",
        f"Created: {manifest['created_utc']}",
        "",
        "This resource provides contextual expression marker sets for DR/FVM/retina/PBMC analyses.",
        "It is not genetic validation, does not encode DR-vs-control marker evidence by default, and does not upgrade RIPPLE claim tiers.",
        "",
        "| Dataset | Scope | Cell type | Marker genes | Target cells | QC status |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in marker_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['dataset_id']} | {row['condition_scope']} | {row['cell_type']} | "
            f"{int(row['n_marker_genes'])} | {int(row['n_target_cells'])} | {row['marker_qc_status']} |"
        )
    (args.out_dir / "reports" / "dr_scrna_cell_type_marker_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote DR scRNA cell-type marker resource to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
