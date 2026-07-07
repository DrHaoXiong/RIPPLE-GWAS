"""SNP-to-gene weight construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class GeneWeightVector:
    """Weights for the SNPs mapped to one gene."""

    gene_id: str
    snp_ids: tuple[str, ...]
    weights: np.ndarray


@dataclass(frozen=True)
class MappingWeightSummary:
    """High-level mapping and multi-mapping summary."""

    n_mapping_rows: int
    n_snps: int
    n_genes: int
    n_multi_mapped_snps: int
    max_genes_per_snp: int


@dataclass(frozen=True)
class SparseMappingMatrix:
    """Sparse SNP-by-gene mapping matrix."""

    matrix: sparse.csr_matrix
    snp_ids: tuple[str, ...]
    gene_ids: tuple[str, ...]


def _require_columns(table: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [col for col in columns if col not in table.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def add_positional_weights(
    mapping: pd.DataFrame,
    *,
    snp_col: str = "snp_id",
    gene_col: str = "gene_id",
    weight_col: str = "weight",
    count_col: str = "snp_mapping_count",
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Add default positional weights `w_sg = 1 / n_genes_mapped_by_snp`.

    This is the frozen RIPPLE V1 positional-mode default.
    """

    _require_columns(mapping, [snp_col, gene_col], "mapping")
    out = mapping.copy()
    if deduplicate:
        out = out.drop_duplicates([snp_col, gene_col]).reset_index(drop=True)

    counts = out.groupby(snp_col, observed=True)[gene_col].transform("count")
    out[count_col] = counts.astype(int)
    out[weight_col] = 1.0 / counts.astype(float)
    return out


def add_split_weights_from_raw(
    mapping: pd.DataFrame,
    *,
    raw_weight_col: str,
    snp_col: str = "snp_id",
    gene_col: str = "gene_id",
    weight_col: str = "weight",
    normalize: str = "absolute_sum",
    deduplicate: bool = True,
) -> pd.DataFrame:
    """Split raw SNP-to-gene weights within each SNP.

    This helper is for later eQTL/TWAS-style weights. For signed weights,
    `normalize='absolute_sum'` preserves sign while keeping per-SNP absolute
    weight mass equal to one.
    """

    _require_columns(mapping, [snp_col, gene_col, raw_weight_col], "mapping")
    if normalize not in {"absolute_sum", "count"}:
        raise ValueError("normalize must be 'absolute_sum' or 'count'.")

    out = mapping.copy()
    if deduplicate:
        out = out.drop_duplicates([snp_col, gene_col]).reset_index(drop=True)

    raw = pd.to_numeric(out[raw_weight_col], errors="raise").astype(float)
    if normalize == "count":
        counts = out.groupby(snp_col, observed=True)[gene_col].transform("count").astype(float)
        out[weight_col] = raw / counts
    else:
        abs_sum = raw.abs().groupby(out[snp_col], observed=True).transform("sum")
        if (abs_sum <= 0).any():
            raise ValueError("raw weights must have nonzero absolute sum within each SNP.")
        out[weight_col] = raw / abs_sum
    return out


def summarize_mapping(
    mapping: pd.DataFrame,
    *,
    snp_col: str = "snp_id",
    gene_col: str = "gene_id",
) -> MappingWeightSummary:
    """Summarize SNP-to-gene mapping multiplicity."""

    _require_columns(mapping, [snp_col, gene_col], "mapping")
    pairs = mapping.drop_duplicates([snp_col, gene_col])
    counts = pairs.groupby(snp_col, observed=True)[gene_col].nunique()
    return MappingWeightSummary(
        n_mapping_rows=int(len(pairs)),
        n_snps=int(pairs[snp_col].nunique()),
        n_genes=int(pairs[gene_col].nunique()),
        n_multi_mapped_snps=int(np.sum(counts > 1)),
        max_genes_per_snp=int(counts.max()) if not counts.empty else 0,
    )


def weights_for_gene(
    mapping: pd.DataFrame,
    gene_id: str,
    *,
    snp_col: str = "snp_id",
    gene_col: str = "gene_id",
    weight_col: str = "weight",
    snp_order: Iterable[str] | None = None,
) -> GeneWeightVector:
    """Return SNP IDs and weights mapped to one gene.

    Duplicate SNP-gene rows are summed to produce one weight per SNP.
    """

    _require_columns(mapping, [snp_col, gene_col, weight_col], "mapping")
    subset = mapping.loc[mapping[gene_col].astype(str) == str(gene_id), [snp_col, weight_col]].copy()
    if subset.empty:
        return GeneWeightVector(gene_id=str(gene_id), snp_ids=(), weights=np.array([], dtype=float))

    subset[weight_col] = pd.to_numeric(subset[weight_col], errors="raise").astype(float)
    grouped = subset.groupby(snp_col, observed=True, sort=False)[weight_col].sum()

    if snp_order is not None:
        order = [str(snp) for snp in snp_order]
        grouped.index = grouped.index.astype(str)
        grouped = grouped.reindex([snp for snp in order if snp in grouped.index])

    return GeneWeightVector(
        gene_id=str(gene_id),
        snp_ids=tuple(str(snp) for snp in grouped.index),
        weights=grouped.to_numpy(dtype=float),
    )


def mapping_to_sparse_matrix(
    mapping: pd.DataFrame,
    *,
    snp_col: str = "snp_id",
    gene_col: str = "gene_id",
    weight_col: str = "weight",
    snp_ids: Iterable[str] | None = None,
    gene_ids: Iterable[str] | None = None,
) -> SparseMappingMatrix:
    """Return sparse SNP-by-gene mapping matrix `M_sg`."""

    _require_columns(mapping, [snp_col, gene_col, weight_col], "mapping")
    out = mapping.copy()
    out[weight_col] = pd.to_numeric(out[weight_col], errors="raise").astype(float)

    snp_index = tuple(str(x) for x in (snp_ids if snp_ids is not None else out[snp_col].drop_duplicates()))
    gene_index = tuple(str(x) for x in (gene_ids if gene_ids is not None else out[gene_col].drop_duplicates()))
    snp_lookup = {snp: i for i, snp in enumerate(snp_index)}
    gene_lookup = {gene: i for i, gene in enumerate(gene_index)}

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for record in out.itertuples(index=False):
        snp = str(getattr(record, snp_col))
        gene = str(getattr(record, gene_col))
        if snp not in snp_lookup or gene not in gene_lookup:
            continue
        rows.append(snp_lookup[snp])
        cols.append(gene_lookup[gene])
        data.append(float(getattr(record, weight_col)))

    matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(len(snp_index), len(gene_index)),
        dtype=float,
    ).tocsr()
    return SparseMappingMatrix(matrix=matrix, snp_ids=snp_index, gene_ids=gene_index)
