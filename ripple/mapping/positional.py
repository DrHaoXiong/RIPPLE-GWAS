"""LD/GWAS build-oriented positional SNP-to-gene mapping."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_POSITIONAL_UPSTREAM_BP: int = 0
DEFAULT_POSITIONAL_DOWNSTREAM_BP: int = 0


def _require_columns(table: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [col for col in columns if col not in table.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def normalize_chromosome(chrom: str | int) -> str:
    """Normalize chromosome labels by removing a leading `chr` prefix."""

    value = str(chrom).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    return value


def _distance_to_gene(pos: int, gene_start: int, gene_end: int) -> int:
    if pos < gene_start:
        return int(gene_start - pos)
    if pos > gene_end:
        return int(pos - gene_end)
    return 0


def positional_map_snps_to_genes(
    snps: pd.DataFrame,
    genes: pd.DataFrame,
    *,
    upstream_bp: int = DEFAULT_POSITIONAL_UPSTREAM_BP,
    downstream_bp: int = DEFAULT_POSITIONAL_DOWNSTREAM_BP,
    snp_id_col: str = "snp_id",
    snp_chrom_col: str = "chrom",
    snp_pos_col: str = "pos",
    gene_id_col: str = "gene_id",
    gene_symbol_col: str | None = "gene_symbol",
    gene_chrom_col: str = "chrom",
    gene_start_col: str = "start",
    gene_end_col: str = "end",
) -> pd.DataFrame:
    """Map SNPs to genes by genomic interval overlap.

    The output is one row per SNP-gene positional assignment. Gene intervals are
    expanded by `upstream_bp` and `downstream_bp` before matching. The function
    is genome-build agnostic, but SNP and gene coordinates must already be on
    the same build.
    """

    if upstream_bp < 0 or downstream_bp < 0:
        raise ValueError("upstream_bp and downstream_bp must be nonnegative.")

    _require_columns(snps, [snp_id_col, snp_chrom_col, snp_pos_col], "snps")
    _require_columns(genes, [gene_id_col, gene_chrom_col, gene_start_col, gene_end_col], "genes")

    snp_work = snps.copy()
    gene_work = genes.copy()

    snp_work["_snp_order"] = np.arange(len(snp_work))
    gene_work["_gene_order"] = np.arange(len(gene_work))
    snp_work["_chrom_norm"] = snp_work[snp_chrom_col].map(normalize_chromosome)
    gene_work["_chrom_norm"] = gene_work[gene_chrom_col].map(normalize_chromosome)
    snp_work["_pos"] = pd.to_numeric(snp_work[snp_pos_col], errors="raise").astype(int)
    gene_work["_gene_start"] = pd.to_numeric(gene_work[gene_start_col], errors="raise").astype(int)
    gene_work["_gene_end"] = pd.to_numeric(gene_work[gene_end_col], errors="raise").astype(int)

    if (gene_work["_gene_start"] > gene_work["_gene_end"]).any():
        raise ValueError("Gene start positions must be <= gene end positions.")

    gene_work["_map_start"] = (gene_work["_gene_start"] - int(upstream_bp)).clip(lower=0)
    gene_work["_map_end"] = gene_work["_gene_end"] + int(downstream_bp)

    output_columns = [
        "snp_id",
        "gene_id",
        "chrom",
        "snp_pos",
        "gene_start",
        "gene_end",
        "map_start",
        "map_end",
        "distance_to_gene",
    ]
    include_symbol = gene_symbol_col is not None and gene_symbol_col in gene_work.columns
    if include_symbol:
        output_columns.insert(2, "gene_symbol")

    rows: list[dict[str, object]] = []
    for chrom, snp_chr in snp_work.groupby("_chrom_norm", sort=False):
        gene_chr = gene_work.loc[gene_work["_chrom_norm"] == chrom].copy()
        if gene_chr.empty or snp_chr.empty:
            continue

        gene_chr = gene_chr.sort_values(["_map_start", "_map_end", gene_id_col]).reset_index(drop=True)
        snp_chr = snp_chr.sort_values(["_pos", "_snp_order"])

        active: list[int] = []
        gene_pointer = 0
        starts = gene_chr["_map_start"].to_numpy()
        ends = gene_chr["_map_end"].to_numpy()

        for _, snp in snp_chr.iterrows():
            pos = int(snp["_pos"])
            while gene_pointer < len(gene_chr) and int(starts[gene_pointer]) <= pos:
                active.append(gene_pointer)
                gene_pointer += 1

            active = [idx for idx in active if int(ends[idx]) >= pos]
            if not active:
                continue

            snp_id = snp[snp_id_col]
            snp_order = int(snp["_snp_order"])
            for idx in active:
                gene = gene_chr.iloc[idx]
                row = {
                    "snp_id": snp_id,
                    "gene_id": gene[gene_id_col],
                    "chrom": chrom,
                    "snp_pos": pos,
                    "gene_start": int(gene["_gene_start"]),
                    "gene_end": int(gene["_gene_end"]),
                    "map_start": int(gene["_map_start"]),
                    "map_end": int(gene["_map_end"]),
                    "distance_to_gene": _distance_to_gene(
                        pos,
                        int(gene["_gene_start"]),
                        int(gene["_gene_end"]),
                    ),
                    "_snp_order": snp_order,
                    "_gene_order": int(gene["_gene_order"]),
                }
                if include_symbol:
                    row["gene_symbol"] = gene[gene_symbol_col]
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=output_columns)

    result = pd.DataFrame(rows)
    result = result.sort_values(["_snp_order", "_gene_order", "gene_id"]).reset_index(drop=True)
    return result.loc[:, output_columns]
