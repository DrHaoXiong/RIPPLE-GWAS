"""Gene coordinate, mappability, and special-region annotation IO."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


CANONICAL_GENE_COLUMNS: tuple[str, ...] = (
    "gene_id",
    "gene_symbol",
    "chrom",
    "start",
    "end",
    "strand",
)

GENE_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "gene_id": ("gene_id", "entrez_id", "entrez", "ensembl_gene_id", "id"),
    "gene_symbol": ("gene_symbol", "symbol", "gene_name", "name"),
    "chrom": ("chrom", "chr", "chromosome", "seqname"),
    "start": ("start", "tx_start", "gene_start"),
    "end": ("end", "tx_end", "gene_end", "stop"),
    "strand": ("strand",),
    "mappability": ("mappability", "map_score", "gene_mappability"),
}


@dataclass(frozen=True)
class GeneAnnotationReport:
    """Summary of gene annotation loading."""

    n_rows_input: int
    n_rows_output: int
    genome_build: str | None
    source_format: str
    dropped_invalid_intervals: int
    duplicated_gene_ids: int


@dataclass(frozen=True)
class GeneAnnotations:
    """Canonical gene annotation table and report."""

    table: pd.DataFrame
    report: GeneAnnotationReport


def normalize_chromosome(chrom: str | int) -> str:
    """Normalize chromosome labels by removing a leading `chr` prefix."""

    value = str(chrom).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    return value


def _normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def infer_gene_column_map(
    columns: Iterable[str],
    *,
    aliases: dict[str, tuple[str, ...]] = GENE_COLUMN_ALIASES,
) -> dict[str, str]:
    """Infer canonical gene annotation columns from common aliases."""

    normalized_to_original = {_normalize_column_name(col): str(col) for col in columns}
    column_map: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            normalized = _normalize_column_name(candidate)
            if normalized in normalized_to_original:
                column_map[canonical] = normalized_to_original[normalized]
                break
    return column_map


def _standardize_gene_table(
    table: pd.DataFrame,
    *,
    column_map: dict[str, str],
    source_format: str,
    genome_build: str | None = None,
    drop_duplicate_gene_ids: bool = True,
) -> GeneAnnotations:
    required = ("gene_id", "chrom", "start", "end")
    missing = [col for col in required if col not in column_map]
    if missing:
        raise ValueError(f"Missing required gene annotation columns: {missing}")

    out = pd.DataFrame(index=table.index)
    for canonical in GENE_COLUMN_ALIASES:
        if canonical in column_map:
            out[canonical] = table[column_map[canonical]]

    out["gene_id"] = out["gene_id"].astype(str)
    if "gene_symbol" not in out.columns:
        out["gene_symbol"] = out["gene_id"]
    else:
        out["gene_symbol"] = out["gene_symbol"].astype(str)
    out["chrom"] = out["chrom"].map(normalize_chromosome)
    out["start"] = pd.to_numeric(out["start"], errors="coerce")
    out["end"] = pd.to_numeric(out["end"], errors="coerce")

    invalid = out["start"].isna() | out["end"].isna() | (out["start"] > out["end"])
    dropped_invalid = int(invalid.sum())
    out = out.loc[~invalid].copy()
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)

    duplicated_gene_ids = int(out["gene_id"].duplicated().sum())
    if drop_duplicate_gene_ids:
        out = out.drop_duplicates("gene_id", keep="first").copy()

    for col in CANONICAL_GENE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    optional_cols = ["mappability"] if "mappability" in out.columns else []
    out = out.loc[:, [*CANONICAL_GENE_COLUMNS, *optional_cols]].sort_values(
        ["chrom", "start", "end", "gene_id"],
        kind="stable",
    )
    out = out.reset_index(drop=True)

    report = GeneAnnotationReport(
        n_rows_input=int(len(table)),
        n_rows_output=int(len(out)),
        genome_build=genome_build,
        source_format=source_format,
        dropped_invalid_intervals=dropped_invalid,
        duplicated_gene_ids=duplicated_gene_ids,
    )
    return GeneAnnotations(table=out, report=report)


def read_magma_gene_loc(
    path: str | Path,
    *,
    genome_build: str | None = "GRCh37",
    drop_duplicate_gene_ids: bool = True,
) -> GeneAnnotations:
    """Read MAGMA `.gene.loc` files such as `NCBI37.3.gene.loc`.

    MAGMA gene loc files contain six whitespace-delimited columns:

    `entrez ID`, `chromosome`, `start`, `end`, `strand`, `gene symbol`.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["gene_id", "chrom", "start", "end", "strand", "gene_symbol"],
        dtype={"gene_id": str, "chrom": str, "strand": str, "gene_symbol": str},
    )
    return _standardize_gene_table(
        raw,
        column_map={
            "gene_id": "gene_id",
            "gene_symbol": "gene_symbol",
            "chrom": "chrom",
            "start": "start",
            "end": "end",
            "strand": "strand",
        },
        source_format="magma_gene_loc",
        genome_build=genome_build,
        drop_duplicate_gene_ids=drop_duplicate_gene_ids,
    )


def read_gene_annotation_table(
    path: str | Path,
    *,
    column_map: dict[str, str] | None = None,
    sep: str | None = None,
    genome_build: str | None = None,
    drop_duplicate_gene_ids: bool = True,
) -> GeneAnnotations:
    """Read and standardize a generic tabular gene annotation file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_csv(
        path,
        sep=sep,
        compression="infer",
        engine="python" if sep is None else "c",
    )
    if column_map is None:
        column_map = infer_gene_column_map(raw.columns)
    return _standardize_gene_table(
        raw,
        column_map=column_map,
        source_format="generic_table",
        genome_build=genome_build,
        drop_duplicate_gene_ids=drop_duplicate_gene_ids,
    )


def expand_gene_intervals(
    genes: pd.DataFrame,
    *,
    upstream_bp: int = 0,
    downstream_bp: int = 0,
    start_col: str = "start",
    end_col: str = "end",
    output_start_col: str = "map_start",
    output_end_col: str = "map_end",
) -> pd.DataFrame:
    """Return a copy with interval columns expanded by upstream/downstream bp."""

    if upstream_bp < 0 or downstream_bp < 0:
        raise ValueError("upstream_bp and downstream_bp must be nonnegative.")
    if start_col not in genes.columns or end_col not in genes.columns:
        raise ValueError(f"Input table must contain {start_col!r} and {end_col!r}.")

    out = genes.copy()
    start = pd.to_numeric(out[start_col], errors="raise").astype(int)
    end = pd.to_numeric(out[end_col], errors="raise").astype(int)
    if (start > end).any():
        raise ValueError("Gene start positions must be <= gene end positions.")

    out[output_start_col] = (start - int(upstream_bp)).clip(lower=0)
    out[output_end_col] = end + int(downstream_bp)
    return out


def read_mappability_table(
    path: str | Path,
    *,
    gene_id_col: str = "gene_id",
    mappability_col: str = "mappability",
    sep: str | None = None,
) -> pd.DataFrame:
    """Read a simple gene-level mappability table."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(
        path,
        sep=sep,
        compression="infer",
        engine="python" if sep is None else "c",
    )
    missing = [col for col in (gene_id_col, mappability_col) if col not in table.columns]
    if missing:
        raise ValueError(f"Missing mappability columns: {missing}")
    out = table.loc[:, [gene_id_col, mappability_col]].copy()
    out.columns = ["gene_id", "mappability"]
    out["gene_id"] = out["gene_id"].astype(str)
    out["mappability"] = pd.to_numeric(out["mappability"], errors="raise").astype(float)
    return out


def attach_mappability(
    genes: pd.DataFrame,
    mappability: pd.DataFrame,
    *,
    gene_id_col: str = "gene_id",
) -> pd.DataFrame:
    """Attach gene-level mappability scores to a canonical gene table."""

    if gene_id_col not in genes.columns:
        raise ValueError(f"genes is missing {gene_id_col!r}.")
    if not {"gene_id", "mappability"}.issubset(mappability.columns):
        raise ValueError("mappability table must contain 'gene_id' and 'mappability'.")
    out = genes.copy()
    out[gene_id_col] = out[gene_id_col].astype(str)
    map_table = mappability.copy()
    map_table["gene_id"] = map_table["gene_id"].astype(str)
    return out.merge(map_table, left_on=gene_id_col, right_on="gene_id", how="left")
