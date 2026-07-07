"""Safety flags for low-information genes, clipping, and special regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpecialRegion:
    """A genomic interval that should be reported separately in RIPPLE V1."""

    label: str
    chrom: str
    start: int
    end: int


@dataclass(frozen=True)
class GeneSafetyFlags:
    """Safety flags for one gene-level signal row."""

    gene_id: str
    n_mapped_snps: int
    m_eff: float
    is_one_snp_gene: bool
    is_low_information: bool
    is_high_snp_count: bool
    is_special_region: bool
    special_region_labels: tuple[str, ...]
    is_p_clipped: bool


@dataclass(frozen=True)
class ClippingSummary:
    """Summary of P-value clipping after normal-score transformation."""

    n_total: int
    n_clipped: int
    fraction_clipped: float


DEFAULT_MIN_M_EFF: float = 2.0
DEFAULT_HIGH_SNP_COUNT_THRESHOLD: int = 5_000

# GRCh37-oriented defaults for the first RIPPLE V1 smoke-test track.
DEFAULT_SPECIAL_REGIONS: tuple[SpecialRegion, ...] = (
    SpecialRegion(label="MHC", chrom="6", start=25_000_000, end=34_000_000),
    SpecialRegion(label="APOE", chrom="19", start=45_000_000, end=46_000_000),
)


def _normalize_chrom(chrom: str | int) -> str:
    value = str(chrom).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    return value


def _is_finite_number(value: float | int | None) -> bool:
    if value is None:
        return False
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def is_low_information_gene(m_eff: float, *, min_m_eff: float = DEFAULT_MIN_M_EFF) -> bool:
    """Return whether a gene has too little effective SNP information."""

    if not _is_finite_number(m_eff):
        return True
    return float(m_eff) < float(min_m_eff)


def is_high_snp_count_gene(
    n_mapped_snps: int,
    *,
    high_snp_count_threshold: int = DEFAULT_HIGH_SNP_COUNT_THRESHOLD,
) -> bool:
    """Return whether a gene has unusually many mapped SNPs."""

    if n_mapped_snps < 0:
        raise ValueError("n_mapped_snps must be nonnegative.")
    return int(n_mapped_snps) > int(high_snp_count_threshold)


def overlapping_special_regions(
    chrom: str | int | None,
    start: int | None,
    end: int | None,
    *,
    regions: Iterable[SpecialRegion] = DEFAULT_SPECIAL_REGIONS,
) -> tuple[str, ...]:
    """Return labels for special regions overlapping a gene interval."""

    if chrom is None or start is None or end is None:
        return ()
    if start > end:
        raise ValueError("start must be <= end.")

    norm_chrom = _normalize_chrom(chrom)
    labels: list[str] = []
    for region in regions:
        if norm_chrom != _normalize_chrom(region.chrom):
            continue
        if int(start) <= region.end and int(end) >= region.start:
            labels.append(region.label)
    return tuple(labels)


def build_gene_safety_flags(
    *,
    gene_id: str,
    n_mapped_snps: int,
    m_eff: float,
    chrom: str | int | None = None,
    start: int | None = None,
    end: int | None = None,
    is_p_clipped: bool = False,
    min_m_eff: float = DEFAULT_MIN_M_EFF,
    high_snp_count_threshold: int = DEFAULT_HIGH_SNP_COUNT_THRESHOLD,
    regions: Iterable[SpecialRegion] = DEFAULT_SPECIAL_REGIONS,
) -> GeneSafetyFlags:
    """Build all V1 safety flags for one gene."""

    if n_mapped_snps < 0:
        raise ValueError("n_mapped_snps must be nonnegative.")
    labels = overlapping_special_regions(chrom, start, end, regions=regions)
    return GeneSafetyFlags(
        gene_id=str(gene_id),
        n_mapped_snps=int(n_mapped_snps),
        m_eff=float(m_eff) if _is_finite_number(m_eff) else float("nan"),
        is_one_snp_gene=int(n_mapped_snps) == 1,
        is_low_information=is_low_information_gene(m_eff, min_m_eff=min_m_eff),
        is_high_snp_count=is_high_snp_count_gene(
            n_mapped_snps,
            high_snp_count_threshold=high_snp_count_threshold,
        ),
        is_special_region=bool(labels),
        special_region_labels=labels,
        is_p_clipped=bool(is_p_clipped),
    )


def summarize_clipping(is_p_clipped: Iterable[bool]) -> ClippingSummary:
    """Summarize P-value clipping flags."""

    flags = np.asarray(list(is_p_clipped), dtype=bool)
    n_total = int(flags.size)
    n_clipped = int(np.sum(flags))
    fraction = float(n_clipped / n_total) if n_total else 0.0
    return ClippingSummary(n_total=n_total, n_clipped=n_clipped, fraction_clipped=fraction)


def append_safety_columns(
    table: pd.DataFrame,
    *,
    gene_id_col: str = "gene_id",
    n_mapped_snps_col: str = "n_mapped_snps",
    m_eff_col: str = "m_eff",
    chrom_col: str = "chrom",
    start_col: str = "start",
    end_col: str = "end",
    p_clipped_col: str = "is_p_clipped",
    min_m_eff: float = DEFAULT_MIN_M_EFF,
    high_snp_count_threshold: int = DEFAULT_HIGH_SNP_COUNT_THRESHOLD,
    regions: Iterable[SpecialRegion] = DEFAULT_SPECIAL_REGIONS,
) -> pd.DataFrame:
    """Return a copy of `table` with V1 gene safety columns appended."""

    required = [gene_id_col, n_mapped_snps_col, m_eff_col]
    missing = [col for col in required if col not in table.columns]
    if missing:
        raise ValueError(f"Missing required safety columns: {missing}")

    out = table.copy()
    flags: list[GeneSafetyFlags] = []
    for _, row in out.iterrows():
        flags.append(
            build_gene_safety_flags(
                gene_id=row[gene_id_col],
                n_mapped_snps=int(row[n_mapped_snps_col]),
                m_eff=float(row[m_eff_col]) if pd.notna(row[m_eff_col]) else float("nan"),
                chrom=row[chrom_col] if chrom_col in out.columns else None,
                start=int(row[start_col]) if start_col in out.columns and pd.notna(row[start_col]) else None,
                end=int(row[end_col]) if end_col in out.columns and pd.notna(row[end_col]) else None,
                is_p_clipped=bool(row[p_clipped_col]) if p_clipped_col in out.columns else False,
                min_m_eff=min_m_eff,
                high_snp_count_threshold=high_snp_count_threshold,
                regions=regions,
            )
        )

    out["is_one_snp_gene"] = [flag.is_one_snp_gene for flag in flags]
    out["is_low_information"] = [flag.is_low_information for flag in flags]
    out["is_high_snp_count"] = [flag.is_high_snp_count for flag in flags]
    out["is_special_region"] = [flag.is_special_region for flag in flags]
    out["special_region_labels"] = [";".join(flag.special_region_labels) for flag in flags]
    return out
