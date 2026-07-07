"""GWAS summary-statistics IO and harmonization.

This module keeps the first RIPPLE V1 GWAS reader deliberately conservative:

* standardize common GWAS column names into a canonical schema;
* compute signed Z scores only when direction is genuinely available;
* keep unsigned P-value-only inputs usable while marking signed analysis as
  unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


CANONICAL_GWAS_COLUMNS: tuple[str, ...] = (
    "snp_id",
    "chrom",
    "pos",
    "effect_allele",
    "other_allele",
    "beta",
    "se",
    "odds_ratio",
    "p_value",
    "z",
    "sample_size",
)

GWAS_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "snp_id": ("snp_id", "snp", "rsid", "rs_id", "id", "markername", "marker", "variant_id"),
    "chrom": ("chrom", "chr", "chromosome", "#chrom"),
    "pos": ("pos", "bp", "position", "base_pair_location", "base_pair_position"),
    "effect_allele": ("effect_allele", "ea", "a1", "allele1", "tested_allele"),
    "other_allele": ("other_allele", "nea", "non_effect_allele", "a2", "allele2"),
    "beta": ("beta", "b", "effect", "effect_size"),
    "se": ("se", "stderr", "standard_error", "standard_error_beta"),
    "odds_ratio": ("or", "odds_ratio"),
    "p_value": ("p", "pval", "p_value", "pvalue", "p-value", "p_bolt_lmm_inf"),
    "z": ("z", "zscore", "z_score"),
    "sample_size": ("n", "n_total", "n_eff", "sample_size", "samplesize"),
}


@dataclass(frozen=True)
class GwasHarmonizationReport:
    """Summary of a GWAS harmonization step."""

    n_rows_input: int
    n_rows_output: int
    signed_available: bool
    z_source: str
    dropped_duplicate_snps: int
    missing_required_columns: tuple[str, ...]


@dataclass(frozen=True)
class HarmonizedGwas:
    """Harmonized GWAS table and report."""

    table: pd.DataFrame
    report: GwasHarmonizationReport


def _normalize_column_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def infer_gwas_column_map(
    columns: Iterable[str],
    *,
    aliases: dict[str, tuple[str, ...]] = GWAS_COLUMN_ALIASES,
) -> dict[str, str]:
    """Infer a canonical-to-source column map from common GWAS aliases."""

    normalized_to_original = {_normalize_column_name(col): str(col) for col in columns}
    column_map: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            normalized = _normalize_column_name(candidate)
            if normalized in normalized_to_original:
                column_map[canonical] = normalized_to_original[normalized]
                break
    return column_map


def read_gwas_table(
    path: str | Path,
    *,
    sep: str | None = None,
    comment: str | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read a GWAS summary-statistics table.

    `sep=None` uses pandas delimiter inference and works for most TSV/CSV files.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        sep=sep,
        comment=comment,
        nrows=nrows,
        compression="infer",
        engine="python" if sep is None else "c",
    )


def normalize_chromosome(chrom: str | int) -> str:
    """Normalize chromosome labels by removing a leading `chr` prefix."""

    value = str(chrom).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    return value


def _coerce_numeric(series: pd.Series, name: str) -> pd.Series:
    coerced = pd.to_numeric(series, errors="coerce")
    if coerced.isna().all():
        raise ValueError(f"Column {name} could not be parsed as numeric.")
    return coerced


def _clip_p_values(p_values: pd.Series, epsilon: float) -> pd.Series:
    p = _coerce_numeric(p_values, "p_value")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("p_value must be in [0, 1].")
    return p.clip(lower=epsilon, upper=1.0 - epsilon)


def _compute_z_score(out: pd.DataFrame, *, p_clip_epsilon: float) -> tuple[pd.Series, str]:
    if "z" in out.columns and out["z"].notna().any():
        return _coerce_numeric(out["z"], "z"), "z"

    has_beta_se = {"beta", "se"}.issubset(out.columns) and out[["beta", "se"]].notna().any().all()
    if has_beta_se:
        beta = _coerce_numeric(out["beta"], "beta")
        se = _coerce_numeric(out["se"], "se")
        if (se <= 0).any():
            raise ValueError("se must be positive when computing z = beta / se.")
        return beta / se, "beta_se"

    has_or_se = (
        {"odds_ratio", "se"}.issubset(out.columns)
        and out[["odds_ratio", "se"]].notna().any().all()
    )
    if has_or_se:
        odds_ratio = _coerce_numeric(out["odds_ratio"], "odds_ratio")
        se = _coerce_numeric(out["se"], "se")
        if (odds_ratio <= 0).any():
            raise ValueError("odds_ratio must be positive when computing log(OR) / se.")
        if (se <= 0).any():
            raise ValueError("se must be positive when computing log(OR) / se.")
        return np.log(odds_ratio) / se, "log_or_se"

    has_p_beta = {"p_value", "beta"}.issubset(out.columns) and out[["p_value", "beta"]].notna().any().all()
    if has_p_beta:
        p = _clip_p_values(out["p_value"], p_clip_epsilon)
        beta = _coerce_numeric(out["beta"], "beta")
        signs = np.sign(beta).replace(0, np.nan)
        return signs * stats.norm.isf(p / 2.0), "p_beta_sign"

    has_p_or = (
        {"p_value", "odds_ratio"}.issubset(out.columns)
        and out[["p_value", "odds_ratio"]].notna().any().all()
    )
    if has_p_or:
        p = _clip_p_values(out["p_value"], p_clip_epsilon)
        odds_ratio = _coerce_numeric(out["odds_ratio"], "odds_ratio")
        signs = np.sign(np.log(odds_ratio)).replace(0, np.nan)
        return signs * stats.norm.isf(p / 2.0), "p_or_sign"

    return pd.Series(np.nan, index=out.index, dtype=float), "unavailable"


def harmonize_gwas_table(
    table: pd.DataFrame,
    *,
    column_map: dict[str, str] | None = None,
    drop_duplicate_snps: bool = True,
    p_clip_epsilon: float = 1e-300,
) -> HarmonizedGwas:
    """Return a canonical RIPPLE GWAS table.

    Required minimal columns are `snp_id`, `chrom`, and `pos`. Signed Z scores
    are produced only from `z`, `beta/se`, `log(OR)/se`, or P value plus an
    effect direction. P-value-only inputs remain valid but `signed_available`
    is false and `z` is missing.
    """

    if column_map is None:
        column_map = infer_gwas_column_map(table.columns)

    required = ("snp_id", "chrom", "pos")
    missing_required = tuple(col for col in required if col not in column_map)
    if missing_required:
        report = GwasHarmonizationReport(
            n_rows_input=int(len(table)),
            n_rows_output=0,
            signed_available=False,
            z_source="unavailable",
            dropped_duplicate_snps=0,
            missing_required_columns=missing_required,
        )
        raise ValueError(f"Missing required GWAS columns: {missing_required}. Report: {report}")

    out = pd.DataFrame(index=table.index)
    for canonical in CANONICAL_GWAS_COLUMNS:
        if canonical in column_map:
            out[canonical] = table[column_map[canonical]]

    out["snp_id"] = out["snp_id"].astype(str)
    out["chrom"] = out["chrom"].map(normalize_chromosome)
    out["pos"] = _coerce_numeric(out["pos"], "pos").astype("Int64")

    if "p_value" in out.columns:
        out["p_value"] = _clip_p_values(out["p_value"], p_clip_epsilon)

    for allele_col in ("effect_allele", "other_allele"):
        if allele_col in out.columns:
            out[allele_col] = out[allele_col].astype(str).str.upper()

    z, z_source = _compute_z_score(out, p_clip_epsilon=p_clip_epsilon)
    out["z"] = z.astype(float)
    out["signed_available"] = out["z"].notna()

    before = len(out)
    if drop_duplicate_snps:
        out = out.drop_duplicates("snp_id", keep="first").reset_index(drop=True)
    dropped = before - len(out)

    ordered_columns = [
        "snp_id",
        "chrom",
        "pos",
        "effect_allele",
        "other_allele",
        "beta",
        "se",
        "odds_ratio",
        "p_value",
        "z",
        "sample_size",
        "signed_available",
    ]
    for col in ordered_columns:
        if col not in out.columns:
            out[col] = pd.NA
    out = out.loc[:, ordered_columns]

    report = GwasHarmonizationReport(
        n_rows_input=int(len(table)),
        n_rows_output=int(len(out)),
        signed_available=bool(out["signed_available"].fillna(False).any()),
        z_source=z_source,
        dropped_duplicate_snps=int(dropped),
        missing_required_columns=(),
    )
    return HarmonizedGwas(table=out, report=report)


def read_and_harmonize_gwas(
    path: str | Path,
    *,
    column_map: dict[str, str] | None = None,
    sep: str | None = None,
    comment: str | None = None,
    nrows: int | None = None,
    drop_duplicate_snps: bool = True,
    p_clip_epsilon: float = 1e-300,
) -> HarmonizedGwas:
    """Read and harmonize a GWAS table in one step."""

    table = read_gwas_table(path, sep=sep, comment=comment, nrows=nrows)
    return harmonize_gwas_table(
        table,
        column_map=column_map,
        drop_duplicate_snps=drop_duplicate_snps,
        p_clip_epsilon=p_clip_epsilon,
    )
