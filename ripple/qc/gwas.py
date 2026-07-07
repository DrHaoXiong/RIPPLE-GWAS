"""GWAS summary-statistics QC and reference harmonization.

The V1 QC target is conservative: autosomal, biallelic, common SNPs that can be
aligned to the 1000G EUR GRCh37 LD reference by rsID. FinnGen R13 GRCh38
coordinates are therefore projected to GRCh37 through the matched 1000G BIM
record rather than by overwriting the raw coordinates in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


AUTOSOMES = {str(chrom) for chrom in range(1, 23)}
VALID_BASES = {"A", "C", "G", "T"}
COMPLEMENT = str.maketrans("ACGT", "TGCA")

MHC_GRCH37 = ("6", 25_000_000, 34_000_000)
APOE_LIKE_GRCH37 = ("19", 44_000_000, 46_500_000)

QC_OUTPUT_COLUMNS: tuple[str, ...] = (
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
    "eaf",
    "maf",
    "info",
    "n_cases",
    "n_controls",
    "source_chrom",
    "source_pos",
    "source_effect_allele",
    "source_other_allele",
    "source_build",
    "analysis_build",
    "source_trait",
    "source_dataset",
    "is_mhc",
    "is_apoe_region",
    "is_palindromic",
    "is_ambiguous_allele",
    "is_indel",
    "is_multiallelic",
    "allele_flip",
    "strand_flip",
    "liftover_status",
    "harmonization_status",
    "qc_pass",
    "qc_fail_reason",
    "p_was_clipped",
)


@dataclass(frozen=True)
class GwasQcConfig:
    """QC thresholds and reference build settings."""

    maf_min: float = 0.01
    info_min: float = 0.8
    ambiguous_palindrome_maf_min: float = 0.42
    ambiguous_palindrome_maf_max: float = 0.58
    p_min_clip: float = 1e-300
    source_build_finngen: str = "GRCh38"
    source_build_pgc: str = "GRCh37"
    source_build_gwas_catalog: str = "GRCh38.p14"
    source_build_ukb: str = "GRCh37"
    analysis_build: str = "GRCh37"


def normalize_chromosome(chrom: object) -> str:
    """Normalize chromosome labels to strings without a leading `chr`."""

    value = str(chrom).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    return value


def normalize_allele(allele: object) -> str:
    """Return an uppercase allele string."""

    return str(allele).strip().upper()


def complement_allele(allele: str) -> str:
    """Return the DNA complement for a single-base or multi-base allele."""

    return allele.translate(COMPLEMENT)


def is_single_base_snp(effect: str, other: str) -> bool:
    """Return true when both alleles are one of A/C/G/T."""

    return len(effect) == 1 and len(other) == 1 and effect in VALID_BASES and other in VALID_BASES


def is_palindromic_pair(effect: str, other: str) -> bool:
    """Return true for A/T, T/A, C/G, or G/C allele pairs."""

    return is_single_base_snp(effect, other) and complement_allele(effect) == other


def is_in_region(chrom: pd.Series, pos: pd.Series, region: tuple[str, int, int]) -> pd.Series:
    """Vectorized closed-interval region membership."""

    region_chrom, start, end = region
    return (chrom.astype(str) == region_chrom) & (pos >= start) & (pos <= end)


def load_snp_set(path: str | Path) -> set[str]:
    """Load a SNP set file, using the first whitespace-delimited column."""

    snps: set[str] = set()
    with Path(path).open("rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if not fields:
                continue
            value = fields[0]
            if value.upper() in {"SNP", "RSID", "SNP_ID"}:
                continue
            if value:
                snps.add(value)
    return snps


def load_reference_bim(
    bim_path: str | Path,
    *,
    include_snps: set[str] | None = None,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Load a 1000G/MAGMA-style PLINK BIM table, optionally restricted by rsID."""

    del chunksize
    rows: list[tuple[str, str, int, str, str]] = []
    seen: set[str] = set()
    with Path(bim_path).open("rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split()
            if len(fields) < 6:
                continue
            chrom, snp_id, _, pos, allele1, allele2 = fields[:6]
            if include_snps is not None and snp_id not in include_snps:
                continue
            if snp_id in seen:
                continue
            chrom = normalize_chromosome(chrom)
            if chrom not in AUTOSOMES:
                continue
            try:
                pos_int = int(pos)
            except ValueError:
                continue
            rows.append(
                (
                    snp_id,
                    chrom,
                    pos_int,
                    normalize_allele(allele1),
                    normalize_allele(allele2),
                )
            )
            seen.add(snp_id)

    if not rows:
        return pd.DataFrame(columns=["snp_id", "chrom_ref", "pos_ref", "ref_a1", "ref_a2"])
    return pd.DataFrame(rows, columns=["snp_id", "chrom_ref", "pos_ref", "ref_a1", "ref_a2"])


def _coerce_numeric(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(pd.NA, index=table.index, dtype="Float64")
    return pd.to_numeric(table[column], errors="coerce")


def _empty_optional(table: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.NA, index=table.index, dtype="Float64")


def _clean_rsid_series(rsids: pd.Series) -> pd.Series:
    return (
        rsids.astype(str)
        .str.strip()
        .replace({"": pd.NA, ".": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def explode_rsids(table: pd.DataFrame, rsid_col: str = "snp_id") -> pd.DataFrame:
    """Explode comma/semicolon separated rsID fields into one row per rsID."""

    out = table.copy()
    out[rsid_col] = _clean_rsid_series(out[rsid_col])
    out = out.dropna(subset=[rsid_col])
    if out.empty:
        return out
    out[rsid_col] = out[rsid_col].str.replace(";", ",", regex=False).str.split(",")
    out = out.explode(rsid_col)
    out[rsid_col] = _clean_rsid_series(out[rsid_col])
    return out.dropna(subset=[rsid_col]).reset_index(drop=True)


def standardize_finngen_chunk(
    chunk: pd.DataFrame,
    *,
    source_trait: str,
    source_dataset: str = "FinnGen_R13",
    config: GwasQcConfig | None = None,
) -> pd.DataFrame:
    """Standardize one FinnGen R13 summary-statistics chunk."""

    cfg = config or GwasQcConfig()
    out = pd.DataFrame(index=chunk.index)
    out["snp_id"] = chunk["rsids"]
    out["source_chrom"] = chunk["#chrom"].map(normalize_chromosome)
    out["source_pos"] = pd.to_numeric(chunk["pos"], errors="coerce")
    out["source_effect_allele"] = chunk["alt"].map(normalize_allele)
    out["source_other_allele"] = chunk["ref"].map(normalize_allele)
    out["beta"] = _coerce_numeric(chunk, "beta")
    out["se"] = _coerce_numeric(chunk, "sebeta")
    out["odds_ratio"] = pd.NA
    out["p_value"] = _coerce_numeric(chunk, "pval")
    out["sample_size"] = pd.NA
    out["eaf"] = _coerce_numeric(chunk, "af_alt")
    out["info"] = pd.NA
    out["n_cases"] = pd.NA
    out["n_controls"] = pd.NA
    out["source_build"] = cfg.source_build_finngen
    out["source_trait"] = source_trait
    out["source_dataset"] = source_dataset
    return explode_rsids(out)


def standardize_pgc_scz_chunk(
    chunk: pd.DataFrame,
    *,
    source_trait: str = "SCZ",
    source_dataset: str = "PGC_SCZ_2022_EUR",
    config: GwasQcConfig | None = None,
) -> pd.DataFrame:
    """Standardize one PGC SCZ 2022 European autosome chunk."""

    cfg = config or GwasQcConfig()
    n_cases = _coerce_numeric(chunk, "NCAS")
    n_controls = _coerce_numeric(chunk, "NCON")
    f_cases = _coerce_numeric(chunk, "FCAS")
    f_controls = _coerce_numeric(chunk, "FCON")
    denom = n_cases + n_controls
    eaf = ((f_cases * n_cases) + (f_controls * n_controls)) / denom.replace(0, np.nan)

    out = pd.DataFrame(index=chunk.index)
    out["snp_id"] = chunk["ID"]
    out["source_chrom"] = chunk["CHROM"].map(normalize_chromosome)
    out["source_pos"] = pd.to_numeric(chunk["POS"], errors="coerce")
    out["source_effect_allele"] = chunk["A1"].map(normalize_allele)
    out["source_other_allele"] = chunk["A2"].map(normalize_allele)
    out["beta"] = _coerce_numeric(chunk, "BETA")
    out["se"] = _coerce_numeric(chunk, "SE")
    out["odds_ratio"] = pd.NA
    out["p_value"] = _coerce_numeric(chunk, "PVAL")
    out["sample_size"] = _coerce_numeric(chunk, "NEFF")
    out["eaf"] = eaf
    out["info"] = _coerce_numeric(chunk, "IMPINFO")
    out["n_cases"] = n_cases
    out["n_controls"] = n_controls
    out["source_build"] = cfg.source_build_pgc
    out["source_trait"] = source_trait
    out["source_dataset"] = source_dataset
    return explode_rsids(out)


def _derive_log_or_se_from_ci(ci_lower: pd.Series, ci_upper: pd.Series) -> pd.Series:
    lower = pd.to_numeric(ci_lower, errors="coerce")
    upper = pd.to_numeric(ci_upper, errors="coerce")
    valid = (lower > 0) & (upper > 0) & (upper > lower)
    out = pd.Series(pd.NA, index=ci_lower.index, dtype="Float64")
    out.loc[valid] = (np.log(upper.loc[valid]) - np.log(lower.loc[valid])) / (2 * 1.96)
    return out


def standardize_gwas_catalog_or_chunk(
    chunk: pd.DataFrame,
    *,
    source_trait: str = "DR_MVP",
    source_dataset: str = "GWAS_Catalog_GCST90475689",
    config: GwasQcConfig | None = None,
) -> pd.DataFrame:
    """Standardize a GWAS Catalog odds-ratio summary-statistics chunk.

    GCST90475689 provides odds ratios and confidence intervals but no explicit
    standard errors. The beta scale is therefore log(OR), and SE is recovered
    from the reported 95% CI when the standard_error column is unavailable.
    """

    cfg = config or GwasQcConfig()
    odds_ratio = _coerce_numeric(chunk, "odds_ratio")
    standard_error = _coerce_numeric(chunk, "standard_error")
    ci_se = _derive_log_or_se_from_ci(chunk["ci_lower"], chunk["ci_upper"])
    se = standard_error.where(standard_error.notna(), ci_se)

    out = pd.DataFrame(index=chunk.index)
    out["snp_id"] = chunk["rsid"]
    out["source_chrom"] = chunk["chromosome"].map(normalize_chromosome)
    out["source_pos"] = pd.to_numeric(chunk["base_pair_location"], errors="coerce")
    out["source_effect_allele"] = chunk["effect_allele"].map(normalize_allele)
    out["source_other_allele"] = chunk["other_allele"].map(normalize_allele)
    out["beta"] = np.log(odds_ratio.where(odds_ratio > 0))
    out["se"] = se
    out["odds_ratio"] = odds_ratio
    out["p_value"] = _coerce_numeric(chunk, "p_value")
    out["sample_size"] = _coerce_numeric(chunk, "n")
    out["eaf"] = _coerce_numeric(chunk, "effect_allele_frequency")
    out["info"] = _coerce_numeric(chunk, "r2")
    out["n_cases"] = _coerce_numeric(chunk, "num_cases")
    out["n_controls"] = _coerce_numeric(chunk, "num_controls")
    out["source_build"] = cfg.source_build_gwas_catalog
    out["source_trait"] = source_trait
    out["source_dataset"] = source_dataset
    return explode_rsids(out)


def standardize_ukb_cai_dr_chunk(
    chunk: pd.DataFrame,
    *,
    source_trait: str = "DR_UKB_CAI_2026",
    source_dataset: str = "UKB_Cai_2026_DR_T2D_H360",
    config: GwasQcConfig | None = None,
) -> pd.DataFrame:
    """Standardize the Cai et al. 2026 UK Biobank T2D-DR GWAS supplement."""

    cfg = config or GwasQcConfig()
    out = pd.DataFrame(index=chunk.index)
    out["snp_id"] = chunk["SNP"]
    out["source_chrom"] = chunk["CHR"].map(normalize_chromosome)
    out["source_pos"] = pd.to_numeric(chunk["POS"], errors="coerce")
    out["source_effect_allele"] = chunk["A1"].map(normalize_allele)
    out["source_other_allele"] = chunk["A2"].map(normalize_allele)
    out["beta"] = _coerce_numeric(chunk, "BETA")
    out["se"] = _coerce_numeric(chunk, "SE")
    out["odds_ratio"] = pd.NA
    out["p_value"] = _coerce_numeric(chunk, "P")
    out["sample_size"] = _coerce_numeric(chunk, "N")
    out["eaf"] = _coerce_numeric(chunk, "AF1")
    out["info"] = _coerce_numeric(chunk, "INFO")
    out["n_cases"] = pd.NA
    out["n_controls"] = pd.NA
    out["source_build"] = cfg.source_build_ukb
    out["source_trait"] = source_trait
    out["source_dataset"] = source_dataset
    return explode_rsids(out)


def _allele_alignment_case(
    effect: str,
    other: str,
    ref_a1: str,
    ref_a2: str,
) -> tuple[bool | None, bool | None]:
    """Return `(allele_flip, strand_flip)` or `(None, None)` for mismatch."""

    comp_effect = complement_allele(effect)
    comp_other = complement_allele(other)
    if effect == ref_a1 and other == ref_a2:
        return False, False
    if effect == ref_a2 and other == ref_a1:
        return True, False
    if comp_effect == ref_a1 and comp_other == ref_a2:
        return False, True
    if comp_effect == ref_a2 and comp_other == ref_a1:
        return True, True
    return None, None


def _alignment_vectors(table: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    allele_flip: list[object] = []
    strand_flip: list[object] = []
    mismatch: list[bool] = []
    for row in table.itertuples(index=False):
        flip, strand = _allele_alignment_case(
            row.source_effect_allele,
            row.source_other_allele,
            row.ref_a1,
            row.ref_a2,
        )
        allele_flip.append(pd.NA if flip is None else flip)
        strand_flip.append(pd.NA if strand is None else strand)
        mismatch.append(flip is None)
    return (
        pd.Series(allele_flip, index=table.index, dtype="boolean"),
        pd.Series(strand_flip, index=table.index, dtype="boolean"),
        pd.Series(mismatch, index=table.index, dtype=bool),
    )


def _add_fail_reason(reason: pd.Series, mask: pd.Series, label: str) -> pd.Series:
    out = reason.copy()
    needs_label = mask & out.eq("")
    out.loc[needs_label] = label
    add_label = mask & ~out.eq("") & ~out.str.contains(label, regex=False)
    out.loc[add_label] = out.loc[add_label] + ";" + label
    return out


def harmonize_to_reference(
    standardized: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    hm3_no_mhc_snps: set[str] | None = None,
    config: GwasQcConfig | None = None,
) -> pd.DataFrame:
    """Harmonize standardized GWAS rows to the GRCh37 1000G EUR reference."""

    cfg = config or GwasQcConfig()
    if standardized.empty:
        return pd.DataFrame(columns=QC_OUTPUT_COLUMNS)

    merged = standardized.merge(reference, on="snp_id", how="left", validate="many_to_one")
    reason = pd.Series("", index=merged.index, dtype=str)
    reason = _add_fail_reason(reason, merged["chrom_ref"].isna(), "not_in_reference_hm3")

    merged["chrom"] = merged["chrom_ref"]
    merged["pos"] = pd.to_numeric(merged["pos_ref"], errors="coerce")
    merged["effect_allele"] = merged["ref_a1"]
    merged["other_allele"] = merged["ref_a2"]
    merged["analysis_build"] = cfg.analysis_build
    merged["liftover_status"] = np.where(
        merged["chrom_ref"].notna(),
        "rsid_to_1000G_EUR_GRCh37",
        "unmapped",
    )

    for col in ("beta", "se", "p_value", "eaf", "info", "sample_size", "n_cases", "n_controls"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    p_zero_or_tiny = merged["p_value"].notna() & (merged["p_value"] < cfg.p_min_clip)
    merged["p_was_clipped"] = p_zero_or_tiny
    merged.loc[p_zero_or_tiny, "p_value"] = cfg.p_min_clip
    reason = _add_fail_reason(reason, merged["p_value"].isna() | (merged["p_value"] > 1), "invalid_p")
    reason = _add_fail_reason(reason, merged["se"].isna() | (merged["se"] <= 0), "invalid_se")
    reason = _add_fail_reason(reason, ~np.isfinite(merged["beta"]), "invalid_beta")
    reason = _add_fail_reason(reason, merged["eaf"].isna() | (merged["eaf"] < 0) | (merged["eaf"] > 1), "invalid_eaf")

    is_snp = pd.Series(
        [
            is_single_base_snp(effect, other)
            for effect, other in zip(merged["source_effect_allele"], merged["source_other_allele"], strict=False)
        ],
        index=merged.index,
        dtype=bool,
    )
    merged["is_indel"] = ~is_snp
    merged["is_multiallelic"] = merged["snp_id"].duplicated(keep=False)
    reason = _add_fail_reason(reason, merged["is_indel"], "non_snp_or_non_acgt")

    allele_flip, strand_flip, allele_mismatch = _alignment_vectors(merged)
    merged["allele_flip"] = allele_flip
    merged["strand_flip"] = strand_flip
    reason = _add_fail_reason(
        reason,
        allele_mismatch & ~merged["is_indel"] & merged["chrom_ref"].notna(),
        "allele_mismatch",
    )

    merged["is_palindromic"] = [
        is_palindromic_pair(effect, other)
        for effect, other in zip(merged["source_effect_allele"], merged["source_other_allele"], strict=False)
    ]
    merged["maf"] = np.minimum(merged["eaf"], 1.0 - merged["eaf"])
    merged["is_ambiguous_allele"] = (
        merged["is_palindromic"]
        & merged["maf"].between(
            cfg.ambiguous_palindrome_maf_min,
            cfg.ambiguous_palindrome_maf_max,
            inclusive="both",
        )
    )
    reason = _add_fail_reason(reason, merged["is_ambiguous_allele"], "ambiguous_palindrome")

    flip_mask = merged["allele_flip"].fillna(False).astype(bool)
    merged.loc[flip_mask, "beta"] = -merged.loc[flip_mask, "beta"]
    merged.loc[flip_mask, "eaf"] = 1.0 - merged.loc[flip_mask, "eaf"]
    merged["maf"] = np.minimum(merged["eaf"], 1.0 - merged["eaf"])
    reason = _add_fail_reason(reason, merged["maf"].isna() | (merged["maf"] < cfg.maf_min), "low_maf")

    has_any_info = merged["info"].notna().any()
    if has_any_info:
        reason = _add_fail_reason(reason, merged["info"].isna() | (merged["info"] < cfg.info_min), "low_or_missing_info")

    mapped_chrom = merged["chrom"].notna()
    reason = _add_fail_reason(
        reason,
        mapped_chrom & ~merged["chrom"].astype(str).isin(AUTOSOMES),
        "non_autosome",
    )
    merged["is_mhc"] = is_in_region(merged["chrom"], merged["pos"], MHC_GRCH37)
    merged["is_apoe_region"] = is_in_region(merged["chrom"], merged["pos"], APOE_LIKE_GRCH37)

    if hm3_no_mhc_snps is not None:
        merged["in_hm3_no_mhc"] = merged["snp_id"].isin(hm3_no_mhc_snps)
    else:
        merged["in_hm3_no_mhc"] = ~merged["is_mhc"]

    merged["z"] = merged["beta"] / merged["se"]
    merged["harmonization_status"] = np.where(reason.eq(""), "aligned_to_reference", "failed")
    merged["qc_pass"] = reason.eq("")
    merged["qc_fail_reason"] = reason

    for col in ("allele_flip", "strand_flip"):
        merged[col] = merged[col].fillna(False).astype(bool)
    for col in ("is_mhc", "is_apoe_region", "is_palindromic", "is_ambiguous_allele", "is_indel", "is_multiallelic"):
        merged[col] = merged[col].fillna(False).astype(bool)

    for col in QC_OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA
    return merged.loc[:, QC_OUTPUT_COLUMNS + ("in_hm3_no_mhc",)].reset_index(drop=True)


def infer_trait_from_path(path: str | Path) -> str:
    """Infer a short trait label from the downloaded file name."""

    name = Path(path).name
    if name.startswith("finngen_R13_"):
        return name.removeprefix("finngen_R13_").removesuffix(".gz")
    if name.startswith("GCST90475689"):
        return "DR_MVP"
    if name.startswith("UKB_Cai2026_DR_T2D_H360") or name.startswith("cai_2026_supplementary_data"):
        return "DR_UKB_CAI_2026"
    if "SCZ" in name.upper():
        return "SCZ"
    return Path(path).stem.replace(".", "_")


def iter_standardized_chunks(
    path: str | Path,
    *,
    chunksize: int,
    config: GwasQcConfig | None = None,
) -> Iterable[pd.DataFrame]:
    """Yield standardized chunks from a supported GWAS file."""

    path = Path(path)
    trait = infer_trait_from_path(path)
    lower_name = path.name.lower()
    if lower_name.startswith("finngen_r13_"):
        for chunk in pd.read_csv(path, sep="\t", compression="infer", chunksize=chunksize):
            yield standardize_finngen_chunk(chunk, source_trait=trait, config=config)
        return

    if "pgc3_scz" in lower_name:
        for chunk in pd.read_csv(
            path,
            sep="\t",
            compression="infer",
            comment="#",
            chunksize=chunksize,
        ):
            yield standardize_pgc_scz_chunk(chunk, source_trait=trait, config=config)
        return

    if lower_name.startswith("gcst90475689"):
        for chunk in pd.read_csv(path, sep="\t", compression="infer", chunksize=chunksize):
            yield standardize_gwas_catalog_or_chunk(chunk, source_trait=trait, config=config)
        return

    if lower_name.startswith("ukb_cai2026_dr_t2d_h360") or lower_name.startswith("cai_2026_supplementary_data"):
        for chunk in pd.read_csv(path, sep="\t", compression="infer", chunksize=chunksize):
            yield standardize_ukb_cai_dr_chunk(chunk, source_trait=trait, config=config)
        return

    raise ValueError(f"Unsupported GWAS file format for {path}")
