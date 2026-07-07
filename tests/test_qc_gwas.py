import pandas as pd
import pytest

from ripple.qc.gwas import (
    GwasQcConfig,
    harmonize_to_reference,
    load_reference_bim,
    load_snp_set,
    standardize_finngen_chunk,
    standardize_pgc_scz_chunk,
)


def test_load_snp_set_uses_first_column_and_skips_header(tmp_path):
    path = tmp_path / "hm3.snplist"
    path.write_text("SNP\tA1\tA2\nrs1\tA\tG\nrs2\n", encoding="utf-8")

    assert load_snp_set(path) == {"rs1", "rs2"}


def test_load_reference_bim_filters_to_requested_autosomal_snps(tmp_path):
    bim = tmp_path / "ref.bim"
    bim.write_text(
        "1\trs1\t0\t100\tA\tG\n"
        "2\trs2\t0\t200\tC\tT\n"
        "X\trsX\t0\t300\tA\tC\n"
        "1\trs3\t0\t400\tG\tA\n",
        encoding="utf-8",
    )

    reference = load_reference_bim(bim, include_snps={"rs1", "rs2", "rsX"})

    assert reference["snp_id"].tolist() == ["rs1", "rs2"]
    assert reference["chrom_ref"].tolist() == ["1", "2"]


def test_standardize_finngen_chunk_explodes_rsids_and_sets_alt_as_effect():
    chunk = pd.DataFrame(
        {
            "#chrom": [1],
            "pos": [123],
            "ref": ["G"],
            "alt": ["A"],
            "rsids": ["rs1,rs2"],
            "pval": [0.5],
            "beta": [0.2],
            "sebeta": [0.1],
            "af_alt": [0.2],
        }
    )

    out = standardize_finngen_chunk(chunk, source_trait="HEIGHT_IRN")

    assert out["snp_id"].tolist() == ["rs1", "rs2"]
    assert out["source_effect_allele"].tolist() == ["A", "A"]
    assert out["source_other_allele"].tolist() == ["G", "G"]
    assert out["source_build"].tolist() == ["GRCh38", "GRCh38"]


def test_standardize_pgc_scz_chunk_computes_weighted_eaf():
    chunk = pd.DataFrame(
        {
            "CHROM": [1],
            "ID": ["rs1"],
            "POS": [100],
            "A1": ["A"],
            "A2": ["G"],
            "FCAS": [0.2],
            "FCON": [0.1],
            "IMPINFO": [0.9],
            "BETA": [0.4],
            "SE": [0.2],
            "PVAL": [0.01],
            "NCAS": [100],
            "NCON": [300],
            "NEFF": [300],
        }
    )

    out = standardize_pgc_scz_chunk(chunk)

    assert out.loc[0, "eaf"] == pytest.approx(0.125)
    assert out.loc[0, "info"] == pytest.approx(0.9)
    assert out.loc[0, "sample_size"] == pytest.approx(300)


def test_harmonize_to_reference_flips_to_reference_effect_allele_and_flags_regions():
    standardized = pd.DataFrame(
        {
            "snp_id": ["rs1", "rs_mhc", "rs_apoe", "rs_low_info", "rs_pal"],
            "source_chrom": ["1", "6", "19", "1", "1"],
            "source_pos": [100, 30_000_000, 45_000_000, 200, 300],
            "source_effect_allele": ["G", "A", "C", "A", "A"],
            "source_other_allele": ["A", "G", "T", "G", "T"],
            "beta": [0.2, 0.3, 0.4, 0.5, 0.6],
            "se": [0.1, 0.1, 0.2, 0.2, 0.2],
            "odds_ratio": [pd.NA] * 5,
            "p_value": [0.05, 1e-400, 0.01, 0.02, 0.03],
            "sample_size": [pd.NA] * 5,
            "eaf": [0.2, 0.3, 0.3, 0.3, 0.5],
            "info": [0.9, 0.9, 0.9, 0.5, 0.9],
            "n_cases": [pd.NA] * 5,
            "n_controls": [pd.NA] * 5,
            "source_build": ["GRCh38"] * 5,
            "source_trait": ["TEST"] * 5,
            "source_dataset": ["toy"] * 5,
        }
    )
    reference = pd.DataFrame(
        {
            "snp_id": ["rs1", "rs_mhc", "rs_apoe", "rs_low_info", "rs_pal"],
            "chrom_ref": ["1", "6", "19", "1", "1"],
            "pos_ref": [100, 30_000_000, 45_000_000, 200, 300],
            "ref_a1": ["A", "A", "C", "A", "A"],
            "ref_a2": ["G", "G", "T", "G", "T"],
        }
    )

    out = harmonize_to_reference(
        standardized,
        reference,
        hm3_no_mhc_snps={"rs1", "rs_apoe", "rs_low_info", "rs_pal"},
        config=GwasQcConfig(),
    )

    passed = out[out["qc_pass"]]
    flipped = passed[passed["snp_id"] == "rs1"].iloc[0]
    assert flipped["beta"] == pytest.approx(-0.2)
    assert flipped["eaf"] == pytest.approx(0.8)
    assert bool(flipped["allele_flip"]) is True
    assert bool(flipped["p_was_clipped"]) is False

    mhc = passed[passed["snp_id"] == "rs_mhc"].iloc[0]
    assert bool(mhc["is_mhc"]) is True
    assert bool(mhc["in_hm3_no_mhc"]) is False
    assert bool(mhc["p_was_clipped"]) is True

    apoe = passed[passed["snp_id"] == "rs_apoe"].iloc[0]
    assert bool(apoe["is_apoe_region"]) is True

    low_info = out[out["snp_id"] == "rs_low_info"].iloc[0]
    assert bool(low_info["qc_pass"]) is False
    assert "low_or_missing_info" in low_info["qc_fail_reason"]

    pal = out[out["snp_id"] == "rs_pal"].iloc[0]
    assert bool(pal["qc_pass"]) is False
    assert "ambiguous_palindrome" in pal["qc_fail_reason"]
