from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ripple.io.gwas import (
    harmonize_gwas_table,
    infer_gwas_column_map,
    read_and_harmonize_gwas,
)
from ripple.io.ld import (
    align_ld_matrix,
    correlation_matrix_from_genotypes,
    filter_bim_to_snps,
    read_bim,
    read_fam,
    read_square_ld_matrix,
    resolve_plink_reference,
    summarize_ld_matrix,
    write_snp_list,
)


def test_infer_gwas_column_map_common_aliases():
    table = pd.DataFrame(columns=["SNP", "CHR", "BP", "A1", "A2", "BETA", "SE", "P"])
    column_map = infer_gwas_column_map(table.columns)

    assert column_map["snp_id"] == "SNP"
    assert column_map["chrom"] == "CHR"
    assert column_map["pos"] == "BP"
    assert column_map["effect_allele"] == "A1"
    assert column_map["other_allele"] == "A2"
    assert column_map["beta"] == "BETA"
    assert column_map["se"] == "SE"
    assert column_map["p_value"] == "P"


def test_harmonize_gwas_computes_z_from_beta_se_and_drops_duplicates():
    table = pd.DataFrame(
        {
            "SNP": ["rs1", "rs1", "rs2"],
            "CHR": ["chr1", "chr1", "2"],
            "BP": [100, 100, 200],
            "A1": ["a", "a", "g"],
            "A2": ["c", "c", "t"],
            "BETA": [0.2, 0.2, -0.4],
            "SE": [0.1, 0.1, 0.2],
            "P": [0.05, 0.05, 0.01],
        }
    )

    result = harmonize_gwas_table(table)
    out = result.table

    assert result.report.z_source == "beta_se"
    assert result.report.signed_available
    assert result.report.dropped_duplicate_snps == 1
    assert list(out["snp_id"]) == ["rs1", "rs2"]
    assert list(out["chrom"]) == ["1", "2"]
    assert list(out["effect_allele"]) == ["A", "G"]
    np.testing.assert_allclose(out["z"].to_numpy(dtype=float), [2.0, -2.0])


def test_harmonize_gwas_p_only_keeps_unsigned_and_marks_signed_unavailable():
    table = pd.DataFrame(
        {
            "snp": ["rs1", "rs2"],
            "chromosome": [1, 1],
            "position": [100, 200],
            "p": [1e-8, 0.5],
        }
    )

    result = harmonize_gwas_table(table)
    assert result.report.z_source == "unavailable"
    assert not result.report.signed_available
    assert result.table["z"].isna().all()
    assert not result.table["signed_available"].any()


def test_harmonize_gwas_computes_signed_z_from_p_and_beta_sign():
    table = pd.DataFrame(
        {
            "snp_id": ["rs1", "rs2"],
            "chrom": ["1", "1"],
            "pos": [100, 200],
            "beta": [0.1, -0.1],
            "p_value": [0.05, 0.05],
        }
    )

    result = harmonize_gwas_table(table)
    expected = stats.norm.isf(0.05 / 2.0)
    np.testing.assert_allclose(result.table["z"].to_numpy(dtype=float), [expected, -expected])
    assert result.report.z_source == "p_beta_sign"


def test_read_and_harmonize_gwas_reads_tsv(tmp_path: Path):
    path = tmp_path / "gwas.tsv"
    path.write_text("SNP\tCHR\tBP\tZ\nrs1\t1\t100\t3.5\n", encoding="utf-8")

    result = read_and_harmonize_gwas(path, sep="\t")
    assert result.table.loc[0, "snp_id"] == "rs1"
    assert result.table.loc[0, "z"] == pytest.approx(3.5)
    assert result.report.z_source == "z"


def test_harmonize_gwas_missing_required_columns_raises():
    table = pd.DataFrame({"snp_id": ["rs1"], "p_value": [0.1]})
    with pytest.raises(ValueError, match="Missing required GWAS columns"):
        harmonize_gwas_table(table)


def _write_plink_triplet(prefix: Path):
    prefix.with_suffix(".bed").write_bytes(b"placeholder")
    prefix.with_suffix(".bim").write_text(
        "1 rs1 0 100 A C\n"
        "1 rs2 0 200 G T\n"
        "2 rs3 0 300 C G\n",
        encoding="utf-8",
    )
    prefix.with_suffix(".fam").write_text(
        "F1 I1 0 0 1 -9\n"
        "F2 I2 0 0 2 -9\n",
        encoding="utf-8",
    )


def test_resolve_plink_reference_and_read_metadata(tmp_path: Path):
    prefix = tmp_path / "ref"
    _write_plink_triplet(prefix)

    reference = resolve_plink_reference(prefix)
    assert reference.bed.exists()
    assert reference.bim.exists()
    assert reference.fam.exists()

    bim = read_bim(prefix)
    fam = read_fam(prefix)
    assert list(bim["snp_id"]) == ["rs1", "rs2", "rs3"]
    assert list(bim["pos"]) == [100, 200, 300]
    assert list(fam["individual_id"]) == ["I1", "I2"]


def test_filter_bim_to_snps_preserves_requested_order(tmp_path: Path):
    prefix = tmp_path / "ref"
    _write_plink_triplet(prefix)
    bim = read_bim(prefix)

    filtered = filter_bim_to_snps(bim, ["rs3", "rs1", "missing"])
    assert list(filtered["snp_id"]) == ["rs3", "rs1"]


def test_align_ld_matrix_reorders_and_symmetrizes():
    matrix = np.array(
        [
            [1.0, 0.2, 0.3],
            [0.1, 1.0, 0.4],
            [0.3, 0.5, 1.0],
        ]
    )

    aligned = align_ld_matrix(matrix, ["rs1", "rs2", "rs3"], ["rs3", "rs1"])
    assert aligned.snp_ids == ("rs3", "rs1")
    np.testing.assert_allclose(aligned.matrix, [[1.0, 0.3], [0.3, 1.0]])


def test_align_ld_matrix_missing_behavior():
    matrix = np.eye(2)
    with pytest.raises(KeyError):
        align_ld_matrix(matrix, ["rs1", "rs2"], ["rs1", "rs3"])

    aligned = align_ld_matrix(matrix, ["rs1", "rs2"], ["rs1", "rs3"], allow_missing=True)
    assert aligned.snp_ids == ("rs1",)
    np.testing.assert_allclose(aligned.matrix, [[1.0]])


def test_read_square_ld_matrix_and_write_snp_list(tmp_path: Path):
    ld_path = tmp_path / "gene.ld"
    np.savetxt(ld_path, np.array([[1.0, 0.25], [0.25, 1.0]]))
    result = read_square_ld_matrix(ld_path, ["rs1", "rs2"])
    assert result.snp_ids == ("rs1", "rs2")
    np.testing.assert_allclose(result.matrix, [[1.0, 0.25], [0.25, 1.0]])

    list_path = write_snp_list(tmp_path / "extract.snplist", ["rs1", "rs2"])
    assert list_path.read_text(encoding="utf-8") == "rs1\nrs2\n"


def test_correlation_matrix_from_genotypes_mean_imputes_missing_values():
    genotypes = np.array([[0.0, 0.0], [1.0, np.nan], [2.0, 2.0]])

    ld = correlation_matrix_from_genotypes(genotypes)

    assert ld.shape == (2, 2)
    np.testing.assert_allclose(np.diag(ld), [1.0, 1.0])
    assert np.all(np.isfinite(ld))


def test_summarize_ld_matrix_returns_effective_number_and_ld_score():
    summary = summarize_ld_matrix(np.eye(3))

    assert summary.m_eff == pytest.approx(3.0)
    assert summary.local_ld_score == pytest.approx(1.0)
