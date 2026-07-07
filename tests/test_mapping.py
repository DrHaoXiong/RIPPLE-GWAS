import numpy as np
import pandas as pd
import pytest

from ripple.mapping.positional import normalize_chromosome, positional_map_snps_to_genes
from ripple.mapping.weights import (
    add_positional_weights,
    add_split_weights_from_raw,
    mapping_to_sparse_matrix,
    summarize_mapping,
    weights_for_gene,
)


def test_normalize_chromosome_removes_chr_prefix():
    assert normalize_chromosome("chr6") == "6"
    assert normalize_chromosome("CHR19") == "19"
    assert normalize_chromosome(1) == "1"


def test_positional_map_snps_to_genes_maps_inside_and_window_hits():
    snps = pd.DataFrame(
        {
            "snp_id": ["rs1", "rs2", "rs3", "rs4"],
            "chrom": ["chr1", "1", "1", "2"],
            "pos": [100, 90, 260, 100],
        }
    )
    genes = pd.DataFrame(
        {
            "gene_id": ["G1", "G2", "G3"],
            "gene_symbol": ["A", "B", "C"],
            "chrom": ["1", "1", "2"],
            "start": [95, 250, 500],
            "end": [120, 270, 600],
        }
    )

    mapping = positional_map_snps_to_genes(snps, genes, upstream_bp=10, downstream_bp=0)
    assert list(mapping["snp_id"]) == ["rs1", "rs2", "rs3"]
    assert list(mapping["gene_id"]) == ["G1", "G1", "G2"]
    assert list(mapping["distance_to_gene"]) == [0, 5, 0]
    assert "rs4" not in set(mapping["snp_id"])


def test_positional_map_snps_to_genes_supports_multi_mapping():
    snps = pd.DataFrame({"snp_id": ["rs1"], "chrom": ["1"], "pos": [105]})
    genes = pd.DataFrame(
        {
            "gene_id": ["G1", "G2"],
            "chrom": ["1", "1"],
            "start": [100, 101],
            "end": [110, 120],
        }
    )

    mapping = positional_map_snps_to_genes(snps, genes)
    assert set(mapping["gene_id"]) == {"G1", "G2"}


def test_positional_map_rejects_negative_window_and_bad_gene_interval():
    snps = pd.DataFrame({"snp_id": ["rs1"], "chrom": ["1"], "pos": [100]})
    genes = pd.DataFrame({"gene_id": ["G1"], "chrom": ["1"], "start": [120], "end": [100]})

    with pytest.raises(ValueError, match="nonnegative"):
        positional_map_snps_to_genes(snps, genes, upstream_bp=-1)
    with pytest.raises(ValueError, match="start positions"):
        positional_map_snps_to_genes(snps, genes)


def test_add_positional_weights_splits_multi_mapped_snps():
    mapping = pd.DataFrame(
        {
            "snp_id": ["rs1", "rs1", "rs2"],
            "gene_id": ["G1", "G2", "G2"],
        }
    )

    weighted = add_positional_weights(mapping)
    observed = dict(zip(zip(weighted["snp_id"], weighted["gene_id"]), weighted["weight"]))
    assert observed[("rs1", "G1")] == pytest.approx(0.5)
    assert observed[("rs1", "G2")] == pytest.approx(0.5)
    assert observed[("rs2", "G2")] == pytest.approx(1.0)
    assert list(weighted["snp_mapping_count"]) == [2, 2, 1]


def test_add_positional_weights_deduplicates_pairs():
    mapping = pd.DataFrame(
        {
            "snp_id": ["rs1", "rs1", "rs1"],
            "gene_id": ["G1", "G1", "G2"],
        }
    )

    weighted = add_positional_weights(mapping)
    assert len(weighted) == 2
    assert set(weighted["weight"]) == {0.5}


def test_add_split_weights_from_raw_preserves_signed_mass():
    mapping = pd.DataFrame(
        {
            "snp_id": ["rs1", "rs1", "rs2"],
            "gene_id": ["G1", "G2", "G1"],
            "raw_weight": [2.0, -1.0, 5.0],
        }
    )

    weighted = add_split_weights_from_raw(mapping, raw_weight_col="raw_weight")
    rs1 = weighted.loc[weighted["snp_id"] == "rs1", "weight"].to_numpy()
    np.testing.assert_allclose(rs1, [2 / 3, -1 / 3])
    assert weighted.loc[weighted["snp_id"] == "rs2", "weight"].iloc[0] == pytest.approx(1.0)


def test_summarize_mapping_reports_multi_mapping():
    mapping = pd.DataFrame(
        {
            "snp_id": ["rs1", "rs1", "rs2", "rs3"],
            "gene_id": ["G1", "G2", "G2", "G2"],
        }
    )

    summary = summarize_mapping(mapping)
    assert summary.n_mapping_rows == 4
    assert summary.n_snps == 3
    assert summary.n_genes == 2
    assert summary.n_multi_mapped_snps == 1
    assert summary.max_genes_per_snp == 2


def test_weights_for_gene_returns_ordered_weights():
    mapping = pd.DataFrame(
        {
            "snp_id": ["rs2", "rs1", "rs3"],
            "gene_id": ["G1", "G1", "G2"],
            "weight": [0.25, 1.0, 0.5],
        }
    )

    vector = weights_for_gene(mapping, "G1", snp_order=["rs1", "rs2", "rs3"])
    assert vector.gene_id == "G1"
    assert vector.snp_ids == ("rs1", "rs2")
    np.testing.assert_allclose(vector.weights, [1.0, 0.25])


def test_mapping_to_sparse_matrix_returns_snp_by_gene_matrix():
    mapping = pd.DataFrame(
        {
            "snp_id": ["rs1", "rs1", "rs2"],
            "gene_id": ["G1", "G2", "G2"],
            "weight": [0.5, 0.5, 1.0],
        }
    )

    matrix = mapping_to_sparse_matrix(mapping, snp_ids=["rs1", "rs2"], gene_ids=["G1", "G2"])
    assert matrix.snp_ids == ("rs1", "rs2")
    assert matrix.gene_ids == ("G1", "G2")
    np.testing.assert_allclose(matrix.matrix.toarray(), [[0.5, 0.5], [0.0, 1.0]])
