from pathlib import Path

import pandas as pd
import pytest

from ripple.io.annotations import (
    attach_mappability,
    expand_gene_intervals,
    infer_gene_column_map,
    read_gene_annotation_table,
    read_magma_gene_loc,
    read_mappability_table,
)


def test_infer_gene_column_map_common_aliases():
    table = pd.DataFrame(columns=["entrez_id", "symbol", "chr", "tx_start", "tx_end", "strand"])
    column_map = infer_gene_column_map(table.columns)

    assert column_map["gene_id"] == "entrez_id"
    assert column_map["gene_symbol"] == "symbol"
    assert column_map["chrom"] == "chr"
    assert column_map["start"] == "tx_start"
    assert column_map["end"] == "tx_end"
    assert column_map["strand"] == "strand"


def test_read_magma_gene_loc_standardizes_columns(tmp_path: Path):
    path = tmp_path / "toy.gene.loc"
    path.write_text(
        "79501\t1\t69091\t70008\t+\tOR4F5\n"
        "100996442\tchr1\t142447\t174392\t-\tLOC100996442\n",
        encoding="utf-8",
    )

    result = read_magma_gene_loc(path)
    out = result.table

    assert result.report.source_format == "magma_gene_loc"
    assert result.report.genome_build == "GRCh37"
    assert result.report.n_rows_output == 2
    assert list(out["gene_id"]) == ["79501", "100996442"]
    assert list(out["gene_symbol"]) == ["OR4F5", "LOC100996442"]
    assert list(out["chrom"]) == ["1", "1"]
    assert list(out["start"]) == [69091, 142447]
    assert list(out["end"]) == [70008, 174392]


def test_read_gene_annotation_table_drops_invalid_and_duplicates(tmp_path: Path):
    path = tmp_path / "genes.tsv"
    path.write_text(
        "gene_id\tgene_name\tchromosome\tstart\tend\n"
        "G1\tA\tchr1\t10\t20\n"
        "G1\tAdup\tchr1\t10\t20\n"
        "G2\tB\tchr2\t30\t25\n",
        encoding="utf-8",
    )

    result = read_gene_annotation_table(path, sep="\t", genome_build="toy")
    out = result.table

    assert result.report.n_rows_input == 3
    assert result.report.dropped_invalid_intervals == 1
    assert result.report.duplicated_gene_ids == 1
    assert len(out) == 1
    assert out.loc[0, "gene_id"] == "G1"
    assert out.loc[0, "gene_symbol"] == "A"


def test_expand_gene_intervals_adds_mapping_bounds():
    genes = pd.DataFrame({"gene_id": ["G1"], "start": [100], "end": [200]})
    out = expand_gene_intervals(genes, upstream_bp=150, downstream_bp=25)

    assert out.loc[0, "map_start"] == 0
    assert out.loc[0, "map_end"] == 225


def test_expand_gene_intervals_rejects_negative_window():
    genes = pd.DataFrame({"gene_id": ["G1"], "start": [100], "end": [200]})
    with pytest.raises(ValueError, match="nonnegative"):
        expand_gene_intervals(genes, upstream_bp=-1)


def test_read_and_attach_mappability(tmp_path: Path):
    path = tmp_path / "mappability.tsv"
    path.write_text("gene_id\tmappability\nG1\t0.9\nG2\t0.5\n", encoding="utf-8")
    mappability = read_mappability_table(path, sep="\t")
    genes = pd.DataFrame({"gene_id": ["G1", "G3"], "chrom": ["1", "1"], "start": [1, 2], "end": [3, 4]})

    out = attach_mappability(genes, mappability)
    assert out.loc[0, "mappability"] == pytest.approx(0.9)
    assert pd.isna(out.loc[1, "mappability"])


def test_real_magma_gene_loc_smoke_if_available():
    path = Path(
        "/path/to/ripple_private_workspace/10_raw_data/reference/genes/"
        "magma_gene_locations/NCBI37.3/NCBI37.3.gene.loc"
    )
    if not path.exists():
        pytest.skip("Private MAGMA gene location file not downloaded.")

    result = read_magma_gene_loc(path)
    out = result.table

    assert result.report.n_rows_output > 15_000
    assert {"gene_id", "gene_symbol", "chrom", "start", "end", "strand"}.issubset(out.columns)
    assert (out["start"] <= out["end"]).all()
    assert "OR4F5" in set(out.head(20)["gene_symbol"])
