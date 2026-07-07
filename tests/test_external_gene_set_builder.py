from scripts.build_external_anchored_gene_sets import (
    broad_rows_to_tables,
    build_go_bp_rows,
    jaccard,
    parse_go_obo,
    prune_redundant_rows,
)


def test_parse_go_obo_and_build_go_bp_rows(tmp_path):
    obo = tmp_path / "go.obo"
    obo.write_text(
        "\n".join(
            [
                "format-version: 1.2",
                "[Term]",
                "id: GO:0000001",
                "name: parent process",
                "namespace: biological_process",
                "[Term]",
                "id: GO:0000002",
                "name: child process",
                "namespace: biological_process",
                "is_a: GO:0000001 ! parent process",
                "[Term]",
                "id: GO:0000003",
                "name: cellular component term",
                "namespace: cellular_component",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    gaf = tmp_path / "goa.gaf.gz"
    import gzip

    with gzip.open(gaf, "wt", encoding="utf-8") as handle:
        handle.write("UniProtKB\tP1\tGENEA\t\tGO:0000002\tPMID:1\tIDA\t\tP\tA\t\tprotein\ttaxon:9606\t20260101\tUniProt\n")
        handle.write("UniProtKB\tP2\tGENEB\t\tGO:0000003\tPMID:1\tIDA\t\tC\tB\t\tprotein\ttaxon:9606\t20260101\tUniProt\n")

    terms = parse_go_obo(obo)
    rows = build_go_bp_rows(obo, gaf)

    assert set(terms) == {"GO:0000001", "GO:0000002", "GO:0000003"}
    parent = next(row for row in rows if row["source_term_id"] == "GO:0000001")
    assert parent["genes"] == {"GENEA"}
    assert all(row["category"] == "go_biological_process" for row in rows)
    assert {row["source_term_id"] for row in rows} == {"GO:0000001", "GO:0000002"}


def test_prune_redundant_rows_prefers_smaller_sets():
    rows = [
        {
            "source_database": "Gene Ontology",
            "source_term_id": "large",
            "source_term_name": "large term",
            "source_url": "",
            "category": "go_biological_process",
            "genes": {"A", "B", "C", "D", "E"},
        },
        {
            "source_database": "Gene Ontology",
            "source_term_id": "small",
            "source_term_name": "small term",
            "source_url": "",
            "category": "go_biological_process",
            "genes": {"A", "B", "C", "D"},
        },
        {
            "source_database": "Reactome",
            "source_term_id": "reactome_large",
            "source_term_name": "reactome large",
            "source_url": "",
            "category": "reactome_pathway",
            "genes": {"A", "B", "C", "D", "E"},
        },
    ]

    kept, dropped = prune_redundant_rows(rows, jaccard_threshold=0.70)

    assert [row["source_term_id"] for row in kept] == ["small", "reactome_large"]
    assert dropped[0][0] == "small"


def test_broad_rows_to_tables_filters_and_records_pruning():
    rows = [
        {
            "source_database": "Gene Ontology",
            "source_term_id": "GO:1",
            "source_term_name": "specific process",
            "source_url": "",
            "category": "go_biological_process",
            "genes": {"A", "B", "C", "D"},
        },
        {
            "source_database": "Gene Ontology",
            "source_term_id": "GO:2",
            "source_term_name": "redundant process",
            "source_url": "",
            "category": "go_biological_process",
            "genes": {"A", "B", "C", "D", "E"},
        },
        {
            "source_database": "Reactome",
            "source_term_id": "R-HSA-1",
            "source_term_name": "tiny pathway",
            "source_url": "",
            "category": "reactome_pathway",
            "genes": {"A"},
        },
    ]

    gene_sets, terms, summary = broad_rows_to_tables(
        rows,
        min_genes=2,
        max_genes=10,
        jaccard_threshold=0.70,
    )

    assert summary["n_source_terms"] == 3
    assert summary["n_size_filtered_terms"] == 2
    assert summary["n_pruned_redundant_terms"] == 1
    assert gene_sets["gene_set"].nunique() == 1
    assert terms.loc[terms["source_term_id"].eq("GO:2"), "drop_reason"].iloc[0] == "jaccard_redundant"
    assert terms.loc[terms["source_term_id"].eq("R-HSA-1"), "drop_reason"].iloc[0] == "outside_gene_count_bounds"


def test_jaccard_handles_empty_sets():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"A"}, {"A", "B"}) == 0.5
