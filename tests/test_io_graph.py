from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from ripple.io.graph import (
    canonicalize_undirected_edges,
    edge_list_to_networkx,
    networkx_to_edge_list,
    read_edge_list,
    read_string_aliases,
    read_string_gene_graph,
    read_string_links,
    read_string_protein_info,
    string_links_to_gene_edges,
    string_protein_to_gene_map,
    write_edge_list,
)


def test_canonicalize_undirected_edges_drops_self_and_aggregates_max():
    edges = pd.DataFrame(
        {
            "a": ["B", "A", "A", "C"],
            "b": ["A", "B", "A", "D"],
            "score": [0.2, 0.8, 1.0, 0.5],
        }
    )

    graph = canonicalize_undirected_edges(
        edges,
        node1_col="a",
        node2_col="b",
        weight_col="score",
        aggregation="max",
    )

    assert graph.report.n_rows_input == 4
    assert graph.report.n_self_edges_dropped == 1
    assert graph.report.n_edges_output == 2
    observed = dict(zip(zip(graph.edges["node1"], graph.edges["node2"]), graph.edges["weight"]))
    assert observed[("A", "B")] == pytest.approx(0.8)
    assert observed[("C", "D")] == pytest.approx(0.5)


def test_read_edge_list_from_tsv(tmp_path: Path):
    path = tmp_path / "edges.tsv"
    path.write_text("node1\tnode2\tweight\nA\tB\t0.1\nB\tC\t0.2\n", encoding="utf-8")

    graph = read_edge_list(path, sep="\t")
    assert graph.report.n_edges_output == 2
    assert set(graph.edges["node1"]) == {"A", "B"}


def test_write_edge_list_round_trip_from_networkx(tmp_path: Path):
    graph = nx.Graph()
    graph.add_edge("B", "A", weight=0.4)
    graph.add_edge("A", "B", weight=0.9)
    graph.add_edge("C", "D")
    path = tmp_path / "analysis_graph_edges.tsv.gz"

    written = write_edge_list(path, networkx_to_edge_list(graph))
    loaded = read_edge_list(path, sep="\t")

    assert written.report.n_edges_output == 2
    observed = dict(zip(zip(loaded.edges["node1"], loaded.edges["node2"]), loaded.edges["weight"]))
    assert observed[("A", "B")] == pytest.approx(0.9)
    assert observed[("C", "D")] == pytest.approx(1.0)


def test_string_links_to_gene_edges_maps_protein_ids_and_scales_scores():
    links = pd.DataFrame(
        {
            "protein1": ["9606.P1", "9606.P2", "9606.P1", "9606.P3"],
            "protein2": ["9606.P2", "9606.P1", "9606.P4", "9606.P3"],
            "combined_score": [200, 900, 500, 800],
        }
    )
    protein_map = {"9606.P1": "A", "9606.P2": "B", "9606.P3": "C"}

    graph = string_links_to_gene_edges(links, protein_map)

    assert graph.report.n_rows_input == 4
    assert graph.report.n_unmapped_edges_dropped == 1
    assert graph.report.n_self_edges_dropped == 1
    assert graph.report.n_edges_output == 1
    assert graph.edges.loc[0, "node1"] == "A"
    assert graph.edges.loc[0, "node2"] == "B"
    assert graph.edges.loc[0, "weight"] == pytest.approx(0.9)


def test_string_protein_to_gene_map_uses_preferred_name():
    info = pd.DataFrame(
        {
            "string_protein_id": ["9606.P1", "9606.P2"],
            "preferred_name": ["GENE1", "GENE2"],
        }
    )

    assert string_protein_to_gene_map(info) == {"9606.P1": "GENE1", "9606.P2": "GENE2"}


def test_read_string_files_and_alias_filter_from_toy_files(tmp_path: Path):
    links_path = tmp_path / "links.txt"
    links_path.write_text(
        "protein1 protein2 combined_score\n"
        "9606.P1 9606.P2 700\n"
        "9606.P1 9606.P3 100\n",
        encoding="utf-8",
    )
    info_path = tmp_path / "info.tsv"
    info_path.write_text(
        "#string_protein_id\tpreferred_name\tprotein_size\tannotation\n"
        "9606.P1\tG1\t100\tanno\n"
        "9606.P2\tG2\t100\tanno\n",
        encoding="utf-8",
    )
    alias_path = tmp_path / "aliases.tsv"
    alias_path.write_text(
        "#string_protein_id\talias\tsource\n"
        "9606.P1\t123\tEnsembl_HGNC_entrez_id\n"
        "9606.P1\tABC\tOther\n",
        encoding="utf-8",
    )

    links = read_string_links(links_path, min_score=500)
    info = read_string_protein_info(info_path)
    aliases = read_string_aliases(
        alias_path,
        source_filter=["Ensembl_HGNC_entrez_id"],
        chunksize=1,
    )

    assert len(links) == 1
    assert list(info["string_protein_id"]) == ["9606.P1", "9606.P2"]
    assert list(aliases["alias"]) == ["123"]


def test_edge_list_to_networkx_builds_weighted_graph():
    edges = pd.DataFrame({"node1": ["A"], "node2": ["B"], "weight": [0.75]})
    graph = edge_list_to_networkx(edges)

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_edges() == 1
    assert graph["A"]["B"]["weight"] == pytest.approx(0.75)


def test_real_string_files_smoke_if_available():
    base = Path("/path/to/ripple_private_workspace/10_raw_data/reference/graphs/string_v12")
    links_path = base / "9606.protein.physical.links.v12.0.txt.gz"
    info_path = base / "9606.protein.info.v12.0.txt.gz"
    aliases_path = base / "9606.protein.aliases.v12.0.txt.gz"
    if not links_path.exists() or not info_path.exists() or not aliases_path.exists():
        pytest.skip("Private STRING v12 files not downloaded.")

    links = read_string_links(links_path, nrows=1000)
    info = read_string_protein_info(info_path, nrows=100)
    aliases = read_string_aliases(
        aliases_path,
        source_filter=["Ensembl_HGNC_entrez_id"],
        nrows=1000,
    )

    assert {"protein1", "protein2", "combined_score"}.issubset(links.columns)
    assert {"string_protein_id", "preferred_name"}.issubset(info.columns)
    assert {"string_protein_id", "alias", "source"}.issubset(aliases.columns)


def test_real_string_gene_graph_smoke_if_available():
    base = Path("/path/to/ripple_private_workspace/10_raw_data/reference/graphs/string_v12")
    links_path = base / "9606.protein.physical.links.v12.0.txt.gz"
    info_path = base / "9606.protein.info.v12.0.txt.gz"
    if not links_path.exists() or not info_path.exists():
        pytest.skip("Private STRING v12 files not downloaded.")

    graph = read_string_gene_graph(links_path, info_path, min_score=400, nrows=2000)

    assert graph.report.n_rows_input <= 2000
    assert graph.report.n_edges_output > 0
    assert {"node1", "node2", "weight"}.issubset(graph.edges.columns)
