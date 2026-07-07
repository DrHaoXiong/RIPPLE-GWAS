"""Biological reference graph IO.

RIPPLE V1 treats biological graphs as reference graphs. This module provides
canonical edge-list IO plus STRING v12 human helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import networkx as nx
import pandas as pd


EdgeAggregation = Literal["max", "mean", "sum"]


@dataclass(frozen=True)
class GraphLoadReport:
    """Summary of graph loading and standardization."""

    source: str
    n_rows_input: int
    n_edges_output: int
    n_nodes_output: int
    n_self_edges_dropped: int
    n_unmapped_edges_dropped: int
    weight_col: str


@dataclass(frozen=True)
class GraphEdges:
    """Canonical undirected weighted edge list."""

    edges: pd.DataFrame
    report: GraphLoadReport


def _require_columns(table: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [col for col in columns if col not in table.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _normalize_node_id(value: object) -> str:
    return str(value).strip()


def canonicalize_undirected_edges(
    edges: pd.DataFrame,
    *,
    node1_col: str = "node1",
    node2_col: str = "node2",
    weight_col: str = "weight",
    source: str = "generic_edge_list",
    aggregation: EdgeAggregation = "max",
    drop_self_edges: bool = True,
) -> GraphEdges:
    """Return canonical undirected edges with sorted node pairs."""

    _require_columns(edges, [node1_col, node2_col, weight_col], "edges")
    if aggregation not in {"max", "mean", "sum"}:
        raise ValueError("aggregation must be one of: max, mean, sum.")

    work = edges.loc[:, [node1_col, node2_col, weight_col]].copy()
    n_input = int(len(work))
    work[node1_col] = work[node1_col].map(_normalize_node_id)
    work[node2_col] = work[node2_col].map(_normalize_node_id)
    work["weight"] = pd.to_numeric(work[weight_col], errors="raise").astype(float)
    work = work[(work[node1_col] != "") & (work[node2_col] != "")]

    n_before_self = len(work)
    if drop_self_edges:
        work = work.loc[work[node1_col] != work[node2_col]].copy()
    n_self_dropped = n_before_self - len(work)

    if work.empty:
        out = pd.DataFrame(columns=["node1", "node2", "weight"])
    else:
        node_min = work[[node1_col, node2_col]].min(axis=1)
        node_max = work[[node1_col, node2_col]].max(axis=1)
        work["node1"] = node_min
        work["node2"] = node_max
        grouped = work.groupby(["node1", "node2"], observed=True, sort=True)["weight"]
        if aggregation == "max":
            out = grouped.max().reset_index()
        elif aggregation == "mean":
            out = grouped.mean().reset_index()
        else:
            out = grouped.sum().reset_index()

    nodes = set(out["node1"]).union(set(out["node2"])) if not out.empty else set()
    report = GraphLoadReport(
        source=source,
        n_rows_input=n_input,
        n_edges_output=int(len(out)),
        n_nodes_output=int(len(nodes)),
        n_self_edges_dropped=int(n_self_dropped),
        n_unmapped_edges_dropped=0,
        weight_col="weight",
    )
    return GraphEdges(edges=out.loc[:, ["node1", "node2", "weight"]], report=report)


def read_edge_list(
    path: str | Path,
    *,
    node1_col: str = "node1",
    node2_col: str = "node2",
    weight_col: str = "weight",
    sep: str | None = None,
    aggregation: EdgeAggregation = "max",
    source: str | None = None,
) -> GraphEdges:
    """Read and canonicalize a generic weighted edge list."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(
        path,
        sep=sep,
        compression="infer",
        engine="python" if sep is None else "c",
    )
    return canonicalize_undirected_edges(
        table,
        node1_col=node1_col,
        node2_col=node2_col,
        weight_col=weight_col,
        source=source or path.name,
        aggregation=aggregation,
    )


def write_edge_list(
    path: str | Path,
    edges: pd.DataFrame,
    *,
    node1_col: str = "node1",
    node2_col: str = "node2",
    weight_col: str = "weight",
    sep: str = "\t",
    aggregation: EdgeAggregation = "max",
    source: str = "generic_edge_list",
) -> GraphEdges:
    """Canonicalize and write an undirected weighted edge list."""

    path = Path(path)
    graph_edges = canonicalize_undirected_edges(
        edges,
        node1_col=node1_col,
        node2_col=node2_col,
        weight_col=weight_col,
        source=source,
        aggregation=aggregation,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    graph_edges.edges.to_csv(path, sep=sep, index=False, compression="infer")
    return graph_edges


def networkx_to_edge_list(graph: nx.Graph, *, weight_attr: str = "weight") -> pd.DataFrame:
    """Convert a NetworkX graph into canonical `node1/node2/weight` rows."""

    rows: list[dict[str, object]] = []
    for node1, node2, data in graph.edges(data=True):
        rows.append(
            {
                "node1": str(node1),
                "node2": str(node2),
                "weight": float(data.get(weight_attr, 1.0)),
            }
        )
    return pd.DataFrame(rows, columns=["node1", "node2", "weight"])


def read_string_links(
    path: str | Path,
    *,
    min_score: int | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """Read STRING `protein.links` or `protein.physical.links` files."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    out = pd.read_csv(
        path,
        sep=r"\s+",
        compression="infer",
        nrows=nrows,
        dtype={"protein1": str, "protein2": str, "combined_score": float},
    )
    _require_columns(out, ["protein1", "protein2", "combined_score"], "STRING links")
    if min_score is not None:
        out = out.loc[out["combined_score"] >= float(min_score)].copy()
    return out.reset_index(drop=True)


def read_string_protein_info(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read STRING `protein.info` metadata."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    out = pd.read_csv(path, sep="\t", compression="infer", nrows=nrows, comment=None)
    if "#string_protein_id" in out.columns:
        out = out.rename(columns={"#string_protein_id": "string_protein_id"})
    _require_columns(out, ["string_protein_id", "preferred_name"], "STRING protein info")
    out["string_protein_id"] = out["string_protein_id"].astype(str)
    out["preferred_name"] = out["preferred_name"].astype(str)
    return out


def read_string_aliases(
    path: str | Path,
    *,
    source_filter: Iterable[str] | None = None,
    nrows: int | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame:
    """Read STRING aliases, optionally filtering by alias source.

    For the full human alias file, use `source_filter` and `chunksize` to avoid
    loading irrelevant aliases into memory.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    sources = set(source_filter) if source_filter is not None else None
    reader = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        nrows=nrows,
        chunksize=chunksize,
        comment=None,
    )

    def clean(chunk: pd.DataFrame) -> pd.DataFrame:
        if "#string_protein_id" in chunk.columns:
            chunk = chunk.rename(columns={"#string_protein_id": "string_protein_id"})
        _require_columns(chunk, ["string_protein_id", "alias", "source"], "STRING aliases")
        if sources is not None:
            chunk = chunk.loc[chunk["source"].isin(sources)].copy()
        chunk["string_protein_id"] = chunk["string_protein_id"].astype(str)
        chunk["alias"] = chunk["alias"].astype(str)
        chunk["source"] = chunk["source"].astype(str)
        return chunk

    if chunksize is None:
        return clean(reader).reset_index(drop=True)

    chunks = [clean(chunk) for chunk in reader]
    if not chunks:
        return pd.DataFrame(columns=["string_protein_id", "alias", "source"])
    return pd.concat(chunks, ignore_index=True)


def string_protein_to_gene_map(
    protein_info: pd.DataFrame,
    *,
    protein_col: str = "string_protein_id",
    gene_col: str = "preferred_name",
) -> dict[str, str]:
    """Build default STRING protein ID -> gene symbol mapping from protein info."""

    _require_columns(protein_info, [protein_col, gene_col], "protein_info")
    pairs = protein_info.loc[:, [protein_col, gene_col]].dropna().drop_duplicates(protein_col)
    return dict(zip(pairs[protein_col].astype(str), pairs[gene_col].astype(str), strict=False))


def string_links_to_gene_edges(
    links: pd.DataFrame,
    protein_to_gene: dict[str, str],
    *,
    score_col: str = "combined_score",
    score_scale: float = 1000.0,
    aggregation: EdgeAggregation = "max",
    source: str = "STRING",
) -> GraphEdges:
    """Map STRING protein links to canonical gene-level weighted edges."""

    _require_columns(links, ["protein1", "protein2", score_col], "STRING links")
    n_input = int(len(links))
    work = links.copy()
    work["node1"] = work["protein1"].map(protein_to_gene)
    work["node2"] = work["protein2"].map(protein_to_gene)
    n_unmapped = int(work[["node1", "node2"]].isna().any(axis=1).sum())
    work = work.dropna(subset=["node1", "node2"]).copy()
    work["weight"] = pd.to_numeric(work[score_col], errors="raise").astype(float) / float(score_scale)

    graph = canonicalize_undirected_edges(
        work,
        node1_col="node1",
        node2_col="node2",
        weight_col="weight",
        source=source,
        aggregation=aggregation,
    )
    report = GraphLoadReport(
        source=graph.report.source,
        n_rows_input=n_input,
        n_edges_output=graph.report.n_edges_output,
        n_nodes_output=graph.report.n_nodes_output,
        n_self_edges_dropped=graph.report.n_self_edges_dropped,
        n_unmapped_edges_dropped=n_unmapped,
        weight_col="weight",
    )
    return GraphEdges(edges=graph.edges, report=report)


def read_string_gene_graph(
    links_path: str | Path,
    protein_info_path: str | Path,
    *,
    min_score: int | None = None,
    nrows: int | None = None,
    aggregation: EdgeAggregation = "max",
) -> GraphEdges:
    """Read STRING links and protein info into a gene-level graph."""

    links = read_string_links(links_path, min_score=min_score, nrows=nrows)
    protein_info = read_string_protein_info(protein_info_path)
    protein_map = string_protein_to_gene_map(protein_info)
    return string_links_to_gene_edges(links, protein_map, aggregation=aggregation, source="STRING")


def edge_list_to_networkx(edges: pd.DataFrame) -> nx.Graph:
    """Convert canonical edge list to an undirected NetworkX graph."""

    _require_columns(edges, ["node1", "node2", "weight"], "edges")
    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.node1), str(row.node2), weight=float(row.weight))
    return graph
