#!/usr/bin/env python
"""Build an FVM high-support graph with a sparse default STRING backbone.

This is a labeled graph-sensitivity construction script. It starts from the FVM
vascular high-support topology and greedily adds default STRING edges that
connect the current largest component to outside components until a target LCC
coverage is reached for the DR_MVP gene-score universe.

The graph is intended as an intermediate sensitivity artifact, not a default
RIPPLE graph.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.graph import preprocess_reference_graph  # noqa: E402
from ripple.io.graph import read_string_gene_graph  # noqa: E402
from run_height_mvp import DEFAULT_STRING_INFO, DEFAULT_STRING_LINKS, write_table  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_HIGH_SUPPORT_EDGES = (
    PRIVATE_ROOT
    / "20_processed_data"
    / "reference_graphs"
    / "fvm_vascular_string"
    / "tables"
    / "fvm_vascular_high_support_string.edges.tsv.gz"
)
DEFAULT_GENE_SUPPORT = (
    PRIVATE_ROOT
    / "20_processed_data"
    / "reference_graphs"
    / "fvm_vascular_string"
    / "tables"
    / "fvm_vascular_gene_support.tsv.gz"
)
DEFAULT_GENE_UNIVERSE = (
    PRIVATE_ROOT
    / "30_analysis"
    / "dr_mvp_analysis_ready"
    / "tables"
    / "DR_MVP.gene_scores.1000G_LD.tsv.gz"
)
DEFAULT_OUT_DIR = PRIVATE_ROOT / "20_processed_data" / "reference_graphs" / "fvm_vascular_backbone"

SCIENCE_BLUE_PALETTE = {
    "primary": "#1F77B4",
    "secondary": "#4C9ED9",
    "dark": "#145A8D",
    "light": "#D6E9F8",
    "accent": "#0B3C5D",
    "neutral": "#6B7280",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high-support-edge-list", type=Path, default=DEFAULT_HIGH_SUPPORT_EDGES)
    parser.add_argument("--gene-support", type=Path, default=DEFAULT_GENE_SUPPORT)
    parser.add_argument("--gene-universe", type=Path, default=DEFAULT_GENE_UNIVERSE)
    parser.add_argument("--string-links", type=Path, default=DEFAULT_STRING_LINKS)
    parser.add_argument("--string-info", type=Path, default=DEFAULT_STRING_INFO)
    parser.add_argument("--string-min-score", type=int, default=400)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-lcc-fraction", type=float, default=0.65)
    parser.add_argument("--max-lcc-fraction", type=float, default=0.70)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    for child in ("tables", "reports"):
        (path / child).mkdir(parents=True, exist_ok=True)


class UnionFind:
    def __init__(self, nodes: list[str]) -> None:
        self.parent = {node: node for node in nodes}
        self.size = {node: 1 for node in nodes}

    def find(self, node: str) -> str:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, left: str, right: str) -> str:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return root_left
        if self.size[root_left] < self.size[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        self.size[root_left] += self.size[root_right]
        del self.size[root_right]
        return root_left

    def component_size(self, node_or_root: str) -> int:
        return self.size[self.find(node_or_root)]

    def largest_root(self) -> str:
        return max(self.size, key=lambda root: (self.size[root], root))


def load_gene_universe(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, sep="\t", compression="infer", usecols=["gene_symbol"])
    genes = sorted(set(table["gene_symbol"].dropna().astype(str)))
    if not genes:
        raise ValueError(f"No gene symbols found in {path}")
    return tuple(genes)


def load_support(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, sep="\t", compression="infer", usecols=["gene_symbol", "vascular_support_score"])
    return dict(zip(table["gene_symbol"].astype(str), table["vascular_support_score"].astype(float), strict=False))


def load_high_support_edges(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, sep="\t", compression="infer")
    missing = [col for col in ("node1", "node2", "weight") if col not in table.columns]
    if missing:
        raise ValueError(f"High-support edge list is missing columns: {missing}")
    table = table.copy()
    table["node1"] = table["node1"].astype(str)
    table["node2"] = table["node2"].astype(str)
    table["weight"] = pd.to_numeric(table["weight"], errors="raise").astype(float)
    if "string_weight" not in table.columns:
        table["string_weight"] = table["weight"]
    if "edge_vascular_support" not in table.columns:
        table["edge_vascular_support"] = 1.0
    table["string_weight"] = pd.to_numeric(table["string_weight"], errors="raise").astype(float)
    table["edge_vascular_support"] = pd.to_numeric(table["edge_vascular_support"], errors="raise").astype(float)
    return table.loc[:, ["node1", "node2", "weight", "string_weight", "edge_vascular_support"]]


def edge_key(node1: str, node2: str) -> tuple[str, str]:
    return (node1, node2) if node1 <= node2 else (node2, node1)


def initial_union_find(edges: pd.DataFrame, universe: tuple[str, ...]) -> tuple[UnionFind, set[tuple[str, str]]]:
    uf = UnionFind(list(universe))
    edge_keys: set[tuple[str, str]] = set()
    universe_set = set(universe)
    for row in edges.itertuples(index=False):
        node1 = str(row.node1)
        node2 = str(row.node2)
        if node1 not in universe_set or node2 not in universe_set or node1 == node2:
            continue
        uf.union(node1, node2)
        edge_keys.add(edge_key(node1, node2))
    return uf, edge_keys


def candidate_backbone_edges(
    *,
    default_edges: pd.DataFrame,
    existing_keys: set[tuple[str, str]],
    support: dict[str, float],
    universe: tuple[str, ...],
) -> pd.DataFrame:
    universe_set = set(universe)
    work = default_edges.loc[
        default_edges["node1"].astype(str).isin(universe_set)
        & default_edges["node2"].astype(str).isin(universe_set)
    ].copy()
    work["node1"] = work["node1"].astype(str)
    work["node2"] = work["node2"].astype(str)
    work["edge_key"] = [edge_key(row.node1, row.node2) for row in work.itertuples(index=False)]
    work = work.loc[~work["edge_key"].isin(existing_keys)].copy()
    work["string_weight"] = pd.to_numeric(work["weight"], errors="raise").astype(float)
    work["node1_support"] = work["node1"].map(support).fillna(0.0).astype(float)
    work["node2_support"] = work["node2"].map(support).fillna(0.0).astype(float)
    work["edge_vascular_support"] = np.sqrt(work["node1_support"] * work["node2_support"])
    work["backbone_priority"] = (0.75 * work["string_weight"]) + (0.25 * work["edge_vascular_support"])
    return work.sort_values(
        ["backbone_priority", "string_weight", "edge_vascular_support"],
        ascending=False,
    ).reset_index(drop=True)


def add_backbone_edges(
    *,
    high_support_edges: pd.DataFrame,
    candidates: pd.DataFrame,
    universe: tuple[str, ...],
    target_fraction: float,
    max_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    if not 0 < target_fraction <= 1:
        raise ValueError("--target-lcc-fraction must be in (0, 1].")
    if not target_fraction <= max_fraction <= 1:
        raise ValueError("--max-lcc-fraction must be >= target and <= 1.")

    uf, existing_keys = initial_union_find(high_support_edges, universe)
    largest_root = uf.largest_root()
    initial_lcc_size = uf.component_size(largest_root)
    target_size = int(np.ceil(target_fraction * len(universe)))
    max_size = int(np.floor(max_fraction * len(universe)))
    selected_rows: list[dict[str, object]] = []

    for row in candidates.itertuples(index=False):
        node1 = str(row.node1)
        node2 = str(row.node2)
        root1 = uf.find(node1)
        root2 = uf.find(node2)
        if root1 == root2:
            continue
        largest_root = uf.largest_root()
        root1_is_lcc = root1 == largest_root
        root2_is_lcc = root2 == largest_root
        if not (root1_is_lcc or root2_is_lcc):
            continue
        outside_root = root2 if root1_is_lcc else root1
        projected_size = uf.component_size(largest_root) + uf.size[outside_root]
        if projected_size > max_size and uf.component_size(largest_root) >= target_size:
            break
        if projected_size > max_size:
            continue

        new_root = uf.union(node1, node2)
        largest_root = new_root if uf.component_size(new_root) >= uf.component_size(largest_root) else largest_root
        selected_rows.append(
            {
                "node1": node1,
                "node2": node2,
                "weight": float(row.string_weight),
                "string_weight": float(row.string_weight),
                "edge_vascular_support": float(row.edge_vascular_support),
                "backbone_priority": float(row.backbone_priority),
                "added_component_size": int(uf.component_size(largest_root)),
            }
        )
        if uf.component_size(largest_root) >= target_size:
            break

    backbone = pd.DataFrame(
        selected_rows,
        columns=[
            "node1",
            "node2",
            "weight",
            "string_weight",
            "edge_vascular_support",
            "backbone_priority",
            "added_component_size",
        ],
    )
    high = high_support_edges.copy()
    high["edge_source"] = "fvm_high_support"
    backbone_for_edges = backbone.loc[:, ["node1", "node2", "weight", "string_weight", "edge_vascular_support"]].copy()
    backbone_for_edges["edge_source"] = "default_string_backbone"
    combined = pd.concat(
        [
            high.loc[:, ["node1", "node2", "weight", "string_weight", "edge_vascular_support", "edge_source"]],
            backbone_for_edges,
        ],
        ignore_index=True,
    )
    final_lcc_size = uf.component_size(uf.largest_root())
    report = {
        "n_gene_universe": int(len(universe)),
        "target_lcc_fraction": float(target_fraction),
        "max_lcc_fraction": float(max_fraction),
        "target_lcc_size": int(target_size),
        "max_lcc_size": int(max_size),
        "initial_lcc_size": int(initial_lcc_size),
        "initial_lcc_fraction": float(initial_lcc_size / len(universe)),
        "final_lcc_size_union_find": int(final_lcc_size),
        "final_lcc_fraction_union_find": float(final_lcc_size / len(universe)),
        "n_backbone_edges_added": int(len(backbone)),
    }
    return combined.sort_values(["node1", "node2", "edge_source"]).reset_index(drop=True), backbone, report


def render_report(summary: dict[str, object]) -> str:
    palette = summary["science_blue_palette"]
    backbone = summary["backbone_report"]
    validation = summary["coverage_validation"]
    return "\n".join(
        [
            "# FVM Vascular High-Support Plus Default Backbone Graph",
            "",
            "This is an intermediate RIPPLE graph-sensitivity artifact. It is not a frozen default graph.",
            "",
            "## Visual Style",
            "",
            f"- Primary science blue: `{palette['primary']}`",
            f"- Secondary science blue: `{palette['secondary']}`",
            f"- Dark science blue: `{palette['dark']}`",
            f"- Light science blue: `{palette['light']}`",
            "",
            "## Backbone Construction",
            "",
            f"- Initial LCC fraction: {backbone['initial_lcc_fraction']:.3f}",
            f"- Target LCC fraction: {backbone['target_lcc_fraction']:.3f}",
            f"- Max LCC fraction: {backbone['max_lcc_fraction']:.3f}",
            f"- Backbone edges added: {backbone['n_backbone_edges_added']:,}",
            f"- Final union-find LCC fraction: {backbone['final_lcc_fraction_union_find']:.3f}",
            "",
            "## Coverage Validation",
            "",
            f"- Input edges: {validation['n_input_edges']:,}",
            f"- Input nodes: {validation['n_input_nodes']:,}",
            f"- Largest component size: {validation['largest_component_size']:,}",
            f"- Largest component gene fraction: {validation['largest_component_gene_fraction']:.3f}",
            "",
            "## Intended Use",
            "",
            "Use this graph with `run_trait_ld_analysis.py --graph-name fvm_vascular_backbone --graph-edge-list <edge-list>`.",
            "The key test is whether this graph keeps better coverage than the FVM high-support topology while reducing topology-null sensitivity relative to default STRING.",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"

    universe = load_gene_universe(args.gene_universe)
    support = load_support(args.gene_support)
    high_support = load_high_support_edges(args.high_support_edge_list)
    default_string = read_string_gene_graph(args.string_links, args.string_info, min_score=args.string_min_score).edges
    _, existing_keys = initial_union_find(high_support, universe)
    candidates = candidate_backbone_edges(
        default_edges=default_string,
        existing_keys=existing_keys,
        support=support,
        universe=universe,
    )
    combined, backbone, backbone_report = add_backbone_edges(
        high_support_edges=high_support,
        candidates=candidates,
        universe=universe,
        target_fraction=args.target_lcc_fraction,
        max_fraction=args.max_lcc_fraction,
    )
    canonical = combined.loc[:, ["node1", "node2", "weight"]].copy()
    pre = preprocess_reference_graph(canonical, gene_universe=universe)
    coverage_validation = asdict(pre.coverage_report)

    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "graph_name": "fvm_vascular_high_support_plus_default_backbone",
        "source_high_support_edge_list": str(args.high_support_edge_list),
        "source_gene_support": str(args.gene_support),
        "source_gene_universe": str(args.gene_universe),
        "string_min_score": int(args.string_min_score),
        "science_blue_palette": SCIENCE_BLUE_PALETTE,
        "backbone_report": backbone_report,
        "coverage_validation": coverage_validation,
        "outputs": {
            "edge_list": str(tables_dir / "fvm_vascular_high_support_plus_default_backbone.edges.tsv.gz"),
            "edge_provenance": str(tables_dir / "fvm_vascular_high_support_plus_default_backbone.edge_provenance.tsv.gz"),
            "backbone_edges": str(tables_dir / "fvm_vascular_default_backbone_edges.tsv.gz"),
            "palette": str(reports_dir / "science_blue_palette.json"),
            "summary": str(reports_dir / "fvm_vascular_backbone.summary.json"),
            "report": str(reports_dir / "fvm_vascular_backbone.report.md"),
        },
    }
    write_table(tables_dir / "fvm_vascular_high_support_plus_default_backbone.edges.tsv.gz", canonical)
    write_table(tables_dir / "fvm_vascular_high_support_plus_default_backbone.edge_provenance.tsv.gz", combined)
    write_table(tables_dir / "fvm_vascular_default_backbone_edges.tsv.gz", backbone)
    (reports_dir / "science_blue_palette.json").write_text(
        json.dumps(SCIENCE_BLUE_PALETTE, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (reports_dir / "fvm_vascular_backbone.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (reports_dir / "fvm_vascular_backbone.report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
