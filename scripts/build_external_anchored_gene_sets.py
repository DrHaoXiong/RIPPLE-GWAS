#!/usr/bin/env python
"""Build external anchored gene sets from Reactome, GO, and optional MSigDB files."""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import urllib.request
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
RAW_PATHWAY_ROOT = PRIVATE_ROOT / "10_raw_data" / "reference" / "pathways"
PROCESSED_ROOT = PRIVATE_ROOT / "20_processed_data" / "reference_pathways" / "anchored_external_v1"
BROAD_PROCESSED_ROOT = PRIVATE_ROOT / "20_processed_data" / "reference_pathways" / "anchored_broad_reactome_go_v1"

REACTOME_GMT_URL = "https://reactome.org/download/current/ReactomePathways.gmt.zip"
GO_OBO_URL = "https://current.geneontology.org/ontology/go-basic.obo"
GOA_HUMAN_GAF_URL = "https://current.geneontology.org/annotations/goa_human.gaf.gz"

CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "retinal_vascular": (
        "retina vasculature",
        "retinal vasculature",
        "retinal blood vessel",
        "retina blood vessel",
        "retinal vascular",
    ),
    "ecm": (
        "extracellular matrix",
        "basement membrane",
        "collagen",
        "elastic fiber",
        "matrix organization",
        "integrin cell surface",
        "cell-matrix adhesion",
    ),
    "angiogenesis": (
        "angiogenesis",
        "blood vessel development",
        "blood vessel morphogenesis",
        "vascular development",
        "vasculogenesis",
        "endothelial cell migration",
        "endothelial cell proliferation",
        "sprouting angiogenesis",
        "vegf",
        "vascular endothelial growth factor",
    ),
    "complement": (
        "complement activation",
        "complement cascade",
        "regulation of complement",
        "classical complement",
        "alternative complement",
        "terminal pathway of complement",
    ),
    "inflammation": (
        "inflammatory response",
        "inflammation",
        "leukocyte migration",
        "leukocyte chemotaxis",
        "cytokine-mediated signaling",
        "chemokine",
        "interleukin",
        "tumor necrosis factor",
        "tnf",
        "nf-kappab",
        "nf-kappa b",
        "toll-like receptor",
    ),
}


@dataclass(frozen=True)
class GoTerm:
    go_id: str
    name: str
    namespace: str
    parents: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_PATHWAY_ROOT)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_ROOT)
    parser.add_argument(
        "--mode",
        choices=("focused_microvascular", "broad_reactome_go"),
        default="focused_microvascular",
    )
    parser.add_argument("--msigdb-dir", type=Path, default=RAW_PATHWAY_ROOT / "msigdb")
    parser.add_argument("--min-genes", type=int, default=5)
    parser.add_argument("--max-genes", type=int, default=500)
    parser.add_argument("--max-union-genes", type=int, default=2000)
    parser.add_argument("--jaccard-threshold", type=float, default=0.70)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    if args.mode == "broad_reactome_go" and args.out_dir == PROCESSED_ROOT:
        args.out_dir = BROAD_PROCESSED_ROOT
    return args


def slug(value: str, *, max_len: int = 80) -> str:
    out = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().upper()).strip("_")
    out = re.sub(r"_+", "_", out)
    return out[:max_len] if len(out) > max_len else out


def category_for_name(name: str) -> str | None:
    lower = name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lower for keyword in keywords):
            return category
    if ("retina" in lower or "retinal" in lower) and any(
        keyword in lower for keyword in ("vascular", "vasculature", "blood vessel", "endothelial")
    ):
        return "retinal_vascular"
    return None


def ensure_download(url: str, path: Path, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not force:
        return
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 RIPPLE-GWAS reference downloader"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def parse_reactome_gmt(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        gmt_names = [name for name in archive.namelist() if name.endswith(".gmt")]
        if not gmt_names:
            raise ValueError(f"No GMT file found in {path}.")
        with archive.open(gmt_names[0]) as handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                parts = [part.strip() for part in line.split("\t") if part.strip()]
                if len(parts) < 3:
                    continue
                name, description, genes = parts[0], parts[1], {gene.upper() for gene in parts[2:]}
                category = category_for_name(name)
                if category is None:
                    continue
                match = re.search(r"R-HSA-\d+", description)
                term_id = match.group(0) if match else ""
                rows.append(
                    {
                        "source_database": "Reactome",
                        "source_term_id": term_id,
                        "source_term_name": name,
                        "source_url": description,
                        "category": category,
                        "genes": genes,
                    }
                )
    return rows


def parse_reactome_gmt_all(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(path) as archive:
        gmt_names = [name for name in archive.namelist() if name.endswith(".gmt")]
        if not gmt_names:
            raise ValueError(f"No GMT file found in {path}.")
        with archive.open(gmt_names[0]) as handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                parts = [part.strip() for part in line.split("\t") if part.strip()]
                if len(parts) < 3:
                    continue
                name, description, genes = parts[0], parts[1], {gene.upper() for gene in parts[2:]}
                match = re.search(r"R-HSA-\d+", description)
                term_id = match.group(0) if match else ""
                rows.append(
                    {
                        "source_database": "Reactome",
                        "source_term_id": term_id,
                        "source_term_name": name,
                        "source_url": description,
                        "category": "reactome_pathway",
                        "genes": genes,
                    }
                )
    return rows


def parse_go_obo(path: Path) -> dict[str, GoTerm]:
    terms: dict[str, GoTerm] = {}
    current: dict[str, object] = {}
    in_term = False

    def commit(stanza: dict[str, object]) -> None:
        if stanza.get("obsolete") or not stanza.get("id") or not stanza.get("name"):
            return
        go_id = str(stanza["id"])
        terms[go_id] = GoTerm(
            go_id=go_id,
            name=str(stanza["name"]),
            namespace=str(stanza.get("namespace", "")),
            parents=tuple(str(parent) for parent in stanza.get("parents", [])),
        )

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "[Term]":
            if in_term and current:
                commit(current)
            current = {}
            in_term = True
            continue
        if line.startswith("["):
            if in_term and current:
                commit(current)
            current = {}
            in_term = False
            continue
        if not in_term or not line or line.startswith("!"):
            continue
        if line.startswith("id: "):
            current["id"] = line.removeprefix("id: ").strip()
        elif line.startswith("name: "):
            current["name"] = line.removeprefix("name: ").strip()
        elif line.startswith("namespace: "):
            current["namespace"] = line.removeprefix("namespace: ").strip()
        elif line.startswith("is_a: "):
            parent = line.removeprefix("is_a: ").split()[0]
            current.setdefault("parents", []).append(parent)
        elif line.startswith("is_obsolete: true"):
            current["obsolete"] = True
    if in_term and current:
        commit(current)
    return terms


def descendants_by_term(terms: dict[str, GoTerm]) -> dict[str, set[str]]:
    children: dict[str, set[str]] = defaultdict(set)
    for term in terms.values():
        for parent in term.parents:
            children[parent].add(term.go_id)
    out: dict[str, set[str]] = {}
    for go_id in terms:
        seen: set[str] = set()
        queue: deque[str] = deque(children.get(go_id, set()))
        while queue:
            child = queue.popleft()
            if child in seen:
                continue
            seen.add(child)
            queue.extend(children.get(child, set()))
        out[go_id] = seen
    return out


def parse_goa_human(path: Path) -> dict[str, set[str]]:
    term_to_genes: dict[str, set[str]] = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line or line.startswith("!"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            qualifier = parts[3]
            if "NOT" in qualifier.split("|"):
                continue
            symbol = parts[2].strip().upper()
            go_id = parts[4].strip()
            if symbol and go_id:
                term_to_genes[go_id].add(symbol)
    return term_to_genes


def build_go_rows(obo_path: Path, gaf_path: Path) -> list[dict[str, object]]:
    terms = parse_go_obo(obo_path)
    descendants = descendants_by_term(terms)
    direct_annotations = parse_goa_human(gaf_path)
    rows: list[dict[str, object]] = []
    for go_id, term in terms.items():
        category = category_for_name(term.name)
        if category is None:
            continue
        term_ids = {go_id} | descendants.get(go_id, set())
        genes: set[str] = set()
        for term_id in term_ids:
            genes.update(direct_annotations.get(term_id, set()))
        rows.append(
            {
                "source_database": "Gene Ontology",
                "source_term_id": go_id,
                "source_term_name": term.name,
                "source_url": f"https://amigo.geneontology.org/amigo/term/{go_id}",
                "category": category,
                "genes": genes,
            }
        )
    return rows


def build_go_bp_rows(obo_path: Path, gaf_path: Path) -> list[dict[str, object]]:
    terms = parse_go_obo(obo_path)
    descendants = descendants_by_term(terms)
    direct_annotations = parse_goa_human(gaf_path)
    rows: list[dict[str, object]] = []
    for go_id, term in terms.items():
        if term.namespace != "biological_process":
            continue
        term_ids = {go_id} | descendants.get(go_id, set())
        genes: set[str] = set()
        for term_id in term_ids:
            genes.update(direct_annotations.get(term_id, set()))
        rows.append(
            {
                "source_database": "Gene Ontology",
                "source_term_id": go_id,
                "source_term_name": term.name,
                "source_url": f"https://amigo.geneontology.org/amigo/term/{go_id}",
                "category": "go_biological_process",
                "genes": genes,
            }
        )
    return rows


def parse_gmt_file(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) < 3:
            continue
        name, description, genes = parts[0], parts[1], {gene.upper() for gene in parts[2:]}
        category = category_for_name(name)
        if category is None and description:
            category = category_for_name(description)
        if category is None:
            continue
        rows.append(
            {
                "source_database": "MSigDB_local",
                "source_term_id": name,
                "source_term_name": name,
                "source_url": description,
                "category": category,
                "genes": genes,
            }
        )
    return rows


def build_msigdb_rows(msigdb_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not msigdb_dir.exists():
        return rows
    for path in sorted(msigdb_dir.glob("*.gmt")):
        rows.extend(parse_gmt_file(path))
    return rows


def rows_to_tables(
    source_rows: list[dict[str, object]],
    *,
    min_genes: int,
    max_genes: int,
    max_union_genes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[dict[str, object]] = []
    term_rows: list[dict[str, object]] = []
    category_union: dict[str, set[str]] = defaultdict(set)
    for row in source_rows:
        genes = {str(gene).upper() for gene in row["genes"]}
        n_genes = len(genes)
        include = min_genes <= n_genes <= max_genes
        term_name = str(row["source_term_name"])
        gene_set = (
            f"{slug(str(row['source_database']))}_{str(row['category']).upper()}_"
            f"{slug(str(row['source_term_id']) or term_name, max_len=40)}_{slug(term_name, max_len=80)}"
        )
        term_rows.append(
            {
                "gene_set": gene_set,
                "source_database": row["source_database"],
                "source_term_id": row["source_term_id"],
                "source_term_name": term_name,
                "source_url": row["source_url"],
                "category": row["category"],
                "n_genes": n_genes,
                "included": include,
                "drop_reason": "" if include else "outside_gene_count_bounds",
            }
        )
        if not include:
            continue
        category_union[str(row["category"])].update(genes)
        for gene in sorted(genes):
            selected_rows.append(
                {
                    "gene_set": gene_set,
                    "gene_symbol": gene,
                    "source_database": row["source_database"],
                    "source_term_id": row["source_term_id"],
                    "source_term_name": term_name,
                    "source_url": row["source_url"],
                    "category": row["category"],
                    "gene_set_scope": "external_term",
                    "module_source": row["source_database"],
                    "annotation_source_type": "independent_external"
                    if row["source_database"] != "MSigDB_local"
                    else "licensed_external",
                }
            )
    for category, genes in sorted(category_union.items()):
        if not (min_genes <= len(genes) <= max_union_genes):
            continue
        gene_set = f"EXTERNAL_UNION_{category.upper()}"
        for gene in sorted(genes):
            selected_rows.append(
                {
                    "gene_set": gene_set,
                    "gene_symbol": gene,
                    "source_database": "Reactome_GO_MSigDB_union",
                    "source_term_id": gene_set,
                    "source_term_name": f"{category} external union",
                    "source_url": "",
                    "category": category,
                    "gene_set_scope": "category_union",
                    "module_source": "external_category_union",
                    "annotation_source_type": "independent_external",
                }
            )
    return pd.DataFrame(selected_rows), pd.DataFrame(term_rows)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return float(len(a & b) / union) if union else 0.0


def prune_redundant_rows(
    source_rows: list[dict[str, object]],
    *,
    jaccard_threshold: float,
) -> tuple[list[dict[str, object]], dict[int, tuple[str, float]]]:
    """Prune highly overlapping gene sets within each source/category group.

    Rows are sorted by source, category, size, then term name. This favors
    smaller and more specific terms when GO/Reactome parent-child sets overlap.
    """

    kept: list[dict[str, object]] = []
    dropped: dict[int, tuple[str, float]] = {}
    rows_with_index = [(int(row.get("_source_index", idx)), row) for idx, row in enumerate(source_rows)]
    rows_with_index.sort(
        key=lambda item: (
            str(item[1]["source_database"]),
            str(item[1]["category"]),
            len(item[1]["genes"]),
            str(item[1]["source_term_name"]),
        )
    )
    kept_by_group: dict[tuple[str, str], list[tuple[int, dict[str, object]]]] = defaultdict(list)
    for original_idx, row in rows_with_index:
        group = (str(row["source_database"]), str(row["category"]))
        genes = {str(gene).upper() for gene in row["genes"]}
        redundant_with = ""
        redundant_jaccard = 0.0
        for kept_idx, kept_row in kept_by_group[group]:
            overlap = jaccard(genes, {str(gene).upper() for gene in kept_row["genes"]})
            if overlap >= jaccard_threshold:
                redundant_with = str(kept_row["source_term_id"] or kept_row["source_term_name"])
                redundant_jaccard = overlap
                break
        if redundant_with:
            dropped[original_idx] = (redundant_with, redundant_jaccard)
            continue
        kept.append(row)
        kept_by_group[group].append((original_idx, row))
    return kept, dropped


def broad_rows_to_tables(
    source_rows: list[dict[str, object]],
    *,
    min_genes: int,
    max_genes: int,
    jaccard_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    size_filtered: list[dict[str, object]] = []
    term_rows: list[dict[str, object]] = []
    for original_idx, row in enumerate(source_rows):
        genes = {str(gene).upper() for gene in row["genes"]}
        n_genes = len(genes)
        include_size = min_genes <= n_genes <= max_genes
        gene_set = (
            f"{slug(str(row['source_database']))}_{slug(str(row['category']))}_"
            f"{slug(str(row['source_term_id']) or str(row['source_term_name']), max_len=40)}_"
            f"{slug(str(row['source_term_name']), max_len=80)}"
        )
        term_rows.append(
            {
                "gene_set": gene_set,
                "source_database": row["source_database"],
                "source_term_id": row["source_term_id"],
                "source_term_name": row["source_term_name"],
                "source_url": row["source_url"],
                "category": row["category"],
                "n_genes": n_genes,
                "included_before_pruning": include_size,
                "included": include_size,
                "drop_reason": "" if include_size else "outside_gene_count_bounds",
                "pruned_by": "",
                "pruned_jaccard": "",
            }
        )
        if include_size:
            row_with_index = dict(row)
            row_with_index["_source_index"] = original_idx
            size_filtered.append(row_with_index)

    pruned_rows, dropped_pruned = prune_redundant_rows(
        size_filtered,
        jaccard_threshold=jaccard_threshold,
    )
    for original_idx, (pruned_by, overlap) in dropped_pruned.items():
        term_rows[int(original_idx)]["included"] = False
        term_rows[int(original_idx)]["drop_reason"] = "jaccard_redundant"
        term_rows[int(original_idx)]["pruned_by"] = pruned_by
        term_rows[int(original_idx)]["pruned_jaccard"] = f"{overlap:.6f}"

    selected_rows: list[dict[str, object]] = []
    for row in pruned_rows:
        genes = {str(gene).upper() for gene in row["genes"]}
        gene_set = (
            f"{slug(str(row['source_database']))}_{slug(str(row['category']))}_"
            f"{slug(str(row['source_term_id']) or str(row['source_term_name']), max_len=40)}_"
            f"{slug(str(row['source_term_name']), max_len=80)}"
        )
        for gene in sorted(genes):
            selected_rows.append(
                {
                    "gene_set": gene_set,
                    "gene_symbol": gene,
                    "source_database": row["source_database"],
                    "source_term_id": row["source_term_id"],
                    "source_term_name": row["source_term_name"],
                    "source_url": row["source_url"],
                    "category": row["category"],
                    "gene_set_scope": "broad_reference_term",
                    "module_source": row["source_database"],
                    "annotation_source_type": "independent_external",
                }
            )
    pruning_summary = {
        "n_source_terms": int(len(source_rows)),
        "n_size_filtered_terms": int(len(size_filtered)),
        "n_pruned_redundant_terms": int(len(dropped_pruned)),
        "n_included_terms_after_pruning": int(len(pruned_rows)),
        "jaccard_threshold": float(jaccard_threshold),
    }
    return pd.DataFrame(selected_rows), pd.DataFrame(term_rows), pruning_summary


def main() -> None:
    args = parse_args()
    raw_reactome = args.raw_root / "reactome_current" / "ReactomePathways.gmt.zip"
    raw_go_obo = args.raw_root / "go_current" / "go-basic.obo"
    raw_goa = args.raw_root / "go_current" / "goa_human.gaf.gz"
    ensure_download(REACTOME_GMT_URL, raw_reactome, force=args.force_download)
    ensure_download(GO_OBO_URL, raw_go_obo, force=args.force_download)
    ensure_download(GOA_HUMAN_GAF_URL, raw_goa, force=args.force_download)

    source_rows = []
    pruning_summary: dict[str, object] = {}
    if args.mode == "broad_reactome_go":
        source_rows.extend(parse_reactome_gmt_all(raw_reactome))
        source_rows.extend(build_go_bp_rows(raw_go_obo, raw_goa))
        gene_sets, terms, pruning_summary = broad_rows_to_tables(
            source_rows,
            min_genes=args.min_genes,
            max_genes=args.max_genes,
            jaccard_threshold=args.jaccard_threshold,
        )
    else:
        source_rows.extend(parse_reactome_gmt(raw_reactome))
        source_rows.extend(build_go_rows(raw_go_obo, raw_goa))
        source_rows.extend(build_msigdb_rows(args.msigdb_dir))
        gene_sets, terms = rows_to_tables(
            source_rows,
            min_genes=args.min_genes,
            max_genes=args.max_genes,
            max_union_genes=args.max_union_genes,
        )

    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = "anchored_broad_reactome_go" if args.mode == "broad_reactome_go" else "anchored_external"
    gene_set_path = tables_dir / f"{stem}_gene_sets.tsv.gz"
    term_path = tables_dir / f"{stem}_selected_terms.tsv"
    gene_sets.to_csv(gene_set_path, sep="\t", index=False, compression="infer")
    terms.to_csv(term_path, sep="\t", index=False)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "mode": args.mode,
        "reactome_url": REACTOME_GMT_URL,
        "go_obo_url": GO_OBO_URL,
        "goa_human_gaf_url": GOA_HUMAN_GAF_URL,
        "msigdb_dir": str(args.msigdb_dir),
        "msigdb_note": "MSigDB files are included only if locally supplied; automatic download is not attempted because MSigDB access is license/login controlled.",
        "n_source_terms_matching_keywords": int(len(source_rows)),
        "n_included_gene_sets": int(gene_sets["gene_set"].nunique()) if not gene_sets.empty else 0,
        "n_gene_set_rows": int(len(gene_sets)),
        "category_counts": gene_sets.groupby("category", observed=True)["gene_set"].nunique().to_dict()
        if not gene_sets.empty
        else {},
        "source_counts": gene_sets.groupby("source_database", observed=True)["gene_set"].nunique().to_dict()
        if not gene_sets.empty
        else {},
        "min_genes": int(args.min_genes),
        "max_genes": int(args.max_genes),
        "max_union_genes": int(args.max_union_genes),
        "jaccard_threshold": float(args.jaccard_threshold),
        "pruning_summary": pruning_summary,
        "outputs": {
            "gene_sets": str(gene_set_path),
            "selected_terms": str(term_path),
        },
    }
    (reports_dir / f"{stem}_gene_sets.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {args.mode} anchored gene sets to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
