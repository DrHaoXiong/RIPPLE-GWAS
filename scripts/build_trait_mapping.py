#!/usr/bin/env python
"""Build positional SNP-to-gene mapping for a QC'ed GWAS trait."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ripple.io.annotations import read_magma_gene_loc  # noqa: E402
from ripple.mapping.weights import add_positional_weights, summarize_mapping  # noqa: E402
from run_height_mvp import DEFAULT_GENE_LOC, fast_positional_map_by_gene, load_height_gwas, write_table  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_GWAS_DIR = PRIVATE_ROOT / "20_processed_data" / "gwas_qc" / "core_hm3_no_mhc"


def trait_slug(trait: str) -> str:
    return trait.lower().replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--gwas", type=Path, default=None)
    parser.add_argument("--gene-loc", type=Path, default=DEFAULT_GENE_LOC)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--upstream-bp", type=int, default=0)
    parser.add_argument("--downstream-bp", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trait = str(args.trait)
    gwas_path = args.gwas or DEFAULT_GWAS_DIR / f"{trait}.tsv.gz"
    out_dir = args.out_dir or PRIVATE_ROOT / "30_analysis" / f"{trait_slug(trait)}_mvp"
    tables_dir = out_dir / "tables"
    reports_dir = out_dir / "reports"
    mapping_path = tables_dir / f"{trait}.gene_body_mapping.tsv.gz"
    if mapping_path.exists() and not args.force:
        raise FileExistsError(f"{mapping_path} exists. Use --force to overwrite.")

    print(f"Loading GWAS: {gwas_path}", flush=True)
    gwas = load_height_gwas(gwas_path)
    print("Loading MAGMA gene locations", flush=True)
    genes = read_magma_gene_loc(args.gene_loc).table
    genes = genes[genes["chrom"].astype(str).isin({str(i) for i in range(1, 23)})].copy()

    print("Building positional SNP-to-gene mapping", flush=True)
    mapping = fast_positional_map_by_gene(
        gwas,
        genes,
        upstream_bp=args.upstream_bp,
        downstream_bp=args.downstream_bp,
    )
    mapping = add_positional_weights(mapping)
    write_table(mapping_path, mapping)
    mapping_summary = summarize_mapping(mapping)

    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "trait": trait,
        "gwas": str(gwas_path),
        "gene_loc": str(args.gene_loc),
        "mapping": str(mapping_path),
        "upstream_bp": int(args.upstream_bp),
        "downstream_bp": int(args.downstream_bp),
        "n_gwas_snps": int(len(gwas)),
        "n_mapping_rows": int(len(mapping)),
        "mapping_summary": asdict(mapping_summary),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{trait}.mapping_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote mapping: {mapping_path}", flush=True)


if __name__ == "__main__":
    main()
