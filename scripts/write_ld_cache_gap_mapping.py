#!/usr/bin/env python
"""Write mapping rows for genes whose mapped SNPs are not covered by LD caches."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_height_mvp import write_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--ld-cache-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-mapping", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def read_cache_snps(cache_dirs: list[Path], gene_id: str) -> tuple[str, set[str]]:
    first_status = "missing_cache"
    for cache_dir in cache_dirs:
        path = cache_dir / "matrices" / f"{gene_id}.ld.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as data:
            status = str(data["status"])
            if first_status == "missing_cache":
                first_status = status
            if status == "computed":
                return status, {str(snp) for snp in data["snp_ids"]}
    return first_status, set()


def main() -> None:
    args = parse_args()
    mapping = pd.read_csv(
        args.mapping,
        sep="\t",
        compression="infer",
        dtype={"gene_id": str, "gene_symbol": str, "snp_id": str},
    )

    rows: list[pd.DataFrame] = []
    report_rows: list[dict[str, object]] = []
    for (gene_id, gene_symbol), group in mapping.groupby(["gene_id", "gene_symbol"], observed=True, sort=True):
        mapped_snps = set(group["snp_id"].astype(str))
        status, cache_snps = read_cache_snps(args.ld_cache_dir, str(gene_id))
        missing = sorted(mapped_snps - cache_snps)
        if status != "computed" or missing:
            rows.append(group)
            report_rows.append(
                {
                    "gene_id": str(gene_id),
                    "gene_symbol": str(gene_symbol),
                    "cache_status": status,
                    "n_mapped_snps": len(mapped_snps),
                    "n_cache_snps": len(cache_snps),
                    "n_missing_from_cache": len(missing),
                    "missing_snps_preview": ",".join(missing[:20]),
                }
            )

    gap_mapping = pd.concat(rows, ignore_index=True) if rows else mapping.iloc[0:0].copy()
    write_table(args.out_mapping, gap_mapping)
    report_table = pd.DataFrame(report_rows)
    write_table(args.report.with_suffix(".tsv"), report_table)
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "mapping": str(args.mapping),
        "ld_cache_dirs": [str(path) for path in args.ld_cache_dir],
        "out_mapping": str(args.out_mapping),
        "n_input_genes": int(mapping["gene_id"].nunique()),
        "n_gap_genes": int(len(report_table)),
        "n_gap_mapping_rows": int(len(gap_mapping)),
        "status_counts": report_table["cache_status"].value_counts(dropna=False).to_dict()
        if not report_table.empty
        else {},
    }
    args.report.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote LD cache gap mapping for {len(report_table):,} genes: {args.out_mapping}", flush=True)


if __name__ == "__main__":
    main()
