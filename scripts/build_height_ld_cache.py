#!/usr/bin/env python
"""Build per-gene 1000G EUR LD cache for the HEIGHT_IRN pilot mapping."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from bed_reader import open_bed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.io.ld import correlation_matrix_from_genotypes, summarize_ld_matrix  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_MAPPING = (
    PRIVATE_ROOT
    / "30_analysis"
    / "height_irn_mvp"
    / "tables"
    / "HEIGHT_IRN.gene_body_mapping.tsv.gz"
)
DEFAULT_BFILE = (
    PRIVATE_ROOT
    / "10_raw_data"
    / "reference"
    / "ld"
    / "1000G_phase3_GRCh37"
    / "EUR"
    / "g1000_eur"
    / "g1000_eur"
)
DEFAULT_OUT_DIR = PRIVATE_ROOT / "30_analysis" / "height_irn_mvp" / "ld_cache_1000G_EUR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--bfile", type=Path, default=DEFAULT_BFILE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-snps", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-gene", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def cache_path(cache_dir: Path, gene_id: str) -> Path:
    return cache_dir / f"{gene_id}.ld.npz"


def load_mapping(path: Path) -> pd.DataFrame:
    columns = ["gene_id", "gene_symbol", "snp_id"]
    mapping = pd.read_csv(path, sep="\t", compression="infer", usecols=columns, dtype=str)
    return mapping.drop_duplicates(["gene_id", "snp_id"]).reset_index(drop=True)


def build_snp_index(bim_path: Path, requested_snps: set[str]) -> dict[str, int]:
    index: dict[str, int] = {}
    with bim_path.open("rt", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            fields = line.split()
            if len(fields) < 2:
                continue
            snp_id = fields[1]
            if snp_id in requested_snps and snp_id not in index:
                index[snp_id] = idx
    return index


def save_cache_entry(
    path: Path,
    *,
    gene_id: str,
    gene_symbol: str,
    snp_ids: tuple[str, ...],
    ld: np.ndarray,
    status: str,
    missing_snps: tuple[str, ...] = (),
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_ld_matrix(ld)
    np.savez_compressed(
        path,
        gene_id=np.array(gene_id),
        gene_symbol=np.array(gene_symbol),
        snp_ids=np.array(snp_ids, dtype=object),
        ld=ld.astype(np.float32),
        m_eff=np.array(summary.m_eff, dtype=float),
        local_ld_score=np.array(summary.local_ld_score, dtype=float),
        status=np.array(status),
        missing_snps=np.array(missing_snps, dtype=object),
    )
    return {
        "gene_id": gene_id,
        "gene_symbol": gene_symbol,
        "n_snps": len(snp_ids),
        "n_missing_snps": len(missing_snps),
        "m_eff": summary.m_eff,
        "local_ld_score": summary.local_ld_score,
        "status": status,
        "path": str(path),
    }


def save_skipped_entry(
    path: Path,
    *,
    gene_id: str,
    gene_symbol: str,
    snp_ids: tuple[str, ...],
    status: str,
    missing_snps: tuple[str, ...] = (),
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        gene_id=np.array(gene_id),
        gene_symbol=np.array(gene_symbol),
        snp_ids=np.array(snp_ids, dtype=object),
        ld=np.empty((0, 0), dtype=np.float32),
        m_eff=np.array(np.nan, dtype=float),
        local_ld_score=np.array(np.nan, dtype=float),
        status=np.array(status),
        missing_snps=np.array(missing_snps, dtype=object),
    )
    return {
        "gene_id": gene_id,
        "gene_symbol": gene_symbol,
        "n_snps": len(snp_ids),
        "n_missing_snps": len(missing_snps),
        "m_eff": np.nan,
        "local_ld_score": np.nan,
        "status": status,
        "path": str(path),
    }


def main() -> None:
    args = parse_args()
    if args.max_snps < 1:
        raise ValueError("--max-snps must be positive.")

    cache_dir = args.out_dir / "matrices"
    manifest_path = args.out_dir / "manifest.tsv"
    report_path = args.out_dir / "ld_cache_report.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading mapping: {args.mapping}", flush=True)
    mapping = load_mapping(args.mapping)
    if args.only_gene:
        wanted = {str(gene) for gene in args.only_gene}
        mapping = mapping[mapping["gene_id"].isin(wanted) | mapping["gene_symbol"].isin(wanted)].copy()
    if mapping.empty:
        raise RuntimeError("No mapping rows selected.")

    groups = list(mapping.groupby(["gene_id", "gene_symbol"], observed=True, sort=True))
    if args.limit is not None:
        groups = groups[: args.limit]

    requested = set(mapping["snp_id"].astype(str))
    print(f"Indexing {len(requested):,} requested SNPs in BIM", flush=True)
    snp_index = build_snp_index(args.bfile.with_suffix(".bim"), requested)
    print(f"Found {len(snp_index):,} requested SNPs in 1000G EUR BIM", flush=True)

    bed = open_bed(str(args.bfile.with_suffix(".bed")))
    rows: list[dict[str, object]] = []
    for counter, ((gene_id, gene_symbol), group) in enumerate(groups, start=1):
        gene_id = str(gene_id)
        gene_symbol = str(gene_symbol)
        out_path = cache_path(cache_dir, gene_id)
        if out_path.exists() and not args.force:
            with np.load(out_path, allow_pickle=True) as data:
                rows.append(
                    {
                        "gene_id": gene_id,
                        "gene_symbol": gene_symbol,
                        "n_snps": int(len(data["snp_ids"])),
                        "n_missing_snps": int(len(data["missing_snps"])),
                        "m_eff": float(data["m_eff"]),
                        "local_ld_score": float(data["local_ld_score"]),
                        "status": str(data["status"]),
                        "path": str(out_path),
                    }
                )
            continue

        snp_ids_all = tuple(dict.fromkeys(group["snp_id"].astype(str)))
        present = tuple(snp for snp in snp_ids_all if snp in snp_index)
        missing = tuple(snp for snp in snp_ids_all if snp not in snp_index)

        if not present:
            rows.append(
                save_skipped_entry(
                    out_path,
                    gene_id=gene_id,
                    gene_symbol=gene_symbol,
                    snp_ids=(),
                    status="skipped_no_reference_snps",
                    missing_snps=missing,
                )
            )
        elif len(present) > args.max_snps:
            rows.append(
                save_skipped_entry(
                    out_path,
                    gene_id=gene_id,
                    gene_symbol=gene_symbol,
                    snp_ids=present,
                    status="skipped_too_many_snps",
                    missing_snps=missing,
                )
            )
        elif len(present) == 1:
            rows.append(
                save_cache_entry(
                    out_path,
                    gene_id=gene_id,
                    gene_symbol=gene_symbol,
                    snp_ids=present,
                    ld=np.ones((1, 1), dtype=float),
                    status="computed",
                    missing_snps=missing,
                )
            )
        else:
            variant_indices = [snp_index[snp] for snp in present]
            genotypes = bed.read(index=(slice(None), variant_indices))
            ld = correlation_matrix_from_genotypes(genotypes)
            rows.append(
                save_cache_entry(
                    out_path,
                    gene_id=gene_id,
                    gene_symbol=gene_symbol,
                    snp_ids=present,
                    ld=ld,
                    status="computed",
                    missing_snps=missing,
                )
            )

        if counter % 500 == 0 or counter == len(groups):
            print(f"Processed {counter:,}/{len(groups):,} genes", flush=True)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_path, sep="\t", index=False)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "mapping": str(args.mapping),
        "bfile": str(args.bfile),
        "out_dir": str(args.out_dir),
        "max_snps": args.max_snps,
        "n_genes": int(len(manifest)),
        "status_counts": manifest["status"].value_counts(dropna=False).to_dict(),
        "n_computed": int((manifest["status"] == "computed").sum()),
        "n_skipped": int((manifest["status"] != "computed").sum()),
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote LD cache manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
