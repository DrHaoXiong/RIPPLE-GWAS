#!/usr/bin/env python
"""Add an external LD-block locus column to a RIPPLE gene-score table."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--ld-block-bed", type=Path, required=True)
    parser.add_argument("--out-score-file", type=Path, required=True)
    parser.add_argument("--out-audit-file", type=Path, required=True)
    parser.add_argument("--out-summary-json", type=Path, required=True)
    parser.add_argument("--locus-column-name", default="eur_ld_block_id")
    parser.add_argument("--prefix", default="EUR_LD")
    return parser.parse_args()


def normalize_chrom(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    if text.endswith(".0"):
        text = text[:-2]
    return text.upper()


def read_ld_blocks(path: Path, prefix: str) -> pd.DataFrame:
    blocks = pd.read_csv(path, sep=r"\s+", engine="python")
    blocks.columns = [str(col).strip().lower() for col in blocks.columns]
    rename = {"stop": "end", "chrom": "chr", "chromosome": "chr"}
    blocks = blocks.rename(columns=rename)
    required = {"chr", "start", "end"}
    missing = sorted(required - set(blocks.columns))
    if missing:
        raise ValueError(f"LD block file is missing required columns: {missing}")
    blocks = blocks.loc[:, ["chr", "start", "end"]].copy()
    blocks["chrom_norm"] = blocks["chr"].map(normalize_chrom)
    blocks["start"] = pd.to_numeric(blocks["start"], errors="coerce").astype("Int64")
    blocks["end"] = pd.to_numeric(blocks["end"], errors="coerce").astype("Int64")
    blocks = blocks.dropna(subset=["start", "end"]).copy()
    blocks = blocks.sort_values(["chrom_norm", "start", "end"]).reset_index(drop=True)
    blocks["block_index"] = blocks.groupby("chrom_norm").cumcount() + 1
    blocks["ld_block_id"] = blocks.apply(
        lambda row: f"{prefix}_chr{row.chrom_norm}_{int(row.block_index):04d}", axis=1
    )
    return blocks


def assign_gene_to_block(row: pd.Series, chrom_blocks: pd.DataFrame) -> dict[str, object]:
    if chrom_blocks.empty:
        return {
            "ld_block_id": f"UNBLOCKED:{row.gene_symbol}",
            "ld_block_start": np.nan,
            "ld_block_end": np.nan,
            "ld_block_assignment": "no_chromosome_blocks",
            "ld_block_overlap_bp": 0,
            "ld_block_n_overlaps": 0,
        }
    start = int(row.gene_start)
    end = int(row.gene_end)
    midpoint = int((start + end) / 2)
    midpoint_hits = chrom_blocks.loc[(chrom_blocks["start"] <= midpoint) & (chrom_blocks["end"] >= midpoint)]
    overlaps = chrom_blocks.loc[(chrom_blocks["start"] <= end) & (chrom_blocks["end"] >= start)].copy()
    if not midpoint_hits.empty:
        chosen = midpoint_hits.iloc[0]
        assignment = "midpoint_block"
    elif not overlaps.empty:
        overlaps["overlap_bp"] = np.minimum(overlaps["end"].astype(int), end) - np.maximum(overlaps["start"].astype(int), start)
        chosen = overlaps.sort_values(["overlap_bp", "start"], ascending=[False, True]).iloc[0]
        assignment = "max_overlap_block"
    else:
        return {
            "ld_block_id": f"UNBLOCKED:{row.gene_symbol}",
            "ld_block_start": np.nan,
            "ld_block_end": np.nan,
            "ld_block_assignment": "no_overlap",
            "ld_block_overlap_bp": 0,
            "ld_block_n_overlaps": 0,
        }
    overlap_bp = max(0, min(int(chosen.end), end) - max(int(chosen.start), start))
    return {
        "ld_block_id": chosen.ld_block_id,
        "ld_block_start": int(chosen.start),
        "ld_block_end": int(chosen.end),
        "ld_block_assignment": assignment,
        "ld_block_overlap_bp": int(overlap_bp),
        "ld_block_n_overlaps": int(len(overlaps)),
    }


def add_ld_block_column(scores: pd.DataFrame, blocks: pd.DataFrame, locus_column_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"gene_symbol", "chrom", "gene_start", "gene_end"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"Score table is missing required columns: {missing}")
    out = scores.copy()
    out["gene_symbol"] = out["gene_symbol"].astype(str).str.upper()
    out["chrom_norm"] = out["chrom"].map(normalize_chrom)
    out["gene_start"] = pd.to_numeric(out["gene_start"], errors="coerce")
    out["gene_end"] = pd.to_numeric(out["gene_end"], errors="coerce")
    valid = out["gene_start"].notna() & out["gene_end"].notna() & out["chrom_norm"].notna()
    rows: list[dict[str, object]] = []
    block_groups = {chrom: group.reset_index(drop=True) for chrom, group in blocks.groupby("chrom_norm", observed=True)}
    for idx, row in out.iterrows():
        if not bool(valid.loc[idx]):
            assigned = {
                "ld_block_id": f"UNBLOCKED:{row.gene_symbol}",
                "ld_block_start": np.nan,
                "ld_block_end": np.nan,
                "ld_block_assignment": "missing_gene_coordinate",
                "ld_block_overlap_bp": 0,
                "ld_block_n_overlaps": 0,
            }
        else:
            assigned = assign_gene_to_block(row, block_groups.get(str(row.chrom_norm), pd.DataFrame()))
        assigned.update({"gene_symbol": row.gene_symbol, "chrom": row.chrom_norm, "gene_start": row.gene_start, "gene_end": row.gene_end})
        rows.append(assigned)
    audit = pd.DataFrame(rows)
    out[locus_column_name] = audit["ld_block_id"].values
    out[f"{locus_column_name}_assignment"] = audit["ld_block_assignment"].values
    out[f"{locus_column_name}_start"] = audit["ld_block_start"].values
    out[f"{locus_column_name}_end"] = audit["ld_block_end"].values
    out[f"{locus_column_name}_overlap_bp"] = audit["ld_block_overlap_bp"].values
    out[f"{locus_column_name}_n_overlaps"] = audit["ld_block_n_overlaps"].values
    out = out.drop(columns=["chrom_norm"])
    return out, audit


def main() -> None:
    args = parse_args()
    scores = pd.read_csv(args.score_file, sep="\t", compression="infer")
    blocks = read_ld_blocks(args.ld_block_bed, args.prefix)
    out, audit = add_ld_block_column(scores, blocks, args.locus_column_name)
    args.out_score_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_audit_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_score_file, sep="\t", index=False, compression="infer")
    audit.to_csv(args.out_audit_file, sep="\t", index=False)
    summary = {
        "score_file": str(args.score_file),
        "ld_block_bed": str(args.ld_block_bed),
        "out_score_file": str(args.out_score_file),
        "out_audit_file": str(args.out_audit_file),
        "locus_column_name": args.locus_column_name,
        "n_genes": int(len(out)),
        "n_blocks": int(len(blocks)),
        "n_unique_assigned_loci": int(out[args.locus_column_name].nunique()),
        "assignment_counts": audit["ld_block_assignment"].value_counts(dropna=False).to_dict(),
        "created_utc": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).resolve()),
    }
    args.out_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
