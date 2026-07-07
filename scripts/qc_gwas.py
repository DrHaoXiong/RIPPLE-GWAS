#!/usr/bin/env python
"""Run RIPPLE V1 GWAS QC for the private downloaded benchmark traits."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.qc.gwas import (  # noqa: E402
    GwasQcConfig,
    infer_trait_from_path,
    iter_standardized_chunks,
    load_reference_bim,
    load_snp_set,
    harmonize_to_reference,
)


DEFAULT_PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_RAW_DIR = DEFAULT_PRIVATE_ROOT / "10_raw_data" / "gwas"
DEFAULT_OUT_DIR = DEFAULT_PRIVATE_ROOT / "20_processed_data" / "gwas_qc"
DEFAULT_BIM = (
    DEFAULT_PRIVATE_ROOT
    / "10_raw_data"
    / "reference"
    / "ld"
    / "1000G_phase3_GRCh37"
    / "EUR"
    / "g1000_eur"
    / "g1000_eur.bim"
)
DEFAULT_HM3 = (
    DEFAULT_PRIVATE_ROOT
    / "10_raw_data"
    / "reference"
    / "snp_sets"
    / "hapmap3"
    / "w_hm3.snplist"
)
DEFAULT_HM3_NO_MHC = (
    DEFAULT_PRIVATE_ROOT
    / "10_raw_data"
    / "reference"
    / "snp_sets"
    / "hapmap3"
    / "hm3_no_MHC.list.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bim", type=Path, default=DEFAULT_BIM)
    parser.add_argument("--hm3", type=Path, default=DEFAULT_HM3)
    parser.add_argument("--hm3-no-mhc", type=Path, default=DEFAULT_HM3_NO_MHC)
    parser.add_argument("--reference-cache", type=Path, default=None)
    parser.add_argument("--rebuild-reference", action="store_true")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--maf-min", type=float, default=0.01)
    parser.add_argument("--info-min", type=float, default=0.8)
    parser.add_argument("--only", nargs="*", default=None, help="Optional trait labels or file substrings.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing QC outputs.")
    return parser.parse_args()


def open_outputs(out_dir: Path, trait: str) -> dict[str, gzip.GzipFile]:
    paths = {
        "with_mhc": out_dir / "harmonized_hm3_with_mhc" / f"{trait}.tsv.gz",
        "no_mhc": out_dir / "core_hm3_no_mhc" / f"{trait}.tsv.gz",
        "no_mhc_no_apoe": out_dir / "core_hm3_no_mhc_no_apoe" / f"{trait}.tsv.gz",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return {name: gzip.open(path, "wt", encoding="utf-8", newline="") for name, path in paths.items()}


def close_outputs(handles: dict[str, gzip.GzipFile]) -> None:
    for handle in handles.values():
        handle.close()


def output_paths(out_dir: Path, trait: str) -> dict[str, Path]:
    return {
        "with_mhc": out_dir / "harmonized_hm3_with_mhc" / f"{trait}.tsv.gz",
        "no_mhc": out_dir / "core_hm3_no_mhc" / f"{trait}.tsv.gz",
        "no_mhc_no_apoe": out_dir / "core_hm3_no_mhc_no_apoe" / f"{trait}.tsv.gz",
        "report": out_dir / "qc_reports" / f"{trait}.qc_report.json",
    }


def write_chunk(handle: gzip.GzipFile, table: pd.DataFrame, *, write_header: bool) -> None:
    table.to_csv(handle, sep="\t", index=False, header=write_header)


def update_fail_counts(counter: Counter[str], fail_reason: pd.Series) -> None:
    for reason in fail_reason.dropna():
        if not reason:
            continue
        for item in str(reason).split(";"):
            if item:
                counter[item] += 1


def select_files(raw_dir: Path, only: list[str] | None) -> list[Path]:
    files = sorted([*raw_dir.glob("*.gz"), *raw_dir.glob("*.zip")])
    if not only:
        return files
    lowered = [item.lower() for item in only]
    selected: list[Path] = []
    for path in files:
        trait = infer_trait_from_path(path).lower()
        name = path.name.lower()
        if any(item in trait or item in name for item in lowered):
            selected.append(path)
    return selected


def write_manifest(out_dir: Path, reports: list[dict[str, object]]) -> Path:
    manifest_rows = []
    for report in reports:
        counts = report["counts"]
        manifest_rows.append(
            {
                "trait": report["trait"],
                "source_file": report["source_file"],
                "with_mhc_rows": counts.get("output_with_mhc_rows", 0),
                "no_mhc_rows": counts.get("output_no_mhc_rows", 0),
                "no_mhc_no_apoe_rows": counts.get("output_no_mhc_no_apoe_rows", 0),
                "p_clipped_rows": counts.get("p_clipped_rows", 0),
                "duplicate_snps_dropped": counts.get("duplicate_snps_dropped", 0),
                "info_filter_applied": report["info_filter_applied"],
                "report_path": str(out_dir / "qc_reports" / f"{report['trait']}.qc_report.json"),
            }
        )
    new_manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.tsv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, sep="\t")
        existing = existing[~existing["trait"].isin(set(new_manifest["trait"]))]
        manifest = pd.concat([existing, new_manifest], ignore_index=True)
    else:
        manifest = new_manifest
    manifest = manifest.sort_values("trait").reset_index(drop=True)
    manifest.to_csv(manifest_path, sep="\t", index=False)
    return manifest_path


def ensure_not_overwriting(paths: dict[str, Path], *, force: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not force:
        pretty = "\n".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs without --force:\n{pretty}")
    for path in existing:
        path.unlink()


def process_file(
    path: Path,
    *,
    reference: pd.DataFrame,
    hm3_no_mhc_snps: set[str],
    out_dir: Path,
    config: GwasQcConfig,
    chunksize: int,
    force: bool,
) -> dict[str, object]:
    trait = infer_trait_from_path(path)
    paths = output_paths(out_dir, trait)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    ensure_not_overwriting(paths, force=force)

    handles = open_outputs(out_dir, trait)
    write_header = {"with_mhc": True, "no_mhc": True, "no_mhc_no_apoe": True}
    seen_snps: set[str] = set()
    reference_snps = set(reference["snp_id"])
    fail_counts: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    info_available = False

    try:
        for standardized in iter_standardized_chunks(path, chunksize=chunksize, config=config):
            counts["standardized_rows"] += int(len(standardized))
            if standardized.empty:
                continue
            in_reference = standardized["snp_id"].isin(reference_snps)
            not_in_reference = int((~in_reference).sum())
            counts["not_in_reference_hm3_premerge_rows"] += not_in_reference
            fail_counts["not_in_reference_hm3"] += not_in_reference
            standardized = standardized.loc[in_reference].copy()
            if standardized.empty:
                continue

            harmonized = harmonize_to_reference(
                standardized,
                reference,
                hm3_no_mhc_snps=hm3_no_mhc_snps,
                config=config,
            )
            counts["harmonized_rows_attempted"] += int(len(harmonized))
            counts["p_clipped_rows"] += int(harmonized["p_was_clipped"].sum())
            counts["info_nonmissing_rows"] += int(harmonized["info"].notna().sum())
            info_available = info_available or bool(harmonized["info"].notna().any())
            update_fail_counts(fail_counts, harmonized["qc_fail_reason"])

            passed = harmonized[harmonized["qc_pass"]].copy()
            if passed.empty:
                continue
            duplicate_mask = passed["snp_id"].isin(seen_snps)
            counts["duplicate_snps_dropped"] += int(duplicate_mask.sum())
            passed = passed.loc[~duplicate_mask].copy()
            seen_snps.update(passed["snp_id"].tolist())
            if passed.empty:
                continue

            with_mhc = passed.drop(columns=["in_hm3_no_mhc"])
            no_mhc_mask = passed["in_hm3_no_mhc"] & ~passed["is_mhc"]
            no_mhc = passed[no_mhc_mask].drop(columns=["in_hm3_no_mhc"])
            no_mhc_no_apoe = passed[
                no_mhc_mask & ~passed["is_apoe_region"]
            ].drop(columns=["in_hm3_no_mhc"])

            for name, table in (
                ("with_mhc", with_mhc),
                ("no_mhc", no_mhc),
                ("no_mhc_no_apoe", no_mhc_no_apoe),
            ):
                counts[f"output_{name}_rows"] += int(len(table))
                if table.empty:
                    continue
                write_chunk(handles[name], table, write_header=write_header[name])
                write_header[name] = False
    finally:
        close_outputs(handles)

    report = {
        "trait": trait,
        "source_file": str(path),
        "created_utc": datetime.now(UTC).isoformat(),
        "config": asdict(config),
        "reference": {
            "bim_rows_loaded": int(len(reference)),
            "analysis_build": config.analysis_build,
            "hm3_no_mhc_snps": int(len(hm3_no_mhc_snps)),
        },
        "info_filter_applied": info_available,
        "info_filter_note": (
            f"Applied INFO >= {config.info_min} where INFO/IMPINFO was present."
            if info_available
            else "INFO column unavailable for this source; INFO filter could not be applied."
        ),
        "counts": dict(counts),
        "fail_counts": dict(fail_counts),
        "outputs": {key: str(value) for key, value in paths.items() if key != "report"},
    }
    paths["report"].write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    config = GwasQcConfig(maf_min=args.maf_min, info_min=args.info_min)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HapMap3 SNP sets from {args.hm3} and {args.hm3_no_mhc}", flush=True)
    hm3_snps = load_snp_set(args.hm3)
    hm3_no_mhc_snps = load_snp_set(args.hm3_no_mhc)
    print(f"Loaded {len(hm3_snps):,} HapMap3 SNPs and {len(hm3_no_mhc_snps):,} no-MHC SNPs", flush=True)

    reference_cache = args.reference_cache or (
        args.out_dir / "reference" / "1000G_EUR_GRCh37_hm3_reference.tsv.gz"
    )
    reference_cache.parent.mkdir(parents=True, exist_ok=True)
    if reference_cache.exists() and args.rebuild_reference:
        reference_cache.unlink()

    if reference_cache.exists():
        print(f"Loading cached 1000G EUR GRCh37 HapMap3 reference: {reference_cache}", flush=True)
        reference = pd.read_csv(
            reference_cache,
            sep="\t",
            dtype={
                "snp_id": str,
                "chrom_ref": str,
                "pos_ref": int,
                "ref_a1": str,
                "ref_a2": str,
            },
            compression="infer",
        )
    else:
        print(f"Building 1000G EUR GRCh37 BIM subset from {args.bim}", flush=True)
        reference = load_reference_bim(args.bim, include_snps=hm3_snps)
        reference.to_csv(reference_cache, sep="\t", index=False, compression="infer")
        print(f"Wrote reference cache: {reference_cache}", flush=True)
    print(f"Loaded {len(reference):,} reference SNPs after HapMap3/autosome filtering", flush=True)
    if reference.empty:
        raise RuntimeError("Reference SNP cache is empty; check HapMap3 SNP IDs and BIM SNP IDs.")

    files = select_files(args.raw_dir, args.only)
    if not files:
        raise FileNotFoundError(f"No GWAS .gz files found in {args.raw_dir}")

    reports: list[dict[str, object]] = []
    for path in files:
        trait = infer_trait_from_path(path)
        print(f"Processing {trait}: {path.name}", flush=True)
        report = process_file(
            path,
            reference=reference,
            hm3_no_mhc_snps=hm3_no_mhc_snps,
            out_dir=args.out_dir,
            config=config,
            chunksize=args.chunksize,
            force=args.force,
        )
        reports.append(report)
        print(
            f"Finished {trait}: "
            f"with-MHC={report['counts'].get('output_with_mhc_rows', 0):,}, "
            f"no-MHC={report['counts'].get('output_no_mhc_rows', 0):,}",
            flush=True,
        )

    manifest_path = write_manifest(args.out_dir, reports)
    print(f"Wrote manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
