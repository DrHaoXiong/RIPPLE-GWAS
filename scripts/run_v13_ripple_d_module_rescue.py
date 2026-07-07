#!/usr/bin/env python
"""Run RIPPLE-D V1.3 locus-aware distributed module rescue diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.modules import RippleDConfig, load_anchored_gene_set_library, ripple_d_module_tests  # noqa: E402
from ripple.modules.anchored import AnchoredModuleLibrary  # noqa: E402

PRIVATE_ROOT = Path("/mnt/d/RIPPLE/RIPPLE_private")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "tier4_v13_locus_distributed_module_rescue_v0_1"
DEFAULT_GENE_SET_FILE = (
    PRIVATE_ROOT
    / "20_processed_data"
    / "reference_pathways"
    / "anchored_broad_reactome_go_v1"
    / "tables"
    / "anchored_broad_reactome_go_gene_sets.tsv.gz"
)

TRAITS = {
    "DR_MVP": {
        "analysis_dir": "dr_mvp_string_final5000",
        "score_file": "DR_MVP.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
    },
    "DR_MVP_NO_MHC_NO_APOE": {
        "analysis_dir": "dr_mvp_no_mhc_no_apoe_final5000",
        "score_file": "DR_MVP_NO_MHC_NO_APOE.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
    },
    "T2D": {
        "analysis_dir": "t2d_analysis_ready",
        "score_file": "T2D.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
    },
    "BMI_IRN": {
        "analysis_dir": "bmi_irn_analysis_ready",
        "score_file": "BMI_IRN.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
    },
    "SCZ": {
        "analysis_dir": "scz_no_mhc_string_final5000",
        "score_file": "SCZ.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
    },
    "HEIGHT_IRN": {
        "analysis_dir": "height_irn_analysis_ready",
        "score_file": "HEIGHT_IRN.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gene-set-file", type=Path, default=DEFAULT_GENE_SET_FILE)
    parser.add_argument("--traits", nargs="*", default=list(TRAITS))
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--min-present", type=int, default=5)
    parser.add_argument("--score-cap", type=float, default=3.0)
    parser.add_argument("--locus-window-bp", type=int, default=500_000)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--property-bins", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument("--write-null-details", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "reports").mkdir(parents=True, exist_ok=True)


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, compression="infer")


def load_scores(trait: str) -> tuple[pd.DataFrame, Path]:
    cfg = TRAITS[trait]
    path = ANALYSIS_ROOT / cfg["analysis_dir"] / "tables" / cfg["score_file"]
    if not path.exists():
        raise FileNotFoundError(path)
    scores = pd.read_csv(path, sep="\t", compression="infer")
    return scores, path


def maybe_limit_library(library: AnchoredModuleLibrary, max_modules: int | None) -> AnchoredModuleLibrary:
    if max_modules is None or max_modules >= len(library.gene_sets):
        return library
    names = list(library.gene_sets)[: int(max_modules)]
    return AnchoredModuleLibrary(
        gene_sets={name: library.gene_sets[name] for name in names},
        module_source={name: library.module_source.get(name, "unspecified") for name in names},
        annotation_source_type={
            name: library.annotation_source_type.get(name, "internal_support") for name in names
        },
        module_category={name: library.module_category.get(name, "unspecified") for name in names},
    )


def render_report(trait: str, modules: pd.DataFrame, summary: dict[str, object]) -> str:
    lines = [
        f"# RIPPLE-D V1.3 Module Rescue Diagnostics: {trait}",
        "",
        "RIPPLE-D tests whether fixed biological modules retain signal after locus collapse, "
        "score capping, top-locus decomposition, and locus-aware null calibration.",
        "",
        "## Summary",
        "",
        f"- Tested modules: {summary.get('n_tested_modules', 0):,}",
        f"- Background genes: {summary.get('n_background_genes', 0):,}",
        f"- Background pseudo-loci: {summary.get('n_background_loci', 0):,}",
        f"- Distributed candidates: {summary.get('n_distributed_weak_signal_module_candidate', 0):,}",
        f"- Top-locus dominant modules: {summary.get('n_top_locus_dominant_module', 0):,}",
        f"- Raw-enrichment-only modules: {summary.get('n_raw_gene_set_enrichment_only', 0):,}",
        "",
        "## Top Rows",
        "",
        "| Rank | Module | Status | n genes | n loci | RIPPLE-D | locus P | leave-top1 P | n eff loci | top1 locus |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    tested = modules.loc[modules.get("module_status", pd.Series(dtype=str)).ne("not_tested_low_overlap")]
    for row in tested.head(20).itertuples(index=False):
        lines.append(
            f"| {int(getattr(row, 'ripple_d_module_rank', 0))} | {row.module_name} | "
            f"{row.module_status} | {int(row.n_present)} | {int(row.n_loci)} | "
            f"{float(row.ripple_d_stat):.3f} | {float(row.locus_robust_empirical_p):.4g} | "
            f"{float(row.leave_top1_locus_empirical_p):.4g} | {float(row.n_effective_loci):.2f} | "
            f"{float(row.top1_locus_contribution):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- `distributed_weak_signal_module_candidate` requires the full RIPPLE-D gate.",
            "- `top_locus_dominant_module` and `raw_gene_set_enrichment_only` are not weak-signal module claims.",
            "- The default locus definition is coordinate-based pseudo-locus and should be replaced or sensitivity-tested with external LD blocks before final submission.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    config = RippleDConfig(
        score_cap=args.score_cap,
        locus_window_bp=args.locus_window_bp,
        degree_bins=args.degree_bins,
        property_bins=args.property_bins,
    )

    library = load_anchored_gene_set_library(args.gene_set_file)
    library = maybe_limit_library(library, args.max_modules)
    run_summaries: list[dict[str, object]] = []
    for trait_idx, trait in enumerate(args.traits):
        if trait not in TRAITS:
            raise ValueError(f"Unknown trait {trait!r}; choose from {sorted(TRAITS)}")
        scores, score_path = load_scores(trait)
        print(f"Running RIPPLE-D for {trait} with {len(library.gene_sets):,} modules", flush=True)
        modules, locus_audit, nulls, summary = ripple_d_module_tests(
            scores,
            library,
            config=config,
            min_present=args.min_present,
            n_null=args.n_null,
            seed=args.seed + trait_idx * 1009,
            return_null_details=args.write_null_details,
        )
        write_table(tables_dir / f"{trait}.v13_ripple_d_module_tests.tsv", modules)
        write_table(tables_dir / f"{trait}.v13_locus_contribution_audit.tsv", locus_audit)
        if args.write_null_details:
            write_table(tables_dir / f"{trait}.v13_locus_nulls.tsv.gz", nulls)
        run_summary = {
            **summary,
            "trait": trait,
            "score_path": str(score_path),
            "gene_set_file": str(args.gene_set_file),
            "output_dir": str(args.out_dir),
            "script_path": str(Path(__file__).resolve()),
            "seed": int(args.seed + trait_idx * 1009),
            "created_utc": now_utc(),
            "write_null_details": bool(args.write_null_details),
        }
        run_summaries.append(run_summary)
        (reports_dir / f"{trait}.v13_ripple_d_summary.json").write_text(
            json.dumps(run_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (reports_dir / f"{trait}.v13_ripple_d_report.md").write_text(
            render_report(trait, modules, run_summary) + "\n",
            encoding="utf-8",
        )

    summary_table = pd.DataFrame(run_summaries)
    write_table(tables_dir / "v13_ripple_d_cross_trait_summary.tsv", summary_table)
    print(f"Wrote RIPPLE-D V1.3 diagnostics to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
