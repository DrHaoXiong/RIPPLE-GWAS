#!/usr/bin/env python
"""Complete V1.6b candidate annotation and pseudo-window sensitivities."""

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

from ripple.modules import (  # noqa: E402
    RippleDConfig,
    add_ripple_d_v16_claim_readiness,
    external_locus_audit_table,
    load_anchored_gene_set_library,
    prepare_locus_inputs,
    ripple_d_module_tests,
)

PRIVATE_ROOT = Path("/mnt/d/RIPPLE/RIPPLE_private")
DEFAULT_OUT_DIR = (
    PRIVATE_ROOT
    / "30_analysis"
    / "tier4_v16_claim_readiness_hardening_v0_1"
    / "v16b_candidate_sensitivity_completion_v0_1"
)

SENSITIVITY_PASS_STATUSES = {
    "manuscript_ready_distributed_candidate",
    "high_confidence_diagnostic_candidate",
    "exploratory_locus_distributed_candidate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trait", required=True)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--gene-set-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--degree-bins", type=int, default=5)
    parser.add_argument("--property-bins", type=int, default=2)
    parser.add_argument("--locus-id-column", type=str, required=True)
    parser.add_argument("--locus-definition-name", type=str, default="ldetect_eur_grch37")
    parser.add_argument("--locus-source", default="LDetect")
    parser.add_argument("--locus-source-version", default="fourier_ls-all")
    parser.add_argument("--genome-build", default="GRCh37")
    parser.add_argument("--ancestry", default="EUR")
    parser.add_argument("--construction-script", default="build_external_ldblock_score_table.py")
    parser.add_argument("--pseudo-windows", default="250000,500000,1000000")
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "reports").mkdir(parents=True, exist_ok=True)


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def run_claim_table(
    *,
    scores: pd.DataFrame,
    library,
    config: RippleDConfig,
    run_label: str,
    seed: int,
    n_null: int,
    external_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    work, background = prepare_locus_inputs(scores, library, config)
    modules, locus_audit, _, _ = ripple_d_module_tests(
        scores,
        library,
        config=config,
        min_present=5,
        n_null=n_null,
        seed=seed,
        return_null_details=False,
        precomputed_work=work,
        precomputed_locus_background=background,
    )
    claim = add_ripple_d_v16_claim_readiness(
        modules,
        config,
        locus_audit=locus_audit,
        external_locus_audit=external_audit,
    )
    claim.insert(0, "run_label", run_label)
    return claim


def subset_for_merge(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = [
        "module_name",
        "module_status",
        "v16b_claim_status",
        "v16b_downgrade_reason",
        "ripple_d_q_full_library",
        "module_specific_rank_q_full_library",
        "ripple_d_v16_empirical_p",
        "module_specific_rank_empirical_p",
        "null_quality_balanced_pass",
        "multiplicity_pass",
        "top_tail_pass",
        "leave_topk_pass",
        "n_loci",
        "n_effective_loci",
        "top1_locus_contribution",
        "top5_locus_contribution",
    ]
    out = table.loc[:, [column for column in keep if column in table.columns]].copy()
    return out.rename(columns={column: f"{prefix}_{column}" for column in out.columns if column != "module_name"})


def classify_completion(row: pd.Series) -> str:
    base_status = str(row.get("base_v16b_claim_status", ""))
    if base_status != "high_confidence_diagnostic_candidate":
        return base_status or "not_tested"
    if not bool(row.get("annotation_sensitivity_balanced_pass", False)):
        return "annotation_dependent_candidate"
    if not bool(row.get("pseudo_window_stability_balanced_pass", False)):
        return "pseudo_window_unstable_candidate"
    if str(row.get("base_module_status", "")) == "distributed_weak_signal_module_candidate":
        return "sensitivity_completed_high_confidence_candidate"
    return "sensitivity_completed_supportive_candidate"


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    scores = pd.read_csv(args.score_file, sep="\t", compression="infer")
    library = load_anchored_gene_set_library(args.gene_set_file)
    windows = [int(value) for value in args.pseudo_windows.split(",") if value.strip()]

    external_config_off = RippleDConfig(
        locus_id_column=args.locus_id_column,
        locus_definition_name=args.locus_definition_name,
        degree_bins=args.degree_bins,
        property_bins=args.property_bins,
        annotation_matching_enabled=False,
    )
    external_config_on = RippleDConfig(
        locus_id_column=args.locus_id_column,
        locus_definition_name=args.locus_definition_name,
        degree_bins=args.degree_bins,
        property_bins=args.property_bins,
        annotation_matching_enabled=True,
    )
    external_work, _ = prepare_locus_inputs(scores, library, external_config_off)
    external_audit = external_locus_audit_table(
        external_work,
        locus_id_column=args.locus_id_column,
        locus_source=args.locus_source,
        locus_source_version=args.locus_source_version,
        genome_build=args.genome_build,
        ancestry=args.ancestry,
        construction_script=args.construction_script,
    )

    base = run_claim_table(
        scores=scores,
        library=library,
        config=external_config_off,
        run_label="external_ldblock_annotation_off",
        seed=args.seed,
        n_null=args.n_null,
        external_audit=external_audit,
    )
    annotation_on = run_claim_table(
        scores=scores,
        library=library,
        config=external_config_on,
        run_label="external_ldblock_annotation_on",
        seed=args.seed + 1009,
        n_null=args.n_null,
        external_audit=external_audit,
    )
    write_table(tables_dir / f"{args.trait}.external_ldblock_annotation_off.v16b_claim_readiness.tsv", base)
    write_table(tables_dir / f"{args.trait}.external_ldblock_annotation_on.v16b_claim_readiness.tsv", annotation_on)

    pseudo_tables: list[pd.DataFrame] = []
    for idx, window in enumerate(windows):
        pseudo_config = RippleDConfig(
            locus_window_bp=window,
            degree_bins=args.degree_bins,
            property_bins=args.property_bins,
            annotation_matching_enabled=False,
        )
        pseudo = run_claim_table(
            scores=scores,
            library=library,
            config=pseudo_config,
            run_label=f"pseudo_window_{window}",
            seed=args.seed + 2000 + idx * 1009,
            n_null=args.n_null,
            external_audit=None,
        )
        pseudo["pseudo_window_bp"] = int(window)
        pseudo_tables.append(pseudo)
        write_table(tables_dir / f"{args.trait}.pseudo_window_{window}.v16b_claim_readiness.tsv", pseudo)

    merged = subset_for_merge(base, "base")
    merged = merged.merge(subset_for_merge(annotation_on, "annotation_on"), on="module_name", how="left")
    annotation_status = merged["annotation_on_v16b_claim_status"].astype(str)
    merged["annotation_sensitivity_balanced_pass"] = annotation_status.isin(SENSITIVITY_PASS_STATUSES)
    merged["annotation_sensitivity_balanced_status"] = merged["annotation_on_v16b_claim_status"].fillna("not_tested")

    pseudo_long = pd.concat(pseudo_tables, ignore_index=True) if pseudo_tables else pd.DataFrame()
    pseudo_rows: list[dict[str, object]] = []
    for module_name, group in pseudo_long.groupby("module_name", observed=True):
        statuses = group["v16b_claim_status"].astype(str)
        pass_mask = statuses.isin(SENSITIVITY_PASS_STATUSES)
        high_mask = statuses.eq("high_confidence_diagnostic_candidate")
        pseudo_rows.append(
            {
                "module_name": module_name,
                "pseudo_window_tested_n": int(len(group)),
                "pseudo_window_pass_n": int(pass_mask.sum()),
                "pseudo_window_high_confidence_n": int(high_mask.sum()),
                "pseudo_window_statuses": ";".join(
                    f"{int(row.pseudo_window_bp)}:{row.v16b_claim_status}" for row in group.itertuples(index=False)
                ),
                "pseudo_window_stability_balanced_pass": int(pass_mask.sum()) >= max(2, len(group) - 1),
            }
        )
    pseudo_summary = pd.DataFrame(pseudo_rows)
    merged = merged.merge(pseudo_summary, on="module_name", how="left")
    merged["pseudo_window_stability_balanced_pass"] = merged["pseudo_window_stability_balanced_pass"].fillna(False).astype(bool)
    merged["v16b_sensitivity_completed_status"] = merged.apply(classify_completion, axis=1)
    write_table(tables_dir / f"{args.trait}.v16b_sensitivity_completion_summary.tsv", merged)
    if not pseudo_long.empty:
        write_table(tables_dir / f"{args.trait}.v16b_pseudo_window_long.tsv", pseudo_long)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "trait": args.trait,
        "score_file": str(args.score_file),
        "gene_set_file": str(args.gene_set_file),
        "out_dir": str(args.out_dir),
        "n_null": int(args.n_null),
        "degree_bins": int(args.degree_bins),
        "property_bins": int(args.property_bins),
        "pseudo_windows": windows,
        "seed": int(args.seed),
    }
    (reports_dir / f"{args.trait}.v16b_sensitivity_completion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    counts = merged["v16b_sensitivity_completed_status"].value_counts(dropna=False)
    report = "\n".join(
        [
            f"# V1.6b candidate sensitivity completion: {args.trait}",
            "",
            f"n_null = {args.n_null}",
            "",
            "## Completed Status Counts",
            "",
            counts.to_string(),
            "",
            "## Top Rows",
            "",
            merged.sort_values(["v16b_sensitivity_completed_status", "base_ripple_d_q_full_library"]).head(30).to_string(
                index=False
            ),
            "",
        ]
    )
    (reports_dir / f"{args.trait}.v16b_sensitivity_completion_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
