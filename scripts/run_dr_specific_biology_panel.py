#!/usr/bin/env python
"""Run the DR-specific biology panel across primary and comparator traits."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
PROJECT_ROOT = PRIVATE_ROOT / "04_private_src" / "ripple_v1"
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
PANEL_ROOT = PRIVATE_ROOT / "20_processed_data" / "reference_pathways" / "dr_specific_biology_panel_v0_1"
DEFAULT_PANEL = PANEL_ROOT / "tables" / "dr_specific_biology_panel_v0_1.gene_sets.tsv"
DEFAULT_REGISTRY = PANEL_ROOT / "tables" / "dr_specific_biology_panel_v0_1.registry.tsv"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "dr_specific_biology_panel_v0_1"
RUN_ANCHORED_SCRIPT = PROJECT_ROOT / "scripts" / "run_v12_anchored_module_test.py"

TRAIT_CONFIGS: dict[str, dict[str, str]] = {
    "DR_MVP": {
        "trait": "DR_MVP",
        "analysis_dir": "dr_mvp_string_final5000",
        "role": "primary_dr",
    },
    "DR_MVP_NO_MHC_NO_APOE": {
        "trait": "DR_MVP_NO_MHC_NO_APOE",
        "analysis_dir": "dr_mvp_no_mhc_no_apoe_final5000",
        "role": "primary_dr_sensitivity",
    },
    "T2D": {
        "trait": "T2D",
        "analysis_dir": "t2d_analysis_ready",
        "role": "diabetic_liability_comparator",
    },
    "BMI_IRN": {
        "trait": "BMI_IRN",
        "analysis_dir": "bmi_irn_analysis_ready",
        "role": "metabolic_comparator",
    },
    "HEIGHT_IRN": {
        "trait": "HEIGHT_IRN",
        "analysis_dir": "height_irn_analysis_ready",
        "role": "non_dr_anthropometric_comparator",
    },
    "SCZ": {
        "trait": "SCZ",
        "analysis_dir": "scz_no_mhc_string_final5000",
        "role": "non_ocular_polygenic_comparator",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traits", nargs="*", default=list(TRAIT_CONFIGS))
    parser.add_argument("--panel-file", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--panel-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-degree-matched-null", type=int, default=5000)
    parser.add_argument("--n-score-permutation-null", type=int, default=5000)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-runs", action="store_true", help="Only summarize existing run outputs.")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, compression="infer")


def run_trait(args: argparse.Namespace, key: str) -> None:
    config = TRAIT_CONFIGS[key]
    trait = config["trait"]
    out_dir = args.out_dir / key
    cmd = [
        sys.executable,
        str(RUN_ANCHORED_SCRIPT),
        "--trait",
        trait,
        "--analysis-dir",
        str(ANALYSIS_ROOT / config["analysis_dir"]),
        "--out-dir",
        str(out_dir),
        "--graph-name",
        "STRING_default",
        "--gene-set-file",
        str(args.panel_file),
        "--external-gene-set-source-type",
        "independent_external_pending_citation",
        "--no-default-dr-panel",
        "--no-louvain-communities",
        "--min-present",
        "5",
        "--n-degree-matched-null",
        str(args.n_degree_matched_null),
        "--n-score-permutation-null",
        str(args.n_score_permutation_null),
        "--degree-bins",
        str(args.degree_bins),
        "--seed",
        str(args.seed + stable_offset(key)),
        "--force",
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def stable_offset(text: str, modulo: int = 100_000) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % modulo


def load_trait_modules(root: Path, key: str) -> pd.DataFrame:
    config = TRAIT_CONFIGS[key]
    table_path = root / key / "tables" / f"{config['trait']}.v12_anchored_module_tests.tsv"
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    table = pd.read_csv(table_path, sep="\t")
    table["analysis_id"] = key
    table["trait"] = config["trait"]
    table["trait_role"] = config["role"]
    table["source_result_path"] = str(table_path)
    return table


def add_registry_fields(modules: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "gene_set",
        "module_category",
        "panel_role",
        "description",
        "citation_status",
        "construction_note",
    ]
    available = [col for col in keep if col in registry.columns]
    registry_subset = registry[available].rename(columns={"module_category": "registry_module_category"})
    annotated = modules.merge(
        registry_subset,
        left_on="module_name",
        right_on="gene_set",
        how="left",
    )
    if "registry_module_category" in annotated.columns:
        if "module_category" not in annotated.columns:
            annotated["module_category"] = annotated["registry_module_category"]
        else:
            module_category = annotated["module_category"].astype(str)
            needs_registry = module_category.isin(["", "nan", "None", "unspecified"])
            annotated.loc[needs_registry, "module_category"] = annotated.loc[
                needs_registry,
                "registry_module_category",
            ]
        annotated = annotated.drop(columns=["registry_module_category"])
    if "gene_set" in annotated.columns:
        annotated = annotated.drop(columns=["gene_set"])
    return annotated


def summarize_cross_trait(modules: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tested = modules.loc[modules["module_status"].astype(str).ne("not_tested_low_overlap")].copy()
    tested["library_familywise_p"] = pd.to_numeric(tested["library_familywise_p"], errors="coerce")
    tested["degree_matched_empirical_p"] = pd.to_numeric(tested["degree_matched_empirical_p"], errors="coerce")
    tested["degree_matched_z"] = pd.to_numeric(tested["degree_matched_z"], errors="coerce")
    tested["observed_value"] = pd.to_numeric(tested["observed_value"], errors="coerce")
    tested = tested.sort_values(
        ["trait", "library_familywise_p", "degree_matched_empirical_p", "observed_value"],
        ascending=[True, True, True, False],
    )
    tested["rank_within_trait"] = tested.groupby("trait", observed=True).cumcount() + 1

    trait_rows = []
    for trait, group in tested.groupby("trait", observed=True):
        best = group.iloc[0]
        trait_rows.append(
            {
                "trait": trait,
                "analysis_id": best["analysis_id"],
                "trait_role": best["trait_role"],
                "n_tested_modules": int(len(group)),
                "n_fixed_degree_supported": int(group["module_status"].eq("fixed_degree_supported").sum()),
                "n_familywise_supported": int(group["module_status"].eq("anchored_familywise_supported").sum()),
                "best_module_name": best["module_name"],
                "best_module_category": best["module_category"],
                "best_panel_role": best.get("panel_role", ""),
                "best_degree_matched_z": best["degree_matched_z"],
                "best_degree_matched_p": best["degree_matched_empirical_p"],
                "best_library_familywise_p": best["library_familywise_p"],
                "best_module_status": best["module_status"],
                "interpretation_note": trait_interpretation(str(trait), group),
            }
        )
    return pd.DataFrame(trait_rows).sort_values("trait").reset_index(drop=True), tested


def trait_interpretation(trait: str, group: pd.DataFrame) -> str:
    supported = group.loc[group["module_status"].eq("anchored_familywise_supported")]
    if supported.empty:
        return "no_panel_familywise_supported_module"
    roles = sorted({str(item) for item in supported.get("panel_role", pd.Series(dtype=str)).dropna()})
    if trait.startswith("DR") and any("dr_retinal" in role or "dr_neuro" in role for role in roles):
        return "dr_panel_support_in_retinal_or_neurovascular_axis"
    if trait.startswith("DR") and any("shared_diabetic" in role for role in roles):
        return "dr_panel_support_limited_to_shared_diabetic_context"
    return "panel_familywise_supported_module"


def dr_specificity_contrast(tested: pd.DataFrame) -> pd.DataFrame:
    pivot_cols = [
        "observed_value",
        "degree_matched_z",
        "degree_matched_empirical_p",
        "library_familywise_p",
        "rank_within_trait",
        "module_status",
    ]
    rows = []
    for module_name, group in tested.groupby("module_name", observed=True):
        by_trait = {str(row["trait"]): row for row in group.to_dict(orient="records")}
        dr = by_trait.get("DR_MVP")
        dr_sens = by_trait.get("DR_MVP_NO_MHC_NO_APOE")
        t2d = by_trait.get("T2D")
        if dr is None:
            continue
        row: dict[str, Any] = {
            "module_name": module_name,
            "module_category": dr.get("module_category", ""),
            "panel_role": dr.get("panel_role", ""),
            "description": dr.get("description", ""),
        }
        for trait_label, source in [
            ("DR_MVP", dr),
            ("DR_MVP_NO_MHC_NO_APOE", dr_sens),
            ("T2D", t2d),
            ("BMI_IRN", by_trait.get("BMI_IRN")),
            ("HEIGHT_IRN", by_trait.get("HEIGHT_IRN")),
            ("SCZ", by_trait.get("SCZ")),
        ]:
            for col in pivot_cols:
                row[f"{trait_label}_{col}"] = source.get(col, "") if source is not None else ""
        row["DR_minus_T2D_observed_value"] = numeric_diff(dr, t2d, "observed_value")
        row["DR_minus_T2D_degree_matched_z"] = numeric_diff(dr, t2d, "degree_matched_z")
        row["DR_rank_minus_T2D_rank"] = numeric_diff(dr, t2d, "rank_within_trait")
        row["DR_specificity_label"] = specificity_label(dr, dr_sens, t2d)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["DR_MVP_library_familywise_p", "T2D_library_familywise_p"],
        na_position="last",
    )


def numeric_diff(left: dict[str, Any] | None, right: dict[str, Any] | None, field: str) -> float:
    if left is None or right is None:
        return float("nan")
    left_value = pd.to_numeric(pd.Series([left.get(field)]), errors="coerce").iloc[0]
    right_value = pd.to_numeric(pd.Series([right.get(field)]), errors="coerce").iloc[0]
    return float(left_value - right_value) if np.isfinite(left_value) and np.isfinite(right_value) else float("nan")


def is_familywise_supported(row: dict[str, Any] | None) -> bool:
    return row is not None and str(row.get("module_status", "")) == "anchored_familywise_supported"


def specificity_label(
    dr: dict[str, Any] | None,
    dr_sens: dict[str, Any] | None,
    t2d: dict[str, Any] | None,
) -> str:
    dr_pos = is_familywise_supported(dr)
    sens_pos = is_familywise_supported(dr_sens)
    t2d_pos = is_familywise_supported(t2d)
    if dr_pos and sens_pos and not t2d_pos:
        return "dr_enriched_panel_axis"
    if dr_pos and t2d_pos:
        return "shared_diabetic_liability_or_common_axis"
    if dr_pos and not sens_pos:
        return "dr_default_only_region_sensitive"
    if not dr_pos and t2d_pos:
        return "t2d_comparator_axis"
    return "not_familywise_supported_in_dr"


def render_report(summary: pd.DataFrame, contrast: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        "# DR-specific biology panel v0.1 cross-trait report",
        "",
        f"Created: {now_utc()}",
        "",
        "This is a fixed-panel sensitivity analysis for DR pathobiology. It does not upgrade "
        "anchored modules to de novo topology-specific PPI modules.",
        "",
        f"Degree-matched nulls: {args.n_degree_matched_null}",
        f"Score-permutation library nulls: {args.n_score_permutation_null}",
        "",
        "## Trait summary",
        "",
        "| Trait | Role | Familywise-supported | Best module | Best family P | Best status | Interpretation |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['trait']} | {row['trait_role']} | {int(row['n_familywise_supported'])} | "
            f"{row['best_module_name']} | {float(row['best_library_familywise_p']):.4g} | "
            f"{row['best_module_status']} | {row['interpretation_note']} |"
        )
    lines.extend(
        [
            "",
            "## DR vs T2D module contrast",
            "",
            "| Module | Role | DR status | DR family P | T2D status | T2D family P | Specificity label |",
            "|---|---|---|---:|---|---:|---|",
        ]
    )
    for row in contrast.to_dict(orient="records"):
        lines.append(
            f"| {row['module_name']} | {row['panel_role']} | {row['DR_MVP_module_status']} | "
            f"{fmt_float(row['DR_MVP_library_familywise_p'])} | {row['T2D_module_status']} | "
            f"{fmt_float(row['T2D_library_familywise_p'])} | {row['DR_specificity_label']} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "- Positive panel results support DR biological plausibility only within the fixed panel.",
            "- Shared DR/T2D positivity is interpreted as diabetic liability or common metabolic axis.",
            "- DR positive and T2D negative axes are candidates for DR-enriched biology, pending citation and external expression support.",
            "- No result in this report supports a de novo topology-specific disease module claim.",
            "",
        ]
    )
    return "\n".join(lines)


def fmt_float(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return f"{float(numeric):.4g}" if np.isfinite(numeric) else ""


def main() -> None:
    args = parse_args()
    if not args.panel_file.exists():
        raise FileNotFoundError(args.panel_file)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["tables", "reports"]:
        (args.out_dir / subdir).mkdir(parents=True, exist_ok=True)

    unknown = [trait for trait in args.traits if trait not in TRAIT_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown trait keys: {unknown}")
    if not args.skip_runs:
        for key in args.traits:
            run_trait(args, key)

    registry = pd.read_csv(args.panel_registry, sep="\t") if args.panel_registry.exists() else pd.DataFrame()
    modules = pd.concat([load_trait_modules(args.out_dir, key) for key in args.traits], ignore_index=True)
    modules["panel_scope"] = args.panel_file.stem
    modules = add_registry_fields(modules, registry)
    summary, tested = summarize_cross_trait(modules)
    contrast = dr_specificity_contrast(tested)

    write_table(args.out_dir / "tables" / "dr_specific_panel_all_module_tests.tsv", modules)
    write_table(args.out_dir / "tables" / "dr_specific_panel_tested_modules.tsv", tested)
    write_table(args.out_dir / "tables" / "dr_specific_panel_cross_trait_summary.tsv", summary)
    write_table(args.out_dir / "tables" / "dr_specific_panel_dr_vs_t2d_contrast.tsv", contrast)
    report = render_report(summary, contrast, args)
    (args.out_dir / "reports" / "dr_specific_panel_cross_trait_report.md").write_text(
        report + "\n",
        encoding="utf-8",
    )
    manifest = {
        "created_utc": now_utc(),
        "script_path": str(Path(__file__).resolve()),
        "panel_file": str(args.panel_file),
        "panel_registry": str(args.panel_registry),
        "out_dir": str(args.out_dir),
        "traits": args.traits,
        "n_degree_matched_null": int(args.n_degree_matched_null),
        "n_score_permutation_null": int(args.n_score_permutation_null),
        "seed": int(args.seed),
        "citation_status": "panel_citations_pending_manual_review",
    }
    (args.out_dir / "reports" / "dr_specific_panel_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote DR-specific biology panel analysis to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
