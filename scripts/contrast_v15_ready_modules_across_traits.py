#!/usr/bin/env python
"""Contrast DR_MVP V1.5 ready modules against T2D and BMI module evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TRAIT_COLUMNS = [
    "module_name",
    "module_status",
    "n_present",
    "n_loci",
    "top1_gene",
    "top1_gene_score",
    "top1_locus",
    "n_effective_loci",
    "top1_locus_contribution",
    "top5_locus_contribution",
    "locus_robust_z",
    "locus_robust_empirical_p",
    "ripple_d_z",
    "ripple_d_empirical_p",
    "module_specific_rank_z",
    "module_specific_rank_empirical_p",
    "leave_top1_locus_empirical_p",
    "moderate_locus_burden_empirical_p",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dr-module-tests", type=Path, required=True)
    parser.add_argument("--t2d-module-tests", type=Path, required=True)
    parser.add_argument("--bmi-module-tests", type=Path, required=True)
    parser.add_argument("--biology-audit", type=Path, required=True)
    parser.add_argument("--out-table", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    return parser.parse_args()


def read_trait_table(path: Path, prefix: str) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t")
    missing = sorted(set(TRAIT_COLUMNS) - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    out = table.loc[:, TRAIT_COLUMNS].copy()
    rename = {col: f"{prefix}_{col}" for col in TRAIT_COLUMNS if col != "module_name"}
    return out.rename(columns=rename)


def is_distributed_positive(row: pd.Series, prefix: str) -> bool:
    status = str(row.get(f"{prefix}_module_status", ""))
    p = pd.to_numeric(pd.Series([row.get(f"{prefix}_ripple_d_empirical_p", np.nan)]), errors="coerce").iloc[0]
    rank_p = pd.to_numeric(
        pd.Series([row.get(f"{prefix}_module_specific_rank_empirical_p", np.nan)]), errors="coerce"
    ).iloc[0]
    leave_p = pd.to_numeric(
        pd.Series([row.get(f"{prefix}_leave_top1_locus_empirical_p", np.nan)]), errors="coerce"
    ).iloc[0]
    return bool(
        status == "distributed_weak_signal_module_candidate"
        and np.isfinite(p)
        and p <= 0.05
        and np.isfinite(rank_p)
        and rank_p <= 0.05
        and np.isfinite(leave_p)
        and leave_p <= 0.10
    )


def classify(row: pd.Series) -> str:
    dr_pos = bool(row["dr_positive"])
    t2d_pos = bool(row["t2d_positive"])
    bmi_pos = bool(row["bmi_positive"])
    z_margin = float(row["dr_vs_max_t2d_bmi_ripple_d_z_margin"])
    if dr_pos and not t2d_pos and not bmi_pos and z_margin >= 1.0:
        return "DR_enriched_candidate"
    if dr_pos and not t2d_pos and not bmi_pos:
        return "DR_only_by_status_small_z_margin"
    if dr_pos and t2d_pos and not bmi_pos:
        return "T2D_shared_candidate"
    if dr_pos and bmi_pos and not t2d_pos:
        return "BMI_shared_candidate"
    if dr_pos and t2d_pos and bmi_pos:
        return "broad_shared_candidate"
    return "not_DR_specific_or_inconclusive"


def main() -> None:
    args = parse_args()
    biology = pd.read_csv(args.biology_audit, sep="\t")
    dr = read_trait_table(args.dr_module_tests, "dr")
    t2d = read_trait_table(args.t2d_module_tests, "t2d")
    bmi = read_trait_table(args.bmi_module_tests, "bmi")
    keep = set(dr["module_name"].astype(str))
    out = biology.merge(dr, on="module_name", how="left").merge(t2d, on="module_name", how="left").merge(
        bmi, on="module_name", how="left"
    )
    out = out.loc[out["module_name"].astype(str).isin(keep)].copy()

    for prefix in ["dr", "t2d", "bmi"]:
        out[f"{prefix}_positive"] = out.apply(is_distributed_positive, axis=1, prefix=prefix)
        for col in [
            "ripple_d_z",
            "ripple_d_empirical_p",
            "module_specific_rank_z",
            "module_specific_rank_empirical_p",
            "leave_top1_locus_empirical_p",
            "top1_locus_contribution",
            "top5_locus_contribution",
            "n_effective_loci",
        ]:
            out[f"{prefix}_{col}"] = pd.to_numeric(out[f"{prefix}_{col}"], errors="coerce")

    out["max_t2d_bmi_ripple_d_z"] = out[["t2d_ripple_d_z", "bmi_ripple_d_z"]].max(axis=1)
    out["dr_vs_t2d_ripple_d_z_margin"] = out["dr_ripple_d_z"] - out["t2d_ripple_d_z"]
    out["dr_vs_bmi_ripple_d_z_margin"] = out["dr_ripple_d_z"] - out["bmi_ripple_d_z"]
    out["dr_vs_max_t2d_bmi_ripple_d_z_margin"] = out["dr_ripple_d_z"] - out["max_t2d_bmi_ripple_d_z"]
    out["specificity_class"] = out.apply(classify, axis=1)

    order = [
        "module_name",
        "specificity_class",
        "biological_theme",
        "theme_reason",
        "diabetes_prior_gene_overlap",
        "dr_microvascular_prior_gene_overlap",
        "dr_positive",
        "t2d_positive",
        "bmi_positive",
        "dr_ripple_d_z",
        "t2d_ripple_d_z",
        "bmi_ripple_d_z",
        "dr_vs_max_t2d_bmi_ripple_d_z_margin",
        "dr_ripple_d_empirical_p",
        "t2d_ripple_d_empirical_p",
        "bmi_ripple_d_empirical_p",
        "dr_module_status",
        "t2d_module_status",
        "bmi_module_status",
        "dr_top1_gene",
        "t2d_top1_gene",
        "bmi_top1_gene",
        "dr_n_effective_loci",
        "t2d_n_effective_loci",
        "bmi_n_effective_loci",
        "dr_top1_locus_contribution",
        "t2d_top1_locus_contribution",
        "bmi_top1_locus_contribution",
        "dr_top5_locus_contribution",
        "t2d_top5_locus_contribution",
        "bmi_top5_locus_contribution",
        "n_extreme_score_ge5",
        "n_upper_tail_score_ge3",
        "n_moderate_score_1_to_3",
        "top10_module_genes_by_score",
    ]
    remaining = [col for col in out.columns if col not in order]
    out = out.loc[:, order + remaining]
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_table, sep="\t", index=False)

    lines = [
        "# V1.5 DR_MVP ready module trait-specificity contrast",
        "",
        f"Modules contrasted: {len(out)}",
        "",
        "## Specificity class counts",
        "",
        out["specificity_class"].value_counts(dropna=False).to_string(),
        "",
        "## Positive status counts",
        "",
        f"DR positive: {int(out['dr_positive'].sum())}",
        f"T2D positive: {int(out['t2d_positive'].sum())}",
        f"BMI positive: {int(out['bmi_positive'].sum())}",
        "",
        "## Top DR-enriched candidates by Z margin",
        "",
        out.sort_values("dr_vs_max_t2d_bmi_ripple_d_z_margin", ascending=False)[
            [
                "module_name",
                "specificity_class",
                "biological_theme",
                "dr_vs_max_t2d_bmi_ripple_d_z_margin",
                "dr_ripple_d_z",
                "t2d_ripple_d_z",
                "bmi_ripple_d_z",
            ]
        ]
        .head(15)
        .to_string(index=False),
        "",
        f"Detailed table: `{args.out_table}`",
    ]
    args.out_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
