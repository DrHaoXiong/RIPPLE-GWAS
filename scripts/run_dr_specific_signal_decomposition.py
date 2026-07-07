#!/usr/bin/env python
"""Decompose DR gene scores against shared metabolic liability signals.

This diagnostic constructs DR-specific residual gene scores by regressing
DR_MVP residualized gene scores on T2D and optionally BMI residualized gene
scores. It then retests curated panels and cell-type marker sets with the same
tail-robust gene-set statistics used for manuscript-facing sensitivity checks.

The output is diagnostic. It does not modify the frozen RIPPLE gene-score layer.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from run_gene_set_tail_robust_diagnostics import (  # noqa: E402
    STATISTICS,
    add_degree_bins,
    add_fdr,
    add_rank_percentile,
    build_claim_table,
    degree_matched_null_statistics,
    empty_stat,
    empty_tail,
    load_gene_sets,
    positive_contribution,
    stable_offset,
    stat_value,
    tail_status,
    write_table,
)

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "dr_specific_signal_decomposition_v0_1"

SCORE_CONFIGS = {
    "DR_MVP": {
        "trait": "DR_MVP",
        "analysis_dir": "dr_mvp_string_final5000",
        "score_file": "DR_MVP.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
    },
    "DR_MVP_NO_MHC_NO_APOE": {
        "trait": "DR_MVP_NO_MHC_NO_APOE",
        "analysis_dir": "dr_mvp_no_mhc_no_apoe_final5000",
        "score_file": "DR_MVP_NO_MHC_NO_APOE.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
    },
    "T2D": {
        "trait": "T2D",
        "analysis_dir": "t2d_analysis_ready",
        "score_file": "T2D.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
    },
    "BMI_IRN": {
        "trait": "BMI_IRN",
        "analysis_dir": "bmi_irn_analysis_ready",
        "score_file": "BMI_IRN.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
    },
}

GENE_SET_INPUTS = {
    "dr_specific_panel": PRIVATE_ROOT
    / "20_processed_data"
    / "reference_pathways"
    / "dr_specific_biology_panel_v0_1"
    / "tables"
    / "dr_specific_biology_panel_v0_1.gene_sets.tsv",
    "dr_retinal_only_panel": PRIVATE_ROOT
    / "20_processed_data"
    / "reference_pathways"
    / "dr_specific_biology_panel_v0_1"
    / "tables"
    / "dr_specific_biology_panel_v0_1.retinal_only.gene_sets.tsv",
    "scrna_cell_type_min100": PRIVATE_ROOT
    / "20_processed_data"
    / "reference_expression"
    / "dr_cell_type_specificity_v0_1_min100"
    / "tables"
    / "dr_scrna_cell_type_marker_gene_sets.tsv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-null", type=int, default=5000)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--min-overlap", type=int, default=5)
    parser.add_argument("--trim-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--include-low-information", action="store_true")
    parser.add_argument("--include-special-region", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def score_path(key: str) -> Path:
    config = SCORE_CONFIGS[key]
    return ANALYSIS_ROOT / str(config["analysis_dir"]) / "tables" / str(config["score_file"])


def load_score_table(key: str, args: argparse.Namespace) -> pd.DataFrame:
    path = score_path(key)
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path, sep="\t")
    required = {"gene_symbol", "assoc_resid_score", "graph_degree"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    keep_cols = ["gene_symbol", "assoc_resid_score", "graph_degree"]
    for column in ["is_low_information", "is_special_region"]:
        if column in table.columns:
            keep_cols.append(column)
    table = table.loc[:, keep_cols].copy()
    table["gene_symbol"] = table["gene_symbol"].astype(str).str.upper()
    table["assoc_resid_score"] = pd.to_numeric(table["assoc_resid_score"], errors="coerce")
    table["graph_degree"] = pd.to_numeric(table["graph_degree"], errors="coerce").fillna(0.0)
    for column in ["is_low_information", "is_special_region"]:
        if column not in table.columns:
            table[column] = False
        table[column] = table[column].fillna(False).astype(bool)
    table = table.dropna(subset=["assoc_resid_score"]).drop_duplicates("gene_symbol")
    if not args.include_low_information:
        table = table.loc[~table["is_low_information"]].copy()
    if not args.include_special_region:
        table = table.loc[~table["is_special_region"]].copy()
    return table.reset_index(drop=True)


def zscore(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    if sd <= 0 or not np.isfinite(sd):
        raise ValueError("Cannot z-score vector with nonpositive standard deviation.")
    return (values - mean) / sd, mean, sd


def regress_residual(y: np.ndarray, predictors: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    design = np.column_stack([np.ones(y.shape[0]), predictors])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    residual = y - fitted
    ss_resid = float(np.sum(residual**2))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_resid / ss_total if ss_total > 0 else np.nan
    return residual, beta, r2


def build_decomposed_scores(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    loaded = {key: load_score_table(key, args) for key in SCORE_CONFIGS}
    base = loaded["T2D"].rename(columns={"assoc_resid_score": "score_T2D"})[
        ["gene_symbol", "score_T2D"]
    ].merge(
        loaded["BMI_IRN"].rename(columns={"assoc_resid_score": "score_BMI_IRN"})[
            ["gene_symbol", "score_BMI_IRN"]
        ],
        on="gene_symbol",
        how="inner",
    )
    long_rows: list[pd.DataFrame] = []
    model_rows: list[dict[str, object]] = []
    for dr_key in ["DR_MVP", "DR_MVP_NO_MHC_NO_APOE"]:
        dr = loaded[dr_key].rename(columns={"assoc_resid_score": "score_DR"})
        merged = dr.merge(base, on="gene_symbol", how="inner")
        y_z, y_mean, y_sd = zscore(merged["score_DR"].to_numpy(dtype=float))
        predictor_sets = {
            "resid_T2D": ["score_T2D"],
            "resid_T2D_BMI": ["score_T2D", "score_BMI_IRN"],
        }
        for mode, predictor_cols in predictor_sets.items():
            predictor_z = []
            predictor_scaling: dict[str, dict[str, float]] = {}
            for column in predictor_cols:
                z, mean, sd = zscore(merged[column].to_numpy(dtype=float))
                predictor_z.append(z)
                predictor_scaling[column] = {"mean": mean, "sd": sd}
            predictors = np.column_stack(predictor_z)
            residual, beta, r2 = regress_residual(y_z, predictors)
            residual_z, residual_mean, residual_sd = zscore(residual)
            out = merged.loc[
                :,
                [
                    "gene_symbol",
                    "graph_degree",
                    "is_low_information",
                    "is_special_region",
                    "score_DR",
                    "score_T2D",
                    "score_BMI_IRN",
                ],
            ].copy()
            out["trait"] = str(SCORE_CONFIGS[dr_key]["trait"])
            out["analysis_id"] = f"{dr_key}_{mode}"
            out["score_mode"] = mode
            out["assoc_resid_score"] = residual_z
            out["source_dr_score_mean"] = y_mean
            out["source_dr_score_sd"] = y_sd
            out["model_r2"] = r2
            out["predictors"] = ",".join(predictor_cols)
            long_rows.append(out)
            model_rows.append(
                {
                    "trait": str(SCORE_CONFIGS[dr_key]["trait"]),
                    "analysis_id": f"{dr_key}_{mode}",
                    "score_mode": mode,
                    "n_genes": int(merged.shape[0]),
                    "predictors": ",".join(predictor_cols),
                    "intercept": float(beta[0]),
                    "coefficients": ",".join(f"{value:.8g}" for value in beta[1:]),
                    "model_r2": r2,
                    "source_dr_score_mean": y_mean,
                    "source_dr_score_sd": y_sd,
                    "residual_prestandardization_mean": residual_mean,
                    "residual_prestandardization_sd": residual_sd,
                    "t2d_score_path": str(score_path("T2D")),
                    "bmi_score_path": str(score_path("BMI_IRN")),
                    "dr_score_path": str(score_path(dr_key)),
                    "script_path": str(Path(__file__).resolve()),
                    "timestamp": now_utc(),
                }
            )
    return pd.concat(long_rows, ignore_index=True), pd.DataFrame(model_rows)


def robust_tests_for_score_table(
    score_table: pd.DataFrame,
    gene_set_file: Path,
    analysis_label: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_sets, gene_set_summary = load_gene_sets(gene_set_file)
    stat_tables: list[pd.DataFrame] = []
    claim_tables: list[pd.DataFrame] = []
    for analysis_id, raw_universe in score_table.groupby("analysis_id", observed=True):
        universe = raw_universe.copy().reset_index(drop=True)
        universe = add_degree_bins(universe, args.degree_bins)
        universe = add_rank_percentile(universe)
        score_by_gene = universe.set_index("gene_symbol", drop=False)
        cap_p99 = float(universe["assoc_resid_score"].quantile(0.99))
        cap_p995 = float(universe["assoc_resid_score"].quantile(0.995))
        stat_rows: list[dict[str, object]] = []
        tail_rows: list[dict[str, object]] = []
        for row in gene_set_summary.to_dict(orient="records"):
            gene_set = str(row["gene_set"])
            genes = set(gene_sets.loc[gene_sets["gene_set"].eq(gene_set), "gene_symbol"].astype(str).str.upper())
            present_genes = sorted(genes.intersection(set(score_by_gene.index)))
            trait = str(universe["trait"].iloc[0])
            base = {
                "trait": trait,
                "analysis_id": str(analysis_id),
                "trait_role": "dr_specific_decomposition",
                "analysis_label": analysis_label,
                "gene_set": gene_set,
                "n_query_genes": int(row.get("n_query_genes", len(genes))),
                "n_present_genes": len(present_genes),
                "score_path": str(args.out_dir / "tables" / "dr_specific_decomposed_gene_scores.tsv.gz"),
                "gene_set_file": str(gene_set_file),
            }
            for key, value in row.items():
                if key not in base:
                    base[key] = value
            if len(present_genes) < args.min_overlap:
                tail_rows.append({**base, **empty_tail("low_gene_overlap")})
                for statistic in STATISTICS:
                    stat_rows.append({**base, **empty_stat(statistic, args.n_null, "low_gene_overlap")})
                continue
            selected = score_by_gene.loc[present_genes].copy()
            rng = np.random.default_rng(args.seed + stable_offset(f"{analysis_id}:{analysis_label}:{gene_set}"))
            nulls = degree_matched_null_statistics(
                universe,
                selected,
                n_null=args.n_null,
                rng=rng,
                trim_fraction=args.trim_fraction,
                cap_p99=cap_p99,
                cap_p995=cap_p995,
            )
            values = selected["assoc_resid_score"].to_numpy(dtype=float)
            rank_values = selected["score_rank_percentile"].to_numpy(dtype=float)
            top = selected.sort_values("assoc_resid_score", ascending=False)
            tail_rows.append(
                {
                    **base,
                    "max_gene_score": float(np.max(values)),
                    "max_gene_symbol": str(top.iloc[0]["gene_symbol"]),
                    "top1_positive_score_contribution": positive_contribution(values, 1),
                    "top5_positive_score_contribution": positive_contribution(values, 5),
                    "top10_positive_score_contribution": positive_contribution(values, 10),
                    "top5_gene_symbols": ",".join(top["gene_symbol"].astype(str).head(5)),
                    "top5_gene_scores": ",".join(f"{value:.5g}" for value in top["assoc_resid_score"].head(5)),
                    "tail_diagnostic_status": tail_status(values),
                    "exclusion_or_na_reason": "none",
                }
            )
            for statistic in STATISTICS:
                observed = stat_value(
                    values,
                    rank_values,
                    statistic,
                    trim_fraction=args.trim_fraction,
                    cap_p99=cap_p99,
                    cap_p995=cap_p995,
                )
                null_values = nulls[statistic]
                finite_null = null_values[np.isfinite(null_values)]
                null_mean = float(np.mean(finite_null)) if finite_null.size else float("nan")
                null_sd = float(np.std(finite_null, ddof=1)) if finite_null.size > 1 else float("nan")
                z = (observed - null_mean) / null_sd if null_sd > 0 else float("nan")
                empirical_p = (
                    float((1 + np.sum(finite_null >= observed)) / (1 + finite_null.size))
                    if np.isfinite(observed) and finite_null.size
                    else float("nan")
                )
                stat_rows.append(
                    {
                        **base,
                        "statistic_name": statistic,
                        "observed_value": observed,
                        "null_mean": null_mean,
                        "null_sd": null_sd,
                        "z": z,
                        "empirical_p": empirical_p,
                        "n_null": int(args.n_null),
                        "statistic_direction": "greater_is_more_extreme",
                        "exclusion_or_na_reason": "none",
                        "script_path": str(Path(__file__).resolve()),
                        "seed": int(args.seed + stable_offset(f"{analysis_id}:{analysis_label}:{gene_set}")),
                        "timestamp": now_utc(),
                    }
                )
        stat_table = add_fdr(pd.DataFrame(stat_rows))
        claim_table = build_claim_table(stat_table, pd.DataFrame(tail_rows))
        stat_tables.append(stat_table)
        claim_tables.append(claim_table)
    return pd.concat(stat_tables, ignore_index=True), pd.concat(claim_tables, ignore_index=True)


def write_report(out_dir: Path, model_table: pd.DataFrame, claim_summaries: list[pd.DataFrame]) -> None:
    lines = [
        "# DR-specific signal decomposition v0.1",
        "",
        f"Created: {now_utc()}",
        "",
        "DR residualized gene scores were standardized and regressed on T2D or T2D+BMI standardized gene scores.",
        "The resulting standardized residuals are diagnostic DR-specific scores and do not replace the frozen RIPPLE gene-score layer.",
        "",
        "## Decomposition models",
        "",
        "| Analysis | Predictors | n genes | R2 |",
        "|---|---|---:|---:|",
    ]
    for row in model_table.to_dict(orient="records"):
        lines.append(
            f"| {row['analysis_id']} | {row['predictors']} | {int(row['n_genes'])} | {float(row['model_r2']):.4g} |"
        )
    lines.extend(["", "## Robust gene-set claim summary", "", "| Analysis label | Analysis | Calibrated | Outlier-driven | Negative |", "|---|---|---:|---:|---:|"])
    if claim_summaries:
        claims = pd.concat(claim_summaries, ignore_index=True)
        for (analysis_label, analysis_id), group in claims.groupby(["analysis_label", "analysis_id"], observed=True):
            lines.append(
                f"| {analysis_label} | {analysis_id} | "
                f"{int(group['tail_robust_claim_status'].eq('calibrated_contextual_or_module_support').sum())} | "
                f"{int(group['tail_robust_claim_status'].eq('outlier_driven_supportive_only').sum())} | "
                f"{int(group['tail_robust_claim_status'].eq('negative').sum())} |"
            )
    (out_dir / "reports" / "dr_specific_signal_decomposition_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    for subdir in ["tables", "reports"]:
        (args.out_dir / subdir).mkdir(parents=True, exist_ok=True)
    decomposed, models = build_decomposed_scores(args)
    write_table(args.out_dir / "tables" / "dr_specific_decomposed_gene_scores.tsv.gz", decomposed)
    write_table(args.out_dir / "tables" / "dr_specific_decomposition_model_summary.tsv", models)
    claim_summaries: list[pd.DataFrame] = []
    for label, gene_set_file in GENE_SET_INPUTS.items():
        stats, claims = robust_tests_for_score_table(decomposed, gene_set_file, label, args)
        write_table(args.out_dir / "tables" / f"{label}.decomposed_tail_robust_statistics.tsv", stats)
        write_table(args.out_dir / "tables" / f"{label}.decomposed_tail_robust_claims.tsv", claims)
        claim_summaries.append(claims)
    manifest = {
        "created_utc": now_utc(),
        "script_path": str(Path(__file__).resolve()),
        "out_dir": str(args.out_dir),
        "n_null": int(args.n_null),
        "degree_bins": int(args.degree_bins),
        "min_overlap": int(args.min_overlap),
        "trim_fraction": float(args.trim_fraction),
        "include_low_information": bool(args.include_low_information),
        "include_special_region": bool(args.include_special_region),
        "decomposition_modes": ["resid_T2D", "resid_T2D_BMI"],
        "gene_set_inputs": {key: str(value) for key, value in GENE_SET_INPUTS.items()},
        "claim_boundary": "diagnostic DR-specific residualization; positive results require independent biological review",
    }
    (args.out_dir / "reports" / "dr_specific_signal_decomposition_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    write_report(args.out_dir, models, claim_summaries)
    print(f"Wrote DR-specific signal decomposition outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
