#!/usr/bin/env python
"""Run contextual cell-type marker enrichment against RIPPLE residualized gene scores.

This is an exploratory contextual screen. The marker sets encode cell-type
identity/support in retina, FVM, or PBMC data; they are not DR-specific genetic
validation unless a downstream DR-specific decomposition and contrast supports
that interpretation.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_MARKERS = (
    PRIVATE_ROOT
    / "20_processed_data"
    / "reference_expression"
    / "dr_cell_type_specificity_v0_1"
    / "tables"
    / "dr_scrna_cell_type_marker_gene_sets.tsv"
)
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "dr_cell_type_specificity_v0_1"

TRAITS = {
    "DR_MVP": {
        "trait": "DR_MVP",
        "analysis_dir": "dr_mvp_string_final5000",
        "score_file": "DR_MVP.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "role": "primary_dr",
    },
    "DR_MVP_NO_MHC_NO_APOE": {
        "trait": "DR_MVP_NO_MHC_NO_APOE",
        "analysis_dir": "dr_mvp_no_mhc_no_apoe_final5000",
        "score_file": "DR_MVP_NO_MHC_NO_APOE.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "role": "primary_dr_sensitivity",
    },
    "T2D": {
        "trait": "T2D",
        "analysis_dir": "t2d_analysis_ready",
        "score_file": "T2D.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "role": "diabetic_liability_comparator",
    },
    "BMI_IRN": {
        "trait": "BMI_IRN",
        "analysis_dir": "bmi_irn_analysis_ready",
        "score_file": "BMI_IRN.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "role": "metabolic_comparator",
    },
    "HEIGHT_IRN": {
        "trait": "HEIGHT_IRN",
        "analysis_dir": "height_irn_analysis_ready",
        "score_file": "HEIGHT_IRN.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "role": "non_dr_anthropometric_comparator",
    },
    "SCZ": {
        "trait": "SCZ",
        "analysis_dir": "scz_no_mhc_string_final5000",
        "score_file": "SCZ.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "role": "non_ocular_polygenic_comparator",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker-gene-sets", type=Path, default=DEFAULT_MARKERS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--traits", nargs="*", default=list(TRAITS))
    parser.add_argument("--n-null", type=int, default=5000)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--top-fractions", nargs="*", type=float, default=[0.01, 0.02, 0.05, 0.10])
    parser.add_argument("--min-overlap", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--include-low-information", action="store_true")
    parser.add_argument("--include-special-region", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def load_scores(config: dict[str, str], args: argparse.Namespace) -> pd.DataFrame:
    path = ANALYSIS_ROOT / config["analysis_dir"] / "tables" / config["score_file"]
    if not path.exists():
        raise FileNotFoundError(path)
    scores = pd.read_csv(path, sep="\t")
    required = {"gene_symbol", "assoc_resid_score", "graph_degree"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    keep_cols = ["gene_symbol", "assoc_resid_score", "graph_degree"]
    for column in ["is_low_information", "is_special_region"]:
        if column in scores.columns:
            keep_cols.append(column)
    scores = scores.loc[:, keep_cols].copy()
    scores["gene_symbol"] = scores["gene_symbol"].astype(str).str.upper()
    scores["assoc_resid_score"] = pd.to_numeric(scores["assoc_resid_score"], errors="coerce")
    scores["graph_degree"] = pd.to_numeric(scores["graph_degree"], errors="coerce").fillna(0.0)
    for column in ["is_low_information", "is_special_region"]:
        if column not in scores.columns:
            scores[column] = False
        scores[column] = scores[column].fillna(False).astype(bool)
    scores = scores.dropna(subset=["assoc_resid_score"]).drop_duplicates("gene_symbol")
    if not args.include_low_information:
        scores = scores.loc[~scores["is_low_information"]].copy()
    if not args.include_special_region:
        scores = scores.loc[~scores["is_special_region"]].copy()
    return scores


def add_degree_bins(scores: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    scores = scores.copy()
    ranks = scores["graph_degree"].rank(method="first")
    scores["degree_bin"] = pd.qcut(ranks, q=min(n_bins, len(scores)), labels=False, duplicates="drop").astype(int)
    return scores


def empirical_p(observed: float, null_values: np.ndarray, direction: str = "greater") -> float:
    if direction != "greater":
        raise ValueError("Only greater-is-more-extreme is implemented.")
    return float((1 + np.sum(null_values >= observed)) / (1 + null_values.shape[0]))


def summarize_marker_sets(markers: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["gene_set", "dataset_id", "tissue_context", "condition_scope", "cell_type"]
    aggregations: dict[str, tuple[str, str]] = {"n_marker_genes": ("gene_symbol", "nunique")}
    for column in ["n_target_cells", "n_background_cells", "marker_qc_status"]:
        if column in markers.columns:
            aggregations[column] = (column, "first")
    summary = markers.groupby(meta_cols, observed=True).agg(**aggregations).reset_index().sort_values(meta_cols)
    if "marker_qc_status" not in summary.columns:
        summary["marker_qc_status"] = "unknown_cell_support"
    return summary


def degree_matched_mean_null(
    scores: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    n_null: int,
    rng: np.random.Generator,
) -> np.ndarray:
    bins = selected["degree_bin"].value_counts().to_dict()
    values: list[float] = []
    pools = {
        int(bin_id): group["assoc_resid_score"].to_numpy(dtype=float)
        for bin_id, group in scores.groupby("degree_bin", observed=True)
    }
    for _ in range(n_null):
        sampled: list[np.ndarray] = []
        for bin_id, count in bins.items():
            pool = pools[int(bin_id)]
            replace = int(count) > pool.shape[0]
            sampled.append(rng.choice(pool, size=int(count), replace=replace))
        values.append(float(np.mean(np.concatenate(sampled))) if sampled else np.nan)
    return np.asarray(values, dtype=float)


def top_fraction_rows(
    trait: str,
    trait_role: str,
    scores: pd.DataFrame,
    genes: set[str],
    *,
    top_fractions: list[float],
) -> list[dict[str, object]]:
    rows = []
    universe = set(scores["gene_symbol"])
    marker_genes = genes.intersection(universe)
    if not marker_genes:
        return rows
    ordered = scores.sort_values("assoc_resid_score", ascending=False).reset_index(drop=True)
    marker_n = len(marker_genes)
    universe_n = int(scores.shape[0])
    for fraction in top_fractions:
        n_top = max(1, int(round(universe_n * fraction)))
        top_genes = set(ordered.head(n_top)["gene_symbol"])
        overlap = len(marker_genes.intersection(top_genes))
        p_value = float(hypergeom.sf(overlap - 1, universe_n, marker_n, n_top)) if overlap > 0 else 1.0
        rows.append(
            {
                "trait": trait,
                "trait_role": trait_role,
                "top_fraction": fraction,
                "universe_n": universe_n,
                "marker_gene_n": marker_n,
                "top_gene_n": n_top,
                "overlap_n": overlap,
                "hypergeom_p": p_value,
                "overlap_fraction_of_marker_set": overlap / marker_n,
            }
        )
    return rows


def run_trait(
    key: str,
    marker_summary: pd.DataFrame,
    markers: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = TRAITS[key]
    scores = add_degree_bins(load_scores(config, args), args.degree_bins)
    score_by_gene = scores.set_index("gene_symbol")
    rng = np.random.default_rng(args.seed + stable_offset(key))
    rows: list[dict[str, object]] = []
    top_rows: list[dict[str, object]] = []
    for marker_set in marker_summary.to_dict(orient="records"):
        genes = set(
            markers.loc[markers["gene_set"].eq(marker_set["gene_set"]), "gene_symbol"].astype(str).str.upper()
        )
        present_genes = sorted(genes.intersection(set(score_by_gene.index)))
        if len(present_genes) < args.min_overlap:
            rows.append(not_tested_row(config, marker_set, present_genes, args, "low_gene_overlap"))
            continue
        selected = score_by_gene.loc[present_genes].reset_index()
        observed_mean = float(selected["assoc_resid_score"].mean())
        observed_sum = float(selected["assoc_resid_score"].sum())
        null_values = degree_matched_mean_null(scores, selected, n_null=args.n_null, rng=rng)
        null_mean = float(np.nanmean(null_values))
        null_sd = float(np.nanstd(null_values, ddof=1))
        z = (observed_mean - null_mean) / null_sd if null_sd > 0 else np.nan
        p_value = empirical_p(observed_mean, null_values)
        rows.append(
            {
                "trait": config["trait"],
                "analysis_id": key,
                "trait_role": config["role"],
                **marker_set,
                "n_present_genes": len(present_genes),
                "present_gene_symbols": ",".join(present_genes),
                "observed_mean_score": observed_mean,
                "observed_sum_score": observed_sum,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "degree_matched_z": z,
                "degree_matched_empirical_p": p_value,
                "degree_matched_fdr_bh": np.nan,
                "n_null": int(args.n_null),
                "statistic_direction": "greater_is_more_extreme",
                "cell_context_status": "pending_fdr",
                "claim_boundary": "contextual_expression_support_not_genetic_validation",
                "analysis_role": "exploratory_contextual_screen_not_claim_source",
            }
        )
        for top_row in top_fraction_rows(
            config["trait"],
            config["role"],
            scores,
            genes,
            top_fractions=args.top_fractions,
        ):
            top_rows.append({**marker_set, **top_row})
    return pd.DataFrame(rows), pd.DataFrame(top_rows)


def not_tested_row(
    config: dict[str, str],
    marker_set: dict[str, Any],
    present_genes: list[str],
    args: argparse.Namespace,
    reason: str,
) -> dict[str, object]:
    return {
        "trait": config["trait"],
        "analysis_id": config["trait"],
        "trait_role": config["role"],
        **marker_set,
        "n_present_genes": len(present_genes),
        "present_gene_symbols": ",".join(present_genes),
        "observed_mean_score": np.nan,
        "observed_sum_score": np.nan,
        "null_mean": np.nan,
        "null_sd": np.nan,
        "degree_matched_z": np.nan,
        "degree_matched_empirical_p": np.nan,
        "degree_matched_fdr_bh": np.nan,
        "n_null": int(args.n_null),
        "statistic_direction": "greater_is_more_extreme",
        "cell_context_status": f"not_tested_{reason}",
        "claim_boundary": "contextual_expression_support_not_genetic_validation",
        "analysis_role": "exploratory_contextual_screen_not_claim_source",
    }


def stable_offset(text: str, modulo: int = 100_000) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % modulo


def add_fdr_and_status(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    for trait, index in results.groupby("trait", observed=True).groups.items():
        tested = results.loc[index]
        mask = tested["degree_matched_empirical_p"].notna().to_numpy()
        tested_index = tested.index[mask]
        p_values = tested.loc[tested_index, "degree_matched_empirical_p"].to_numpy(dtype=float)
        q_values = bh_fdr(p_values)
        results.loc[tested_index, "degree_matched_fdr_bh"] = q_values
    z = pd.to_numeric(results["degree_matched_z"], errors="coerce")
    p = pd.to_numeric(results["degree_matched_empirical_p"], errors="coerce")
    q = pd.to_numeric(results["degree_matched_fdr_bh"], errors="coerce")
    status = np.full(results.shape[0], "negative", dtype=object)
    status[(z >= 2.5) & (q <= 0.10)] = "context_supported"
    status[(status == "negative") & (z >= 2.0) & (p <= 0.05)] = "nominal_suggestive"
    status[results["cell_context_status"].astype(str).str.startswith("not_tested").to_numpy()] = results.loc[
        results["cell_context_status"].astype(str).str.startswith("not_tested"),
        "cell_context_status",
    ]
    results["cell_context_status"] = status
    return results


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    n = p_values.shape[0]
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def summarize_cross_trait(results: pd.DataFrame) -> pd.DataFrame:
    tested = results.loc[~results["cell_context_status"].astype(str).str.startswith("not_tested")].copy()
    tested = tested.sort_values(
        ["trait", "degree_matched_empirical_p", "degree_matched_z"],
        ascending=[True, True, False],
    )
    rows = []
    for trait, group in tested.groupby("trait", observed=True):
        best = group.iloc[0]
        rows.append(
            {
                "trait": trait,
                "analysis_id": best["analysis_id"],
                "trait_role": best["trait_role"],
                "n_tested_marker_sets": int(group.shape[0]),
                "n_context_supported": int(group["cell_context_status"].eq("context_supported").sum()),
                "n_nominal_suggestive": int(group["cell_context_status"].eq("nominal_suggestive").sum()),
                "best_gene_set": best["gene_set"],
                "best_dataset_id": best["dataset_id"],
                "best_tissue_context": best["tissue_context"],
                "best_condition_scope": best["condition_scope"],
                "best_cell_type": best["cell_type"],
                "best_degree_matched_z": best["degree_matched_z"],
                "best_degree_matched_empirical_p": best["degree_matched_empirical_p"],
                "best_status": best["cell_context_status"],
            }
        )
    return pd.DataFrame(rows).sort_values("trait")


def dr_vs_t2d_contrast(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene_set, group in results.groupby("gene_set", observed=True):
        by_trait = {str(row["trait"]): row for row in group.to_dict(orient="records")}
        dr = by_trait.get("DR_MVP")
        t2d = by_trait.get("T2D")
        if dr is None:
            continue
        rows.append(
            {
                "gene_set": gene_set,
                "dataset_id": dr.get("dataset_id", ""),
                "tissue_context": dr.get("tissue_context", ""),
                "condition_scope": dr.get("condition_scope", ""),
                "cell_type": dr.get("cell_type", ""),
                "DR_MVP_z": dr.get("degree_matched_z", np.nan),
                "DR_MVP_p": dr.get("degree_matched_empirical_p", np.nan),
                "DR_MVP_status": dr.get("cell_context_status", ""),
                "T2D_z": t2d.get("degree_matched_z", np.nan) if t2d else np.nan,
                "T2D_p": t2d.get("degree_matched_empirical_p", np.nan) if t2d else np.nan,
                "T2D_status": t2d.get("cell_context_status", "") if t2d else "",
                "DR_minus_T2D_z": numeric(dr.get("degree_matched_z")) - numeric(t2d.get("degree_matched_z") if t2d else np.nan),
                "specificity_label": specificity_label(dr, t2d),
                "claim_boundary": "requires_dr_specific_decomposition_before_dr_specific_interpretation",
            }
        )
    return pd.DataFrame(rows).sort_values(["DR_MVP_p", "DR_minus_T2D_z"], ascending=[True, False])


def numeric(value: object) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else np.nan


def specificity_label(dr: dict[str, Any], t2d: dict[str, Any] | None) -> str:
    dr_status = str(dr.get("cell_context_status", ""))
    t2d_status = str(t2d.get("cell_context_status", "")) if t2d else ""
    if dr_status == "context_supported" and t2d_status != "context_supported":
        return "dr_context_enriched"
    if dr_status == "context_supported" and t2d_status == "context_supported":
        return "shared_or_non_specific_context"
    if dr_status == "nominal_suggestive" and t2d_status not in {"context_supported", "nominal_suggestive"}:
        return "dr_suggestive_context"
    return "not_dr_specific_context_supported"


def render_report(summary: pd.DataFrame, contrast: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        "# DR cell-type contextual marker enrichment v0.1",
        "",
        f"Created: {now_utc()}",
        "",
        "This exploratory analysis tests whether RIPPLE residualized GWAS gene scores are enriched in scRNA-derived cell-type marker sets.",
        "It is a contextual annotation layer only; marker sets are not DR-vs-control markers, do not validate genetic modules, and do not upgrade topology-specific claims.",
        "",
        f"Degree-matched nulls per marker set: {args.n_null}",
        "",
        "## Cross-trait summary",
        "",
        "| Trait | Context-supported | Nominal suggestive | Best cell type | Best dataset | Best Z | Best P | Status |",
        "|---|---:|---:|---|---|---:|---:|---|",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            f"| {row['trait']} | {int(row['n_context_supported'])} | {int(row['n_nominal_suggestive'])} | "
            f"{row['best_cell_type']} | {row['best_dataset_id']}:{row['best_condition_scope']} | "
            f"{float(row['best_degree_matched_z']):.3g} | {float(row['best_degree_matched_empirical_p']):.4g} | "
            f"{row['best_status']} |"
        )
    lines.extend(
        [
            "",
            "## Top DR versus T2D contexts",
            "",
            "| Gene set | Tissue | Scope | Cell type | DR Z | DR P | T2D Z | T2D P | Label |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in contrast.head(20).to_dict(orient="records"):
        lines.append(
            f"| {row['gene_set']} | {row['tissue_context']} | {row['condition_scope']} | {row['cell_type']} | "
            f"{fmt(row['DR_MVP_z'])} | {fmt(row['DR_MVP_p'])} | {fmt(row['T2D_z'])} | {fmt(row['T2D_p'])} | "
            f"{row['specificity_label']} |"
        )
    return "\n".join(lines)


def fmt(value: object) -> str:
    number = numeric(value)
    return f"{number:.4g}" if np.isfinite(number) else ""


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["tables", "reports"]:
        (args.out_dir / subdir).mkdir(parents=True, exist_ok=True)
    markers = pd.read_csv(args.marker_gene_sets, sep="\t")
    markers["gene_symbol"] = markers["gene_symbol"].astype(str).str.upper()
    marker_summary = summarize_marker_sets(markers)
    unknown = [trait for trait in args.traits if trait not in TRAITS]
    if unknown:
        raise ValueError(f"Unknown traits: {unknown}")
    result_tables = []
    top_tables = []
    for key in args.traits:
        results, top = run_trait(key, marker_summary, markers, args)
        result_tables.append(results)
        top_tables.append(top)
    all_results = add_fdr_and_status(pd.concat(result_tables, ignore_index=True))
    top_overlap = pd.concat(top_tables, ignore_index=True) if top_tables else pd.DataFrame()
    summary = summarize_cross_trait(all_results)
    contrast = dr_vs_t2d_contrast(all_results)
    write_table(args.out_dir / "tables" / "dr_cell_type_marker_enrichment.tsv", all_results)
    write_table(args.out_dir / "tables" / "dr_cell_type_top_fraction_overlap.tsv", top_overlap)
    write_table(args.out_dir / "tables" / "dr_cell_type_cross_trait_summary.tsv", summary)
    write_table(args.out_dir / "tables" / "dr_vs_t2d_cell_type_specificity.tsv", contrast)
    report = render_report(summary, contrast, args)
    (args.out_dir / "reports" / "dr_cell_type_specificity_report.md").write_text(report + "\n", encoding="utf-8")
    manifest = {
        "created_utc": now_utc(),
        "script_path": str(Path(__file__).resolve()),
        "marker_gene_sets": str(args.marker_gene_sets),
        "out_dir": str(args.out_dir),
        "traits": args.traits,
        "n_null": int(args.n_null),
        "degree_bins": int(args.degree_bins),
        "top_fractions": args.top_fractions,
        "seed": int(args.seed),
        "include_low_information": bool(args.include_low_information),
        "include_special_region": bool(args.include_special_region),
        "analysis_role": "exploratory_contextual_screen_not_claim_source",
        "claim_boundary": "single-cell enrichment is contextual support only; DR-specific interpretation requires decomposition/contrast",
    }
    (args.out_dir / "reports" / "dr_cell_type_specificity_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote DR cell-type specificity enrichment to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
