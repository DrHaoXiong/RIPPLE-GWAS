#!/usr/bin/env python
"""Audit whether anchored module signals are driven by top GWAS gene signals.

This diagnostic does not change RIPPLE results. It checks whether supported
anchored modules mostly overlap the strongest residualized gene scores.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
ANCHOR_ROOT = ANALYSIS_ROOT / "tier4_v12_anchored_broad_reactome_go_cross_trait_n500"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "module_signal_overlap_audit_v0_1"

TRAITS = {
    "DR_MVP": {
        "trait": "DR_MVP",
        "score_path": ANALYSIS_ROOT
        / "dr_mvp_string_final5000"
        / "tables"
        / "DR_MVP.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "module_path": ANCHOR_ROOT / "DR_MVP" / "tables" / "DR_MVP.v12_anchored_module_tests.tsv",
    },
    "T2D": {
        "trait": "T2D",
        "score_path": ANALYSIS_ROOT
        / "t2d_analysis_ready"
        / "tables"
        / "T2D.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "module_path": ANCHOR_ROOT / "T2D" / "tables" / "T2D.v12_anchored_module_tests.tsv",
    },
    "BMI_IRN": {
        "trait": "BMI_IRN",
        "score_path": ANALYSIS_ROOT
        / "bmi_irn_analysis_ready"
        / "tables"
        / "BMI_IRN.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "module_path": ANCHOR_ROOT / "BMI_IRN" / "tables" / "BMI_IRN.v12_anchored_module_tests.tsv",
    },
    "SCZ": {
        "trait": "SCZ",
        "score_path": ANALYSIS_ROOT
        / "scz_no_mhc_string_final5000"
        / "tables"
        / "SCZ.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        "module_path": ANCHOR_ROOT / "SCZ_NO_MHC" / "tables" / "SCZ.v12_anchored_module_tests.tsv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-fixed-modules", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def parse_genes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [gene.strip().upper() for gene in str(value).split(",") if gene.strip()]


def load_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path, sep="\t", compression="infer")
    required = {"gene_symbol", "assoc_resid_score", "assoc_normal_score_g", "assoc_p_g"}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    scores = scores.copy()
    scores["gene_symbol"] = scores["gene_symbol"].astype(str).str.upper()
    scores["assoc_resid_score"] = pd.to_numeric(scores["assoc_resid_score"], errors="coerce")
    scores["assoc_normal_score_g"] = pd.to_numeric(scores["assoc_normal_score_g"], errors="coerce")
    scores["assoc_p_g"] = pd.to_numeric(scores["assoc_p_g"], errors="coerce")
    scores = scores.dropna(subset=["assoc_resid_score"]).drop_duplicates("gene_symbol")
    scores = scores.sort_values("assoc_resid_score", ascending=False).reset_index(drop=True)
    scores["resid_rank"] = np.arange(1, len(scores) + 1)
    scores["resid_rank_fraction"] = scores["resid_rank"] / len(scores)
    scores["is_top_100"] = scores["resid_rank"].le(100)
    scores["is_top_1pct"] = scores["resid_rank_fraction"].le(0.01)
    scores["is_top_5pct"] = scores["resid_rank_fraction"].le(0.05)
    return scores


def select_modules(modules: pd.DataFrame, *, top_fixed_modules: int) -> pd.DataFrame:
    modules = modules.copy()
    modules["is_primary_familywise"] = modules["module_status"].eq("anchored_familywise_supported")
    fixed = modules.loc[modules["module_status"].eq("fixed_degree_supported")].copy()
    fixed = fixed.sort_values(
        ["library_familywise_p", "degree_matched_empirical_p", "observed_value"],
        ascending=[True, True, False],
    ).head(top_fixed_modules)
    selected = pd.concat(
        [modules.loc[modules["is_primary_familywise"]], fixed],
        ignore_index=True,
    ).drop_duplicates("module_name")
    return selected.sort_values(
        ["is_primary_familywise", "library_familywise_p", "degree_matched_empirical_p", "observed_value"],
        ascending=[False, True, True, False],
    ).reset_index(drop=True)


def positive_contribution(values: pd.Series, top_n: int) -> float:
    pos = np.clip(values.to_numpy(dtype=float), 0.0, None)
    denom = float(pos.sum())
    if denom <= 0:
        return float("nan")
    return float(np.sort(pos)[::-1][:top_n].sum() / denom)


def audit_module(
    trait: str,
    module: pd.Series,
    scores: pd.DataFrame,
    score_by_gene: dict[str, pd.Series],
) -> dict[str, object]:
    genes = parse_genes(module["present_genes"])
    rows = [score_by_gene[gene] for gene in genes if gene in score_by_gene]
    selected = pd.DataFrame(rows)
    selected = selected.sort_values("assoc_resid_score", ascending=False).reset_index(drop=True)
    if selected.empty:
        return {
            "trait": trait,
            "module_name": module["module_name"],
            "module_status": module["module_status"],
            "n_module_genes_scored": 0,
        }
    top5 = selected.head(5)
    module_mean = float(selected["assoc_resid_score"].mean())
    leave_top1_mean = float(selected.iloc[1:]["assoc_resid_score"].mean()) if len(selected) > 1 else float("nan")
    leave_top5_mean = float(selected.iloc[5:]["assoc_resid_score"].mean()) if len(selected) > 5 else float("nan")
    best = selected.iloc[0]
    return {
        "trait": trait,
        "module_name": module["module_name"],
        "module_category": module.get("module_category", ""),
        "module_source": module.get("module_source", ""),
        "module_status": module["module_status"],
        "library_familywise_p": module.get("library_familywise_p", np.nan),
        "degree_matched_z": module.get("degree_matched_z", np.nan),
        "degree_matched_empirical_p": module.get("degree_matched_empirical_p", np.nan),
        "observed_value": module.get("observed_value", np.nan),
        "n_present": int(module.get("n_present", len(genes))),
        "n_module_genes_scored": int(len(selected)),
        "n_top100_genes": int(selected["is_top_100"].sum()),
        "fraction_top100_genes": float(selected["is_top_100"].mean()),
        "n_top1pct_genes": int(selected["is_top_1pct"].sum()),
        "fraction_top1pct_genes": float(selected["is_top_1pct"].mean()),
        "n_top5pct_genes": int(selected["is_top_5pct"].sum()),
        "fraction_top5pct_genes": float(selected["is_top_5pct"].mean()),
        "best_gene": best["gene_symbol"],
        "best_gene_rank": int(best["resid_rank"]),
        "best_gene_rank_fraction": float(best["resid_rank_fraction"]),
        "best_gene_score": float(best["assoc_resid_score"]),
        "best_gene_p": float(best["assoc_p_g"]),
        "module_mean_score": module_mean,
        "leave_top1_mean_score": leave_top1_mean,
        "leave_top5_mean_score": leave_top5_mean,
        "top1_positive_score_contribution": positive_contribution(selected["assoc_resid_score"], 1),
        "top5_positive_score_contribution": positive_contribution(selected["assoc_resid_score"], 5),
        "top5_genes_by_score": ",".join(top5["gene_symbol"].astype(str)),
        "top5_gene_ranks": ",".join(str(int(rank)) for rank in top5["resid_rank"]),
        "top5_gene_scores": ",".join(f"{score:.4g}" for score in top5["assoc_resid_score"]),
    }


def top_gene_table(trait: str, scores: pd.DataFrame, n: int = 50) -> pd.DataFrame:
    top = scores.head(n).copy()
    top.insert(0, "trait", trait)
    return top.loc[
        :,
        [
            "trait",
            "gene_symbol",
            "resid_rank",
            "resid_rank_fraction",
            "assoc_resid_score",
            "assoc_normal_score_g",
            "assoc_p_g",
        ],
    ]


def summarize_trait(trait: str, audit: pd.DataFrame) -> dict[str, object]:
    familywise = audit.loc[audit["module_status"].eq("anchored_familywise_supported")]
    supported = audit.loc[
        audit["module_status"].isin(["anchored_familywise_supported", "fixed_degree_supported"])
    ]
    if familywise.empty:
        primary = supported.sort_values(
            ["library_familywise_p", "degree_matched_empirical_p"],
            ascending=[True, True],
        ).head(1)
    else:
        primary = familywise.sort_values("library_familywise_p").head(1)
    row = primary.iloc[0].to_dict() if not primary.empty else {}
    return {
        "trait": trait,
        "n_audited_modules": int(len(audit)),
        "n_familywise_modules": int(familywise.shape[0]),
        "median_top1_contribution": float(supported["top1_positive_score_contribution"].median())
        if not supported.empty
        else np.nan,
        "median_top5_contribution": float(supported["top5_positive_score_contribution"].median())
        if not supported.empty
        else np.nan,
        "median_fraction_top1pct": float(supported["fraction_top1pct_genes"].median())
        if not supported.empty
        else np.nan,
        "median_fraction_top5pct": float(supported["fraction_top5pct_genes"].median())
        if not supported.empty
        else np.nan,
        "primary_module_name": row.get("module_name", ""),
        "primary_module_status": row.get("module_status", ""),
        "primary_best_gene": row.get("best_gene", ""),
        "primary_best_gene_rank": row.get("best_gene_rank", ""),
        "primary_top1_contribution": row.get("top1_positive_score_contribution", np.nan),
        "primary_top5_contribution": row.get("top5_positive_score_contribution", np.nan),
        "primary_fraction_top1pct": row.get("fraction_top1pct_genes", np.nan),
        "primary_fraction_top5pct": row.get("fraction_top5pct_genes", np.nan),
    }


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.out_dir} exists and is not empty. Use --force to overwrite.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    tables_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    audit_rows: list[dict[str, object]] = []
    top_gene_rows: list[pd.DataFrame] = []
    for trait, cfg in TRAITS.items():
        scores = load_scores(cfg["score_path"])
        score_by_gene = {row.gene_symbol: row for row in scores.itertuples(index=False)}
        modules = pd.read_csv(cfg["module_path"], sep="\t")
        selected_modules = select_modules(modules, top_fixed_modules=args.top_fixed_modules)
        audit_rows.extend(
            audit_module(trait, module, scores, score_by_gene)
            for _, module in selected_modules.iterrows()
        )
        top_gene_rows.append(top_gene_table(trait, scores))

    audit = pd.DataFrame(audit_rows)
    top_genes = pd.concat(top_gene_rows, ignore_index=True)
    summary = pd.DataFrame([summarize_trait(trait, audit.loc[audit["trait"].eq(trait)]) for trait in TRAITS])
    write_table(tables_dir / "module_signal_overlap_audit.tsv", audit)
    write_table(tables_dir / "module_signal_overlap_summary.tsv", summary)
    write_table(tables_dir / "top50_gene_scores_by_trait.tsv", top_genes)
    manifest = {
        "created_utc": now_utc(),
        "script_path": str(Path(__file__).resolve()),
        "output_dir": str(args.out_dir),
        "traits": {trait: {key: str(value) for key, value in cfg.items()} for trait, cfg in TRAITS.items()},
        "interpretation": (
            "High top1/top5 positive-score contribution or high fractions of top-ranked genes indicate "
            "that an anchored module signal may largely track the strongest GWAS gene signals."
        ),
    }
    (reports_dir / "module_signal_overlap_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote module signal overlap audit to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
