#!/usr/bin/env python
"""Summarize decision stability from gene-score tail calibration audits."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRIVATE_ROOT = (
    Path("D:/path/to/ripple_private_workspace")
    if Path("D:/path/to/ripple_private_workspace").exists()
    else Path("/path/to/ripple_private_workspace")
)
MANUSCRIPT_ROOT = (
    Path("D:/RIPPLE/RIPPLE_manuscript")
    if Path("D:/RIPPLE/RIPPLE_manuscript").exists()
    else Path("/path/to/ripple_manuscript_workspace")
)
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
REVIEW_ROOT = ANALYSIS_ROOT / "review_driven_revision_v1"
DEFAULT_TAIL_TABLE = REVIEW_ROOT / "gene_score_tail_calibration" / "gene_score_tail_calibration.tsv"
DEFAULT_OUT_DIR = REVIEW_ROOT / "gene_score_tail_decision_stability"
DEFAULT_SUPPLEMENT_DIR = MANUSCRIPT_ROOT / "supplementary_files"
FINAL_CLAIM_TABLE = ANALYSIS_ROOT / "manuscript_ready_v1_2_review" / "final_claim_audit.tsv"

ANALYSIS_DIRS = {
    "DR_MVP": ANALYSIS_ROOT / "dr_mvp_string_final5000",
    "DR_MVP_NO_MHC_NO_APOE": ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000",
    "SCZ": ANALYSIS_ROOT / "scz_no_mhc_string_final5000",
    "HEIGHT_IRN": ANALYSIS_ROOT / "height_irn_analysis_ready",
    "BMI_IRN": ANALYSIS_ROOT / "bmi_irn_analysis_ready",
    "T2D": ANALYSIS_ROOT / "t2d_analysis_ready",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail-table", type=Path, default=DEFAULT_TAIL_TABLE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--supplement-dir", type=Path, default=DEFAULT_SUPPLEMENT_DIR)
    parser.add_argument("--copy-to-supplement", action="store_true", default=True)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def read_table(path: Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", compression="infer", usecols=usecols)


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def score_path(trait: str) -> Path:
    return ANALYSIS_DIRS[trait] / "tables" / f"{trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(values.size, dtype=int)
    ranks[order] = np.arange(1, values.size + 1)
    return ranks


def alternative_rank_detail(trait: str, tail: pd.DataFrame) -> pd.DataFrame:
    path = score_path(trait)
    scores = read_table(path, usecols=["gene_id", "gene_symbol", "assoc_normal_score_g", "assoc_resid_score"])
    scores["gene_id"] = scores["gene_id"].astype(str)
    scores["gene_symbol"] = scores["gene_symbol"].astype(str)
    base_values = pd.to_numeric(scores["assoc_normal_score_g"], errors="coerce").to_numpy(dtype=float)
    base_ranks = rank_desc(base_values)
    gene_to_idx = dict(zip(scores["gene_id"], range(len(scores)), strict=True))
    rows: list[dict[str, Any]] = []
    tail = tail.loc[tail["calibration_status"].astype(str).eq("complete")].copy()
    for alt_col in ["normal_saddlepoint", "normal_satterthwaite", "normal_parametric_sim"]:
        alt_values = base_values.copy()
        changed_indices: list[int] = []
        for item in tail.to_dict(orient="records"):
            idx = gene_to_idx.get(str(item["gene_id"]))
            alt = item.get(alt_col)
            if idx is None or pd.isna(alt):
                continue
            alt_values[idx] = float(alt)
            changed_indices.append(idx)
        alt_ranks = rank_desc(alt_values)
        for idx in sorted(set(changed_indices)):
            original_rank = int(base_ranks[idx])
            alt_rank = int(alt_ranks[idx])
            rows.append(
                {
                    "trait": trait,
                    "gene_id": scores.iloc[idx]["gene_id"],
                    "gene_symbol": scores.iloc[idx]["gene_symbol"],
                    "alternative_method": alt_col.replace("normal_", ""),
                    "original_assoc_normal_score": float(base_values[idx]),
                    "alternative_normal_score": float(alt_values[idx]),
                    "original_rank": original_rank,
                    "alternative_rank": alt_rank,
                    "absolute_rank_displacement": abs(alt_rank - original_rank),
                    "original_top_1pct": original_rank <= max(1, int(np.ceil(0.01 * len(scores)))),
                    "alternative_top_1pct": alt_rank <= max(1, int(np.ceil(0.01 * len(scores)))),
                    "original_top_5pct": original_rank <= max(1, int(np.ceil(0.05 * len(scores)))),
                    "alternative_top_5pct": alt_rank <= max(1, int(np.ceil(0.05 * len(scores)))),
                    "n_lcc_genes": int(len(scores)),
                }
            )
    return pd.DataFrame(rows)


def claim_status_for_trait(final_claims: pd.DataFrame, trait: str) -> dict[str, str]:
    if final_claims.empty:
        return {}
    subset = final_claims.loc[final_claims["trait"].astype(str).eq(trait)].copy()
    out: dict[str, str] = {}
    for _, row in subset.iterrows():
        tier = str(row.get("claim_tier", ""))
        out[f"{tier}_status"] = str(row.get("claim_status", ""))
        out[f"{tier}_z"] = str(row.get("z", ""))
    return out


def summarize_trait(trait: str, tail: pd.DataFrame, detail: pd.DataFrame, final_claims: pd.DataFrame) -> dict[str, Any]:
    complete = tail.loc[tail["calibration_status"].astype(str).eq("complete")].copy()
    max_delta_saddle = pd.to_numeric(complete["abs_delta_normal_liu_vs_saddlepoint"], errors="coerce").max()
    max_delta_sim = pd.to_numeric(complete["abs_delta_normal_liu_vs_sim"], errors="coerce").max()
    detail_trait = detail.loc[detail["trait"].astype(str).eq(trait)].copy()
    max_rank_displacement = pd.to_numeric(detail_trait["absolute_rank_displacement"], errors="coerce").max()
    top1_changed = (
        detail_trait["original_top_1pct"].astype(str).ne(detail_trait["alternative_top_1pct"].astype(str)).sum()
        if not detail_trait.empty
        else 0
    )
    top5_changed = (
        detail_trait["original_top_5pct"].astype(str).ne(detail_trait["alternative_top_5pct"].astype(str)).sum()
        if not detail_trait.empty
        else 0
    )
    risk_count = int(complete["tail_decision_risk"].astype(str).eq("review").sum())
    status = "requires_conservative_tail_language" if risk_count else "low_tail_decision_risk"
    return {
        "trait": trait,
        "n_audited_genes": int(len(tail)),
        "n_complete_tail_audits": int(len(complete)),
        "n_review_risk_genes": risk_count,
        "max_abs_delta_normal_liu_vs_saddlepoint": float(max_delta_saddle) if pd.notna(max_delta_saddle) else np.nan,
        "max_abs_delta_normal_liu_vs_sim": float(max_delta_sim) if pd.notna(max_delta_sim) else np.nan,
        "max_alternative_rank_displacement": int(max_rank_displacement) if pd.notna(max_rank_displacement) else 0,
        "n_top1pct_membership_changes_among_audited_genes": int(top1_changed),
        "n_top5pct_membership_changes_among_audited_genes": int(top5_changed),
        "tier_decision_recompute_performed": False,
        "decision_stability_status": status,
        "manuscript_interpretation": (
            "Tail audit found numerical differences for extreme genes; tier claims are retained under the "
            "primary calibrated pipeline but gene-level extremes require conservative interpretation."
            if risk_count
            else "Tail audit did not identify material instability among audited genes."
        ),
        **claim_status_for_trait(final_claims, trait),
        "source_result_path": str(DEFAULT_OUT_DIR / "gene_score_tail_decision_stability_summary.tsv"),
        "script_path": str(Path(__file__).resolve()),
        "seed": 20260706,
        "timestamp": now_utc(),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tail = read_table(args.tail_table)
    final_claims = read_table(FINAL_CLAIM_TABLE) if FINAL_CLAIM_TABLE.exists() else pd.DataFrame()
    details: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for trait, group in tail.groupby("trait", observed=True):
        trait = str(trait)
        if trait not in ANALYSIS_DIRS or not score_path(trait).exists():
            continue
        detail = alternative_rank_detail(trait, group)
        details.append(detail)
        summaries.append(summarize_trait(trait, group, detail, final_claims))
    detail_all = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    summary_all = pd.DataFrame(summaries)
    summary_path = args.out_dir / "gene_score_tail_decision_stability_summary.tsv"
    detail_path = args.out_dir / "gene_score_tail_rank_displacement.tsv"
    write_table(summary_path, summary_all)
    write_table(detail_path, detail_all)
    manifest = {
        "created_utc": now_utc(),
        "script_path": str(Path(__file__).resolve()),
        "tail_table": str(args.tail_table),
        "summary": str(summary_path),
        "rank_displacement": str(detail_path),
        "scope": "post hoc decision-stability summary from existing high-risk gene tail audit",
    }
    (args.out_dir / "gene_score_tail_decision_stability_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    if args.copy_to_supplement:
        write_table(args.supplement_dir / "gene_score_tail_decision_stability_summary.tsv", summary_all)
        write_table(args.supplement_dir / "gene_score_tail_rank_displacement.tsv", detail_all)
    print(f"Wrote tail decision-stability summary to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
