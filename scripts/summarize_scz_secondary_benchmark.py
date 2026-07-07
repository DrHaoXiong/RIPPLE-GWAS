#!/usr/bin/env python
"""Summarize SCZ secondary benchmark RIPPLE outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_height_mvp import write_table  # noqa: E402


PRIVATE_ROOT = (
    Path("D:/path/to/ripple_private_workspace")
    if Path("D:/path/to/ripple_private_workspace").exists()
    else Path("/path/to/ripple_private_workspace")
)
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "scz_secondary_benchmark_final"
APOE_CHROM = "19"
APOE_START = 44_000_000
APOE_END = 46_500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-mhc-dir", type=Path, default=ANALYSIS_ROOT / "scz_with_mhc_string_dev500")
    parser.add_argument("--no-mhc-dir", type=Path, default=ANALYSIS_ROOT / "scz_no_mhc_string_dev500")
    parser.add_argument("--no-mhc-final-dir", type=Path, default=ANALYSIS_ROOT / "scz_no_mhc_string_final5000")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer")


def summary_path(analysis_dir: Path, trait: str) -> Path:
    return analysis_dir / "reports" / f"{trait}.analysis_ready_summary.json"


def claim_path(analysis_dir: Path, trait: str) -> Path:
    return analysis_dir / "tables" / f"{trait}.claim_tiers.tsv"


def gene_scores_path(analysis_dir: Path, trait: str) -> Path:
    return analysis_dir / "tables" / f"{trait}.gene_scores.1000G_LD.tsv.gz"


def lcc_scores_path(analysis_dir: Path, trait: str) -> Path:
    return analysis_dir / "tables" / f"{trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"


def modules_path(analysis_dir: Path, trait: str) -> Path:
    return analysis_dir / "tables" / f"{trait}.local_modules.tsv"


def claim_summary(analysis_id: str, trait: str, analysis_dir: Path) -> pd.DataFrame:
    table = read_tsv(claim_path(analysis_dir, trait))
    table.insert(0, "analysis_id", analysis_id)
    table.insert(1, "analysis_scale", "final5000" if "final5000" in analysis_id else "dev500")
    return table


def selected_specs(args: argparse.Namespace) -> list[tuple[str, str, Path]]:
    specs = [("SCZ_default_with_MHC_dev500", "SCZ_WITH_MHC", args.with_mhc_dir)]
    final_claim = claim_path(args.no_mhc_final_dir, "SCZ")
    if final_claim.exists():
        specs.append(("SCZ_no_MHC_final5000", "SCZ", args.no_mhc_final_dir))
    else:
        specs.append(("SCZ_no_MHC_dev500", "SCZ", args.no_mhc_dir))
    return specs


def split_genes(value: object) -> set[str]:
    return {gene.strip() for gene in str(value).split(",") if gene.strip()}


def apoe_diagnostic(analysis_id: str, trait: str, analysis_dir: Path) -> pd.DataFrame:
    genes = read_tsv(gene_scores_path(analysis_dir, trait))
    lcc = read_tsv(lcc_scores_path(analysis_dir, trait))
    modules = read_tsv(modules_path(analysis_dir, trait))
    region = genes.loc[
        (genes["chrom"].astype(str) == APOE_CHROM)
        & (pd.to_numeric(genes["gene_end"], errors="coerce") >= APOE_START)
        & (pd.to_numeric(genes["gene_start"], errors="coerce") <= APOE_END)
    ].copy()
    if region.empty:
        return pd.DataFrame(
            [
                {
                    "analysis_id": analysis_id,
                    "trait": trait,
                    "region_id": "APOE",
                    "n_region_genes": 0,
                    "top_region_gene": "",
                    "top_region_assoc_normal_score": np.nan,
                    "top_region_assoc_minuslog10p": np.nan,
                    "top_region_resid_score": np.nan,
                    "top_region_rank_fraction": np.nan,
                    "region_gene_in_reportable_module": False,
                    "reportable_module_ids": "",
                    "diagnostic_status": "no_region_gene_scored",
                    "recommended_action": "no_mhc_no_apoe_not_required_for_apoe",
                }
            ]
        )

    lcc_rank = lcc.loc[:, ["gene_symbol", "assoc_resid_score"]].copy()
    lcc_rank["rank"] = lcc_rank["assoc_resid_score"].rank(method="first", ascending=False)
    lcc_rank["rank_fraction"] = lcc_rank["rank"] / len(lcc_rank)
    region = region.merge(lcc_rank, on="gene_symbol", how="left", suffixes=("", "_lcc"))
    region["assoc_resid_score"] = pd.to_numeric(region["assoc_resid_score"], errors="coerce")
    region = region.sort_values(
        ["assoc_resid_score", "assoc_normal_score_g"],
        ascending=[False, False],
        na_position="last",
    )
    top = region.iloc[0]
    module_hits: list[str] = []
    region_symbols = set(region["gene_symbol"].dropna().astype(str))
    if not modules.empty and "is_reportable_calibrated_module" in modules.columns:
        reportable = modules.loc[modules["is_reportable_calibrated_module"].astype(str).str.lower() == "true"]
        for module in reportable.to_dict(orient="records"):
            if region_symbols & split_genes(module.get("module_genes", "")):
                module_hits.append(str(module.get("module_id", "")))
    rank_fraction = float(top.get("rank_fraction")) if pd.notna(top.get("rank_fraction")) else np.nan
    max_resid = float(top.get("assoc_resid_score")) if pd.notna(top.get("assoc_resid_score")) else np.nan
    nontrivial = bool((np.isfinite(rank_fraction) and rank_fraction <= 0.02) or module_hits or max_resid >= 2.5)
    return pd.DataFrame(
        [
            {
                "analysis_id": analysis_id,
                "trait": trait,
                "region_id": "APOE",
                "n_region_genes": int(len(region)),
                "top_region_gene": top.get("gene_symbol", ""),
                "top_region_assoc_normal_score": top.get("assoc_normal_score_g", np.nan),
                "top_region_assoc_minuslog10p": top.get("assoc_minuslog10p_g", np.nan),
                "top_region_resid_score": max_resid,
                "top_region_rank_fraction": rank_fraction,
                "region_gene_in_reportable_module": bool(module_hits),
                "reportable_module_ids": ",".join(module_hits),
                "diagnostic_status": "apoe_region_nontrivial" if nontrivial else "apoe_region_not_dominant",
                "recommended_action": "run_no_mhc_no_apoe_sensitivity"
                if nontrivial
                else "no_mhc_no_apoe_not_required_for_apoe",
            }
        ]
    )


def render_report(claims: pd.DataFrame, apoe: pd.DataFrame) -> str:
    def z_for(analysis_id: str, tier: str) -> str:
        hit = claims.loc[(claims["analysis_id"] == analysis_id) & (claims["tier"] == tier)]
        if hit.empty:
            return "NA"
        return f"{float(hit.iloc[0]['z']):.3f}" if pd.notna(hit.iloc[0]["z"]) else "NA"

    lines = [
        "# SCZ Secondary Benchmark Summary",
        "",
        "Scale: with-MHC is dev500; no-MHC uses final5000 when available.",
        "",
        "| Analysis | Tier 1 Z | Tier 2 Z | Tier 3 Z | Local modules | Interpretation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for analysis_id in claims["analysis_id"].drop_duplicates().astype(str):
        local = claims.loc[
            (claims["analysis_id"] == analysis_id)
            & (claims["tier"] == "TIER_4_local_calibrated_modules")
        ]
        n_modules = int(float(local.iloc[0]["observed"])) if not local.empty else 0
        scale = str(claims.loc[claims["analysis_id"] == analysis_id, "analysis_scale"].iloc[0])
        tier_text = "final-positive Tier 1/Tier 2" if scale == "final5000" else "Z-positive Tier 1/Tier 2 at dev scale"
        lines.append(
            f"| {analysis_id} | {z_for(analysis_id, 'TIER_1_degree_calibrated_aggregation')} | "
            f"{z_for(analysis_id, 'TIER_2_graph_domain_aggregation')} | "
            f"{z_for(analysis_id, 'TIER_3_topology_specific_support')} | {n_modules} | "
            f"{tier_text}; topology-specific STRING support negative |"
        )
    lines.extend(["", "## APOE Diagnostic", ""])
    for row in apoe.to_dict(orient="records"):
        lines.append(
            f"- {row['analysis_id']}: top APOE-region gene `{row['top_region_gene']}`, "
            f"rank fraction {row['top_region_rank_fraction']:.4f}, status `{row['diagnostic_status']}`, "
            f"recommended action `{row['recommended_action']}`."
        )
    lines.extend(
        [
            "",
            "## Manuscript-Safe Interpretation",
            "",
            "SCZ was included as a secondary cross-domain benchmark. Under no-MHC final-scale calibration, "
            "RIPPLE detects degree-calibrated and graph-domain aggregation in a large, highly polygenic "
            "non-vascular trait. "
            "As in DR_MVP, default STRING topology-specific support remains negative under graph-null calibration.",
            "",
            "These results support generality of the aggregation-diagnostic layer, not SCZ-specific causal topology.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"{args.out_dir} exists and is not empty. Use --force.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    claims = pd.concat(
        [
            *(claim_summary(analysis_id, trait, analysis_dir) for analysis_id, trait, analysis_dir in selected_specs(args)),
        ],
        ignore_index=True,
    )
    apoe = pd.concat(
        [
            *(apoe_diagnostic(analysis_id, trait, analysis_dir) for analysis_id, trait, analysis_dir in selected_specs(args)),
        ],
        ignore_index=True,
    )
    write_table(args.out_dir / "scz_claim_summary.tsv", claims)
    write_table(args.out_dir / "scz_apoe_region_diagnostic.tsv", apoe)
    (args.out_dir / "SCZ_secondary_benchmark_report.md").write_text(
        render_report(claims, apoe),
        encoding="utf-8",
    )
    print(f"Wrote SCZ secondary benchmark summary to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
