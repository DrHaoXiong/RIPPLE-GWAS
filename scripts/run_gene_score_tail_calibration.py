#!/usr/bin/env python
"""Compare Liu gene-level P values against stronger tail-calibration checks."""

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

from ripple.signals.unsigned import (  # noqa: E402
    default_weight_matrix,
    mixture_eigenvalues,
    normal_score_from_p_value,
    quadratic_p_value,
)

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
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "review_driven_revision_v1" / "gene_score_tail_calibration"
DEFAULT_SUPPLEMENT_DIR = MANUSCRIPT_ROOT / "supplementary_files"


DEFAULT_ANALYSES = {
    "DR_MVP": ANALYSIS_ROOT / "dr_mvp_string_final5000",
    "DR_MVP_NO_MHC_NO_APOE": ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000",
    "SCZ": ANALYSIS_ROOT / "scz_no_mhc_string_final5000",
    "HEIGHT_IRN": ANALYSIS_ROOT / "height_irn_analysis_ready",
    "BMI_IRN": ANALYSIS_ROOT / "bmi_irn_analysis_ready",
    "T2D": ANALYSIS_ROOT / "t2d_analysis_ready",
}

MAPPING_FALLBACKS = {
    "HEIGHT_IRN": ANALYSIS_ROOT / "height_irn_mvp" / "tables" / "HEIGHT_IRN.gene_body_mapping.tsv.gz",
    "BMI_IRN": ANALYSIS_ROOT / "bmi_irn_mvp" / "tables" / "BMI_IRN.gene_body_mapping.tsv.gz",
    "T2D": ANALYSIS_ROOT / "t2d_mvp" / "tables" / "T2D.gene_body_mapping.tsv.gz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--supplement-dir", type=Path, default=DEFAULT_SUPPLEMENT_DIR)
    parser.add_argument("--analysis", action="append", default=[])
    parser.add_argument("--max-genes-per-category", type=int, default=5)
    parser.add_argument("--n-sim", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--copy-to-supplement", action="store_true", default=True)
    return parser.parse_args()


def read_tsv(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", compression="infer", nrows=nrows)


def gene_scores_path(analysis_dir: Path, trait: str) -> Path:
    return analysis_dir / "tables" / f"{trait}.gene_scores.1000G_LD.tsv.gz"


def lcc_scores_path(analysis_dir: Path, trait: str) -> Path:
    return analysis_dir / "tables" / f"{trait}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"


def mapping_path(analysis_dir: Path, trait: str) -> Path:
    direct = analysis_dir / "tables" / f"{trait}.gene_body_mapping.tsv.gz"
    if direct.exists():
        return direct
    summaries = sorted((analysis_dir / "reports").glob("*.analysis_ready_summary.json"))
    for summary_path in summaries:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            mapping_value = str(summary.get("mapping", "") or "")
            if not mapping_value:
                continue
            mapped = Path(mapping_value)
            if mapped.is_file():
                return mapped
        except Exception:
            pass
    fallback = MAPPING_FALLBACKS.get(trait)
    if fallback is not None:
        return fallback
    return direct


def choose_genes(gene_scores: pd.DataFrame, lcc_scores: pd.DataFrame, *, n: int) -> pd.DataFrame:
    work = gene_scores.merge(
        lcc_scores.loc[:, ["gene_id", "gene_symbol", "assoc_resid_score"]],
        on=["gene_id", "gene_symbol"],
        how="left",
    )
    candidates: list[pd.DataFrame] = []
    selectors = {
        "top_residualized_score": work.dropna(subset=["assoc_resid_score"]).nlargest(n, "assoc_resid_score"),
        "top_quadratic_statistic": work.nlargest(n, "quadratic_statistic"),
        "high_mapped_snp_count": work.nlargest(n, "n_mapped_snps"),
        "high_local_ld_score": work.nlargest(n, "local_ld_score"),
        "low_m_eff": work.nsmallest(n, "m_eff"),
        "one_snp_gene": work.loc[work["is_one_snp_gene"].astype(bool)].head(n),
        "p_value_clipped": work.loc[work["is_p_clipped"].astype(bool)].head(n),
    }
    for category, table in selectors.items():
        if table.empty:
            continue
        tmp = table.copy()
        tmp["selection_category"] = category
        candidates.append(tmp)
    if not candidates:
        return pd.DataFrame()
    out = pd.concat(candidates, ignore_index=True)
    out = out.drop_duplicates(["gene_id", "gene_symbol", "selection_category"])
    out = out.drop_duplicates(["gene_id", "gene_symbol"], keep="first")
    return out.reset_index(drop=True)


def load_ld(path: str | Path) -> tuple[np.ndarray, tuple[str, ...]]:
    with np.load(path, allow_pickle=True) as data:
        ld = np.asarray(data["ld"], dtype=float)
        snps = tuple(str(value) for value in data["snp_ids"])
    return ld, snps


def weights_for_gene(mapping: pd.DataFrame, gene_id: str, snps: tuple[str, ...]) -> tuple[np.ndarray, int]:
    subset = mapping.loc[mapping["gene_id"].astype(str).eq(str(gene_id)), ["snp_id", "weight"]].copy()
    if subset.empty:
        raise ValueError("no mapping rows for gene")
    by_snp = subset.drop_duplicates("snp_id").set_index("snp_id")["weight"].astype(float).to_dict()
    missing = [snp for snp in snps if snp not in by_snp]
    return np.asarray([by_snp.get(snp, 0.0) for snp in snps], dtype=float), len(missing)


def simulation_p_value(
    observed: float,
    lambdas: np.ndarray,
    *,
    n_sim: int,
    rng: np.random.Generator,
) -> float:
    draws = rng.chisquare(df=1.0, size=(int(n_sim), len(lambdas))) @ lambdas
    return float((1 + np.count_nonzero(draws >= observed)) / (1 + len(draws)))


def normal(p_value: float) -> float:
    return normal_score_from_p_value(float(p_value), epsilon=1e-300)[0]


def calibrate_gene(row: pd.Series, mapping: pd.DataFrame, *, n_sim: int, rng: np.random.Generator) -> dict[str, Any]:
    base = {
        "gene_id": str(row["gene_id"]),
        "gene_symbol": str(row["gene_symbol"]),
        "selection_category": str(row["selection_category"]),
        "n_mapped_snps": int(row["n_mapped_snps"]),
        "m_eff": float(row["m_eff"]),
        "local_ld_score": float(row["local_ld_score"]),
        "ld_status": str(row["ld_status"]),
        "quadratic_statistic": float(row["quadratic_statistic"]),
        "pipeline_assoc_p_g": float(row["assoc_p_g"]),
        "pipeline_assoc_normal_score_g": float(row["assoc_normal_score_g"]),
        "assoc_resid_score": float(row["assoc_resid_score"]) if pd.notna(row.get("assoc_resid_score")) else np.nan,
        "ld_cache_path": str(row["ld_cache_path"]),
    }
    try:
        ld, snps = load_ld(row["ld_cache_path"])
        weights, n_ld_snps_missing_mapping_weight = weights_for_gene(mapping, str(row["gene_id"]), snps)
        lambdas = mixture_eigenvalues(ld, default_weight_matrix(weights), shrinkage=float(row["ld_shrinkage"]))
        p_liu = quadratic_p_value(float(row["quadratic_statistic"]), lambdas, method="liu")
        p_saddle = quadratic_p_value(float(row["quadratic_statistic"]), lambdas, method="saddlepoint")
        p_satt = quadratic_p_value(float(row["quadratic_statistic"]), lambdas, method="satterthwaite")
        p_sim = simulation_p_value(float(row["quadratic_statistic"]), lambdas, n_sim=n_sim, rng=rng)
        z_liu = normal(p_liu)
        z_saddle = normal(p_saddle)
        z_satt = normal(p_satt)
        z_sim = normal(p_sim)
        return {
            **base,
            "calibration_status": "complete",
            "n_ld_snps_missing_mapping_weight": int(n_ld_snps_missing_mapping_weight),
            "n_mixture_eigenvalues": int(len(lambdas)),
            "p_liu": p_liu,
            "p_saddlepoint": p_saddle,
            "p_satterthwaite": p_satt,
            "p_parametric_sim": p_sim,
            "normal_liu": z_liu,
            "normal_saddlepoint": z_saddle,
            "normal_satterthwaite": z_satt,
            "normal_parametric_sim": z_sim,
            "abs_delta_normal_liu_vs_saddlepoint": abs(z_liu - z_saddle),
            "abs_delta_normal_liu_vs_sim": abs(z_liu - z_sim),
            "tail_decision_risk": "review" if max(abs(z_liu - z_saddle), abs(z_liu - z_sim)) >= 0.25 else "low",
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "calibration_status": "failed",
            "n_ld_snps_missing_mapping_weight": np.nan,
            "n_mixture_eigenvalues": np.nan,
            "p_liu": np.nan,
            "p_saddlepoint": np.nan,
            "p_satterthwaite": np.nan,
            "p_parametric_sim": np.nan,
            "normal_liu": np.nan,
            "normal_saddlepoint": np.nan,
            "normal_satterthwaite": np.nan,
            "normal_parametric_sim": np.nan,
            "abs_delta_normal_liu_vs_saddlepoint": np.nan,
            "abs_delta_normal_liu_vs_sim": np.nan,
            "tail_decision_risk": "not_evaluable",
            "error": str(exc),
        }


def selected_analyses(requested: list[str]) -> dict[str, Path]:
    if not requested:
        return DEFAULT_ANALYSES
    out = {}
    for name in requested:
        if name not in DEFAULT_ANALYSES:
            raise ValueError(f"Unknown analysis {name}; available: {sorted(DEFAULT_ANALYSES)}")
        out[name] = DEFAULT_ANALYSES[name]
    return out


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []
    for trait, analysis_dir in selected_analyses(args.analysis).items():
        if not gene_scores_path(analysis_dir, trait).exists() or not mapping_path(analysis_dir, trait).exists():
            continue
        gene_scores = read_tsv(gene_scores_path(analysis_dir, trait))
        lcc_scores = read_tsv(lcc_scores_path(analysis_dir, trait))
        mapping = read_tsv(mapping_path(analysis_dir, trait))
        selected = choose_genes(gene_scores, lcc_scores, n=args.max_genes_per_category)
        for _, row in selected.iterrows():
            out = calibrate_gene(row, mapping, n_sim=args.n_sim, rng=rng)
            out["trait"] = trait
            out["analysis_dir"] = str(analysis_dir)
            out["n_parametric_sim"] = int(args.n_sim)
            out["script_path"] = str(Path(__file__).resolve())
            out["seed"] = int(args.seed)
            out["timestamp"] = datetime.now(UTC).isoformat()
            rows.append(out)
    table = pd.DataFrame(rows)
    out_path = args.out_dir / "gene_score_tail_calibration.tsv"
    write_table(out_path, table)
    if args.copy_to_supplement:
        write_table(args.supplement_dir / "gene_score_tail_calibration.tsv", table)
    print(f"Wrote gene-score tail calibration to {out_path}", flush=True)


if __name__ == "__main__":
    main()
