#!/usr/bin/env python
"""Residualize a target RIPPLE gene-score table against covariate score tables."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-score-file", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--covariate-score-files", nargs="+", type=Path, required=True)
    parser.add_argument("--covariate-names", nargs="+", required=True)
    parser.add_argument("--out-score-file", type=Path, required=True)
    parser.add_argument("--out-model-json", type=Path, required=True)
    return parser.parse_args()


def zscore(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Cannot z-score values with nonpositive standard deviation.")
    return (arr - mean) / sd, mean, sd


def read_score(path: Path, name: str) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", compression="infer")
    required = {"gene_symbol", "assoc_resid_score"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    out = table.loc[:, ["gene_symbol", "assoc_resid_score"]].copy()
    out["gene_symbol"] = out["gene_symbol"].astype(str).str.upper()
    out[f"{name}_assoc_resid_score"] = pd.to_numeric(out["assoc_resid_score"], errors="coerce")
    out = out.drop(columns=["assoc_resid_score"]).dropna().drop_duplicates("gene_symbol")
    return out


def main() -> None:
    args = parse_args()
    if len(args.covariate_score_files) != len(args.covariate_names):
        raise ValueError("--covariate-score-files and --covariate-names must have the same length.")

    target = pd.read_csv(args.target_score_file, sep="\t", compression="infer")
    if "gene_symbol" not in target.columns or "assoc_resid_score" not in target.columns:
        raise ValueError("Target score table must contain gene_symbol and assoc_resid_score.")
    target = target.copy()
    target["gene_symbol"] = target["gene_symbol"].astype(str).str.upper()
    target[f"{args.target_name}_assoc_resid_score_raw"] = pd.to_numeric(target["assoc_resid_score"], errors="coerce")

    merged = target.loc[:, ["gene_symbol", f"{args.target_name}_assoc_resid_score_raw"]].copy()
    covariate_paths: dict[str, str] = {}
    for name, path in zip(args.covariate_names, args.covariate_score_files, strict=True):
        covariate_paths[name] = str(path)
        merged = merged.merge(read_score(path, name), on="gene_symbol", how="inner")

    merged = merged.dropna().copy()
    y, target_mean, target_sd = zscore(merged[f"{args.target_name}_assoc_resid_score_raw"].to_numpy(dtype=float))
    covariate_z: list[np.ndarray] = []
    covariate_scaling: dict[str, dict[str, float]] = {}
    for name in args.covariate_names:
        z, mean, sd = zscore(merged[f"{name}_assoc_resid_score"].to_numpy(dtype=float))
        covariate_z.append(z)
        covariate_scaling[name] = {"mean": mean, "sd": sd}
    predictors = np.column_stack(covariate_z)
    design = np.column_stack([np.ones(y.shape[0]), predictors])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    residual = y - fitted
    ss_resid = float(np.sum(residual**2))
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_resid / ss_total if ss_total > 0 else float("nan")
    residual_z, residual_mean, residual_sd = zscore(residual)

    residual_by_gene = pd.DataFrame(
        {
            "gene_symbol": merged["gene_symbol"].to_numpy(),
            "assoc_resid_score_residualized": residual_z,
            "target_z_score_used_for_model": y,
            "target_fitted_z_from_covariates": fitted,
        }
    )
    for name in args.covariate_names:
        residual_by_gene[f"{name}_assoc_resid_score"] = merged[f"{name}_assoc_resid_score"].to_numpy(dtype=float)

    out = target.merge(residual_by_gene, on="gene_symbol", how="inner")
    out["assoc_resid_score_original"] = out["assoc_resid_score"]
    out["assoc_resid_score"] = out["assoc_resid_score_residualized"]
    out["residualization_target"] = args.target_name
    out["residualization_covariates"] = ",".join(args.covariate_names)
    out["residualization_model_r2"] = r2
    out["residualization_intercept"] = float(beta[0])
    for idx, name in enumerate(args.covariate_names, start=1):
        out[f"residualization_beta_{name}"] = float(beta[idx])

    args.out_score_file.parent.mkdir(parents=True, exist_ok=True)
    args.out_model_json.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_score_file, sep="\t", index=False, compression="infer")
    model = {
        "target_name": args.target_name,
        "target_score_file": str(args.target_score_file),
        "covariate_score_files": covariate_paths,
        "out_score_file": str(args.out_score_file),
        "n_aligned_genes": int(len(out)),
        "target_mean": target_mean,
        "target_sd": target_sd,
        "covariate_scaling": covariate_scaling,
        "intercept": float(beta[0]),
        "coefficients": {name: float(beta[idx + 1]) for idx, name in enumerate(args.covariate_names)},
        "model_r2": r2,
        "residual_prestandardization_mean": residual_mean,
        "residual_prestandardization_sd": residual_sd,
        "created_utc": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).resolve()),
    }
    args.out_model_json.write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(model, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
