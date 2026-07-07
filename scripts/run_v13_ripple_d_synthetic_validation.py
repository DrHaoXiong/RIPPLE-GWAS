#!/usr/bin/env python
"""Synthetic validation for RIPPLE-D locus-aware distributed module statistics."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.modules import RippleDConfig, ripple_d_module_tests  # noqa: E402
from ripple.modules.anchored import AnchoredModuleLibrary  # noqa: E402

PRIVATE_ROOT = Path("/mnt/d/RIPPLE/RIPPLE_private")
DEFAULT_OUT_DIR = (
    PRIVATE_ROOT / "30_analysis" / "tier4_v13_locus_distributed_module_rescue_v0_1" / "synthetic"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)


def make_base_scores(rng: np.random.Generator, *, n_genes: int = 260) -> pd.DataFrame:
    genes = [f"G{i:04d}" for i in range(n_genes)]
    scores = rng.normal(0.0, 0.55, size=n_genes)
    chroms = np.repeat(np.arange(1, 23), int(np.ceil(n_genes / 22)))[:n_genes]
    starts = np.arange(n_genes) * 1_300_000 + 100_000
    return pd.DataFrame(
        {
            "gene_symbol": genes,
            "assoc_resid_score": scores,
            "chrom": chroms,
            "gene_start": starts,
            "gene_end": starts + 25_000,
            "graph_degree": rng.integers(1, 60, size=n_genes),
            "gene_length": 25_000,
            "n_mapped_snps": rng.integers(5, 80, size=n_genes),
            "local_ld_score": rng.uniform(1.0, 8.0, size=n_genes),
        }
    )


def scenario_scores(name: str, rng: np.random.Generator) -> tuple[pd.DataFrame, set[str]]:
    scores = make_base_scores(rng)
    module = {f"G{i:04d}" for i in range(30)}
    if name == "pure_null":
        return scores, module
    if name == "single_top_locus_artifact":
        module = {f"G{i:04d}" for i in range(5)}
        scores.loc[scores["gene_symbol"].eq("G0000"), "assoc_resid_score"] = 20.0
        return scores, module
    if name == "sparse_top_locus_mixture":
        module = {f"G{i:04d}" for i in range(12)}
        for idx, value in [(0, 10.0), (1, 7.5), (2, 6.0)]:
            scores.loc[scores["gene_symbol"].eq(f"G{idx:04d}"), "assoc_resid_score"] = value
        return scores, module
    if name == "distributed_8_locus_moderate":
        for idx in range(8):
            scores.loc[scores["gene_symbol"].eq(f"G{idx:04d}"), "assoc_resid_score"] = 2.15
        return scores, module
    if name == "distributed_15_locus_weak":
        for idx in range(15):
            scores.loc[scores["gene_symbol"].eq(f"G{idx:04d}"), "assoc_resid_score"] = 1.65
        return scores, module
    if name == "mixed_top_locus_plus_distributed":
        scores.loc[scores["gene_symbol"].eq("G0000"), "assoc_resid_score"] = 9.0
        for idx in range(1, 10):
            scores.loc[scores["gene_symbol"].eq(f"G{idx:04d}"), "assoc_resid_score"] = 1.85
        return scores, module
    raise ValueError(f"Unknown synthetic scenario: {name}")


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    config = RippleDConfig(locus_window_bp=500_000, score_cap=3.0, degree_bins=5, property_bins=3)
    scenarios = [
        "pure_null",
        "single_top_locus_artifact",
        "sparse_top_locus_mixture",
        "distributed_8_locus_moderate",
        "distributed_15_locus_weak",
        "mixed_top_locus_plus_distributed",
    ]
    rows: list[dict[str, object]] = []
    for idx, scenario in enumerate(scenarios):
        rng = np.random.default_rng(args.seed + idx * 1009)
        scores, module = scenario_scores(scenario, rng)
        library = AnchoredModuleLibrary(
            gene_sets={
                "oracle_module": module,
                "background_decoy": {f"G{i:04d}" for i in range(80, 120)},
            },
            module_source={"oracle_module": "synthetic", "background_decoy": "synthetic"},
            annotation_source_type={
                "oracle_module": "internal_support",
                "background_decoy": "internal_support",
            },
            module_category={"oracle_module": scenario, "background_decoy": "negative_control"},
        )
        modules, _, _, summary = ripple_d_module_tests(
            scores,
            library,
            config=config,
            min_present=5,
            n_null=args.n_null,
            seed=args.seed + idx * 1009,
        )
        oracle = modules.loc[modules["module_name"].eq("oracle_module")].iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "module_status": oracle["module_status"],
                "raw_gene_empirical_p": oracle["raw_gene_empirical_p"],
                "locus_robust_empirical_p": oracle["locus_robust_empirical_p"],
                "moderate_locus_burden_empirical_p": oracle["moderate_locus_burden_empirical_p"],
                "leave_top1_locus_empirical_p": oracle["leave_top1_locus_empirical_p"],
                "n_effective_loci": oracle["n_effective_loci"],
                "top1_locus_contribution": oracle["top1_locus_contribution"],
                "top5_locus_contribution": oracle["top5_locus_contribution"],
                "ripple_d_stat": oracle["ripple_d_stat"],
                "n_null": int(args.n_null),
                "seed": int(args.seed + idx * 1009),
                "summary": json.dumps(summary, sort_keys=True),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(args.out_dir / "v13_synthetic_validation.tsv", sep="\t", index=False)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "n_null": int(args.n_null),
        "seed": int(args.seed),
        "output_path": str(args.out_dir / "v13_synthetic_validation.tsv"),
    }
    (args.out_dir / "v13_synthetic_validation.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote synthetic RIPPLE-D validation to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
