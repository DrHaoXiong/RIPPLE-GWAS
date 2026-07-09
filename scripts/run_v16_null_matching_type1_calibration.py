#!/usr/bin/env python
"""Type I calibration for RIPPLE-D V1.6 null-matching configurations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.modules import (  # noqa: E402
    RippleDConfig,
    add_ripple_d_v16_claim_readiness,
    external_locus_audit_table,
    ripple_d_module_tests,
)
from ripple.modules.anchored import AnchoredModuleLibrary  # noqa: E402

PRIVATE_ROOT = Path("/mnt/d/RIPPLE/RIPPLE_private")
DEFAULT_OUT_DIR = (
    PRIVATE_ROOT
    / "30_analysis"
    / "tier4_v16_claim_readiness_hardening_v0_1"
    / "null_matching_type1_calibration_v0_1"
)


@dataclass(frozen=True)
class Type1Config:
    degree_bins: int
    property_bins: int
    annotation_matching_enabled: bool
    n_loci: int
    genes_per_locus: int
    n_modules: int
    module_size: int
    n_null: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-outer", type=int, default=100)
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--n-loci", type=int, default=520)
    parser.add_argument("--genes-per-locus", type=int, default=2)
    parser.add_argument("--n-modules", type=int, default=80)
    parser.add_argument("--module-size", type=int, default=24)
    parser.add_argument("--degree-bins", type=int, default=5)
    parser.add_argument("--property-bins", type=int, default=2)
    parser.add_argument("--annotation-matching", choices=["on", "off"], default="off")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def ensure_out_dir(path: Path, *, force: bool) -> None:
    if path.exists() and any(path.iterdir()) and not force:
        raise FileExistsError(f"{path} exists and is not empty. Use --force to overwrite.")
    path.mkdir(parents=True, exist_ok=True)
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "reports").mkdir(parents=True, exist_ok=True)


def make_null_scores(config: Type1Config, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for locus_idx in range(config.n_loci):
        chrom = (locus_idx % 22) + 1
        rank_on_chrom = locus_idx // 22
        locus_start = rank_on_chrom * 1_650_000 + 100_000
        for gene_idx in range(config.genes_per_locus):
            start = locus_start + gene_idx * 40_000
            rows.append(
                {
                    "gene_symbol": f"L{locus_idx:04d}_G{gene_idx}",
                    "assoc_resid_score": float(rng.normal(0.0, 0.55)),
                    "chrom": chrom,
                    "gene_start": start,
                    "gene_end": start + 20_000,
                    "graph_degree": int(rng.integers(1, 80)),
                    "gene_length": int(rng.integers(5_000, 120_000)),
                    "n_mapped_snps": int(rng.integers(5, 120)),
                    "local_ld_score": float(rng.uniform(0.5, 8.0)),
                    "synthetic_ld_block_id": f"SYNLD_{locus_idx:04d}",
                }
            )
    return pd.DataFrame(rows)


def make_random_library(config: Type1Config, rng: np.random.Generator) -> AnchoredModuleLibrary:
    gene_sets: dict[str, set[str]] = {}
    module_source: dict[str, str] = {}
    annotation_source_type: dict[str, str] = {}
    module_category: dict[str, str] = {}
    all_loci = np.arange(config.n_loci)
    for module_idx in range(config.n_modules):
        size = min(config.module_size, config.n_loci)
        loci = rng.choice(all_loci, size=size, replace=False)
        genes = {f"L{int(locus_idx):04d}_G0" for locus_idx in loci}
        name = f"null_random_module_{module_idx:04d}"
        gene_sets[name] = genes
        module_source[name] = "synthetic_null_library"
        annotation_source_type[name] = "independent_external"
        module_category[name] = "synthetic_type1_null"
    return AnchoredModuleLibrary(
        gene_sets=gene_sets,
        module_source=module_source,
        annotation_source_type=annotation_source_type,
        module_category=module_category,
    )


def run_outer_replicate(rep_idx: int, config: Type1Config) -> tuple[dict[str, object], pd.DataFrame]:
    seed = config.seed + rep_idx * 10007
    rng = np.random.default_rng(seed)
    scores = make_null_scores(config, rng)
    library = make_random_library(config, rng)
    ripple_config = RippleDConfig(
        locus_id_column="synthetic_ld_block_id",
        locus_definition_name="synthetic_external_ld_blocks",
        degree_bins=config.degree_bins,
        property_bins=config.property_bins,
        annotation_matching_enabled=config.annotation_matching_enabled,
    )
    modules, locus_audit, _, _ = ripple_d_module_tests(
        scores,
        library,
        config=ripple_config,
        min_present=5,
        n_null=config.n_null,
        seed=seed + 17,
        return_null_details=False,
    )
    external_audit = external_locus_audit_table(
        scores.rename(columns={"synthetic_ld_block_id": "synthetic_ld_block_id"}),
        locus_id_column="synthetic_ld_block_id",
        locus_source="synthetic",
        locus_source_version="v16_type1",
        genome_build="synthetic",
        ancestry="synthetic",
        construction_script=Path(__file__).name,
    )
    claim = add_ripple_d_v16_claim_readiness(
        modules,
        ripple_config,
        locus_audit=locus_audit,
        external_locus_audit=external_audit,
    )
    statuses = claim["v16_claim_status"].astype(str)
    statuses_v16b = claim.get("v16b_claim_status", claim["v16_claim_status"]).astype(str)
    row = {
        "replicate": rep_idx,
        "seed": seed,
        "n_modules": int(len(claim)),
        "n_high_confidence": int(statuses.eq("high_confidence_diagnostic_candidate").sum()),
        "n_manuscript_ready": int(statuses.eq("manuscript_ready_distributed_candidate").sum()),
        "n_exploratory": int(statuses.eq("exploratory_locus_distributed_candidate").sum()),
        "v16b_n_high_confidence": int(statuses_v16b.eq("high_confidence_diagnostic_candidate").sum()),
        "v16b_n_manuscript_ready": int(statuses_v16b.eq("manuscript_ready_distributed_candidate").sum()),
        "v16b_n_exploratory": int(statuses_v16b.eq("exploratory_locus_distributed_candidate").sum()),
        "n_multi_strong": int(statuses.eq("multi_strong_locus_pathway_overlap").sum()),
        "n_nonnegative": int(statuses.ne("negative").sum() - statuses.eq("not_tested").sum()),
        "any_high_confidence": bool(statuses.eq("high_confidence_diagnostic_candidate").any()),
        "any_manuscript_ready": bool(statuses.eq("manuscript_ready_distributed_candidate").any()),
        "any_exploratory": bool(statuses.eq("exploratory_locus_distributed_candidate").any()),
        "v16b_any_high_confidence": bool(statuses_v16b.eq("high_confidence_diagnostic_candidate").any()),
        "v16b_any_manuscript_ready": bool(statuses_v16b.eq("manuscript_ready_distributed_candidate").any()),
        "v16b_any_exploratory": bool(statuses_v16b.eq("exploratory_locus_distributed_candidate").any()),
        "null_quality_pass_n": int(claim["null_quality_pass"].astype(bool).sum()),
        "null_quality_balanced_pass_n": int(claim["null_quality_balanced_pass"].astype(bool).sum()),
        "multiplicity_pass_n": int(claim["multiplicity_pass"].astype(bool).sum()),
        "top_tail_pass_n": int(claim["top_tail_pass"].astype(bool).sum()),
        "leave_topk_pass_n": int(claim["leave_topk_pass"].astype(bool).sum()),
        "min_ripple_d_q_full_library": float(pd.to_numeric(claim["ripple_d_q_full_library"], errors="coerce").min()),
        "min_module_specific_rank_q_full_library": float(
            pd.to_numeric(claim["module_specific_rank_q_full_library"], errors="coerce").min()
        ),
    }
    keep = claim.loc[
        statuses.isin(
            {
                "high_confidence_diagnostic_candidate",
                "manuscript_ready_distributed_candidate",
                "exploratory_locus_distributed_candidate",
                "multi_strong_locus_pathway_overlap",
            }
        )
    ].copy()
    keep.insert(0, "replicate", rep_idx)
    keep.insert(1, "replicate_seed", seed)
    return row, keep


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def summarize_type1(rows: pd.DataFrame, config: Type1Config) -> pd.DataFrame:
    metrics = [
        ("any_high_confidence", "replicate_level_high_confidence_fpr"),
        ("any_manuscript_ready", "replicate_level_manuscript_ready_fpr"),
        ("any_exploratory", "replicate_level_exploratory_fpr"),
        ("v16b_any_high_confidence", "v16b_replicate_level_high_confidence_fpr"),
        ("v16b_any_manuscript_ready", "v16b_replicate_level_manuscript_ready_fpr"),
        ("v16b_any_exploratory", "v16b_replicate_level_exploratory_fpr"),
    ]
    out: list[dict[str, object]] = []
    n = len(rows)
    for column, label in metrics:
        k = int(rows[column].astype(bool).sum())
        lo, hi = wilson_ci(k, n)
        fpr = k / n if n else float("nan")
        out.append(
            {
                "config_name": f"{config.degree_bins}x{config.property_bins}_ann_{'on' if config.annotation_matching_enabled else 'off'}",
                "metric": label,
                "false_positive_count": k,
                "n_outer": n,
                "fpr": fpr,
                "binomial_95ci_low": lo,
                "binomial_95ci_high": hi,
                "mc_se": math.sqrt(fpr * (1 - fpr) / n) if n else float("nan"),
                **asdict(config),
            }
        )
    out.append(
        {
            "config_name": f"{config.degree_bins}x{config.property_bins}_ann_{'on' if config.annotation_matching_enabled else 'off'}",
            "metric": "module_level_high_confidence_rate",
            "false_positive_count": int(rows["n_high_confidence"].sum()),
            "n_outer": n,
            "fpr": float(rows["n_high_confidence"].sum() / max(1, rows["n_modules"].sum())),
            "binomial_95ci_low": float("nan"),
            "binomial_95ci_high": float("nan"),
            "mc_se": float("nan"),
            **asdict(config),
        }
    )
    out.append(
        {
            "config_name": f"{config.degree_bins}x{config.property_bins}_ann_{'on' if config.annotation_matching_enabled else 'off'}",
            "metric": "v16b_module_level_high_confidence_rate",
            "false_positive_count": int(rows["v16b_n_high_confidence"].sum()),
            "n_outer": n,
            "fpr": float(rows["v16b_n_high_confidence"].sum() / max(1, rows["n_modules"].sum())),
            "binomial_95ci_low": float("nan"),
            "binomial_95ci_high": float("nan"),
            "mc_se": float("nan"),
            **asdict(config),
        }
    )
    return pd.DataFrame(out)


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir, force=args.force)
    config = Type1Config(
        degree_bins=args.degree_bins,
        property_bins=args.property_bins,
        annotation_matching_enabled=args.annotation_matching == "on",
        n_loci=args.n_loci,
        genes_per_locus=args.genes_per_locus,
        n_modules=args.n_modules,
        module_size=args.module_size,
        n_null=args.n_null,
        seed=args.seed,
    )
    rows: list[dict[str, object]] = []
    candidates: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=max(1, args.n_jobs)) as executor:
        futures = {executor.submit(run_outer_replicate, idx, config): idx for idx in range(args.n_outer)}
        for future in as_completed(futures):
            row, keep = future.result()
            rows.append(row)
            if not keep.empty:
                candidates.append(keep)
            print(
                f"[{datetime.now(UTC).isoformat()}] replicate {row['replicate']} "
                f"high_conf={row['n_high_confidence']} exploratory={row['n_exploratory']}",
                flush=True,
            )
    replicate_table = pd.DataFrame(rows).sort_values("replicate").reset_index(drop=True)
    candidate_table = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    summary = summarize_type1(replicate_table, config)
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    replicate_table.to_csv(tables_dir / "v16_null_matching_type1_replicates.tsv", sep="\t", index=False)
    candidate_table.to_csv(tables_dir / "v16_null_matching_type1_nonnegative_modules.tsv", sep="\t", index=False)
    summary.to_csv(tables_dir / "v16_null_matching_type1_summary.tsv", sep="\t", index=False)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "out_dir": str(args.out_dir),
        "n_jobs": int(args.n_jobs),
        **asdict(config),
    }
    (reports_dir / "v16_null_matching_type1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = "\n".join(
        [
            "# RIPPLE-D V1.6 null-matching Type I calibration",
            "",
            f"Configuration: degree_bins={config.degree_bins}, property_bins={config.property_bins}, "
            f"annotation_matching={config.annotation_matching_enabled}",
            "",
            "## Summary",
            "",
            summary.to_string(index=False),
            "",
            "## Replicate Count Summary",
            "",
            replicate_table.describe(include="all").to_string(),
            "",
        ]
    )
    (reports_dir / "v16_null_matching_type1_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
