#!/usr/bin/env python
"""Run actual MAGMA v1.10 external baseline analyses for RIPPLE V1."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ripple.io.annotations import read_magma_gene_loc  # noqa: E402
from ripple.modules.discovery import DEFAULT_DR_GENE_SETS  # noqa: E402


PRIVATE_ROOT = (
    Path("D:/path/to/ripple_private_workspace")
    if Path("D:/path/to/ripple_private_workspace").exists()
    else Path("/path/to/ripple_private_workspace")
)
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
GWAS_QC_ROOT = PRIVATE_ROOT / "20_processed_data" / "gwas_qc"
DEFAULT_MAGMA = PRIVATE_ROOT / "02_environment" / "tools" / "magma_v1.10" / "magma"
DEFAULT_GENE_LOC = (
    PRIVATE_ROOT
    / "10_raw_data"
    / "reference"
    / "genes"
    / "magma_gene_locations"
    / "NCBI37.3"
    / "NCBI37.3.gene.loc"
)
DEFAULT_BFILE = (
    PRIVATE_ROOT
    / "10_raw_data"
    / "reference"
    / "ld"
    / "1000G_phase3_GRCh37"
    / "EUR"
    / "g1000_eur"
    / "g1000_eur"
)
DEFAULT_SYNONYMS = DEFAULT_BFILE.with_suffix(".synonyms")
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "external_baselines" / "magma_v1.10"


@dataclass(frozen=True)
class TraitSpec:
    trait: str
    analysis_id: str
    gwas: Path
    ripple_scores: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magma", type=Path, default=DEFAULT_MAGMA)
    parser.add_argument("--gene-loc", type=Path, default=DEFAULT_GENE_LOC)
    parser.add_argument("--bfile", type=Path, default=DEFAULT_BFILE)
    parser.add_argument("--synonyms", type=Path, default=DEFAULT_SYNONYMS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--analysis-set", choices=["primary", "all"], default="primary")
    parser.add_argument("--trait", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="Only prepare inputs and command manifest.")
    return parser.parse_args()


def default_trait_specs() -> list[TraitSpec]:
    return [
        TraitSpec(
            trait="DR_MVP",
            analysis_id="DR_MVP_default_final5000",
            gwas=GWAS_QC_ROOT / "core_hm3_no_mhc" / "DR_MVP.tsv.gz",
            ripple_scores=ANALYSIS_ROOT
            / "dr_mvp_string_final5000"
            / "tables"
            / "DR_MVP.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        ),
        TraitSpec(
            trait="DR_MVP_NO_MHC_NO_APOE",
            analysis_id="DR_MVP_no_MHC_no_APOE_final5000",
            gwas=GWAS_QC_ROOT / "core_hm3_no_mhc_no_apoe" / "DR_MVP.tsv.gz",
            ripple_scores=ANALYSIS_ROOT
            / "dr_mvp_no_mhc_no_apoe_final5000"
            / "tables"
            / "DR_MVP_NO_MHC_NO_APOE.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        ),
        TraitSpec(
            trait="SCZ",
            analysis_id="SCZ_no_MHC_final5000",
            gwas=GWAS_QC_ROOT / "core_hm3_no_mhc" / "SCZ.tsv.gz",
            ripple_scores=ANALYSIS_ROOT
            / "scz_no_mhc_string_final5000"
            / "tables"
            / "SCZ.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
        ),
    ]


def all_trait_specs() -> list[TraitSpec]:
    specs = default_trait_specs()
    specs.extend(
        [
            TraitSpec(
                trait="HEIGHT_IRN",
                analysis_id="HEIGHT_IRN_analysis_ready",
                gwas=GWAS_QC_ROOT / "core_hm3_no_mhc" / "HEIGHT_IRN.tsv.gz",
                ripple_scores=ANALYSIS_ROOT
                / "height_irn_analysis_ready"
                / "tables"
                / "HEIGHT_IRN.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
            ),
            TraitSpec(
                trait="BMI_IRN",
                analysis_id="BMI_IRN_analysis_ready",
                gwas=GWAS_QC_ROOT / "core_hm3_no_mhc" / "BMI_IRN.tsv.gz",
                ripple_scores=ANALYSIS_ROOT
                / "bmi_irn_analysis_ready"
                / "tables"
                / "BMI_IRN.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
            ),
            TraitSpec(
                trait="T2D",
                analysis_id="T2D_analysis_ready",
                gwas=GWAS_QC_ROOT / "core_hm3_no_mhc" / "T2D.tsv.gz",
                ripple_scores=ANALYSIS_ROOT
                / "t2d_analysis_ready"
                / "tables"
                / "T2D.lcc_gene_scores.1000G_LD.residualized.tsv.gz",
            ),
        ]
    )
    return specs


def select_specs(args: argparse.Namespace) -> list[TraitSpec]:
    specs = all_trait_specs() if args.analysis_set == "all" else default_trait_specs()
    if not args.trait:
        return specs
    requested = set(args.trait)
    selected = [spec for spec in specs if spec.trait in requested or spec.analysis_id in requested]
    missing = sorted(requested - {spec.trait for spec in selected} - {spec.analysis_id for spec in selected})
    if missing:
        raise ValueError(f"Unknown requested traits/analysis IDs: {missing}")
    return selected


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    table.to_csv(path, sep="\t", index=False, compression=compression)


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer")


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def prepare_magma_inputs(spec: TraitSpec, out_dir: Path) -> dict[str, Any]:
    gwas = read_tsv(spec.gwas)
    required = {"snp_id", "chrom", "pos", "p_value"}
    missing = sorted(required - set(gwas.columns))
    if missing:
        raise ValueError(f"{spec.gwas} missing required columns: {missing}")

    input_dir = out_dir / spec.analysis_id / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    pval_path = input_dir / f"{spec.trait}.magma.pval.tsv"
    snp_loc_path = input_dir / f"{spec.trait}.magma.snp_loc.tsv"

    p = pd.to_numeric(gwas["p_value"], errors="coerce")
    valid = gwas.loc[p.notna() & (p > 0) & (p <= 1), ["snp_id", "chrom", "pos", "p_value"]].copy()
    valid = valid.drop_duplicates(subset=["snp_id"], keep="first")
    if "sample_size" in gwas.columns:
        n = pd.to_numeric(gwas.loc[valid.index, "sample_size"], errors="coerce")
    elif {"n_cases", "n_controls"}.issubset(gwas.columns):
        n = pd.to_numeric(gwas.loc[valid.index, "n_cases"], errors="coerce") + pd.to_numeric(
            gwas.loc[valid.index, "n_controls"], errors="coerce"
        )
    else:
        n = pd.Series(np.nan, index=valid.index)
    if n.notna().any():
        valid["N"] = np.rint(n.fillna(n.median())).astype(int).clip(lower=1)
    else:
        raise ValueError(f"No usable sample size columns for {spec.trait}")

    pval = valid.loc[:, ["snp_id", "p_value", "N"]].rename(columns={"snp_id": "SNP", "p_value": "P"})
    snp_loc = valid.loc[:, ["snp_id", "chrom", "pos"]].rename(
        columns={"snp_id": "SNP", "chrom": "CHR", "pos": "BP"}
    )
    write_table(pval_path, pval)
    snp_loc.to_csv(snp_loc_path, sep="\t", index=False, header=False)
    return {
        "n_input_rows": int(len(gwas)),
        "n_valid_p_rows": int(len(valid)),
        "median_n": float(valid["N"].median()),
        "pval_path": str(pval_path),
        "snp_loc_path": str(snp_loc_path),
    }


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def command_strings(commands: list[list[str]]) -> str:
    return "\n".join(" ".join(cmd) for cmd in commands)


def run_magma_trait(
    spec: TraitSpec,
    *,
    args: argparse.Namespace,
    input_report: dict[str, Any],
    out_dir: Path,
    set_annot: Path,
) -> dict[str, Any]:
    trait_dir = out_dir / spec.analysis_id
    magma_prefix = trait_dir / "magma" / spec.trait
    magma_prefix.parent.mkdir(parents=True, exist_ok=True)
    annot_prefix = magma_prefix.with_name(f"{spec.trait}.annot")
    gene_prefix = magma_prefix.with_name(f"{spec.trait}.genes")
    gene_set_prefix = magma_prefix.with_name(f"{spec.trait}.dr_panel_gene_sets")
    bfile_args = [str(args.bfile)]
    if args.synonyms.exists():
        bfile_args.append(f"synonyms={args.synonyms}")

    commands = [
        [
            str(args.magma),
            "--annotate",
            "--snp-loc",
            str(input_report["snp_loc_path"]),
            "--gene-loc",
            str(args.gene_loc),
            "--out",
            str(annot_prefix),
        ],
        [
            str(args.magma),
            "--bfile",
            *bfile_args,
            "--pval",
            str(input_report["pval_path"]),
            "ncol=N",
            "--gene-annot",
            str(annot_prefix) + ".genes.annot",
            "--out",
            str(gene_prefix),
        ],
        [
            str(args.magma),
            "--gene-results",
            str(gene_prefix) + ".genes.raw",
            "--set-annot",
            str(set_annot),
            "--out",
            str(gene_set_prefix),
        ],
    ]
    if not args.skip_run:
        expected = [Path(str(gene_prefix) + ".genes.out"), Path(str(gene_set_prefix) + ".gsa.out")]
        if args.force or any(not path.exists() for path in expected):
            for idx, command in enumerate(commands, start=1):
                run_command(command, cwd=PROJECT_ROOT, log_path=trait_dir / "logs" / f"magma_step{idx}.log")
    return {
        "annot_prefix": str(annot_prefix),
        "gene_prefix": str(gene_prefix),
        "gene_set_prefix": str(gene_set_prefix),
        "commands": command_strings(commands),
    }


def load_gene_symbol_map(gene_loc: Path) -> pd.DataFrame:
    table = read_magma_gene_loc(gene_loc).table.copy()
    table["gene_id"] = table["gene_id"].astype(str)
    return table.loc[:, ["gene_id", "gene_symbol", "chrom", "start", "end"]].drop_duplicates("gene_id")


def write_dr_panel_set_annot(gene_loc: Path, out_dir: Path) -> Path:
    gene_map = load_gene_symbol_map(gene_loc)
    symbol_to_id = dict(zip(gene_map["gene_symbol"].astype(str), gene_map["gene_id"].astype(str), strict=False))
    set_path = out_dir / "reference" / "dr_panel_entrez_gene_sets.tsv"
    rows: list[str] = []
    dropped: list[dict[str, str]] = []
    for name, symbols in DEFAULT_DR_GENE_SETS.items():
        ids = []
        for symbol in sorted(symbols):
            gene_id = symbol_to_id.get(symbol)
            if gene_id is None:
                dropped.append({"gene_set": name, "gene_symbol": symbol, "drop_reason": "not_in_magma_gene_loc"})
            else:
                ids.append(gene_id)
        rows.append("\t".join([name, *ids]))
    set_path.parent.mkdir(parents=True, exist_ok=True)
    set_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    write_table(set_path.with_suffix(".dropped.tsv"), pd.DataFrame(dropped))
    return set_path


def parse_magma_gene_results(spec: TraitSpec, out_dir: Path, gene_loc: Path) -> pd.DataFrame:
    path = out_dir / spec.analysis_id / "magma" / f"{spec.trait}.genes.genes.out"
    raw = pd.read_csv(path, sep=r"\s+", comment="#", dtype={"GENE": str})
    gene_map = load_gene_symbol_map(gene_loc)
    out = raw.merge(gene_map, left_on="GENE", right_on="gene_id", how="left")
    out.insert(0, "trait", spec.trait)
    out.insert(1, "analysis_id", spec.analysis_id)
    rename = {
        "GENE": "magma_gene_id",
        "NSNPS": "magma_n_snps",
        "ZSTAT": "magma_z",
        "P": "magma_p",
    }
    out = out.rename(columns=rename)
    return out


def parse_magma_gene_set_results(spec: TraitSpec, out_dir: Path) -> pd.DataFrame:
    path = out_dir / spec.analysis_id / "magma" / f"{spec.trait}.dr_panel_gene_sets.gsa.out"
    raw = pd.read_csv(path, sep=r"\s+", comment="#")
    raw.insert(0, "trait", spec.trait)
    raw.insert(1, "analysis_id", spec.analysis_id)
    raw.insert(2, "gene_set_scope", "DR_panel_internal")
    return raw


def compare_to_ripple(spec: TraitSpec, magma_genes: pd.DataFrame) -> pd.DataFrame:
    ripple = read_tsv(spec.ripple_scores)
    joined = ripple.merge(
        magma_genes.loc[:, ["gene_symbol", "magma_z", "magma_p"]],
        on="gene_symbol",
        how="inner",
    )
    rows: list[dict[str, Any]] = []
    if joined.empty:
        return pd.DataFrame()
    rho, p_value = spearmanr(joined["assoc_resid_score"], -np.log10(joined["magma_p"].clip(lower=1e-300)))
    rows.append(
        {
            "trait": spec.trait,
            "analysis_id": spec.analysis_id,
            "baseline_tool": "MAGMA",
            "comparison_name": "spearman_ripple_resid_vs_magma_minuslog10p",
            "observed_value": float(rho),
            "p_value": float(p_value),
            "n_genes": int(len(joined)),
        }
    )
    for fraction in [0.01, 0.02, 0.05, 0.10]:
        k = max(1, int(np.ceil(len(joined) * fraction)))
        ripple_top = set(joined.nlargest(k, "assoc_resid_score")["gene_symbol"].astype(str))
        magma_top = set(joined.nsmallest(k, "magma_p")["gene_symbol"].astype(str))
        rows.append(
            {
                "trait": spec.trait,
                "analysis_id": spec.analysis_id,
                "baseline_tool": "MAGMA",
                "comparison_name": f"top_{fraction:g}_gene_overlap",
                "observed_value": len(ripple_top & magma_top) / k,
                "p_value": "",
                "n_genes": int(len(joined)),
                "k": k,
                "overlap_count": len(ripple_top & magma_top),
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    manifests: list[dict[str, Any]],
    gene_tables: list[pd.DataFrame],
    gene_set_tables: list[pd.DataFrame],
    comparison_tables: list[pd.DataFrame],
) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "baseline_tool": "MAGMA",
        "baseline_version": "v1.10 linux static",
        "n_traits": len(manifests),
        "traits": [item["trait"] for item in manifests],
        "n_gene_rows": int(sum(len(table) for table in gene_tables)),
        "n_gene_set_rows": int(sum(len(table) for table in gene_set_tables)),
        "n_comparison_rows": int(sum(len(table) for table in comparison_tables)),
        "manifests": manifests,
    }


def main() -> None:
    args = parse_args()
    require_path(args.magma, "MAGMA binary")
    require_path(args.gene_loc, "MAGMA gene location file")
    require_path(args.bfile.with_suffix(".bed"), "MAGMA 1000G BED")
    require_path(args.bfile.with_suffix(".bim"), "MAGMA 1000G BIM")
    require_path(args.bfile.with_suffix(".fam"), "MAGMA 1000G FAM")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = select_specs(args)
    set_annot = write_dr_panel_set_annot(args.gene_loc, out_dir)

    manifests: list[dict[str, Any]] = []
    gene_tables: list[pd.DataFrame] = []
    gene_set_tables: list[pd.DataFrame] = []
    comparison_tables: list[pd.DataFrame] = []
    for spec in specs:
        require_path(spec.gwas, f"{spec.trait} GWAS")
        require_path(spec.ripple_scores, f"{spec.trait} RIPPLE LCC score table")
        input_report = prepare_magma_inputs(spec, out_dir)
        run_report = run_magma_trait(spec, args=args, input_report=input_report, out_dir=out_dir, set_annot=set_annot)
        manifest = {
            "trait": spec.trait,
            "analysis_id": spec.analysis_id,
            "created_utc": datetime.now(UTC).isoformat(),
            "gwas": str(spec.gwas),
            "ripple_scores": str(spec.ripple_scores),
            "set_annot": str(set_annot),
            **input_report,
            **run_report,
            "run_status": "prepared_only" if args.skip_run else "complete",
        }
        manifests.append(manifest)
        if not args.skip_run:
            gene_table = parse_magma_gene_results(spec, out_dir, args.gene_loc)
            gene_set_table = parse_magma_gene_set_results(spec, out_dir)
            comparison_table = compare_to_ripple(spec, gene_table)
            gene_tables.append(gene_table)
            gene_set_tables.append(gene_set_table)
            comparison_tables.append(comparison_table)
            write_table(out_dir / spec.analysis_id / "tables" / f"{spec.trait}.magma_gene_results.tsv.gz", gene_table)
            write_table(
                out_dir / spec.analysis_id / "tables" / f"{spec.trait}.magma_dr_panel_gene_sets.tsv",
                gene_set_table,
            )
            write_table(
                out_dir / spec.analysis_id / "tables" / f"{spec.trait}.magma_ripple_comparison.tsv",
                comparison_table,
            )
        trait_manifest_path = out_dir / spec.analysis_id / "reports" / f"{spec.trait}.magma_manifest.json"
        trait_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        trait_manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if gene_tables:
        write_table(out_dir / "magma_gene_results.all_traits.tsv.gz", pd.concat(gene_tables, ignore_index=True))
    if gene_set_tables:
        write_table(out_dir / "magma_dr_panel_gene_sets.all_traits.tsv", pd.concat(gene_set_tables, ignore_index=True))
    if comparison_tables:
        write_table(out_dir / "magma_ripple_comparison.all_traits.tsv", pd.concat(comparison_tables, ignore_index=True))
    summary = build_summary(manifests, gene_tables, gene_set_tables, comparison_tables)
    (out_dir / "magma_baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote MAGMA baseline outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
