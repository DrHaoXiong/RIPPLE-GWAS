#!/usr/bin/env python
"""Run PascalX/Pascal-method external baseline analyses for RIPPLE V1."""

from __future__ import annotations

import argparse
import gzip
import json
import os
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

from ripple.modules.discovery import DEFAULT_DR_GENE_SETS  # noqa: E402


PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
ANALYSIS_ROOT = PRIVATE_ROOT / "30_analysis"
GWAS_QC_ROOT = PRIVATE_ROOT / "20_processed_data" / "gwas_qc"
PASCALX_ROOT = PRIVATE_ROOT / "02_environment" / "tools" / "PascalX"
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
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "external_baselines" / "pascalx"
DEFAULT_REFPANEL_PREFIX = DEFAULT_OUT_DIR / "refpanel_1000G_EUR_GRCh37" / "EUR.1KG.GRCh37"


@dataclass(frozen=True)
class TraitSpec:
    trait: str
    analysis_id: str
    gwas: Path
    ripple_scores: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pascalx-root", type=Path, default=PASCALX_ROOT)
    parser.add_argument("--gene-loc", type=Path, default=DEFAULT_GENE_LOC)
    parser.add_argument("--bfile", type=Path, default=DEFAULT_BFILE)
    parser.add_argument("--refpanel-prefix", type=Path, default=DEFAULT_REFPANEL_PREFIX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--analysis-set", choices=["primary"], default="primary")
    parser.add_argument("--trait", action="append", default=[])
    parser.add_argument("--chromosomes", default="1-22")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--maf", type=float, default=0.01)
    parser.add_argument("--varcutoff", type=float, default=0.99)
    parser.add_argument("--method", default="saddle")
    parser.add_argument("--prepare-refpanel", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--force", action="store_true")
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


def parse_chromosomes(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", maxsplit=1))
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))
    invalid = [chrom for chrom in out if chrom < 1 or chrom > 22]
    if invalid:
        raise ValueError(f"Invalid autosome values: {invalid}")
    return sorted(set(out))


def select_specs(args: argparse.Namespace) -> list[TraitSpec]:
    specs = default_trait_specs()
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


def prepare_environment(pascalx_root: Path) -> None:
    lib = pascalx_root / "build" / "lib"
    python_dir = pascalx_root / "python"
    os.environ["LD_LIBRARY_PATH"] = f"{lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


def prepare_pascalx_gwas(spec: TraitSpec, trait_dir: Path) -> dict[str, Any]:
    gwas = read_tsv(spec.gwas)
    valid = gwas.loc[:, ["snp_id", "p_value"]].copy()
    p = pd.to_numeric(valid["p_value"], errors="coerce")
    valid = valid.loc[p.notna() & (p > 0) & (p <= 1)].drop_duplicates("snp_id")
    out = trait_dir / "inputs" / f"{spec.trait}.pascalx_gwas.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    valid.to_csv(out, sep="\t", index=False, header=False)
    return {"gwas_input": str(out), "n_input_snps": int(len(gwas)), "n_valid_snps": int(len(valid))}


def prepare_dr_panel_pathway(out_dir: Path) -> Path:
    path = out_dir / "reference" / "dr_panel_symbols.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, genes in DEFAULT_DR_GENE_SETS.items():
            handle.write("\t".join([name, "DR_panel_internal", *sorted(genes)]) + "\n")
    return path


def refpanel_files(prefix: Path, chrom: int) -> dict[str, Path]:
    base = Path(f"{prefix}.chr{chrom}")
    return {
        "tped": Path(str(base) + ".tped"),
        "tped_gz": Path(str(base) + ".tped.gz"),
        "tfam": Path(str(base) + ".tfam"),
        "db": Path(str(base) + ".db"),
        "idx": Path(str(base) + ".idx.gz"),
        "log": Path(str(base) + ".plink.log"),
    }


def prepare_refpanel(args: argparse.Namespace, chromosomes: list[int]) -> list[dict[str, Any]]:
    args.refpanel_prefix.parent.mkdir(parents=True, exist_ok=True)
    reports = []
    for chrom in chromosomes:
        files = refpanel_files(args.refpanel_prefix, chrom)
        if files["tped_gz"].exists() and not args.force:
            reports.append({"chrom": chrom, "status": "tped_exists", "tped_gz": str(files["tped_gz"])})
            continue
        command = [
            "plink",
            "--bfile",
            str(args.bfile),
            "--chr",
            str(chrom),
            "--recode",
            "12",
            "transpose",
            "--out",
            str(Path(f"{args.refpanel_prefix}.chr{chrom}")),
        ]
        with files["log"].open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if files["tped_gz"].exists():
            files["tped_gz"].unlink()
        with files["tped"].open("rb") as source, gzip.open(files["tped_gz"], "wb") as dest:
            dest.writelines(source)
        files["tped"].unlink()
        reports.append(
            {
                "chrom": chrom,
                "status": "tped_built",
                "tped_gz": str(files["tped_gz"]),
                "tfam": str(files["tfam"]),
                "log": str(files["log"]),
            }
        )
    return reports


def run_pascalx_trait(
    spec: TraitSpec,
    *,
    args: argparse.Namespace,
    chromosomes: list[int],
    gwas_report: dict[str, Any],
    pathway_file: Path,
) -> dict[str, Any]:
    from PascalX import genescorer, pathway  # noqa: PLC0415

    trait_dir = args.out_dir / spec.analysis_id
    output_prefix = trait_dir / "pascalx" / spec.trait
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    scorer = genescorer.chi2sum(window=args.window, varcutoff=args.varcutoff, MAF=args.maf)
    scorer.load_genome(str(args.gene_loc))
    scorer.load_refpanel(str(args.refpanel_prefix), parallel=args.parallel, chrlist=chromosomes)
    scorer.load_GWAS(str(gwas_report["gwas_input"]), rscol=0, pcol=1, delimiter="\t", header=False)
    result = scorer.score_chr(
        chrs=chromosomes,
        parallel=args.parallel,
        nobar=True,
        method=args.method,
        autorescore=True,
    )
    gene_table = pascalx_gene_table(spec, result, chromosomes)
    write_table(trait_dir / "tables" / f"{spec.trait}.pascalx_gene_results.tsv.gz", gene_table)
    scorer.save_scores(str(output_prefix) + ".scores.tsv")

    pscorer = pathway.chi2rank(scorer)
    modules = pscorer.load_modules(str(pathway_file), 0, 2)
    pathway_result = pscorer.score(
        modules,
        parallel=args.parallel,
        nobar=True,
        chrs_only=[str(chrom) for chrom in chromosomes],
        genes_only=False,
        method=args.method,
        autorescore=True,
    )
    pathway_table = pascalx_pathway_table(spec, pathway_result, chromosomes)
    write_table(trait_dir / "tables" / f"{spec.trait}.pascalx_dr_panel_pathways.tsv", pathway_table)
    comparison = compare_to_ripple(spec, gene_table)
    write_table(trait_dir / "tables" / f"{spec.trait}.pascalx_ripple_comparison.tsv", comparison)
    return {
        "n_success_genes": int(len(result[0])),
        "n_fail_genes": int(len(result[1])),
        "n_totalfail_genes": int(len(result[2])),
        "n_pathway_rows": int(len(pathway_table)),
        "gene_results": str(trait_dir / "tables" / f"{spec.trait}.pascalx_gene_results.tsv.gz"),
        "pathway_results": str(trait_dir / "tables" / f"{spec.trait}.pascalx_dr_panel_pathways.tsv"),
        "comparison": str(trait_dir / "tables" / f"{spec.trait}.pascalx_ripple_comparison.tsv"),
    }


def pascalx_gene_table(spec: TraitSpec, result: list[Any], chromosomes: list[int]) -> pd.DataFrame:
    success = pd.DataFrame(result[0], columns=["gene_symbol", "pascalx_p", "pascalx_n_snps"])
    if success.empty:
        success = pd.DataFrame(columns=["gene_symbol", "pascalx_p", "pascalx_n_snps"])
    success.insert(0, "trait", spec.trait)
    success.insert(1, "analysis_id", spec.analysis_id)
    success["chromosomes_scored"] = ",".join(str(chrom) for chrom in chromosomes)
    return success


def pascalx_pathway_table(spec: TraitSpec, result: list[Any], chromosomes: list[int]) -> pd.DataFrame:
    rows = result[0] if isinstance(result, list) and result else result
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(
            columns=[
                "pathway_name",
                "scored_genes",
                "gene_rank_pvalues",
                "pascalx_p",
                "n_scored_genes",
            ]
        )
    elif table.shape[1] >= 4:
        table = table.rename(
            columns={
                0: "pathway_name",
                1: "scored_genes",
                2: "gene_rank_pvalues",
                3: "pascalx_p",
            }
        )
        table["n_scored_genes"] = table["scored_genes"].map(len)
    elif table.shape[1] >= 2:
        table = table.rename(columns={0: "pathway_name", 1: "pascalx_p"})
        table["n_scored_genes"] = np.nan
    table.insert(0, "trait", spec.trait)
    table.insert(1, "analysis_id", spec.analysis_id)
    table.insert(2, "gene_set_scope", "DR_panel_internal")
    table["chromosomes_scored"] = ",".join(str(chrom) for chrom in chromosomes)
    return table


def compare_to_ripple(spec: TraitSpec, pascalx_genes: pd.DataFrame) -> pd.DataFrame:
    if pascalx_genes.empty:
        return pd.DataFrame()
    ripple = read_tsv(spec.ripple_scores)
    joined = ripple.merge(
        pascalx_genes.loc[:, ["gene_symbol", "pascalx_p"]],
        on="gene_symbol",
        how="inner",
    )
    if joined.empty:
        return pd.DataFrame()
    minuslog = -np.log10(pd.to_numeric(joined["pascalx_p"], errors="coerce").clip(lower=1e-300))
    rho, p_value = spearmanr(joined["assoc_resid_score"], minuslog)
    rows: list[dict[str, Any]] = [
        {
            "trait": spec.trait,
            "analysis_id": spec.analysis_id,
            "baseline_tool": "PascalX",
            "comparison_name": "spearman_ripple_resid_vs_pascalx_minuslog10p",
            "observed_value": float(rho),
            "p_value": float(p_value),
            "n_genes": int(len(joined)),
        }
    ]
    for fraction in [0.01, 0.02, 0.05, 0.10]:
        k = max(1, int(np.ceil(len(joined) * fraction)))
        ripple_top = set(joined.nlargest(k, "assoc_resid_score")["gene_symbol"].astype(str))
        pascalx_top = set(joined.nsmallest(k, "pascalx_p")["gene_symbol"].astype(str))
        rows.append(
            {
                "trait": spec.trait,
                "analysis_id": spec.analysis_id,
                "baseline_tool": "PascalX",
                "comparison_name": f"top_{fraction:g}_gene_overlap",
                "observed_value": len(ripple_top & pascalx_top) / k,
                "p_value": "",
                "n_genes": int(len(joined)),
                "k": k,
                "overlap_count": len(ripple_top & pascalx_top),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    prepare_environment(args.pascalx_root)
    require_path(args.gene_loc, "gene annotation")
    require_path(args.bfile.with_suffix(".bed"), "PLINK BED")
    require_path(args.bfile.with_suffix(".bim"), "PLINK BIM")
    require_path(args.bfile.with_suffix(".fam"), "PLINK FAM")
    chromosomes = parse_chromosomes(args.chromosomes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ref_reports = prepare_refpanel(args, chromosomes) if args.prepare_refpanel else []
    pathway_file = prepare_dr_panel_pathway(args.out_dir)

    manifests: list[dict[str, Any]] = []
    gene_tables: list[pd.DataFrame] = []
    pathway_tables: list[pd.DataFrame] = []
    comparison_tables: list[pd.DataFrame] = []
    for spec in select_specs(args):
        trait_dir = args.out_dir / spec.analysis_id
        gwas_report = prepare_pascalx_gwas(spec, trait_dir)
        score_report: dict[str, Any] = {"run_status": "prepared_only"}
        if args.score:
            score_report = run_pascalx_trait(
                spec,
                args=args,
                chromosomes=chromosomes,
                gwas_report=gwas_report,
                pathway_file=pathway_file,
            )
            gene_tables.append(read_tsv(Path(score_report["gene_results"])))
            pathway_tables.append(read_tsv(Path(score_report["pathway_results"])))
            comparison_tables.append(read_tsv(Path(score_report["comparison"])))
            score_report["run_status"] = "complete"
        manifest = {
            "created_utc": datetime.now(UTC).isoformat(),
            "baseline_tool": "PascalX",
            "baseline_version": "0.0.5",
            "trait": spec.trait,
            "analysis_id": spec.analysis_id,
            "chromosomes": ",".join(str(chrom) for chrom in chromosomes),
            "genome_build": "GRCh37",
            "window": args.window,
            "maf": args.maf,
            "varcutoff": args.varcutoff,
            "method": args.method,
            "gwas": str(spec.gwas),
            "ripple_scores": str(spec.ripple_scores),
            "refpanel_prefix": str(args.refpanel_prefix),
            "pathway_file": str(pathway_file),
            **gwas_report,
            **score_report,
        }
        manifests.append(manifest)
        report_path = trait_dir / "reports" / f"{spec.trait}.pascalx_manifest.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if gene_tables:
        write_table(args.out_dir / "pascalx_gene_results.all_traits.tsv.gz", pd.concat(gene_tables, ignore_index=True))
    if pathway_tables:
        write_table(args.out_dir / "pascalx_dr_panel_pathways.all_traits.tsv", pd.concat(pathway_tables, ignore_index=True))
    if comparison_tables:
        write_table(args.out_dir / "pascalx_ripple_comparison.all_traits.tsv", pd.concat(comparison_tables, ignore_index=True))
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "baseline_tool": "PascalX",
        "baseline_version": "0.0.5",
        "chromosomes": ",".join(str(chrom) for chrom in chromosomes),
        "refpanel_reports": ref_reports,
        "manifests": manifests,
    }
    (args.out_dir / "pascalx_baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote PascalX baseline outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
