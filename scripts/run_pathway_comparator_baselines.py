#!/usr/bin/env python
"""Run standard pathway comparators against the RIPPLE anchored library."""

from __future__ import annotations

import argparse
import ctypes
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
from scipy import stats

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
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "review_driven_revision_v1" / "pathway_comparator_baselines"
DEFAULT_SUPPLEMENT_DIR = MANUSCRIPT_ROOT / "supplementary_files"
DEFAULT_LIBRARY = (
    ANALYSIS_ROOT
    / "tier4_v12_anchored_broad_reactome_go_cross_trait_n500"
    / "DR_MVP"
    / "tables"
    / "DR_MVP.v12_anchored_module_library.tsv.gz"
)
DEFAULT_MAGMA = PRIVATE_ROOT / "02_environment" / "tools" / "magma_v1.10" / "magma"
DEFAULT_PASCALX_ROOT = PRIVATE_ROOT / "02_environment" / "tools" / "PascalX"
DEFAULT_GENE_LOC = (
    PRIVATE_ROOT
    / "10_raw_data"
    / "reference"
    / "genes"
    / "magma_gene_locations"
    / "NCBI37.3"
    / "NCBI37.3.gene.loc"
)
MAGMA_BASELINE_DIR = ANALYSIS_ROOT / "external_baselines" / "magma_v1.10"
PASCALX_BASELINE_DIR = ANALYSIS_ROOT / "external_baselines" / "pascalx"


@dataclass(frozen=True)
class TraitSpec:
    trait: str
    analysis_id: str
    magma_raw: Path
    pascalx_scores: Path


def default_specs() -> list[TraitSpec]:
    return [
        TraitSpec(
            "DR_MVP",
            "DR_MVP_default_final5000",
            MAGMA_BASELINE_DIR / "DR_MVP_default_final5000" / "magma" / "DR_MVP.genes.genes.raw",
            PASCALX_BASELINE_DIR / "DR_MVP_default_final5000" / "pascalx" / "DR_MVP.scores.tsv",
        ),
        TraitSpec(
            "DR_MVP_NO_MHC_NO_APOE",
            "DR_MVP_no_MHC_no_APOE_final5000",
            MAGMA_BASELINE_DIR
            / "DR_MVP_no_MHC_no_APOE_final5000"
            / "magma"
            / "DR_MVP_NO_MHC_NO_APOE.genes.genes.raw",
            PASCALX_BASELINE_DIR
            / "DR_MVP_no_MHC_no_APOE_final5000"
            / "pascalx"
            / "DR_MVP_NO_MHC_NO_APOE.scores.tsv",
        ),
        TraitSpec(
            "SCZ",
            "SCZ_no_MHC_final5000",
            MAGMA_BASELINE_DIR / "SCZ_no_MHC_final5000" / "magma" / "SCZ.genes.genes.raw",
            PASCALX_BASELINE_DIR / "SCZ_no_MHC_final5000" / "pascalx" / "SCZ.scores.tsv",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--magma", type=Path, default=DEFAULT_MAGMA)
    parser.add_argument("--pascalx-root", type=Path, default=DEFAULT_PASCALX_ROOT)
    parser.add_argument("--gene-loc", type=Path, default=DEFAULT_GENE_LOC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--supplement-dir", type=Path, default=DEFAULT_SUPPLEMENT_DIR)
    parser.add_argument("--trait", action="append", default=[])
    parser.add_argument("--min-present", type=int, default=5)
    parser.add_argument("--max-sets", type=int, default=0, help="Debug cap; 0 means all eligible sets.")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--pascalx-method", default="saddle")
    parser.add_argument("--skip-magma", action="store_true")
    parser.add_argument("--skip-pascalx", action="store_true")
    parser.add_argument("--copy-to-supplement", action="store_true", default=True)
    return parser.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", compression="infer")


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def load_library(path: Path, *, max_sets: int = 0) -> pd.DataFrame:
    table = read_tsv(path)
    source = table["module_source"].astype(str).str.lower()
    category = table["module_category"].astype(str).str.lower()
    keep = source.str.contains("gene ontology|reactome", regex=True) | category.str.contains(
        "go_|reactome", regex=True
    )
    table = table.loc[keep].copy()
    table = table.drop_duplicates("module_name", keep="first")
    if max_sets > 0:
        table = table.head(max_sets).copy()
    return table.reset_index(drop=True)


def gene_sets_from_library(library: pd.DataFrame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for row in library.itertuples(index=False):
        genes = {gene.strip().upper() for gene in str(row.query_genes).split(",") if gene.strip()}
        if genes:
            out[str(row.module_name)] = genes
    return out


def write_pascalx_modules(path: Path, gene_sets: dict[str, set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, genes in sorted(gene_sets.items()):
            handle.write("\t".join([name, "Reactome_GO_fixed_library", *sorted(genes)]) + "\n")


def symbol_to_entrez_map() -> dict[str, str]:
    genes = read_tsv(MAGMA_BASELINE_DIR / "magma_gene_results.all_traits.tsv.gz")
    genes = genes.dropna(subset=["gene_symbol", "magma_gene_id"])
    genes = genes.drop_duplicates("gene_symbol", keep="first")
    return dict(zip(genes["gene_symbol"].astype(str).str.upper(), genes["magma_gene_id"].astype(str), strict=False))


def write_magma_set_annot(path: Path, gene_sets: dict[str, set[str]]) -> pd.DataFrame:
    symbol_to_entrez = symbol_to_entrez_map()
    rows = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for name, genes in sorted(gene_sets.items()):
            ids = sorted({symbol_to_entrez[gene] for gene in genes if gene in symbol_to_entrez})
            rows.append(
                {
                    "module_name": name,
                    "n_query_genes": len(genes),
                    "n_entrez_mapped": len(ids),
                    "dropped_genes": ",".join(sorted(genes - set(symbol_to_entrez))),
                }
            )
            if len(ids) >= 5:
                handle.write("\t".join([name, *ids]) + "\n")
    return pd.DataFrame(rows)


def parse_magma_gsa(path: Path, spec: TraitSpec) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path, sep=r"\s+", comment="#")
    table.insert(0, "trait", spec.trait)
    table.insert(1, "analysis_id", spec.analysis_id)
    table.insert(2, "comparator", "MAGMA_competitive_gene_set")
    return table


def run_magma(spec: TraitSpec, args: argparse.Namespace, set_annot: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    out_prefix = args.out_dir / spec.analysis_id / "magma" / f"{spec.trait}.reactome_go"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    log_path = out_prefix.with_suffix(".log")
    if args.skip_magma:
        return pd.DataFrame(), {"trait": spec.trait, "tool": "MAGMA", "run_status": "skipped_by_user"}
    if not args.magma.exists() or not spec.magma_raw.exists():
        return pd.DataFrame(), {
            "trait": spec.trait,
            "tool": "MAGMA",
            "run_status": "not_tested_missing_input",
            "magma_raw": str(spec.magma_raw),
        }
    command = [
        str(args.magma),
        "--gene-results",
        str(spec.magma_raw),
        "--set-annot",
        str(set_annot),
        "--out",
        str(out_prefix),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result_path = Path(str(out_prefix) + ".gsa.out")
    table = parse_magma_gsa(result_path, spec)
    return table, {
        "trait": spec.trait,
        "tool": "MAGMA",
        "run_status": "complete",
        "command": " ".join(command),
        "log_path": str(log_path),
        "result_path": str(result_path),
    }


def prepare_pascalx_environment(root: Path) -> None:
    lib = root / "build" / "lib"
    python_dir = root / "python"
    os.environ["LD_LIBRARY_PATH"] = f"{lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    ruben = lib / "libruben.so"
    if ruben.exists():
        ctypes.CDLL(str(ruben), mode=ctypes.RTLD_GLOBAL)
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


def reexec_with_pascalx_library_path(args: argparse.Namespace) -> None:
    if args.skip_pascalx or os.environ.get("RIPPLE_PASCALX_ENV_READY") == "1":
        return
    lib = str(args.pascalx_root / "build" / "lib")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if lib in current.split(":"):
        os.environ["RIPPLE_PASCALX_ENV_READY"] = "1"
        return
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{lib}:{current}" if current else lib
    env["RIPPLE_PASCALX_ENV_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def parse_pascalx_pathway_result(result: Any, spec: TraitSpec) -> pd.DataFrame:
    rows = result[0] if isinstance(result, list) and result else result
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=["pathway_name", "pascalx_p", "n_scored_genes"])
    elif table.shape[1] >= 4:
        table = table.rename(
            columns={0: "pathway_name", 1: "scored_genes", 2: "gene_rank_pvalues", 3: "pascalx_p"}
        )
        table["n_scored_genes"] = table["scored_genes"].map(len)
    elif table.shape[1] >= 2:
        table = table.rename(columns={0: "pathway_name", 1: "pascalx_p"})
        table["n_scored_genes"] = np.nan
    table.insert(0, "trait", spec.trait)
    table.insert(1, "analysis_id", spec.analysis_id)
    table.insert(2, "comparator", "PascalX_pathway")
    return table


def run_pascalx(spec: TraitSpec, args: argparse.Namespace, module_file: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.skip_pascalx:
        return pd.DataFrame(), {"trait": spec.trait, "tool": "PascalX", "run_status": "skipped_by_user"}
    if not spec.pascalx_scores.exists():
        return pd.DataFrame(), {
            "trait": spec.trait,
            "tool": "PascalX",
            "run_status": "not_tested_missing_pascalx_scores",
            "pascalx_scores": str(spec.pascalx_scores),
        }
    try:
        prepare_pascalx_environment(args.pascalx_root)
        from PascalX import genescorer, pathway  # noqa: PLC0415

        scorer = genescorer.chi2sum()
        scorer.load_genome(str(args.gene_loc))
        scorer.load_scores(str(spec.pascalx_scores), gcol=0, pcol=1, header=False)
        pscorer = pathway.chi2rank(scorer, fuse=False)
        modules = pscorer.load_modules(str(module_file), 0, 2)
        result = pscorer.score(
            modules,
            parallel=args.parallel,
            nobar=True,
            genes_only=False,
            method=args.pascalx_method,
            autorescore=False,
        )
        table = parse_pascalx_pathway_result(result, spec)
        path = args.out_dir / spec.analysis_id / "tables" / f"{spec.trait}.pascalx_reactome_go_pathways.tsv"
        write_table(path, table)
        return table, {
            "trait": spec.trait,
            "tool": "PascalX",
            "run_status": "complete",
            "scores": str(spec.pascalx_scores),
            "result_path": str(path),
        }
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame(), {
            "trait": spec.trait,
            "tool": "PascalX",
            "run_status": "failed",
            "scores": str(spec.pascalx_scores),
            "error": str(exc),
        }


def rankset_comparator(
    gene_scores: pd.DataFrame,
    *,
    score_col: str,
    p_col: str,
    comparator: str,
    spec: TraitSpec,
    gene_sets: dict[str, set[str]],
    min_present: int,
) -> pd.DataFrame:
    work = gene_scores.dropna(subset=["gene_symbol", p_col]).copy()
    work["gene_symbol"] = work["gene_symbol"].astype(str).str.upper()
    work["score"] = -np.log10(pd.to_numeric(work[p_col], errors="coerce").clip(lower=1e-300))
    work = work.dropna(subset=["score"]).drop_duplicates("gene_symbol", keep="first")
    by_gene = dict(zip(work["gene_symbol"], work["score"], strict=False))
    background = set(by_gene)
    rows = []
    for name, genes in gene_sets.items():
        present = sorted(genes & background)
        if len(present) < min_present:
            continue
        module_values = np.asarray([by_gene[gene] for gene in present], dtype=float)
        background_values = np.asarray([by_gene[gene] for gene in sorted(background - set(present))], dtype=float)
        _, p_value = stats.mannwhitneyu(module_values, background_values, alternative="greater")
        rows.append(
            {
                "trait": spec.trait,
                "analysis_id": spec.analysis_id,
                "comparator": comparator,
                "module_name": name,
                "n_present": len(present),
                "background_size": len(background),
                "mean_minuslog10p": float(np.mean(module_values)),
                "rankset_p": float(p_value),
                "present_genes": ",".join(present),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rankset_fdr"] = stats.false_discovery_control(out["rankset_p"].to_numpy(dtype=float))
    return out


def load_magma_gene_scores(spec: TraitSpec) -> pd.DataFrame:
    all_genes = read_tsv(MAGMA_BASELINE_DIR / "magma_gene_results.all_traits.tsv.gz")
    return all_genes.loc[all_genes["analysis_id"].eq(spec.analysis_id)].copy()


def load_pascalx_gene_scores(spec: TraitSpec) -> pd.DataFrame:
    all_genes = read_tsv(PASCALX_BASELINE_DIR / "pascalx_gene_results.all_traits.tsv.gz")
    return all_genes.loc[all_genes["analysis_id"].eq(spec.analysis_id)].copy()


def select_specs(args: argparse.Namespace) -> list[TraitSpec]:
    specs = default_specs()
    if not args.trait:
        return specs
    requested = set(args.trait)
    selected = [spec for spec in specs if spec.trait in requested or spec.analysis_id in requested]
    missing = sorted(requested - {spec.trait for spec in selected} - {spec.analysis_id for spec in selected})
    if missing:
        raise ValueError(f"Unknown traits/analysis IDs: {missing}")
    return selected


def main() -> None:
    args = parse_args()
    reexec_with_pascalx_library_path(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    library = load_library(args.library, max_sets=args.max_sets)
    gene_sets = gene_sets_from_library(library)
    module_file = args.out_dir / "reference" / "reactome_go_pascalx_modules.tsv"
    set_annot = args.out_dir / "reference" / "reactome_go_magma_set_annot.tsv"
    write_pascalx_modules(module_file, gene_sets)
    mapping_report = write_magma_set_annot(set_annot, gene_sets)
    write_table(args.out_dir / "reference" / "reactome_go_magma_set_mapping.tsv", mapping_report)

    magma_tables: list[pd.DataFrame] = []
    pascalx_tables: list[pd.DataFrame] = []
    rankset_tables: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for spec in select_specs(args):
        magma_table, magma_manifest = run_magma(spec, args, set_annot)
        manifests.append(magma_manifest)
        if not magma_table.empty:
            magma_tables.append(magma_table)
        pascalx_table, pascalx_manifest = run_pascalx(spec, args, module_file)
        manifests.append(pascalx_manifest)
        if not pascalx_table.empty:
            pascalx_tables.append(pascalx_table)
        try:
            rankset_tables.append(
                rankset_comparator(
                    load_magma_gene_scores(spec),
                    score_col="magma_p",
                    p_col="magma_p",
                    comparator="MAGMA_gene_score_rankset",
                    spec=spec,
                    gene_sets=gene_sets,
                    min_present=args.min_present,
                )
            )
        except Exception as exc:  # noqa: BLE001
            manifests.append({"trait": spec.trait, "tool": "MAGMA_rankset", "run_status": "failed", "error": str(exc)})
        try:
            rankset_tables.append(
                rankset_comparator(
                    load_pascalx_gene_scores(spec),
                    score_col="pascalx_p",
                    p_col="pascalx_p",
                    comparator="PascalX_gene_score_rankset",
                    spec=spec,
                    gene_sets=gene_sets,
                    min_present=args.min_present,
                )
            )
        except Exception as exc:  # noqa: BLE001
            manifests.append({"trait": spec.trait, "tool": "PascalX_rankset", "run_status": "failed", "error": str(exc)})

    if magma_tables:
        write_table(args.out_dir / "magma_reactome_go_gene_sets.all_traits.tsv", pd.concat(magma_tables, ignore_index=True))
    if pascalx_tables:
        write_table(args.out_dir / "pascalx_reactome_go_pathways.all_traits.tsv", pd.concat(pascalx_tables, ignore_index=True))
    if rankset_tables:
        rankset = pd.concat([t for t in rankset_tables if not t.empty], ignore_index=True)
        write_table(args.out_dir / "magma_pascalx_rankset_comparator.all_traits.tsv", rankset)
        if args.copy_to_supplement:
            write_table(args.supplement_dir / "pathway_comparator_rankset_summary.tsv", rankset)

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "library": str(args.library),
        "n_library_sets": int(len(gene_sets)),
        "set_annot": str(set_annot),
        "pascalx_modules": str(module_file),
        "manifests": manifests,
    }
    manifest_path = args.out_dir / "pathway_comparator_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.copy_to_supplement:
        (args.supplement_dir / "pathway_comparator_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
    print(f"Wrote pathway comparator outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
