#!/usr/bin/env python
"""Run a true external dmGWAS 3.0 network-method baseline."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
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
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "review_driven_revision_v1" / "dmgwas_external_baseline"
DEFAULT_SUPPLEMENT_DIR = MANUSCRIPT_ROOT / "supplementary_files"
R_WRAPPER = SCRIPT_DIR / "run_dmgwas_external_baseline.R"
DMGWAS_P_CLIP_MIN = 1e-16
DMGWAS_P_CLIP_MAX = 1 - 1e-16

ANALYSES = {
    "DR_MVP": {
        "analysis_id": "DR_MVP_default_final5000",
        "analysis_dir": ANALYSIS_ROOT / "dr_mvp_string_final5000",
        "score_prefix": "DR_MVP",
        "edge_dir": ANALYSIS_ROOT / "dr_mvp_string_final5000",
        "edge_prefix": "DR_MVP",
        "trait": "DR_MVP",
    },
    "DR_MVP_NO_MHC_NO_APOE": {
        "analysis_id": "DR_MVP_no_MHC_no_APOE_final5000",
        "analysis_dir": ANALYSIS_ROOT / "dr_mvp_no_mhc_no_apoe_final5000",
        "score_prefix": "DR_MVP_NO_MHC_NO_APOE",
        "edge_dir": ANALYSIS_ROOT / "dr_mvp_string_final5000",
        "edge_prefix": "DR_MVP",
        "trait": "DR_MVP_NO_MHC_NO_APOE",
    },
    "SCZ": {
        "analysis_id": "SCZ_no_MHC_final5000",
        "analysis_dir": ANALYSIS_ROOT / "scz_no_mhc_string_final5000",
        "score_prefix": "SCZ",
        "edge_dir": ANALYSIS_ROOT / "scz_no_mhc_string_final5000",
        "edge_prefix": "SCZ",
        "trait": "SCZ",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--supplement-dir", type=Path, default=DEFAULT_SUPPLEMENT_DIR)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--traits", nargs="+", default=list(ANALYSES))
    parser.add_argument("--max-genes", type=int, default=0, help="Restrict to top N genes for smoke testing.")
    parser.add_argument("--r", type=float, default=0.1, help="dmGWAS module expansion increment cutoff.")
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--copy-to-supplement", action="store_true", default=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", compression="infer")


def prepare_inputs(analysis: dict[str, Any], out_dir: Path, max_genes: int) -> dict[str, Path | int | float]:
    trait = str(analysis["trait"])
    analysis_dir = Path(analysis["analysis_dir"])
    score_prefix = str(analysis.get("score_prefix", trait))
    edge_dir = Path(analysis.get("edge_dir", analysis_dir))
    edge_prefix = str(analysis.get("edge_prefix", trait))
    tables_dir = analysis_dir / "tables"
    score_path = tables_dir / f"{score_prefix}.lcc_gene_scores.1000G_LD.residualized.tsv.gz"
    edge_path = edge_dir / "tables" / f"{edge_prefix}.analysis_graph_edges.tsv.gz"
    if not score_path.exists():
        raise FileNotFoundError(score_path)
    if not edge_path.exists():
        raise FileNotFoundError(edge_path)

    scores = read_table(score_path)
    p_col = "assoc_p_g_clipped" if "assoc_p_g_clipped" in scores.columns else "assoc_p_g"
    keep_cols = ["gene_symbol", p_col, "assoc_resid_score"]
    gene_p = scores[keep_cols].copy()
    gene_p = gene_p.rename(columns={"gene_symbol": "gene", p_col: "weight"})
    raw_p = pd.to_numeric(gene_p["weight"], errors="coerce")
    low_clip_count = int(raw_p.lt(DMGWAS_P_CLIP_MIN).sum())
    high_clip_count = int(raw_p.gt(DMGWAS_P_CLIP_MAX).sum())
    gene_p["weight"] = raw_p.clip(DMGWAS_P_CLIP_MIN, DMGWAS_P_CLIP_MAX)
    gene_p["assoc_resid_score"] = pd.to_numeric(gene_p["assoc_resid_score"], errors="coerce")
    gene_p = gene_p.dropna(subset=["gene", "weight"])
    gene_p = gene_p.drop_duplicates("gene")

    if max_genes and max_genes > 0:
        gene_p = gene_p.sort_values("weight", ascending=True).head(max_genes)

    edges = read_table(edge_path)
    edges = edges.rename(columns={edges.columns[0]: "node1", edges.columns[1]: "node2"})
    edges = edges[["node1", "node2"]].dropna().drop_duplicates()
    genes = set(gene_p["gene"].astype(str))
    edges = edges[edges["node1"].isin(genes) & edges["node2"].isin(genes)].copy()

    # Drop genes outside the induced graph, because dmGWAS integrates weights only over graph nodes.
    covered = set(edges["node1"]).union(set(edges["node2"]))
    gene_p = gene_p[gene_p["gene"].isin(covered)].copy()
    edges = edges[edges["node1"].isin(covered) & edges["node2"].isin(covered)].copy()

    prefix = trait if not max_genes else f"{trait}.top{max_genes}"
    input_dir = out_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    gene_p_path = input_dir / f"{prefix}.dmGWAS_gene_p.tsv"
    edge_out_path = input_dir / f"{prefix}.dmGWAS_network.tsv"
    write_table(gene_p_path, gene_p[["gene", "weight"]])
    write_table(edge_out_path, edges)
    return {
        "gene_p_path": gene_p_path,
        "network_path": edge_out_path,
        "score_source_path": score_path,
        "edge_source_path": edge_path,
        "n_genes": int(len(gene_p)),
        "n_edges": int(len(edges)),
        "max_genes_requested": int(max_genes),
        "analysis_scope": "top_ranked_graph_induced" if max_genes and max_genes > 0 else "full_graph_covered_lcc",
        "scope_note": (
            f"Official dmGWAS 3.0 run on the graph-induced subset of the top {max_genes} genes by RIPPLE "
            "gene-level P value; full 12k-node STRING runs are recorded as computationally deferred."
            if max_genes and max_genes > 0
            else "Official dmGWAS 3.0 run on all graph-covered genes in the analysis LCC."
        ),
        "full_scale_attempt_status": "computationally_deferred" if max_genes and max_genes > 0 else "complete",
        "dmGWAS_p_clip_min": DMGWAS_P_CLIP_MIN,
        "dmGWAS_p_clip_max": DMGWAS_P_CLIP_MAX,
        "dmGWAS_p_low_clip_count": low_clip_count,
        "dmGWAS_p_high_clip_count": high_clip_count,
    }


def run_trait(args: argparse.Namespace, trait_key: str) -> dict[str, Any]:
    analysis = ANALYSES[trait_key]
    trait = str(analysis["trait"])
    analysis_id = str(analysis["analysis_id"])
    inputs = prepare_inputs(analysis, args.out_dir, args.max_genes)
    trait_out = args.out_dir / "tables"
    trait_out.mkdir(parents=True, exist_ok=True)
    modules_path = trait_out / f"{trait}.dmGWAS_modules.tsv"
    summary_path = trait_out / f"{trait}.dmGWAS_summary.tsv"
    if args.force:
        modules_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    if summary_path.exists() and modules_path.exists() and not args.force:
        return {
            "trait": trait,
            "analysis_id": analysis_id,
            "status": "skipped_existing",
            "summary_path": str(summary_path),
            "modules_path": str(modules_path),
            **inputs,
        }

    command = [
        args.rscript,
        str(R_WRAPPER),
        str(inputs["network_path"]),
        str(inputs["gene_p_path"]),
        str(trait_out),
        trait,
        str(args.r),
    ]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        timeout=args.timeout_seconds if args.timeout_seconds > 0 else None,
    )
    status = "complete" if proc.returncode == 0 and summary_path.exists() else "failed"
    return {
        "trait": trait,
        "analysis_id": analysis_id,
        "status": status,
        "returncode": proc.returncode,
        "command": " ".join(command),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "summary_path": str(summary_path),
        "modules_path": str(modules_path),
        **inputs,
    }


def build_combined_outputs(out_dir: Path, run_records: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    modules: list[pd.DataFrame] = []
    for record in run_records:
        summary_path = Path(str(record.get("summary_path", "")))
        modules_path = Path(str(record.get("modules_path", "")))
        if summary_path.exists():
            summary = read_table(summary_path)
            summary.insert(1, "analysis_id", record["analysis_id"])
            summary["actual_external_network_method_run"] = True
            summary["allowed_language"] = "actual dmGWAS 3.0 node-only run; external network-method baseline"
            summary["source_result_path"] = str(summary_path)
            summary["script_path"] = str(Path(__file__).resolve())
            summary["seed"] = 20260725
            summary["timestamp"] = now_utc()
            summary["analysis_scope"] = record.get("analysis_scope", "")
            summary["max_genes_requested"] = record.get("max_genes_requested", "")
            summary["scope_note"] = record.get("scope_note", "")
            summary["full_scale_attempt_status"] = record.get("full_scale_attempt_status", "")
            summary["score_source_path"] = str(record.get("score_source_path", ""))
            summary["edge_source_path"] = str(record.get("edge_source_path", ""))
            summary["dmGWAS_p_clip_min"] = record.get("dmGWAS_p_clip_min", "")
            summary["dmGWAS_p_clip_max"] = record.get("dmGWAS_p_clip_max", "")
            summary["dmGWAS_p_low_clip_count"] = record.get("dmGWAS_p_low_clip_count", "")
            summary["dmGWAS_p_high_clip_count"] = record.get("dmGWAS_p_high_clip_count", "")
            summaries.append(summary)
        if modules_path.exists():
            module = read_table(modules_path)
            module.insert(1, "analysis_id", record["analysis_id"])
            module["source_result_path"] = str(modules_path)
            module["script_path"] = str(Path(__file__).resolve())
            module["seed"] = 20260725
            module["timestamp"] = now_utc()
            module["analysis_scope"] = record.get("analysis_scope", "")
            module["max_genes_requested"] = record.get("max_genes_requested", "")
            module["dmGWAS_p_clip_min"] = record.get("dmGWAS_p_clip_min", "")
            module["dmGWAS_p_clip_max"] = record.get("dmGWAS_p_clip_max", "")
            modules.append(module)
    summary_all = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    modules_all = pd.concat(modules, ignore_index=True) if modules else pd.DataFrame()
    write_table(out_dir / "dmgwas_external_baseline_summary.tsv", summary_all)
    write_table(out_dir / "dmgwas_external_baseline_modules.tsv", modules_all)
    return summary_all, modules_all


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested = [trait for trait in args.traits if trait in ANALYSES]
    if not requested:
        raise ValueError(f"No valid traits requested. Valid traits: {sorted(ANALYSES)}")
    run_records = [run_trait(args, trait) for trait in requested]
    summary_all, modules_all = build_combined_outputs(args.out_dir, run_records)
    manifest = {
        "created_utc": now_utc(),
        "script_path": str(Path(__file__).resolve()),
        "r_wrapper": str(R_WRAPPER),
        "baseline_method": "dmGWAS_3.0_node_only",
        "r": args.r,
        "max_genes": args.max_genes,
        "traits": requested,
        "run_records": [
            {key: str(value) if isinstance(value, Path) else value for key, value in record.items()}
            for record in run_records
        ],
        "combined_summary": str(args.out_dir / "dmgwas_external_baseline_summary.tsv"),
        "combined_modules": str(args.out_dir / "dmgwas_external_baseline_modules.tsv"),
        "n_summary_rows": int(len(summary_all)),
        "n_module_rows": int(len(modules_all)),
    }
    manifest_path = args.out_dir / "dmgwas_external_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.copy_to_supplement:
        args.supplement_dir.mkdir(parents=True, exist_ok=True)
        for path in [
            args.out_dir / "dmgwas_external_baseline_summary.tsv",
            args.out_dir / "dmgwas_external_baseline_modules.tsv",
            manifest_path,
        ]:
            if path.exists():
                shutil.copy2(path, args.supplement_dir / path.name)
    print(f"Wrote dmGWAS external baseline outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
