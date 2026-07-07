#!/usr/bin/env python
"""Attempt a true dmGWAS/DMS baseline, with labelled DMS-style fallback."""

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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

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
DEFAULT_OUT_DIR = ANALYSIS_ROOT / "review_driven_revision_v1" / "true_network_method_baseline"
DEFAULT_SUPPLEMENT_DIR = MANUSCRIPT_ROOT / "supplementary_files"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--supplement-dir", type=Path, default=DEFAULT_SUPPLEMENT_DIR)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--force-fallback", action="store_true")
    parser.add_argument("--n-node-null", type=int, default=500)
    parser.add_argument("--n-degree-graph-null", type=int, default=50)
    parser.add_argument("--copy-to-supplement", action="store_true", default=True)
    return parser.parse_args()


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def probe_r_package(rscript: str, package: str) -> dict[str, Any]:
    command = [rscript, "-e", f"cat(requireNamespace('{package}', quietly=TRUE))"]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
        return {
            "package": package,
            "available": proc.returncode == 0 and "TRUE" in proc.stdout,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"package": package, "available": False, "error": str(exc)}


def run_fallback(args: argparse.Namespace) -> dict[str, Any]:
    fallback_dir = args.out_dir / "faithful_dms_style_fallback"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_dense_module_competitor.py"),
        "--out-dir",
        str(fallback_dir),
        "--n-node-null",
        str(args.n_node_null),
        "--n-degree-graph-null",
        str(args.n_degree_graph_null),
    ]
    subprocess.run(command, check=True)
    summary_path = fallback_dir / "dense_module_competitor_summary.all_traits.tsv"
    summary = pd.read_csv(summary_path, sep="\t") if summary_path.exists() else pd.DataFrame()
    if not summary.empty:
        summary.insert(0, "baseline_adapter", "faithful_DMS_style_fallback")
        summary["actual_dmGWAS_or_DMS_run"] = False
        summary["allowed_language"] = (
            "faithful DMS-style fallback; do not describe as actual dmGWAS or EW_dmGWAS"
        )
        write_table(args.out_dir / "true_network_method_baseline_summary.tsv", summary)
        if args.copy_to_supplement:
            write_table(args.supplement_dir / "true_network_method_baseline_summary.tsv", summary)
    return {
        "fallback_status": "complete",
        "fallback_command": " ".join(command),
        "fallback_summary": str(summary_path),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    probes = [probe_r_package(args.rscript, package) for package in ["dmGWAS", "DMS", "EW_dmGWAS"]]
    available = [probe for probe in probes if probe.get("available")]
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(UTC).isoformat(),
        "script_path": str(Path(__file__).resolve()),
        "strategy": "dmGWAS_DMS_first_then_labelled_fallback",
        "rscript": args.rscript,
        "package_probes": probes,
    }
    if available and not args.force_fallback:
        manifest["run_status"] = "blocked_adapter_not_implemented_for_installed_package"
        manifest["available_packages"] = available
        manifest["next_action"] = (
            "A real dmGWAS/DMS package is installed; implement a package-specific wrapper before "
            "using fallback outputs in the manuscript."
        )
        pd.DataFrame(
            [
                {
                    "baseline_adapter": "dmGWAS_DMS_probe",
                    "actual_dmGWAS_or_DMS_run": False,
                    "run_status": manifest["run_status"],
                    "available_packages": ",".join(str(p["package"]) for p in available),
                    "allowed_language": "not_tested_until_package_wrapper_is_implemented",
                }
            ]
        ).to_csv(args.out_dir / "true_network_method_baseline_summary.tsv", sep="\t", index=False)
    else:
        manifest["run_status"] = "fallback_used"
        manifest.update(run_fallback(args))
    manifest_path = args.out_dir / "true_network_method_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.copy_to_supplement:
        shutil.copy2(manifest_path, args.supplement_dir / "true_network_method_baseline_manifest.json")
    print(f"Wrote true network-method baseline adapter outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
