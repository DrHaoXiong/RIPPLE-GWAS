#!/usr/bin/env python
"""Build the prespecified DR-specific module library v1.

This library is a fixed hypothesis space for DR-context RIPPLE-D testing. It
extends the earlier curated vascular DR panel with RPE/melanosome, visual-cycle,
phagolysosome, cholesterol/LXR/APOE, and complement axes. The output schema is
compatible with the anchored/RIPPLE-D gene-set loader.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_dr_specific_biology_panel import PANEL as V0_PANEL  # noqa: E402

PRIVATE_ROOT = Path("/mnt/d/RIPPLE/RIPPLE_private")
DEFAULT_OUT_DIR = PRIVATE_ROOT / "20_processed_data" / "reference_pathways" / "dr_specific_library_v1"


ADDITIONAL_PANEL: dict[str, dict[str, object]] = {
    "DR_RPE_MELANOSOME_BIOGENESIS_TRAFFICKING": {
        "category": "rpe_melanosome_biogenesis_trafficking",
        "panel_role": "dr_retinal_rpe_pathobiology",
        "description": "RPE melanosome and organelle-trafficking biology relevant to retinal pigment stress.",
        "genes": {
            "AP3B1",
            "AP3B2",
            "AP3D1",
            "AP3M1",
            "AP3M2",
            "AP3S1",
            "AP3S2",
            "AP1G1",
            "AP1S1",
            "BLOC1S1",
            "BLOC1S2",
            "BLOC1S3",
            "BLOC1S4",
            "BLOC1S5",
            "BLOC1S6",
            "DTNBP1",
            "HPS1",
            "HPS3",
            "HPS4",
            "HPS5",
            "HPS6",
            "LYST",
            "OCA2",
            "TYR",
            "TYRP1",
            "DCT",
            "PMEL",
            "RAB27A",
            "MLPH",
            "MYO5A",
            "SLC45A2",
            "SLC24A5",
        },
    },
    "DR_RPE_VISUAL_CYCLE_RETINOID_METABOLISM": {
        "category": "rpe_visual_cycle_retinoid_metabolism",
        "panel_role": "dr_retinal_rpe_pathobiology",
        "description": "RPE visual-cycle and retinoid metabolism genes.",
        "genes": {
            "RPE65",
            "LRAT",
            "RLBP1",
            "RGR",
            "RDH5",
            "RDH8",
            "RDH10",
            "RDH11",
            "ABCA4",
            "STRA6",
            "TTR",
            "RBP1",
            "RBP3",
            "RBP4",
            "CRX",
            "NRL",
            "BEST1",
            "PRPH2",
            "ROM1",
            "PROM1",
            "IMPG1",
            "IMPG2",
        },
    },
    "DR_RPE_PHAGOLYSOSOME_AUTOPHAGY": {
        "category": "rpe_phagolysosome_autophagy",
        "panel_role": "dr_retinal_rpe_pathobiology",
        "description": "RPE phagocytosis, lysosome and autophagy programs relevant to photoreceptor outer-segment turnover.",
        "genes": {
            "MERTK",
            "GAS6",
            "PROS1",
            "AXL",
            "TYRO3",
            "ITGAV",
            "ITGB5",
            "MFGE8",
            "LAMP1",
            "LAMP2",
            "CTSB",
            "CTSD",
            "CTSL",
            "TFEB",
            "TFE3",
            "BECN1",
            "ATG5",
            "ATG7",
            "ATG12",
            "MAP1LC3B",
            "SQSTM1",
            "RAB7A",
            "RAB5A",
            "VPS35",
            "VPS26A",
            "VPS29",
            "OPTN",
            "CALR",
        },
    },
    "DR_RETINAL_CHOLESTEROL_EFFLUX_LXR_APOE": {
        "category": "retinal_cholesterol_efflux_lxr_apoe",
        "panel_role": "dr_retinal_lipid_pathobiology",
        "description": "Retinal lipid handling, cholesterol efflux and LXR/APOE-related pathways.",
        "genes": {
            "APOE",
            "APOC1",
            "APOC2",
            "APOC3",
            "APOA1",
            "APOA4",
            "APOB",
            "ABCA1",
            "ABCG1",
            "ABCG5",
            "ABCG8",
            "NR1H2",
            "NR1H3",
            "RXRA",
            "RXRB",
            "LPL",
            "LDLR",
            "LRP1",
            "CLU",
            "TREM2",
            "LIPA",
            "SOAT1",
            "SOAT2",
            "SREBF1",
            "SREBF2",
            "INSIG1",
            "SCARB1",
            "CETP",
        },
    },
    "DR_RPE_COMPLEMENT_ALTERNATIVE_PATHWAY": {
        "category": "rpe_complement_alternative_pathway",
        "panel_role": "dr_retinal_inflammation_pathobiology",
        "description": "Alternative-complement and RPE-associated inflammatory susceptibility genes.",
        "genes": {
            "CFH",
            "CFHR1",
            "CFHR2",
            "CFHR3",
            "CFHR4",
            "CFHR5",
            "CFI",
            "CFB",
            "C3",
            "C5",
            "C2",
            "CD46",
            "CD55",
            "CD59",
            "SERPING1",
            "C1QA",
            "C1QB",
            "C1QC",
            "C1R",
            "C1S",
            "HTRA1",
            "ARMS2",
            "TIMP3",
        },
    },
    "DR_RETINAL_WNT_NORRIN_BARRIER_ANGIOGENESIS": {
        "category": "retinal_wnt_norrin_barrier_angiogenesis",
        "panel_role": "dr_retinal_vascular_pathobiology",
        "description": "Norrin/Wnt signaling and retinal vascular barrier programs.",
        "genes": {
            "NDP",
            "FZD4",
            "LRP5",
            "LRP6",
            "TSPAN12",
            "CTNNB1",
            "GPR124",
            "REEP6",
            "KDR",
            "FLT1",
            "VEGFA",
            "DLL4",
            "NOTCH1",
            "JAG1",
            "JAG2",
            "CLDN5",
            "MFSD2A",
            "PLVAP",
            "SOX17",
            "FOXC1",
            "FOXF2",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def write_table(path: Path, table: pd.DataFrame, *, gzip: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip:
        table.to_csv(path, sep="\t", index=False, compression="gzip")
    else:
        table.to_csv(path, sep="\t", index=False)


def build_panel() -> dict[str, dict[str, object]]:
    panel: dict[str, dict[str, object]] = {name: dict(spec) for name, spec in V0_PANEL.items()}
    panel.update(ADDITIONAL_PANEL)
    return panel


def main() -> None:
    args = parse_args()
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    panel = build_panel()
    rows: list[dict[str, object]] = []
    registry_rows: list[dict[str, object]] = []
    for gene_set, spec in sorted(panel.items()):
        genes = sorted({str(gene).upper() for gene in spec["genes"]})
        registry_rows.append(
            {
                "gene_set": gene_set,
                "source_database": "curated_dr_specific_library_v1",
                "source_term_id": gene_set,
                "source_term_name": str(spec["description"]),
                "source_url": "pending_manual_citation_review",
                "module_source": "curated_dr_specific_library_v1",
                "annotation_source_type": "independent_external_pending_citation",
                "category": spec["category"],
                "gene_set_scope": "dr_specific_fixed_hypothesis_library_v1",
                "panel_role": spec["panel_role"],
                "description": spec["description"],
                "n_query_genes": len(genes),
                "citation_status": "pending_manual_citation_review",
            }
        )
        for gene in genes:
            rows.append(
                {
                    "gene_set": gene_set,
                    "gene_symbol": gene,
                    "source_database": "curated_dr_specific_library_v1",
                    "source_term_id": gene_set,
                    "source_term_name": str(spec["description"]),
                    "source_url": "pending_manual_citation_review",
                    "category": spec["category"],
                    "gene_set_scope": "dr_specific_fixed_hypothesis_library_v1",
                    "module_source": "curated_dr_specific_library_v1",
                    "annotation_source_type": "independent_external_pending_citation",
                    "panel_role": spec["panel_role"],
                    "citation_status": "pending_manual_citation_review",
                }
            )

    gene_sets = pd.DataFrame(rows).sort_values(["gene_set", "gene_symbol"]).reset_index(drop=True)
    registry = pd.DataFrame(registry_rows).sort_values("gene_set").reset_index(drop=True)
    retinal = gene_sets.loc[gene_sets["panel_role"].ne("shared_diabetic_liability_context")].reset_index(drop=True)
    retinal_registry = registry.loc[registry["panel_role"].ne("shared_diabetic_liability_context")].reset_index(drop=True)

    write_table(tables_dir / "dr_specific_library_v1.gene_sets.tsv", gene_sets)
    write_table(tables_dir / "dr_specific_library_v1.gene_sets.tsv.gz", gene_sets, gzip=True)
    write_table(tables_dir / "dr_specific_library_v1.registry.tsv", registry)
    write_table(tables_dir / "dr_specific_library_v1.retinal_only.gene_sets.tsv", retinal)
    write_table(tables_dir / "dr_specific_library_v1.retinal_only.gene_sets.tsv.gz", retinal, gzip=True)
    write_table(tables_dir / "dr_specific_library_v1.retinal_only.registry.tsv", retinal_registry)

    manifest = {
        "created_utc": now_utc(),
        "script_path": str(Path(__file__).resolve()),
        "output_dir": str(args.out_dir),
        "n_gene_sets": int(registry.shape[0]),
        "n_unique_genes": int(gene_sets["gene_symbol"].nunique()),
        "retinal_only_n_gene_sets": int(retinal_registry.shape[0]),
        "retinal_only_n_unique_genes": int(retinal["gene_symbol"].nunique()),
        "citation_status": "pending_manual_citation_review",
        "claim_boundary": "fixed DR-context hypothesis library; not independent validation until citations and external evidence are finalized",
    }
    (reports_dir / "dr_specific_library_v1.manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    lines = [
        "# DR-specific module library v1",
        "",
        f"Created: {manifest['created_utc']}",
        "",
        "This fixed library is intended for DR-context RIPPLE-D sensitivity testing.",
        "It should not be described as independent validation until citations and source provenance are manually finalized.",
        "",
        "| Gene set | Role | Category | Genes |",
        "|---|---|---|---:|",
    ]
    for row in registry.to_dict(orient="records"):
        lines.append(f"| {row['gene_set']} | {row['panel_role']} | {row['category']} | {int(row['n_query_genes'])} |")
    (reports_dir / "dr_specific_library_v1.report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote DR-specific library v1 to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
