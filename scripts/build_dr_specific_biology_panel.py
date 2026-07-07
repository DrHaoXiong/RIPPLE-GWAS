#!/usr/bin/env python
"""Build a preregistered DR-specific biology panel for anchored module tests.

The panel is deliberately small and pathobiology-oriented. It is intended as a
fixed-library sensitivity layer for DR biology, not as a discovery result.
Literature citations are tracked as pending manual review before manuscript use.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PRIVATE_ROOT = Path("/path/to/ripple_private_workspace")
DEFAULT_OUT_DIR = PRIVATE_ROOT / "20_processed_data" / "reference_pathways" / "dr_specific_biology_panel_v0_1"


PANEL: dict[str, dict[str, object]] = {
    "DR_RETINAL_ANGIOGENESIS_VEGF_HIF": {
        "category": "retinal_angiogenesis_vegf_hif",
        "panel_role": "dr_retinal_vascular_pathobiology",
        "description": "Retinal angiogenesis, hypoxia and VEGF/angiopoietin signaling.",
        "genes": {
            "VEGFA",
            "VEGFB",
            "VEGFC",
            "PGF",
            "KDR",
            "FLT1",
            "FLT4",
            "HIF1A",
            "EPAS1",
            "ANGPT1",
            "ANGPT2",
            "TEK",
            "TIE1",
            "NRP1",
            "NRP2",
            "DLL4",
            "NOTCH1",
            "ESM1",
            "PDGFB",
            "PDGFRB",
        },
    },
    "DR_ENDOTHELIAL_DYSFUNCTION_ACTIVATION": {
        "category": "endothelial_dysfunction_activation",
        "panel_role": "dr_retinal_vascular_pathobiology",
        "description": "Endothelial identity, permeability and inflammatory activation.",
        "genes": {
            "PECAM1",
            "VWF",
            "CDH5",
            "ICAM1",
            "VCAM1",
            "SELE",
            "SELP",
            "ESAM",
            "ENG",
            "NOS3",
            "EDN1",
            "PLVAP",
            "EMCN",
            "ROBO4",
            "ERG",
            "SOX17",
            "KLF2",
            "KLF4",
            "ACKR1",
            "THBD",
        },
    },
    "DR_PERICYTE_MURAL_CELL_BIOLOGY": {
        "category": "pericyte_mural_cell_biology",
        "panel_role": "dr_retinal_vascular_pathobiology",
        "description": "Pericyte, mural cell and vascular smooth muscle biology relevant to capillary stability.",
        "genes": {
            "PDGFRB",
            "PDGFB",
            "RGS5",
            "CSPG4",
            "ACTA2",
            "MCAM",
            "NOTCH3",
            "ABCC9",
            "KCNJ8",
            "DES",
            "MYH11",
            "TAGLN",
            "MYL9",
            "TPM2",
            "ANGPT1",
            "TEK",
            "CD248",
            "FOXC1",
            "FOXF2",
        },
    },
    "DR_BLOOD_RETINA_BARRIER_TIGHT_JUNCTION": {
        "category": "blood_retina_barrier_tight_junction",
        "panel_role": "dr_retinal_vascular_pathobiology",
        "description": "Blood-retina barrier integrity, tight junctions and endothelial barrier transport.",
        "genes": {
            "TJP1",
            "TJP2",
            "OCLN",
            "CLDN5",
            "CLDN12",
            "JAM2",
            "JAM3",
            "F11R",
            "CDH5",
            "CTNNB1",
            "PLVAP",
            "MFSD2A",
            "LRP5",
            "NDP",
            "FZD4",
            "TSPAN12",
            "ABCB1",
            "SLC2A1",
            "CAV1",
        },
    },
    "DR_BASEMENT_MEMBRANE_ECM_REMODELING": {
        "category": "basement_membrane_ecm_remodeling",
        "panel_role": "dr_retinal_vascular_pathobiology",
        "description": "Basement membrane, collagen, laminin and matrix remodeling.",
        "genes": {
            "COL4A1",
            "COL4A2",
            "COL4A3",
            "COL4A4",
            "COL4A5",
            "COL18A1",
            "LAMA1",
            "LAMA2",
            "LAMA4",
            "LAMA5",
            "LAMB1",
            "LAMB2",
            "LAMC1",
            "NID1",
            "NID2",
            "HSPG2",
            "SPARC",
            "FN1",
            "MMP2",
            "MMP9",
            "MMP14",
            "TIMP1",
            "TIMP2",
            "ITGA5",
            "ITGB1",
            "LOX",
            "ELN",
        },
    },
    "DR_INFLAMMATION_COMPLEMENT": {
        "category": "inflammation_complement",
        "panel_role": "dr_inflammation_pathobiology",
        "description": "Complement cascade, cytokine and innate immune activation.",
        "genes": {
            "C1QA",
            "C1QB",
            "C1QC",
            "C1R",
            "C1S",
            "C2",
            "C3",
            "C4A",
            "C4B",
            "C5",
            "C6",
            "C7",
            "C8A",
            "C8B",
            "C8G",
            "C9",
            "CFB",
            "CFH",
            "CFI",
            "CD46",
            "CD55",
            "CD59",
            "SERPING1",
            "MASP1",
            "MASP2",
            "NFKB1",
            "TNF",
            "IL1B",
            "IL6",
            "CCL2",
            "CCR2",
            "CXCL8",
            "CXCL10",
            "ICAM1",
            "VCAM1",
            "TLR4",
            "MYD88",
            "PTGS2",
        },
    },
    "DR_OXIDATIVE_STRESS_MITOCHONDRIAL_INJURY": {
        "category": "oxidative_stress_mitochondrial_injury",
        "panel_role": "dr_metabolic_stress_pathobiology",
        "description": "Oxidative stress response, antioxidant enzymes and mitochondrial injury.",
        "genes": {
            "NFE2L2",
            "KEAP1",
            "SOD1",
            "SOD2",
            "SOD3",
            "GPX1",
            "GPX3",
            "GPX4",
            "CAT",
            "PRDX1",
            "PRDX2",
            "PRDX3",
            "TXN",
            "TXNRD1",
            "HMOX1",
            "NQO1",
            "NOX1",
            "CYBB",
            "NOX4",
            "DUOX1",
            "DUOX2",
            "PPARGC1A",
            "TFAM",
        },
    },
    "DR_MULLER_GLIA_NEUROVASCULAR_UNIT": {
        "category": "muller_glia_neurovascular_unit",
        "panel_role": "dr_neurovascular_unit_pathobiology",
        "description": "Muller glia and neurovascular-unit support programs.",
        "genes": {
            "RLBP1",
            "GLUL",
            "AQP4",
            "SLC1A3",
            "SLC1A2",
            "GFAP",
            "VIM",
            "CLU",
            "SOX9",
            "CRYAB",
            "APOE",
            "ALDH1L1",
            "LRAT",
            "CA2",
            "KCNJ10",
            "GLS",
            "GLUD1",
        },
    },
    "DR_MICROGLIA_MACROPHAGE_ACTIVATION": {
        "category": "microglia_macrophage_activation",
        "panel_role": "dr_inflammation_pathobiology",
        "description": "Microglia, macrophage and phagocytic inflammatory activation.",
        "genes": {
            "AIF1",
            "CX3CR1",
            "TYROBP",
            "ITGAM",
            "CD68",
            "CSF1R",
            "C1QA",
            "C1QB",
            "C1QC",
            "TREM2",
            "APOE",
            "LPL",
            "CTSS",
            "FCGR3A",
            "MS4A7",
            "LYZ",
            "TLR2",
            "TLR4",
            "HLA-DRA",
            "HLA-DRB1",
        },
    },
    "DR_NEURONAL_PHOTORECEPTOR_STRESS": {
        "category": "neuronal_photoreceptor_stress",
        "panel_role": "dr_neuroretinal_pathobiology",
        "description": "Photoreceptor and retinal neuronal stress markers.",
        "genes": {
            "RHO",
            "OPN1LW",
            "OPN1MW",
            "OPN1SW",
            "PDE6A",
            "PDE6B",
            "GNAT1",
            "GNAT2",
            "CNGA1",
            "CNGB1",
            "RCVRN",
            "ARR3",
            "CRX",
            "NRL",
            "VSX2",
            "VSX1",
            "POU4F2",
            "NEFL",
            "NEFM",
            "SNCG",
            "THY1",
        },
    },
    "DIABETIC_LIABILITY_CONTEXT": {
        "category": "diabetic_liability_context",
        "panel_role": "shared_diabetic_liability_context",
        "description": "Metabolic and beta-cell diabetic liability context used to contrast DR-specific axes.",
        "genes": {
            "INS",
            "INSR",
            "IRS1",
            "IRS2",
            "TCF7L2",
            "SLC2A2",
            "GCK",
            "PDX1",
            "MAFA",
            "KCNJ11",
            "ABCC8",
            "HNF1A",
            "HNF4A",
            "GLP1R",
            "IGF1",
            "IGF1R",
            "AKT2",
            "MTOR",
            "PPARG",
            "SLC30A8",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def write_table(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def main() -> None:
    args = parse_args()
    tables_dir = args.out_dir / "tables"
    reports_dir = args.out_dir / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    registry_rows: list[dict[str, object]] = []
    for gene_set, spec in PANEL.items():
        genes = sorted({str(gene).upper() for gene in spec["genes"]})
        registry_rows.append(
            {
                "gene_set": gene_set,
                "module_source": "curated_dr_pathobiology_panel_v0_1",
                "annotation_source_type": "independent_external_pending_citation",
                "module_category": spec["category"],
                "panel_role": spec["panel_role"],
                "description": spec["description"],
                "n_query_genes": len(genes),
                "citation_status": "pending_manual_citation_review",
                "construction_note": (
                    "Curated from prespecified diabetic-retinopathy pathobiology axes; "
                    "intended for fixed-panel sensitivity testing before manuscript citation finalization."
                ),
            }
        )
        for gene in genes:
            rows.append(
                {
                    "gene_set": gene_set,
                    "gene_symbol": gene,
                    "module_source": "curated_dr_pathobiology_panel_v0_1",
                    "annotation_source_type": "independent_external_pending_citation",
                    "module_category": spec["category"],
                    "panel_role": spec["panel_role"],
                    "source_database": "curated_dr_pathobiology_panel_v0_1",
                    "citation_status": "pending_manual_citation_review",
                }
            )
    gene_sets = pd.DataFrame(rows).sort_values(["gene_set", "gene_symbol"]).reset_index(drop=True)
    registry = pd.DataFrame(registry_rows).sort_values("gene_set").reset_index(drop=True)
    write_table(tables_dir / "dr_specific_biology_panel_v0_1.gene_sets.tsv", gene_sets)
    write_table(tables_dir / "dr_specific_biology_panel_v0_1.registry.tsv", registry)

    retinal_only_gene_sets = gene_sets.loc[
        gene_sets["panel_role"].ne("shared_diabetic_liability_context")
    ].reset_index(drop=True)
    retinal_only_registry = registry.loc[
        registry["panel_role"].ne("shared_diabetic_liability_context")
    ].reset_index(drop=True)
    write_table(
        tables_dir / "dr_specific_biology_panel_v0_1.retinal_only.gene_sets.tsv",
        retinal_only_gene_sets,
    )
    write_table(
        tables_dir / "dr_specific_biology_panel_v0_1.retinal_only.registry.tsv",
        retinal_only_registry,
    )

    manifest = {
        "created_utc": now_utc(),
        "script_path": str(Path(__file__).resolve()),
        "output_dir": str(args.out_dir),
        "n_gene_sets": int(registry.shape[0]),
        "n_unique_genes": int(gene_sets["gene_symbol"].nunique()),
        "retinal_only_n_gene_sets": int(retinal_only_registry.shape[0]),
        "retinal_only_n_unique_genes": int(retinal_only_gene_sets["gene_symbol"].nunique()),
        "citation_status": "pending_manual_citation_review",
    }
    (reports_dir / "dr_specific_biology_panel_v0_1.manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# DR-specific biology panel v0.1",
        "",
        f"Created: {manifest['created_utc']}",
        "",
        "This fixed panel is intended to test DR pathobiology axes before any claim upgrade.",
        "All gene-set citations remain pending manual citation review.",
        "",
        "A retinal-only sensitivity file is also written by excluding the shared diabetic liability context.",
        "",
        "| Gene set | Role | Category | Genes |",
        "|---|---|---|---:|",
    ]
    for row in registry.to_dict(orient="records"):
        lines.append(
            f"| {row['gene_set']} | {row['panel_role']} | {row['module_category']} | {int(row['n_query_genes'])} |"
        )
    (reports_dir / "dr_specific_biology_panel_v0_1.report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote DR-specific biology panel to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
