#!/usr/bin/env python
"""Audit V1.5 ready modules for strong-signal and biological-theme patterns."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DIABETES_GENES = {
    "ABCC8",
    "ADCY5",
    "ADAMTS9",
    "ARAP1",
    "BCL11A",
    "CDKAL1",
    "CDKN2A",
    "CDKN2B",
    "CENTD2",
    "CCND2",
    "DGKB",
    "FTO",
    "GCK",
    "GCKR",
    "GLIS3",
    "HHEX",
    "HMGA2",
    "IGF2BP2",
    "INS",
    "IRS1",
    "IRS2",
    "JAZF1",
    "KCNJ11",
    "KCNQ1",
    "MTNR1B",
    "NOTCH2",
    "PPARG",
    "SLC30A8",
    "TCF7L2",
    "THADA",
    "WFS1",
}

DR_MICROVASCULAR_GENES = {
    "ACE",
    "AGER",
    "ANGPT1",
    "ANGPT2",
    "APOE",
    "ARMS2",
    "C3",
    "C5",
    "CD34",
    "CDH2",
    "CDH5",
    "CFB",
    "CFH",
    "CFI",
    "CLDN5",
    "COL18A1",
    "COL4A1",
    "COL4A2",
    "CTGF",
    "ENG",
    "EPAS1",
    "ERBB3",
    "FLT1",
    "FLT4",
    "FN1",
    "HIF1A",
    "HTRA1",
    "ICAM1",
    "IGF1",
    "IGF2",
    "IL6",
    "KDR",
    "LAMA1",
    "LAMB1",
    "LAMC1",
    "MMP2",
    "MMP9",
    "NOS3",
    "PDGFB",
    "PDGFRB",
    "PECAM1",
    "PTPN22",
    "RGS5",
    "SERPINE1",
    "SH2B3",
    "SMAD2",
    "SMAD3",
    "SMAD4",
    "TEK",
    "TGFB1",
    "TGFB2",
    "TGFBR1",
    "TGFBR2",
    "TGFBR3",
    "TIMP3",
    "VCAM1",
    "VEGFA",
    "VWF",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ready-modules", type=Path, required=True)
    parser.add_argument("--module-tests", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--out-table", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    return parser.parse_args()


def parse_genes(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [gene.strip().upper() for gene in str(value).split(",") if gene.strip()]


def theme_from_name(name: str, diabetes_overlap: set[str], dr_overlap: set[str]) -> tuple[str, str]:
    upper = name.upper()
    if any(key in upper for key in ["ANGIO", "VASC", "ENDOTHEL", "COMPLEMENT", "COAGUL", "CELL_ADHESION"]):
        return "dr_microvascular_direct", "vascular/endothelial/complement keyword"
    if any(key in upper for key in ["TGF", "JAK_STAT", "NFE2L2", "KEAP1", "AUTOPHAGY", "APOPTOTIC", "INFLAM", "IMMUNE"]):
        return "dr_context_plausible", "stress/inflammation/TGF/autophagy keyword"
    if any(key in upper for key in ["SYNAPTIC", "NEURON", "AXON", "EYE", "RETINA"]):
        return "retinal_neural_plausible", "neural/eye keyword"
    if any(key in upper for key in ["INSULIN", "GLUCOSE", "ADIPOCYTE", "CHOLESTEROL", "LIPID", "WNT", "PI3K"]):
        return "diabetes_or_metabolic", "metabolic/diabetes keyword"
    if any(key in upper for key in ["CELL_CYCLE", "MITOTIC", "CENTROSOME", "SENESCENCE", "RB1", "DNA_REPAIR"]):
        return "generic_proliferation_or_genome_maintenance", "cell-cycle/DNA-repair keyword"
    if dr_overlap:
        return "dr_context_gene_overlap", "contains DR/microvascular prior genes"
    if diabetes_overlap:
        return "diabetes_gene_overlap", "contains diabetes prior genes"
    return "generic_cellular_process", "no direct DR/diabetes keyword"


def summarize_module_genes(module_genes: list[str], score_map: dict[str, float]) -> dict[str, object]:
    scored = [(gene, float(score_map.get(gene, np.nan))) for gene in module_genes]
    scored = [(gene, score) for gene, score in scored if np.isfinite(score)]
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    high_5 = [gene for gene, score in ranked if score >= 5.0]
    high_3 = [gene for gene, score in ranked if score >= 3.0]
    moderate = [gene for gene, score in ranked if 1.0 <= score < 3.0]
    positive = [gene for gene, score in ranked if score > 0.0]
    return {
        "n_scored_genes": len(scored),
        "n_extreme_score_ge5": len(high_5),
        "n_upper_tail_score_ge3": len(high_3),
        "n_moderate_score_1_to_3": len(moderate),
        "n_positive_score_gt0": len(positive),
        "fraction_upper_tail_score_ge3": len(high_3) / len(scored) if scored else np.nan,
        "top10_module_genes_by_score": ";".join(f"{gene}:{score:.3g}" for gene, score in ranked[:10]),
        "extreme_score_ge5_genes": ",".join(high_5),
        "upper_tail_score_ge3_genes": ",".join(high_3),
    }


def plain_table(frame: pd.DataFrame) -> str:
    return frame.to_string()


def main() -> None:
    args = parse_args()
    ready = pd.read_csv(args.ready_modules, sep="\t")
    modules = pd.read_csv(args.module_tests, sep="\t")
    scores = pd.read_csv(args.score_file, sep="\t", compression="infer")
    scores["gene_symbol"] = scores["gene_symbol"].astype(str).str.upper()
    score_map = dict(zip(scores["gene_symbol"], pd.to_numeric(scores["assoc_resid_score"], errors="coerce"), strict=False))
    module_map = modules.set_index("module_name", drop=False)

    rows: list[dict[str, object]] = []
    for row in ready.itertuples(index=False):
        module_name = str(row.module_name)
        module_row = module_map.loc[module_name]
        genes = parse_genes(module_row.present_genes)
        diabetes_overlap = set(genes) & DIABETES_GENES
        dr_overlap = set(genes) & DR_MICROVASCULAR_GENES
        theme, reason = theme_from_name(module_name, diabetes_overlap, dr_overlap)
        gene_summary = summarize_module_genes(genes, score_map)
        rows.append(
            {
                "module_name": module_name,
                "biological_theme": theme,
                "theme_reason": reason,
                "n_present": int(row.n_present),
                "n_loci": float(row.n_loci),
                "n_effective_loci": float(row.n_effective_loci),
                "top1_gene": row.top1_gene,
                "top1_gene_score": float(row.top1_gene_score),
                "top1_locus_contribution": float(row.top1_locus_contribution),
                "top5_locus_contribution": float(row.top5_locus_contribution),
                "ripple_d_empirical_p": float(row.ripple_d_empirical_p),
                "module_specific_rank_empirical_p": float(row.module_specific_rank_empirical_p),
                "leave_top1_locus_empirical_p": float(module_row.leave_top1_locus_empirical_p),
                "moderate_locus_burden_empirical_p": float(module_row.moderate_locus_burden_empirical_p),
                "diabetes_prior_gene_overlap_n": len(diabetes_overlap),
                "diabetes_prior_gene_overlap": ",".join(sorted(diabetes_overlap)),
                "dr_microvascular_prior_gene_overlap_n": len(dr_overlap),
                "dr_microvascular_prior_gene_overlap": ",".join(sorted(dr_overlap)),
                **gene_summary,
            }
        )
    out = pd.DataFrame(rows)
    args.out_table.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_table, sep="\t", index=False)

    theme_counts = out["biological_theme"].value_counts(dropna=False)
    top_gene_counts = out["top1_gene"].value_counts(dropna=False).head(20)
    strong_summary = out[
        [
            "n_extreme_score_ge5",
            "n_upper_tail_score_ge3",
            "n_moderate_score_1_to_3",
            "top1_locus_contribution",
            "top5_locus_contribution",
            "n_effective_loci",
        ]
    ].describe()
    report = [
        "# V1.5 ready module biology audit",
        "",
        f"Ready modules audited: {len(out)}",
        "",
        "## Theme counts",
        "",
        plain_table(theme_counts.to_frame("n")),
        "",
        "## Top-gene counts",
        "",
        plain_table(top_gene_counts.to_frame("n")),
        "",
        "## Strong-signal summary",
        "",
        plain_table(strong_summary),
        "",
        f"Detailed table: `{args.out_table}`",
    ]
    args.out_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
