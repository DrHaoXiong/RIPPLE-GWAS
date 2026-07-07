import numpy as np
import pandas as pd

from ripple.diagnostics.score_checks import (
    gene_score_clipping_diagnostics,
    gene_score_transform_sensitivity,
    residualization_diagnostics,
)


def diagnostic_table():
    n = 20
    x = np.linspace(-2, 2, n)
    return pd.DataFrame(
        {
            "gene_symbol": [f"G{i}" for i in range(n)],
            "assoc_p_g": np.linspace(1e-320, 0.9, n),
            "assoc_p_g_clipped": np.clip(np.linspace(1e-320, 0.9, n), 1e-300, 1 - 1e-16),
            "assoc_normal_score_g": x + np.linspace(0, 1, n),
            "assoc_resid_score": x,
            "log_gene_length": np.linspace(0, 1, n),
            "log_mapped_snp_count": np.linspace(1, 2, n),
            "log_m_eff": np.linspace(2, 3, n),
            "local_ld_score": np.linspace(3, 4, n),
            "mappability": np.linspace(0.5, 1.0, n),
            "graph_degree": np.arange(n) % 5,
        }
    )


def test_gene_score_clipping_diagnostics_counts_low_clips():
    table = diagnostic_table()
    result = gene_score_clipping_diagnostics(table, trait="TEST", p_clip_min=1e-300)

    assert result.loc[0, "n_genes"] == len(table)
    assert result.loc[0, "n_clipped_low"] >= 1
    assert result.loc[0, "fraction_clipped_low"] > 0


def test_gene_score_clipping_diagnostics_counts_high_clips():
    table = diagnostic_table()
    table.loc[0, "assoc_p_g"] = 1.0
    table.loc[0, "assoc_p_g_clipped"] = 1 - 1e-16
    result = gene_score_clipping_diagnostics(table, trait="TEST", p_clip_min=1e-300, p_clip_max=1 - 1e-16)

    assert result.loc[0, "p_clip_max"] == np.nextafter(1.0, 0.0)
    assert result.loc[0, "n_clipped_high"] >= 1


def test_residualization_diagnostics_reports_before_after_correlations():
    table = diagnostic_table()
    result = residualization_diagnostics(table, trait="TEST")

    assert {"pearson_before", "spearman_after", "covariate"}.issubset(result.columns)
    assert "graph_degree" in set(result["covariate"])


def test_gene_score_transform_sensitivity_reports_correlations_and_top_overlap():
    table = diagnostic_table()
    result = gene_score_transform_sensitivity(table, trait="TEST")

    assert set(result["score_transform"]) == {"normal_score", "minuslog10p", "rank_normal"}
    assert result["top_1pct_jaccard"].between(0, 1).all()
    assert "diffusion_T_max" in result.columns
