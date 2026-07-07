"""Schema constants and validation helpers for manuscript-ready RIPPLE outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

COMMON_RESULT_SCHEMA: tuple[str, ...] = (
    "trait",
    "analysis_id",
    "graph_id",
    "score_stream",
    "null_type",
    "statistic_name",
    "statistic_direction",
    "observed_value",
    "null_mean",
    "null_sd",
    "z",
    "empirical_p",
    "n_null",
    "threshold",
    "claim_tier",
    "claim_status",
    "exclusion_or_na_reason",
    "not_tested_reason",
    "source_result_path",
    "script_path",
    "seed",
    "timestamp",
)

FIGURE_SOURCE_SCHEMA: tuple[str, ...] = (
    "figure_id",
    "panel_id",
    "plot_type",
    "trait",
    "analysis_id",
    "graph_id",
    "x_value",
    "y_value",
    "group",
    "label",
    "error_low",
    "error_high",
    "source_table",
    "source_result_path",
    "script_path",
    "timestamp",
)

FIGURE_CLAIM_MAP_SCHEMA: tuple[str, ...] = (
    "figure_id",
    "panel_id",
    "result_subheading",
    "claim_id",
    "claim_strength",
    "source_table",
    "source_result_path",
    "script_path",
    "missing_validation",
    "allowed_language",
    "banned_language",
)

INFERENCE_FAMILY_SCHEMA: tuple[str, ...] = (
    "family_id",
    "analysis_role",
    "trait",
    "analysis_id",
    "graph_id",
    "score_stream",
    "null_type",
    "statistic_name",
    "grid_or_selection",
    "error_control_scope",
    "claim_tier",
    "allowed_claim_strength",
    "source_table",
    "source_result_path",
    "notes",
)

PARAMETER_TABLE_SCHEMA: tuple[str, ...] = (
    "analysis_id",
    "trait",
    "parameter_name",
    "parameter_value",
    "parameter_type",
    "scope",
    "default_or_override",
    "source_config_or_script",
    "notes",
)

INPUT_CHECKSUM_SCHEMA: tuple[str, ...] = (
    "input_id",
    "file_path",
    "file_type",
    "data_role",
    "genome_build",
    "source",
    "version",
    "checksum_algorithm",
    "checksum",
    "file_size_bytes",
    "last_modified",
    "redistributable",
    "notes",
)

REPRODUCIBILITY_MANIFEST_SCHEMA: tuple[str, ...] = (
    "analysis_id",
    "trait",
    "run_stage",
    "script_path",
    "command",
    "working_directory",
    "conda_or_python_env",
    "software_versions",
    "random_seed",
    "input_ids",
    "output_paths",
    "log_paths",
    "runtime_seconds",
    "run_status",
    "timestamp",
)

CLAIM_EVIDENCE_SCHEMA: tuple[str, ...] = (
    "claim_id",
    "manuscript_sentence",
    "allowed_strength",
    "required_tables",
    "required_figures",
    "source_files",
    "code_path",
    "pass_fail",
    "banned_language_check",
)

GRAPH_REGISTRY_SCHEMA: tuple[str, ...] = (
    "graph_id",
    "graph_name",
    "source",
    "version",
    "construction_script",
    "construction_date",
    "node_count",
    "edge_count",
    "lcc_node_count",
    "lcc_edge_count",
    "edge_weight_rule",
    "expression_filter_rule",
    "constructed_before_final_dr_mvp",
    "graph_role",
    "allowed_claim_level",
)

GENE_ID_MAPPING_SCHEMA: tuple[str, ...] = (
    "input_symbol",
    "mapped_ensembl_id",
    "mapped_hgnc_symbol",
    "mapping_status",
    "mapping_source",
    "mapping_version",
    "drop_reason",
)

PUBLIC_RELEASE_AUDIT_SCHEMA: tuple[str, ...] = (
    "file_path",
    "file_type",
    "contains_raw_gwas",
    "contains_private_path",
    "contains_licensed_data",
    "contains_individual_level_data",
    "redistributable",
    "public_release_action",
)

REGION_EXCLUSION_SCHEMA: tuple[str, ...] = (
    "region_id",
    "region_name",
    "genome_build",
    "chromosome",
    "start_bp",
    "end_bp",
    "reason",
    "default_action",
    "source",
    "notes",
)

TYPE1_UNCERTAINTY_SCHEMA: tuple[str, ...] = (
    "scenario",
    "z_threshold",
    "claim_tier",
    "fpr",
    "n_outer",
    "false_positive_count",
    "binomial_95ci_low",
    "binomial_95ci_high",
    "mc_se",
)

MODULE_CLAIM_POLICY_SCHEMA: tuple[str, ...] = (
    "module_status",
    "tier",
    "allowed_language",
    "forbidden_language",
    "minimum_statistical_requirement",
    "controls_selection_process",
    "controls_topology_specificity",
    "manuscript_role",
    "allowed_figure_role",
    "upgrade_requirement",
    "downgrade_condition",
)


def ensure_columns(table: pd.DataFrame, schema: Sequence[str]) -> pd.DataFrame:
    """Return a copy with all schema columns present and ordered first."""

    out = table.copy()
    for col in schema:
        if col not in out.columns:
            out[col] = ""
    extra = [col for col in out.columns if col not in schema]
    return out.loc[:, list(schema) + extra]


def validate_columns(table: pd.DataFrame, schema: Sequence[str], *, table_name: str) -> None:
    """Raise if any required schema column is absent."""

    missing = [col for col in schema if col not in table.columns]
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def validate_vocabulary(
    table: pd.DataFrame,
    vocab_by_column: Mapping[str, set[str]],
    *,
    table_name: str,
) -> None:
    """Validate controlled vocabulary columns, ignoring blank cells."""

    for col, allowed in vocab_by_column.items():
        if col not in table.columns:
            continue
        values = {str(value) for value in table[col].dropna().unique() if str(value) != ""}
        invalid = sorted(values - allowed)
        if invalid:
            raise ValueError(f"{table_name}.{col} has invalid values: {invalid}")
