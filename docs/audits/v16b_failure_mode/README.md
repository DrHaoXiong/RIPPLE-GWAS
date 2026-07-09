# RIPPLE-D V1.6b Failure-Mode Audit Package

This package is designed for focused external review of why RIPPLE-D V1.6b has not yet recovered a strong DR-specific weak-signal module narrative.

## What Is Included

- Compact derived tables only.
- No raw GWAS summary statistics.
- No LD matrices or genotype reference files.
- No full gene-score tables.
- No private absolute file paths.

## Current Empirical Pattern

1. V1.6b improves the false-negative problem relative to strict V1.6.
2. Synthetic distributed weak-signal scenarios are recovered under V1.6b.
3. DR_MVP broad GO/Reactome candidates contain some multi-locus evidence, but many are generic cell-cycle, senescence, mitochondrial, or lipid modules.
4. A prespecified retinal-only DR-specific library does not produce high-confidence V1.6b DR modules at n_null=5000.
5. The key unresolved question is whether this reflects DR_MVP biology/power, module definition, statistic design, null calibration, or implementation error.

## Files

- `tables/dr_mvp_v16b_high8_specificity_compact.tsv`
- `tables/dr_specific_library_v1_cross_context_compact.tsv`
- `tables/v16b_synthetic_validation_compact.tsv`
- `tables/v16b_type1_calibration_compact.tsv`
- `tables/v16b_sensitivity_completion_compact.tsv`
- `tables/v16b_failure_mode_questions.tsv`
- `tables/v16b_failure_mode_manifest.tsv`
- `AI_REVIEW_PROMPT.md`

## Review Goal

Do not perform a generic code review. The target is to identify mathematical or implementation choices that prevent RIPPLE-D from detecting realistic distributed weak-signal modules in real GWAS data despite synthetic recovery.
