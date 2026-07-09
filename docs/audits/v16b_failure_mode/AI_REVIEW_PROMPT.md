# AI Review Prompt: RIPPLE-D V1.6b Failure-Mode Audit

You are reviewing the RIPPLE-GWAS public repository, focusing on RIPPLE-D V1.6b. Your task is not to give a generic code review. Your task is to identify what mathematical, statistical, or implementation choices may be preventing RIPPLE-D from supporting a strong distributed weak-signal module narrative in real DR GWAS data.

## Repository Context

RIPPLE-GWAS is intended to test graph-domain aggregation and module-level distributed weak-signal patterns from GWAS summary statistics. The V1.6b layer attempts to rescue weak-signal module detection using locus-aware score capping, locus collapse, contribution dispersion diagnostics, leave-top-locus tests, matched locus nulls, multiplicity correction, annotation sensitivity, and pseudo-window/external locus sensitivity.

## Empirical Problem

Observed pattern:

1. Synthetic distributed 8-locus and 15-locus scenarios pass V1.6b high-confidence behavior.
2. Top-locus artifacts are mostly blocked or downgraded.
3. DR_MVP broad GO/Reactome candidates produce some V1.6b high-confidence signals after sensitivity completion.
4. However, many broad candidates are generic cell-cycle/senescence/mitochondrial/lipid modules.
5. A prespecified retinal-only DR-specific library produces no V1.6b high-confidence DR module at n_null=5000.
6. The best DR-specific library signals are only exploratory: oxidative stress/mitochondrial injury and ECM remodeling.

The central question is:

Why can V1.6b recover distributed synthetic signal but fail to produce strong DR-specific weak-signal module evidence in real DR_MVP?

## Code Areas To Audit

Prioritize:

- `ripple/modules/distributed.py`
- `scripts/run_v16_ripple_d_module_rescue.py`
- `scripts/run_v16_ripple_d_synthetic_validation.py`
- `scripts/run_v16_null_matching_sensitivity.py`
- `scripts/run_v16_null_matching_type1_calibration.py`
- `scripts/run_v16b_candidate_sensitivity_completion.py`
- `scripts/build_dr_specific_library_v1.py`
- `scripts/summarize_dr_specific_library_v1_v16b.py`
- `scripts/summarize_v16b_high8_specificity.py`

## Specific Questions

1. Does `T_V1.6` still behave like a pathway burden statistic rather than a distributed weak-signal module statistic?
2. Does using `max(capped gene score)` per locus discard true moderate polygenic signal within loci?
3. Are top1/top5 penalties, effective-loci penalties, leave-top-k gates, and top-conditioned nulls jointly over-conservative?
4. Does the matched-locus null overmatch biology by conditioning on degree/property/annotation/locus structure too strongly?
5. Is full-library BH-FDR the right multiplicity correction for a structured gene-set hierarchy, or should source-family/hierarchical FDR or max-null be used?
6. Are synthetic scenarios too idealized compared with real pathways that contain many passenger genes and sparse causal loci?
7. Does residualizing DR against T2D/BMI remove real DR biology as well as confounding?
8. Are binary gene-set memberships too crude, and should RIPPLE-D use weighted retinal/FVM/cell-type module membership?
9. Are pseudo-window and external LD-block sensitivity tests aligned with the observed module definition, or do they introduce an inconsistent gate?
10. Is there any implementation bug in empirical P values, plus-one formula, directionality, q-value denominator, null reuse, replacement, or annotation matching?

## Expected Output

Please return:

1. A ranked list of likely blockers.
2. For each blocker, state whether it is a mathematical design issue, statistical calibration issue, implementation bug, or data/phenotype limitation.
3. Point to exact code functions or scripts.
4. Suggest minimal diagnostic experiments to distinguish false-negative overcorrection from true lack of DR-specific signal.
5. Recommend V1.7 changes only if justified by the evidence.

Avoid overclaiming. The goal is to find the bottleneck, not to force a positive result.
