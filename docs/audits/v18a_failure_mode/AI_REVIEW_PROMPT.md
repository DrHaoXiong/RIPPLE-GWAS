# External Review Prompt: RIPPLE-D V1.8a Discrimination Bottleneck

You are reviewing the RIPPLE-GWAS public repository at the V1.8a experimental stage. Do not perform a generic repository review. Determine why the profile-mixture module test cannot yet distinguish distributed weak/moderate locus enrichment from several strong-locus artifacts on a real GWAS score background.

## Required reading

1. `docs/audits/v18a_failure_mode/README.md`
2. `docs/audits/v18a_failure_mode/tables/v18a_phase2_discriminant_summary.tsv`
3. `docs/audits/v18a_failure_mode/tables/v18a_acceptance_gate_audit.tsv`
4. `ripple/experimental/v18_mixture.py`
5. `ripple/experimental/v18a_raw_tail.py`
6. `ripple/experimental/v18a_joint.py`
7. `ripple/experimental/v175.py`
8. `ripple/modules/adaptive.py`
9. `scripts/run_v18_fixed_module_synthetic.py`

## Observed failure pattern

At 20 held-out replicates and 1,000 matched nulls per fixed-module test:

- pure-null nominal weak calls: 0%;
- five-strong-locus artifact calls: 30%;
- eight-strong-locus artifact calls: 30%;
- 15/30 effect-1.0 power: 35%;
- 8/30 effect-1.5 power: 25%;
- 15/30 effect-2.0 power: 75%.

The frozen targets are artifact false calls <=5%, 15/30 effect-1.0 power >=65%, 8/30 effect-1.5 power >=60%, and 15/30 effect-2.0 power >=85%.

## Audit objectives

Return a ranked list of blockers. For each blocker:

1. classify it as mathematical identifiability, null/exchangeability, implementation, optimization, or synthetic-design issue;
2. cite exact functions and lines or code blocks;
3. explain the predicted effect on strong-artifact false calls and moderate-signal power separately;
4. propose the smallest discriminant experiment that could falsify your diagnosis;
5. propose a repair only if the diagnosis is testable without looking at real-trait outcomes.

Explicitly check:

- whether capped evidence and binary raw-tail evidence identify separate weak and strong components;
- whether pooled position-wise rank normalization is valid and sufficiently informative;
- whether H0/H1 are numerically nested and boundary fits are handled correctly;
- whether EM monotonicity, convergence and posterior responsibilities are correct;
- whether signal injection and matched-null construction preserve the intended estimand;
- whether empirical P values use the correct plus-one, greater-is-more-extreme rule;
- whether additive injections on a real residual-score background are an adequate ground truth;
- whether continuous bivariate, orthogonalized, or semiparametric alternatives have a defensible calibration path.

## Required recommendation

Choose one:

- `IMPLEMENT_MINIMAL_REPAIR`
- `REDESIGN_THE_LIKELIHOOD`
- `REDESIGN_SYNTHETIC_GROUND_TRUTH_FIRST`
- `ABANDON_MIXTURE_DIRECTION`

State the evidence threshold required before formal 100 x 5,000 validation resumes. Do not recommend relaxing the frozen artifact or power criteria based on DR_MVP results.
