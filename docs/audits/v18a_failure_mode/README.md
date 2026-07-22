# RIPPLE-D V1.8a Failure-Mode Audit

## Executive conclusion

RIPPLE-D V1.8a remains an experimental mathematical reconstruction. The code is numerically stable in the current smoke and medium-scale runs, but the current joint capped-score/raw-tail profile mixture does not pass the pre-specified Phase 2 gate. It therefore must not be used for real-trait claims or described as calibrated weak-signal module discovery.

The principal bottleneck is discrimination, not optimizer failure. On a real DR_MVP score background, the model must distinguish a distributed weak/moderate component from several strong loci after matched-locus conditioning. The current binary raw-tail channel is too coarse: it does not adequately prevent strong-locus artifacts from entering the weak component, while weak/moderate power remains below the frozen targets.

## Scope of this public audit

Included:

- V1.8 finite-grid profile-mixture implementation.
- V1.8a hard-tail and joint soft-tail diagnostics.
- Fixed-module synthetic runner and focused tests.
- Aggregate synthetic results and pre-specified gate decisions.
- A targeted external-review prompt.

Excluded:

- Raw GWAS summary statistics.
- Full gene-score tables.
- Genotype or LD-reference data.
- Large null matrices and per-replicate private outputs.
- Private absolute paths.

The real-background score input is not redistributed. Its SHA-256 is `98c7eda008af9a74caeed84b4e41b6459d4ca9bf98292b6367ea5d9b2d55a025`. The registered broad GO/Reactome library fingerprint is `b761144289ce4b8de859ec372635f4101d8f5c62b7920c399c83237bd794c1ef`; its source-file SHA-256 is `effdcdf8e0ac112330b1393315a5e9082db5eeecc16dc50089a099d7778a5a72`.

## Frozen V1.8 target

V1.8 tests whether an anchored module contains a weak/moderate-enriched locus component after allowing a strong/outlier component:

\[
H_0: \pi_W=0,\ \pi_S\geq 0
\]

\[
H_1: \pi_W>0,\ \pi_S\geq 0.
\]

The original V1.8 implementation rank-normalizes capped locus scores against matched null rows, then fits fixed null, weak and strong densities with deterministic EM. The primary statistic is the profile likelihood-ratio statistic for weak evidence given a fitted strong component.

V1.8a explored two minimal repairs:

1. `v18a_raw_tail`: hard exclusion or conditioning using uncapped locus tails. This reduced strong-artifact leakage but removed too much moderate signal.
2. `v18a_joint`: a capped-score weak channel plus a binary uncapped raw-tail strong channel. H1 is warm-started from H0 and numerical nesting is enforced.

## Failure localization chronology

### 1. Original V1.8

The score cap erased much of the magnitude distinction between moderate and very strong loci. Strong-artifact scenarios were consequently assigned to the weak component. An early EM stopping defect was found and fixed: convergence now requires at least two iterations and profile-likelihood stability. Re-running after the fix showed that the major artifact behavior remained, so early stopping was not the root cause.

### 2. V1.8a hard-tail repair

An absolute uncapped-score tail exclusion suppressed strong-artifact calls, but it also removed moderate distributed evidence. A conditional rank-based tail threshold did not reliably identify synthetically strengthened loci against heterogeneous matched-locus backgrounds. This repair was rejected as over-correcting.

### 3. V1.8a joint repair

The joint model improved one-replicate directional behavior, but the medium-scale paired experiment still failed both sides of the discrimination target. Pure-null behavior was controlled in 20 replicates, yet strong-artifact distributed calls remained at 30%, and moderate distributed-signal power was 25-35% for two central scenarios.

## Current quantitative evidence

The current discriminant run used 20 held-out replicates per scenario, 1,000 matched nulls per fixed-module test, four workers and a real DR_MVP score background. Total runtime was 204.3 seconds. Full values are in `tables/v18a_phase2_discriminant_summary.tsv`.

| Scenario | Intended role | Nominal weak pass rate | Frozen target | Decision |
|---|---|---:|---:|---|
| pure null | null control | 0.00 | <=0.05 | provisional pass |
| five strong loci | artifact control | 0.30 | <=0.05 | fail |
| eight strong loci | artifact control | 0.30 | <=0.05 | fail |
| 15/30 loci, effect 1.0 | moderate power | 0.35 | >=0.65 | fail |
| 8/30 loci, effect 1.5 | moderate power | 0.25 | >=0.60 | fail |
| 15/30 loci, effect 2.0 | strong distributed power | 0.75 | >=0.85 | fail |

These are localization estimates, not final Type I or power estimates. Their Monte Carlo resolution is insufficient for release claims, but the direction and magnitude are enough to reject progression to the planned 100-replicate by 5,000-null Phase 2 run without another mathematical repair.

## Most likely mathematical bottleneck

The model currently tries to identify weak versus strong evidence from two imperfect summaries:

- a capped, position-wise conditional rank-normal score;
- a binary indicator that the uncapped score exceeds an absolute threshold.

This creates overlapping component likelihoods on a heterogeneous real background:

1. Capping removes strong-signal magnitude from the weak channel.
2. Position-wise rank normalization can make an absolutely strong injected locus non-extreme relative to its matched position.
3. The binary raw-tail channel discards the amount of excess above the threshold.
4. An absolute hard-tail rule catches some strong loci but also deletes moderate evidence.
5. The weak component can absorb residual strong-locus evidence because the fitted strong channel is under-informed.

The present evidence therefore supports a statistic-identifiability problem rather than a plus-one P-value, BH-direction, or EM-convergence bug.

## Code map for review

- `ripple/experimental/v18_mixture.py`: original fixed-grid profile mixture, EM, posterior summaries and matched-null wrapper.
- `ripple/experimental/v18a_raw_tail.py`: hard and conditional raw-tail repairs.
- `ripple/experimental/v18a_joint.py`: current capped-score/raw-tail joint likelihood.
- `ripple/experimental/v175.py`: conditional rank-normalization inherited by V1.8.
- `ripple/modules/adaptive.py`: matched-locus context, observed collapse and null sampling.
- `ripple/modules/distributed.py`: locus construction/collapse and shared sampling utilities.
- `scripts/run_v18_fixed_module_synthetic.py`: exact-locus outer permutation, signal injection and paired synthetic execution.
- `ripple/config/v18_profile_mixture_rc.json`: frozen V1.8 RC configuration and claim boundary.

## High-priority external-review questions

1. Is the weak/strong mixture identifiable after capping and position-wise rank normalization?
2. Should the strong channel use continuous uncapped excess rather than a binary threshold?
3. Should the null density be estimated conditionally by matched-locus stratum instead of fixed as standard normal after pooled ranks?
4. Would orthogonalizing capped evidence against uncapped tail magnitude yield a valid weak-given-strong score?
5. Does additive injection into real residual scores create ambiguous strong scenarios, and should the validation instead simulate from an explicit generative mixture?
6. Are fixed location mixtures adequate, or is a constrained semiparametric density-ratio/local-FDR model required?
7. Does fitting all loci as exchangeable ignore locus-specific measurement uncertainty and null heterogeneity?
8. Are the matched null rows genuinely exchangeable after within-locus subset sampling and conditional rank transformation?
9. Is H1 warm-starting/nesting implemented correctly at boundary solutions?
10. What minimal redesign can jointly attain artifact false-call <=0.05 and the frozen moderate-power targets without using real-trait outcomes for tuning?

## Candidate next directions, not implemented

- Continuous bivariate evidence: model capped conditional evidence jointly with uncapped excess above its matched null distribution.
- Cross-fitted conditional empirical Bayes: learn null/strong nuisance densities from null draws and test a weak component on held-out rows.
- Orthogonalized weak evidence: residualize capped evidence against continuous tail strength within matching strata before module aggregation.
- Locus-specific measurement model: incorporate the uncertainty and discreteness of each locus score rather than treating transformed loci as identically measured.
- Generative validation redesign: simulate weak and strong loci from frozen component distributions in addition to additive score injections.

Any replacement must preserve observed/null pipeline identity, plus-one empirical calibration and held-out nested Type I validation. No real-trait result should be used to select the replacement model.

## Current decision

`STOP_BEFORE_FORMAL_PHASE2_AND_REAL_TRAITS`

V1.8/V1.8a should remain in the experimental namespace. The next work item is a minimal mathematical discrimination repair followed by the same paired synthetic gate. Formal 100 x 5,000 validation, nested Type I calibration and DR_MVP full-library analysis remain blocked until the artifact and power criteria are met.
