# Pre-registration — Three-Framing Identification Battery (R12)

Filed 2026-07-22, BEFORE the run. This document freezes the predictions and the
decision rule. Its git commit timestamp is the pre-registration record. No result
existed when this was written.

## Hypothesis

The effect of telling a model about an injected concept depends on WHETHER the
framing invokes the assistant persona or supplies a task model of the
manipulation. Persona-invoking framing ("your thoughts / your mind") triggers the
scripted assistant disclaimer and does not help identification; mechanism-
explaining framing (a concept-agnostic description of the injection) helps.

## Design (frozen)

Model: Gemma-2-2B-it, 8-bit, layer 13, alpha 1.0. Concepts: the 16 dev-bank
concepts. Orders: 6 randomised option orders (seed 0). Three framings, identical
except the instruction text (neutral / introspective / informative, verbatim in
the design spec 2026-07-22-three-framing-battery-design.md). Greedy decoding.
Estimator: `mirror.prior_null.fit` with covariates [wordfreq Zipf, binary
abstractness, is_injected]; contrasts via `gamma_difference_ci` (bootstrap over
trials, 200 resamples, seed fixed in the notebook).

## Predictions (frozen)

Let gN, gIntro, gInfo be fitted gamma under neutral, introspective, informative.

- **P1:** gInfo - gN > 0, 95% CI excludes 0 positively. (Mechanism helps.)
- **P2:** gIntro - gN <= 0, 95% CI not strictly positive. (Persona does not help;
  replicates R11, where gIntro +1.99 < gN +2.57.)
- **P3 (primary):** gInfo - gIntro > 0, 95% CI excludes 0 positively. (Same
  surface act, opposite effect, conditioned on persona vs task model.)

## Decision rule (frozen)

- P1 and P2 and P3 all hold -> persona-vs-mechanism dissociation confirmed;
  reconciles R11 with Pearson-Vogel; supports H7.
- P1 null -> mechanism does not help on Gemma-2-2B; recorded as a flat negative
  and a non-replication of Pearson-Vogel's effect on this model. NOT reinterpreted
  post hoc.
- P2 violated (gIntro - gN strictly positive) -> failure to replicate our own
  R11; halt and investigate before any further claim.
- Any outcome not covered above is reported descriptively and labelled
  exploratory.

## Scope of the claim

This tests one wording of each framing on one model, one seed, 16 concepts, one
layer/strength, with proxy covariates. The framing CONTRAST is robust to the
covariate proxy (priors held fixed across framings); the gamma LEVELS are not.
A null does not establish that no mechanistic explanation helps.
