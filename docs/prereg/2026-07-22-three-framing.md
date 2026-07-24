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

---

## OUTCOME (appended 2026-07-22, after the run — predictions above UNEDITED)

Run R12. Internal consistency check passed: neutral gamma +2.574 and
introspective +1.988 reproduce R10/R11 exactly (greedy decoding).

Fitted gamma: neutral +2.574 [+2.163, +2.999], introspective +1.988
[+1.476, +2.478], informative +1.645 [+1.117, +2.152]. Hit rates 0.433 / 0.302 /
0.247. Unparseable 6 / 0 / 15 (of 96).

Scored against the frozen rule:
- **P1 FALSIFIED.** Predicted gInfo - gN > 0; observed -0.930, CI [-1.757,
  -0.250], excluding 0 NEGATIVELY. Mechanism-explaining did not help; it hurt.
- **P2 HELD.** Predicted gIntro - gN <= 0; observed -0.586, CI [-1.148, -0.007].
  Replicates R11.
- **P3 (primary) FALSIFIED.** Predicted gInfo - gIntro > 0; observed -0.343, CI
  [-1.054, +0.426], includes 0 (and the point estimate is negative).

Verdict: the pre-registered persona-vs-mechanism dissociation does NOT hold. Both
framings lowered gamma relative to neutral, and the mechanistic framing lowered
it most. The primary hypothesis is falsified.

EXPLORATORY (not predicted; must be re-tested before it is a claim): the three
framings are monotonic, neutral > introspective > informative — the more the
prompt directs attention to introspection or the injection, the lower the
identification. Consistent with "any instruction to introspect degrades the pure
steering signal; none unlock access", which strengthens the confabulation
account. Treated as exploratory per the frozen rule.

NON-REPLICATION OF PEARSON-VOGEL. arXiv:2602.20031 reported informative framing
raising injection sensitivity 0.3% -> 39.9% on Qwen-32B; we observe the opposite
sign on Gemma-2-2B. Candidate explanations, none yet tested: (a) scale (32B vs
2B); (b) their detection task differs from our closed-list forced choice; (c)
their briefing was more elaborate than our single-sentence mechanistic framing;
(d) our informative framing raised refusals (15/96 unparseable), biasing its
usable subset. This tension is a reportable pre-registered result and directly
motivates re-running the informative arm at larger scale (PES Titan / IISc)
once available.
