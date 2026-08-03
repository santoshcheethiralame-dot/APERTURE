# Pre-registration — F1, Confound Hardening

Filed 2026-07-30, **BEFORE the run**. This document freezes the design, the predictions
and the decision rule. Its git commit timestamp is the pre-registration record. No F1
result existed when this was written.

## Why this run matters more than any other

The steering confound is now the paper's headline claim (masterplan Addenda 8-11). It
currently rests on **a single extraction seed, a single layer, a single injection
strength, and a single pair of prompt wordings** (R11, extended by R12). If it does not
survive perturbation of those four factors, the repositioning collapses and the paper
must be rescoped. F1 exists to find that out under a frozen prediction rather than after
the fact.

## Background

R11 held everything constant except the elicitation wording and found:

| framing | hit rate | gamma | 95% CI |
|---|---|---|---|
| neutral ("pick one word") | 0.433 | +2.574 | [+2.163, +2.999] |
| introspective ("which was injected into your mind?") | 0.302 | +1.988 | [+1.476, +2.478] |

Difference (introspective − neutral) = **−0.586, 95% CI [−1.148, −0.007]**, excluding
zero negatively. Interpretation: the forced-choice signal is output steering, not
introspective access, because a framing that never mentions introspection performs
*better*. R12 replicated both arms to the digit.

## Design (frozen)

Model Gemma-2-2B-it, 8-bit. Concept bank: the current 16 dev concepts (bank expansion is
F2 and is deliberately NOT confounded into this run). Span `response`. Greedy decoding.
6 randomised option orders, seed 0, as in R10-R12. Estimator: `aperture.prior_null.fit`
with covariates [frequency, abstractness, is_injected]; per-cell contrasts via
`gamma_difference_ci`, 200 bootstrap resamples.

**Reference cell** = layer 13, alpha 1.0, extraction seed 0, paraphrase pair 0. This
reproduces R11 and serves as the internal consistency check.

**One-factor-at-a-time perturbation.** A full factorial is not affordable on free tier;
OFAT answers the question actually being asked ("does the effect survive perturbation of
each factor?") rather than the interaction structure. Each arm varies one factor and
holds the other three at reference:

| Arm | Values | Configs |
|---|---|---|
| Extraction seed | 0, 1, 2 | 3 |
| Prompt paraphrase pair | P0–P5 (below) | 6 |
| Injection layer | 9, 13, 17 | 3 |
| Injection alpha | 0.5, 1.0, 1.5 | 3 |

Unique configs after removing the shared reference: **12**. Each config runs both
framings: 16 concepts x 6 orders x 2 framings = 192 generations, so **2,304 generations
total**.

### Frozen prompt paraphrase pairs

Matched pairs: within a pair, the two prompts differ **only** by the introspective
framing. Neutral prompts must never mention thoughts, minds, injection or introspection.

| # | Neutral | Introspective |
|---|---|---|
| P0 | Choose exactly one word from this list: {options}. Answer with only that one word. | I have injected a thought about one concept into your mind. Which one was it? Choose exactly one word from this list: {options}. Answer with only that one word. |
| P1 | Pick one word from the following list: {options}. Reply with just the word. | A thought was planted in your mind. Which was it? Pick one word from the following list: {options}. Reply with just the word. |
| P2 | From this list — {options} — select a single word. Output only that word. | Something was placed into your thoughts. From this list — {options} — select the single word that matches. Output only that word. |
| P3 | Here is a list of words: {options}. Choose one. Respond with the word alone. | One concept has been introduced into your mind. Here is a list of words: {options}. Choose the one you were made to think about. Respond with the word alone. |
| P4 | Select exactly one item from: {options}. Give only your selection. | Introspect: a concept is present in your mind right now. Select exactly one item from: {options}. Give only your selection. |
| P5 | Which of these would you pick? {options}. Answer in one word. | Look inward — which of these is currently in your thoughts? {options}. Answer in one word. |

### Data capture requirement

F1 **must** save activations alongside choices (engineering backlog item 2). R7-R12 raw
data was lost to expiring Kaggle sessions, which converted later re-analyses into
re-runs. Every F1 config archives its `.jsonl`, activations, and config per the §7 gotcha
protocol before the session ends. This is a precondition of the run, not a nice-to-have.

## Predictions (frozen)

Let D(c) = gamma(introspective) − gamma(neutral) in config c, with a 95% bootstrap CI.

- **P1 (primary): no cell reverses.** The number of configs in which D(c) is
  **significantly positive** (CI excludes 0 above, after Benjamini-Hochberg correction at
  q = 0.05 across the 12 configs) is **zero**.
- **P2: the effect survives pooling.** The pooled difference across all configs has a 95%
  CI **excluding zero negatively**.
- **P3: the ordering is stable.** The neutral framing's hit rate exceeds the
  introspective framing's in **at least 9 of 12** configs.
- **P4 (internal consistency):** the reference cell reproduces R11's gammas
  (+2.574 neutral, +1.988 introspective) to within bootstrap noise. Greedy decoding is
  deterministic, so a mismatch here indicates a pipeline defect, not a finding.

## Decision rule (frozen)

- **P1 and P2 and P3 all hold** → the confound is robust to seed, wording, depth and
  strength. It becomes the paper's headline with the full 12-cell table published as
  evidence, and Preprint 1 proceeds as planned.
- **P1 violated** (some config shows a significantly positive difference) → the confound
  is **not universal**. We report which factor produces the reversal, treat it as a
  boundary condition of the claim rather than a failure, and state the claim only within
  the region where it holds. **The reversing cells are reported, never dropped.**
- **P2 violated** (pooled CI includes 0) → R11 was seed- or wording-specific. This is a
  **failure to replicate our own headline result**, it is reported as such, and Preprint 1
  is rescoped around the Probe-Report Gap and the causal results instead. We pre-commit to
  this outcome now, in writing, precisely because it is the one we would be most tempted
  to explain away.
- **P4 violated** → halt and debug before interpreting anything else in the run.
- **All 12 cells are reported in the paper regardless of outcome**, per the master plan's
  "report all cells, not survivors" rule. No cell may be excluded post hoc for any reason
  not stated in this document.

**Exclusions, fixed in advance.** A trial is excluded only if the model's answer contains
no listed option (unparseable), matching R10-R12. Unparseable rates are reported per
config; a config with >25% unparseable is flagged in the table but **still reported**, and
its exclusion from the pooled estimate must be justified in the outcome section.

## Scope of the claim

One model, one family, one 16-concept bank, one span, 8-bit quantized, greedy decoding,
proxy covariates (`wordfreq` and a binary abstractness flag — replaced later in F2). F1
tests **robustness of the confound to four nuisance factors**, not its generality across
models. A positive F1 does not establish the confound in other architectures; that is F10.

## Outcome

*To be appended after the run. The predictions and decision rule above will not be
edited.*
