# Milestone 9 — Uncertainty Quantification

Status: approved 2026-07-15
Scope reference: master plan §6 (bootstrap CI practice, Efron-Tibshirani);
addresses the workshop-outline gap "every headline number is n=1"

## Goal

Attach confidence intervals to the pilot's headline numbers so claims move from
"n=1, could be anything" to "robust across the concepts tested". Provide one
bootstrap CI function, apply it to the R8 and R9 results already in hand, and
make future runs report CIs natively.

## In scope

- `mirror/stats.py`: a single `bootstrap_ci` function.
- CIs computed for R8 (patching) and R9 (naturalistic) from existing data.
- Notebook analysis cells updated to report CIs.
- Lab-notebook entries updated with CIs and the honest limit below.

## Out of scope (deferred)

Extraction-seed variation (needs re-runs; the next milestone), FDR correction
across cells, the gamma forced-choice arm, analytic/parametric intervals.

## Why bootstrap

Nonparametric, no distributional assumption, works identically for means and
proportions, and matches both the existing `prior_null.gamma_ci` pattern and the
master plan's stated bootstrap practice.

## The statistic

One function serves all three needs; everything else is caller-side arithmetic:

- `bootstrap_ci(values, n_boot=2000, rng=None) -> (lo, hi)`: resamples `values`
  with replacement `n_boot` times, takes the mean of each resample, and returns
  the 2.5th and 97.5th percentiles. `rng` defaults to
  `numpy.random.default_rng()`.

Applications:
- R8 mean self-delta: `bootstrap_ci(self_deltas)`.
- R8 paired self-vs-control (the real test): `bootstrap_ci([s - c for s, c in zip(selfs, controls)])`;
  the effect is robust across concepts if the interval excludes 0.
- R9 identifiability vs chance: `bootstrap_ci(hits)` where `hits` are 0/1 per
  concept; the effect beats chance if the interval excludes 1/16 = 0.0625.

## The honest limit (must accompany every CI reported)

These intervals capture CONCEPT-LEVEL variance only — they answer "does this
effect hold across the concepts we tested?". They do NOT capture variance from
extraction-seed sampling, prompt paraphrase, or model choice. They therefore
upgrade the claims from "single point estimate" to "robust across concepts", and
must not be presented as fully-seeded confirmatory intervals. Note that
generation is greedy (`do_sample=False`), so generation seeds contribute no
variance at all; "more seeds" on the report side would produce identical results.

## Testing (TDD, synthetic, seeded numpy)

- Recovers a known mean: 200 samples from Normal(mu=5, sigma=1) gives an interval
  containing 5.
- Degenerate sample: all values identical gives an interval collapsing to that
  value (lo == hi == value).
- Proportion behaviour: 16 binary values with 11 ones gives an interval
  containing 0.688 and excluding 0.0625.
- Determinism: the same seed gives the same interval.

## Deliverable

CIs for the R8 and R9 headline numbers recorded in the lab notebook alongside the
point estimates and the limit statement, and notebook analysis cells that print
CIs for future runs.

## Conventions

Self-describing names, no comments/docstrings. numpy only (already present). Do
not modify the TL modules.
