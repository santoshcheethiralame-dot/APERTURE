# Milestone 3 — Prior-Guessing Null Model (B4, estimator)

Status: approved 2026-07-13
Scope reference: build guide B4; master plan H1/H2, RQ1

## Goal

Build and prove the gamma access-parameter estimator: the softmax choice model
that predicts which concept a model reports, with gamma measuring the effect of
a concept being the injected one beyond frequency/concreteness/similarity
priors. gamma is the paper's confirmatory statistic (H1 gamma~0 = confabulation,
H2 gamma>0 = access). This milestone builds the estimator and validates it on
simulation with known ground truth; it does not touch real data feeds.

## The statistical object

For trial t, one concept c from the bank is reported:

P(report = c | t) proportional to exp(
    b1 * log_freq(c) + b2 * concreteness(c)
    + b3 * sim(c, prompt_t) + b4 * sim(c, context_t)
    + gamma * 1[c = injected_t] )

A softmax choice model over concepts. gamma is the coefficient on the
injected-identity indicator, estimated after the prior covariates are
accounted for.

## In scope

- `prior_null` module: negative log-likelihood, fit, simulation, bootstrap CI.
- Simulation-based validation (the estimator recovers known coefficients).
- pytest suite, CPU, seeded numpy. New dependency: scipy.

## Out of scope (deferred)

Live infini-gram frequency calls, Brysbaert concreteness data, embedding-based
similarities, fitting on real Gemma runs, FDR correction across cells, and the
PyMC hierarchical version (B14). The estimator is data-source-agnostic: it
takes a feature array and never fetches anything.

## Data shapes

- `X`: numpy array `[n_trials, n_concepts, n_features]`. Each row
  `X[t, c]` is the feature vector for candidate concept c in trial t:
  `[log_freq, concreteness, sim_prompt, sim_context, is_injected]`. The caller
  fills these columns; the estimator is agnostic to their source.
- `y`: numpy int array `[n_trials]`, the index of the reported concept.
- `theta`: numpy array `[n_features]`, the fitted coefficients; `gamma` is the
  coefficient on the `is_injected` column, which by convention is the last
  feature.

## Components (`src/mirror/prior_null.py`)

- `neg_log_likelihood(theta, X, y) -> float`: for each trial, softmax over
  concepts of `X[t] @ theta`; NLL is the summed negative log-probability of the
  reported concept.
- `fit(X, y) -> Fit`: minimize NLL with scipy L-BFGS from a zero start. `Fit`
  is a dataclass with `theta` (array), `gamma` (float, `theta[-1]`), and
  `loglik` (float).
- `simulate_reports(X, theta_true, rng) -> y`: for each trial, draw a reported
  concept from the softmax over `X[t] @ theta_true`. `rng` is a
  `numpy.random.Generator`.
- `gamma_ci(X, y, n_boot=200, rng=None) -> (lo, hi)`: 95% bootstrap percentile
  interval for gamma, resampling trials with replacement and refitting.

## Validation (simulation, the misspecification defense)

- Pure prior: `theta_true` with `gamma = 0` -> fitted `gamma` near 0, CI covers 0.
- Signal present: `theta_true` with `gamma = 2` -> fitted `gamma` near 2, CI
  covers 2 and excludes 0.
- Prior identifiable: a strong frequency coefficient is recovered.
- These simulation checks are the appendix evidence that the null model is not
  misspecified.

## Testing (TDD, CPU, seeded numpy)

- `neg_log_likelihood` at the true theta is lower than at a random theta on
  simulated data.
- A one-trial, two-concept, one-feature case matches a hand-computed softmax
  NLL.
- `simulate_reports` with a large gamma makes the injected concept the modal
  report.
- `fit` recovers `gamma ~ 0` (tol 0.3) from pure-prior simulation with enough
  trials.
- `fit` recovers `gamma ~ 2` (tol 0.4) from signal simulation.
- `fit` recovers a strong frequency coefficient.
- `gamma_ci` on signal simulation returns an interval that excludes 0; on
  pure-prior simulation returns an interval that includes 0.

Tolerances assume a few thousand simulated trials with a fixed seed.

## Conventions

Self-describing names, no comments/docstrings. Dependencies add scipy
(numpy already present via torch). Estimator uses only numpy + scipy.
