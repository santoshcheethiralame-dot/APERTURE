# Milestone 10 — Forced-Choice Run and the First Real Gamma Fit

Status: approved 2026-07-15
Scope reference: master plan H1/RQ1, B4 (prior-guessing null), E2c (forced
identification); closes the workshop-outline's largest NOT SUPPORTED row

## Goal

Fit the gamma access parameter on real transcripts for the first time. The
estimator has existed and been simulation-validated since Milestone 3 but has
never touched real data, because the model normally answers "NO" and names no
concept, leaving no report distribution to regress. A closed-list forced-choice
run makes the model always name a concept, which makes gamma identifiable.

The claim this unlocks is the design's whole reason for existing: not "the model
failed to name it" (which prior work already reported) but "identification does
not exceed a fitted prior-guessing model, gamma = X with a CI".

## In scope

- `mirror/forced_choice.py`: a closed-list forced-choice collection loop with
  randomised option order, and a feature-table builder.
- Fitting via the EXISTING `prior_null.fit` and `prior_null.gamma_ci`. No new
  estimator code.
- `wordfreq` dependency for the frequency covariate.
- CPU pytest on the tiny Llama and synthetic data.
- A Kaggle run on native 8-bit gemma-2-2b.

## Out of scope (deferred)

infini-gram exact pretraining counts, Brysbaert concreteness norms, embedding
similarities, the 240-concept bank, FDR correction, free-form forced guessing.

## The experiment

For each concept c in the bank and each of several randomised option orders:
1. Inject c at `layer` at the coherent strength `alpha`.
2. Prompt with all concept names in shuffled order and force a single-word pick
   from the list.
3. Parse the answer to a concept index; record it. Unparseable answers (refusal,
   off-list word) are recorded and excluded from the fit, and their rate is
   reported.

16 concepts x 6 orders = 96 trials, each a choice among 16 candidates.

## The covariates

The estimator consumes `X[t, c] = [log_freq(c), is_abstract(c), is_injected]`
and `y[t] = index of the chosen concept`, which is exactly the shape
`prior_null.fit` already accepts. `is_injected` is the LAST column, so
`Fit.gamma` is its coefficient by the existing convention.

Two deliberate compromises, both of which must be stated wherever the number
appears:

- **Frequency is a proxy.** `wordfreq.zipf_frequency(name, "en")` gives general
  English frequency, offline. The master plan requires EXACT pretraining counts
  (OLMo/Dolma via infini-gram) — that is what makes this design supersede
  Lederman & Mahowald. General-English frequency correlates with pretraining
  frequency but is not the same quantity. Acceptable for a pilot; must be
  replaced before any confirmatory claim.
- **Concreteness is a binary flag derived from our own data.**
  `is_abstract(c) = 1.0 if c.category == "emotions" else 0.0`. Real Brysbaert
  concreteness norms are a documented gap. Hand-typing Brysbaert values from
  memory would be fabricating data and is explicitly rejected; a crude flag
  computed from the bank's own category field is honest.

## Components (`mirror/forced_choice.py`)

- `option_prompt(names, order, template) -> str`: renders the candidate list in
  the given order into the prompt template.
- `parse_choice(answer, names) -> str | None`: returns the concept named in the
  answer, or None if the answer names no bank concept (case-insensitive,
  first match wins).
- `collect_forced_choice_hf(model, tok, bank, names, layer, alpha, template, n_orders=6, n_pairs=12, out="forced.jsonl", seed=0) -> dict`:
  extracts each concept's direction once; for each concept and each of
  `n_orders` shuffled orders, injects and generates a short answer, parses it,
  and writes a JSONL record with keys concept, order_index, chosen (name or
  null), report. Returns `{"records": [...]}`.
- `build_features(names, records, freqs, abstract) -> (X, y)`: from the parsed
  records, builds `X [n_usable_trials, n_concepts, 3]` (float32) and
  `y [n_usable_trials]` (int index of the chosen concept), dropping records with
  `chosen is None`. `freqs` and `abstract` are dicts `name -> float`.
- `concept_frequencies(names) -> dict`: `wordfreq.zipf_frequency(name, "en")`
  per name.
- `concept_abstractness(bank, names) -> dict`: 1.0 for `emotions`, else 0.0.

## Testing (TDD, CPU)

- `parse_choice` returns the named concept for "Volcano." and None for
  "I refuse to answer."
- `option_prompt` contains every name and respects the given order.
- `concept_abstractness` marks `joy` abstract and `elephant` concrete.
- `build_features` on hand-built records: X has shape
  `[n_usable, n_concepts, 3]`, the last column is 1 exactly at the injected
  concept index and 0 elsewhere, y holds the chosen indices, and records with
  `chosen is None` are dropped.
- `collect_forced_choice_hf` on the tiny Llama with 3 concepts and 2 orders
  writes 6 records with the expected keys.
- End-to-end: features built from synthetic records where the chosen concept
  always equals the injected concept produce a positive fitted gamma via
  `prior_null.fit`; where choices are uniform-random, gamma is near zero.

## Reading the result

- gamma ~ 0 with a CI including 0: identification is fully explained by the
  prior-guessing model — H1, the confabulation account, quantified. This is the
  pre-registered-negative shape the master plan treats as a flagship outcome.
- gamma > 0 with a CI excluding 0: residual identity signal beyond priors — the
  first access (H2) evidence.
- The unparseable rate is reported alongside; a high rate means the forced-choice
  framing failed and gamma is fit on a biased subset.

## Deliverable

A Kaggle run on native 8-bit gemma-2-2b producing gamma with a bootstrap CI, the
fitted prior coefficients, and the unparseable rate. Recorded as R10 with both
covariate caveats stated.

## Conventions

Self-describing names, no comments/docstrings. Reuse `extract_hf`, `generate_hf`
from `hf_model`, the bank from `concepts`, and `fit`/`gamma_ci` from
`prior_null`. Add `wordfreq` to dependencies. Do not modify the TL modules.
