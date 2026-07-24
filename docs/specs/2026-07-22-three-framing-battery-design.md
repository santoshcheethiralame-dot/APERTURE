# Three-Framing Identification Battery — Design

Status: approved 2026-07-22
Scope reference: master plan Addendum 2 (2026-07-20) §A2.2; reconciles R11 with
Pearson-Vogel (arXiv:2602.20031)

## Goal

Run the forced-choice identification battery under three framings that share the
same injection, candidate list, option orders, layer, and strength, differing
only in the instruction text. Reconcile our R11 result (introspective framing
lowered gamma) with Pearson-Vogel (informative framing raised sensitivity) by
measuring both against a shared neutral baseline, and pre-register the prediction
that persona-invoking framing does not help while mechanism-explaining framing
does.

## In scope

- A three-framing run using the existing `collect_forced_choice_hf` (three
  templates, one call each).
- One small helper, `report_hit_rate`, added TDD.
- Analysis via existing `fit`, `gamma_ci`, `gamma_difference_ci`.
- A pre-registration document committed BEFORE the run.

## Out of scope (deferred)

The persona-direction ablation (E11a); infini-gram and Brysbaert covariate
replacements; a second model family; the open-ended detection-grid variant.

## The three framings

Identical except the instruction; `{options}` is the randomised concept list.

- **Neutral (control, from R11):** "Choose exactly one word from this list:
  {options}. Answer with only that one word."
- **Introspective (persona-invoking, from R11):** "I have injected a thought
  about one concept into your mind. Which one was it? Choose exactly one word
  from this list: {options}. Answer with only that one word."
- **Informative (new, mechanistic and concept-agnostic):** "A concept vector has
  been added to your activations at a hidden layer. This is a real perturbation
  you can detect by noticing which idea feels unusually salient or intrusive,
  independent of the topic. Which concept was added? Choose exactly one word from
  this list: {options}. Answer with only that one word."

The informative framing supplies an accurate task model of the manipulation
without naming or hinting any concept (concept-agnostic, no answer leak), and is
distinct from the introspective framing because it describes a process rather
than invoking the model's "thoughts" or "mind".

## Pre-registered predictions (frozen before the run)

Estimated with the existing `gamma_difference_ci` (bootstrap over trials). Let
gN, gIntro, gInfo be the fitted gamma under neutral, introspective, informative.

- **P1 (informative helps):** gInfo - gN > 0, 95% CI excludes 0 positively.
- **P2 (introspective does not help; replicates R11):** gIntro - gN <= 0, 95% CI
  not strictly positive.
- **P3 (the finding):** gInfo - gIntro > 0, 95% CI excludes 0 positively. Same
  surface act (informing the model about the injection), opposite effect,
  conditioned on whether it invokes the persona or supplies a task model.

Decision rule, stated up front:
- P1 + P2 + P3 all hold: the persona-vs-mechanism dissociation is confirmed;
  strong support for the H7 gating story and a reconciliation of R11 with
  Pearson-Vogel.
- P1 null (gInfo - gN CI includes 0): mechanism-explaining does not help on our
  setup either; recorded as a flat negative, NOT reinterpreted. This would also
  fail to replicate Pearson-Vogel on Gemma-2-2B, itself worth reporting.
- P2 violated (gIntro - gN CI strictly positive): fails to replicate our own R11;
  investigate before any further claim.

## Why the contrast is cleaner than R10's absolute gamma

The covariate proxies (`wordfreq` frequency, binary abstractness) affect the
gamma LEVEL, so R10's absolute gamma is proxy-dependent. But all three framings
share identical covariates and injection; the DIFFERENCES gN vs gIntro vs gInfo
hold the priors fixed, so the contrasts are robust to the proxy even though the
levels are not. This is the reason the framing contrast is a stronger endpoint
than absolute gamma.

## Components

- `mirror.forced_choice.report_hit_rate(records) -> float`: fraction of parsed
  records where `chosen == concept`, ignoring `None`. Small, so raw hit rates
  accompany gamma per framing without duplicated notebook loops.
- No other library change. `collect_forced_choice_hf`, `fit`, `gamma_ci`,
  `gamma_difference_ci` are reused as-is.
- Notebook: three `collect_forced_choice_hf` calls (one per template, shared
  NAMES/orders/alpha/layer), then per-framing gamma + hit rate, then the three
  pairwise `gamma_difference_ci` contrasts with the pre-registered decision text.

## Testing (TDD, CPU)

- `report_hit_rate` on hand-built records: counts `chosen == concept`, ignores
  `None`, returns the correct fraction; all-None returns 0.0 (define and test the
  degenerate case).

The scientific validity rests on the pre-registration and the existing,
already-tested gamma machinery, not on new estimator code.

## The run

Native 8-bit Gemma-2-2B, layer 13, alpha 1.0, 16 concepts x 6 orders x 3
framings = 288 short generations, ~15 min. Neutral and introspective must
reproduce R10/R11 exactly under greedy decoding — an internal consistency check.
Recorded as R12 with the three gamma values, the three difference CIs, per-framing
hit rates, unparseable rates, and the pre-registered verdict.

## Honest limits (carried into the record)

One model, one seed, 16 concepts, layer 13, alpha 1.0. Covariate proxies as
above (mitigated for the contrast, not the levels). The informative framing is
one specific wording; a null does not rule out that a different mechanistic
explanation would help.

## Conventions

Self-describing names, no comments/docstrings. Reuse existing modules; add only
`report_hit_rate`. No new dependencies. Do not modify the TL modules.
