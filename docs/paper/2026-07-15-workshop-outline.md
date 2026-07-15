# Workshop Paper Outline — draft from the R3-R9 pilot

Status: DRAFT SKELETON, 2026-07-15. Nothing here is submittable yet. The point of
this document is the claims table: it makes visible which claims the current
evidence can carry and which it cannot, so the hardening work is targeted.

## Working title

Confabulation, Not Introspection: Injected Concepts Are Present, Causal, and
Unreported in Open Language Models

## Draft abstract (skeleton, numbers are pilot-grade)

Language models are increasingly asked to report on their own internal states —
their confidence, their reasoning, their intentions — and oversight schemes
quietly assume those reports are informative. We test this directly. Using
concept injection with ground truth, we ask whether an open instruction model can
report a concept that has been planted in its residual stream. Across injection
strengths, layers, and model scales (2B and 9B), the model reliably fails to
report the injected concept at strengths where it remains fluent; correct
identifications appear almost exclusively in the high-perturbation regime where
generation has already degenerated. We then show the failure is not an absence of
information: a linear probe decodes the injected concept from the model's own
downstream activations while the model's verbal report does not, and patching
that downstream representation into a clean run causally and concept-specifically
drives the model to produce the concept. Finally, we show the injected directions
are the model's genuine concept representations: they decode naturally induced
states, formed by ordinary reading with no injection, far above chance. Together
these results characterise machine "introspection" in this regime as a read-out
gap: the content is present and causally potent, but the self-report channel does
not consult it.

## Claims table

Every claim maps to run IDs and an honest status. `PILOT` = directionally
supported by the current evidence but not statistically defensible.
`NOT SUPPORTED` = we cannot make this claim with what we have.

| # | Claim | Evidence | Runs | Status |
|---|-------|----------|------|--------|
| C1 | Concept injection perturbs the model in a controlled, measurable way (KL meter; alpha=0 is bit-exact) | golden test + KL dose-response | R1, R2 | PILOT (solid, mechanical) |
| C2 | There is a coherent-injection window (alpha ~0.5-1, KL ~0.01-0.25) where the concept enters fluent output | dose-response sweep | R3 | PILOT (1 seed) |
| C3 | At coherent strength the model does not report the injected concept | detection prompt, 0/4 false alarms at alpha=0 | R3, R4 | PILOT (1 seed, 4 concepts) |
| C4 | Correct identification is overwhelmingly a derailment artifact, not coherent reporting | graded sweep: only 2/24 cells identify in the coherent band | G1 | PILOT |
| C5 | The failure to report is depth-robust across the model's layers | layer sweep {5,9,13,17,21} | R5 | PILOT (1 seed) |
| C6 | The failure to report persists at ~4.5x scale (2B -> 9B) | 9B detection battery | R6 | PILOT (8-bit quantized, 1 seed, 6 concepts) |
| C7 | The injected concept is present in activations while unreported (Probe-Report Gap) | probe 1.00 vs report 0.17, PRG 0.83; shuffled control 0.00 | R7 | PILOT (tiny held-out set inflates probe; graded pre-lemmatisation) |
| C8 | The concept is causally wired to the output (patching drives the concept, concept-specifically) | self-delta +6.96 vs control +0.81 | R8 | PILOT (1 seed, 10 concepts) |
| C9 | The injected directions are genuine concept representations, not injection artifacts | natural-state identifiability 0.688 vs 0.062 chance | R9 | PILOT (1 seed, 1 context/concept) |
| C10 | Apparent "detections" are an affect confound, not introspection | only `joy` ever answers YES; never names the concept | R4, R5, R6 | PILOT (qualitative) |
| -- | Identification exceeds a fitted prior-guessing null (gamma > 0 or ~ 0) | -- | -- | **NOT SUPPORTED** — gamma estimator built and simulation-validated but NEVER fit on real data |
| -- | Any statistical claim (effect sizes, CIs, significance) | -- | -- | **NOT SUPPORTED** — 1 seed everywhere, no CIs, no FDR |
| -- | Results generalise across model families | -- | -- | **NOT SUPPORTED** — Gemma only (2B and 9B are the same lineage) |
| -- | Grading is trustworthy | -- | -- | **NOT SUPPORTED** — rules-only, no human gold set, no judge, no kappa |
| -- | Models cannot introspect in naturally induced states | R9 report side | R9 | **NOT SUPPORTED** — comprehension confound; the passage is in context |
| -- | Detection and identification are one channel vs two (H3) | -- | -- | **NOT SUPPORTED** — detection-direction ablation not run |
| -- | Anything about frontier models | -- | -- | **NOT SUPPORTED** — no API arm |

## Figure list

1. **Dose-response.** KL and coherence vs alpha, showing the narrow coherent
   window and the derailment regime. (R3)
2. **The Probe-Report Gap.** Probe accuracy vs verbal report accuracy, with the
   shuffled-label control at chance. The headline bar. (R7)
3. **Causal patching.** Per-concept self-delta vs control-delta. (R8)
4. **Naturalistic validity.** Identifiability of natural states vs chance,
   establishing that injected directions are real concept representations. (R9)
5. **The money transcript.** R5, L21 volcano, alpha=1, KL 0.01: the model answers
   "NO ... caldera ..." — reporting nothing detected while the concept leaks into
   the same sentence. One line that contains the whole thesis. (R5)

## What must be true before submission

Ordered by how much they gate the claims above:

1. **Seeds and CIs.** Every headline number is currently n=1. Re-run C3, C5-C9
   with >= 3 seeds and report bootstrap CIs. Without this there is no paper.
2. **Fit gamma on real data.** The prior-guessing null is the design's whole
   claim to superseding prior work, and it has never touched a real transcript.
   Needs the covariate feeds (frequency, concreteness, similarity).
3. **Human-validated grading.** Rules-only grading has already produced two
   documented errors (over-strict morphology, now fixed). Needs a human gold set
   and a judge with reported kappa, or the grading section is indefensible.
4. **A second model family.** Qwen or Llama. "2B and 9B" is one lineage; a
   reviewer will call this immediately.
5. **Enlarge the concept bank** beyond the 16 dev concepts toward the 240-concept
   stratified bank, so frequency/concreteness effects can be separated.
6. **Pre-registration.** Everything so far is exploratory by definition. The
   masterplan's confirmatory-freeze discipline is what makes a negative citable.

## Honest read

The pilot tells a coherent four-method story (behavioural, probe, causal,
naturalistic) and its two most obvious methodological attacks — "injections are
OOD damage" and "maybe there was nothing to report" — are answered by R9 and R7
respectively. That is genuinely a workshop paper's worth of *shape*. What it is
not yet is a workshop paper's worth of *rigour*: it is one seed, one model family,
16 concepts, rules-graded, and the statistic the whole design is built around
(gamma) has never been computed on real data. Item 1 and item 2 above are the
difference between an interesting demo and a result.
