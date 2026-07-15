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
| C8 | The concept is causally wired to the output (patching drives the concept, concept-specifically) | paired self-control +6.15, CI [+4.50, +7.89] excludes 0; control CI [-0.19, +1.80] includes 0 | R8 | PILOT + **concept-level CI** (10 concepts; no extraction-seed/prompt/family variance) |
| C9 | The injected directions are genuine concept representations, not injection artifacts | identifiability 0.688, CI [0.438, 0.875] excludes chance 0.062 | R9 | PILOT + **concept-level CI** (16 concepts, 1 context each) |
| C10 | Apparent "detections" are an affect confound, not introspection | only `joy` ever answers YES; never names the concept | R4, R5, R6 | PILOT (qualitative) |
| C11 | Under FORCED choice, the model's pick tracks the injected concept beyond frequency/concreteness priors | gamma +1.99, CI [+1.48, +2.48] excludes 0; hit 0.302 vs 0.062 chance; 0% unparseable | R10 | PILOT, **CONFOUNDED** — see below |
| -- | Forced-choice gamma > 0 demonstrates introspective ACCESS | -- | R10 | **NOT SUPPORTED** — R8 shows injection raises the concept's output token by ~7 nats, so gamma > 0 is exactly what pure output-steering predicts with zero introspection. Needs the R11 non-introspective-framing control to separate steering from access. |
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

1. **Variance beyond concepts.** PARTLY DONE (2026-07-15): C8 and C9 now carry
   95% bootstrap CIs over concepts, and both survive (patching paired CI excludes
   0; identifiability CI excludes chance). Note "more seeds" is a non-issue on the
   report side — generation is greedy, so generation seeds produce identical
   output. What remains is variance from (a) extraction-pair sampling, which IS
   seeded and does vary, (b) prompt paraphrase, and (c) model family. C7's probe
   CI needs leave-one-prompt-out CV and is not yet computed.
2. **Separate steering from access (THE priority, new after R10).** gamma has now
   been fit on real data (R10: +1.99, CI excludes 0) — but it is confounded,
   because injection directly steers the output token distribution (R8). Run the
   R11 control: identical setup, non-introspective framing ("Pick any one word
   from this list"). The gamma DIFFERENCE between framings is the introspective
   component; the shared part is steering. Until then neither the access nor the
   confabulation claim is settled under forced choice. Also replace the covariate
   proxies (infini-gram, Brysbaert).
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
