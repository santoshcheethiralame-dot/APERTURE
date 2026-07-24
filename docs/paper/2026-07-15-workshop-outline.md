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
| C11 | Under forced choice the model's pick tracks the injected concept far above chance (gamma > 0) | gamma +1.99 (introspective) / +2.57 (neutral), both CIs exclude 0; hit 0.302 / 0.433 vs 0.062 chance | R10, R11 | PILOT (this is a STEERING effect, not access — see C12) |
| C12 | That effect is OUTPUT STEERING, not introspective access | neutral framing ("pick any word", no mention of introspection) gives gamma +2.57, HIGHER than the introspective framing; difference -0.59, CI [-1.15, -0.007] rules out any positive access effect | R11 | PILOT, **properly controlled** — the strongest result in the pilot |
| C13 | Explaining the injection mechanism does NOT rescue identification on Gemma-2-2B | PRE-REGISTERED (prereg 2026-07-22). gamma neutral +2.57 > introspective +1.99 > informative +1.64; informative - neutral -0.93 CI [-1.76, -0.25] excludes 0 negatively. Primary prediction (informative > introspective) falsified; CI [-1.05, +0.43] includes 0. Non-replication of Pearson-Vogel's 0.3->39.9% on Qwen-32B | R12 | PILOT, pre-registered negative — mechanism framing hurt, did not help |
| -- | Forced-choice gamma > 0 demonstrates introspective ACCESS | -- | R10, R11 | **REFUTED BY OUR OWN CONTROL** — R11 shows the introspective framing adds nothing (point estimate negative). R10's apparent access signal was an artifact of output steering (R8: injection raises the concept's output token ~7 nats). |
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
2. **Separate steering from access.** DONE (R11, 2026-07-15). The neutral-framing
   control yields a HIGHER gamma (+2.57) than the introspective framing (+1.99);
   the difference CI [-1.15, -0.007] rules out any positive access effect. The
   forced-choice gamma is output steering, not introspection. Remaining: replace
   the covariate proxies (infini-gram exact pretraining counts, Brysbaert
   concreteness) and redo the steering/access separation once they are in place.
3. **Rule out prompt uninformativeness as the cause of the null.** Pearson-Vogel
   et al., *Latent Introspection* (arXiv:2602.20031), report that Qwen-32B denies
   an injection in plain output while a logit lens still finds the detection
   signal in the residual stream — and that telling the model, accurately, how AI
   introspection works raises injection sensitivity from 0.3% to 39.9% at +0.6%
   false positives. If that transfers, C3/C4/C10 are underdetermined: our null may
   be a property of our detection prompt rather than of the read-out channel. The
   cheap test is an informative-prompt arm alongside the current detection prompt
   on the existing R3/R5 grid; the null only survives if it holds under the prompt
   most favourable to reporting. Note this cuts both ways — a large prompt effect
   is itself evidence about what the self-report channel consults, and the
   Probe-Report Gap (C7) is the measurement that does not depend on the prompt.
4. **Human-validated grading.** Rules-only grading has already produced two
   documented errors (over-strict morphology, now fixed). Needs a human gold set
   and a judge with reported kappa, or the grading section is indefensible.
5. **A second model family.** Qwen or Llama. "2B and 9B" is one lineage; a
   reviewer will call this immediately. Note Pearson-Vogel et al. run Qwen-32B,
   so the Qwen arm doubles as the replication target for item 3.
6. **Enlarge the concept bank** beyond the 16 dev concepts toward the 240-concept
   stratified bank, so frequency/concreteness effects can be separated.
7. **Pre-registration.** Everything so far is exploratory by definition. The
   masterplan's confirmatory-freeze discipline is what makes a negative citable.
   For H7 specifically, the prediction ("report accuracy rises while probe
   accuracy stays flat under persona suppression") must be registered in writing
   BEFORE E11a is run — a confirmed pre-registered prediction is far stronger
   evidence than the same result found and explained afterwards.

Resource-gated items (see `docs/RESOURCES.md`): items 2 and 5 need the free
infini-gram API and Brysbaert norms; item 4 (second family) and the multi-seed
work need the PES Titan box or rented mid-tier hours; item 3 needs ~200 in-team
human labels.

## Discussion framing — agent oversight (no new experiments required)

Decided 2026-07-22 (masterplan Addendum 2): rather than adding an agentic arm, the
existing results are framed as an agent-safety finding in the discussion. This is
free, and unlike a bolted-on arm it is true of the data we have.

> Agentic deployments increasingly rely on models explaining their own actions
> for oversight. Our results show the self-report channel does not consult the
> model's actual internal state — even when that state is decodable (R7) and
> causally driving the output (R8). Oversight schemes built on "ask the agent
> why" inherit these error bars. Our forced-choice control (R11) further shows
> that the obvious way to test this — forcing the model to name a cause —
> measures output steering rather than self-knowledge unless a neutral-framing
> condition is included.

The agentic experiment proper (inject a goal mid-episode during tool use, then
ask "why did you do that?", ground-truthed against the injected vector) is
sequenced as paper #2, after the core result. It maps onto E7 / ladder rung L3,
which the plan already contains.

## Honest read

The pilot tells a coherent four-method story (behavioural, probe, causal,
naturalistic) and its two most obvious methodological attacks — "injections are
OOD damage" and "maybe there was nothing to report" — are answered by R9 and R7
respectively. That is genuinely a workshop paper's worth of *shape*. What it is
not yet is a workshop paper's worth of *rigour*: it is one seed, one model family,
16 concepts, rules-graded, and the statistic the whole design is built around
(gamma) has never been computed on real data. Item 1 and item 2 above are the
difference between an interesting demo and a result.
