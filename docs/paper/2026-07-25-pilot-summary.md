# Project APERTURE — Pilot Summary

**Can a language model report its own internal state, or does it confabulate?**

Santosh Cheethirala · PES University · July 2026
Two-page summary of pre-semester pilot work, prepared for mentor review.

---

## 1. The question

Language models are increasingly asked to report on themselves: *how confident are
you, why did you do that, what are you thinking?* A growing number of AI-safety
proposals treat those answers as informative — using self-reported confidence to
decide when to trust a model, or asking an agent to explain an action in order to
catch mistakes.

Every one of those proposals assumes the model can **introspect**: that its
self-report is causally connected to its actual internal state.

There is a competing possibility, and it is the default in the human literature.
Nisbett and Wilson's classic work showed people confidently explain choices using
reasons that demonstrably were not the cause — **confabulation**: a sincere,
fluent, and wrong account of one's own processing. If language models are
confabulators, then "ask the model about itself" is an unreliable foundation for
oversight.

**This project is designed to settle which one is happening, with ground truth.**

## 2. Why the question is answerable now

Normally you cannot check a model's self-report, because you do not know what is
actually inside it. Our method removes that problem.

We use **concept injection**. A transformer carries a running internal vector (the
"residual stream") that each layer adds to. Using standard interpretability
techniques, we construct a direction in that space corresponding to a specific
concept — say *volcano* — and add it to the model's internal activity mid-generation.

Because we planted it, **we know the ground truth exactly.** We then ask the model
what it notices and check whether the answer reflects what we planted.

The apparatus quantifies its own intrusiveness: every trial logs a **KL divergence**
measuring how far the injection moved the model's output distribution, so we can
demonstrate that our results come from *gentle* perturbations where the model still
speaks fluently, rather than from breaking it.

## 3. What was built (pre-semester)

A complete, tested experimental pipeline, developed test-first:

- Concept-vector extraction with three automatic validity checks per vector
- Injection with a calibrated strength parameter and a KL perturbation meter
- Automated grading of model reports
- Linear **probes** that decode the injected concept directly from activations
- A fitted statistical null model (below) with bootstrap confidence intervals
- **Activation patching** for causal tests
- A naturalistic arm using states induced by ordinary reading, with no injection

Two backends — a research-interpretability stack for development and a quantized
production stack for larger models. **98 automated tests pass.** All experiments ran
on free-tier Kaggle GPUs at zero cost.

## 4. What the pilot found

Twelve logged experiments on Gemma-2 (2B, with a 9B check). Four independent methods
converge on one picture.

**(a) Behaviourally, the model does not report the injection.** At injection
strengths where it remains fluent, it answers "no, nothing unusual" — while the
planted concept demonstrably shapes its words. In one representative transcript, the
model denies detecting anything and uses the word *caldera* in the same sentence,
with *volcano* injected. This holds across five injection depths and across a 4.5×
increase in model size.

**(b) But the information is present.** A linear probe recovers the injected concept
from the model's own downstream activations almost perfectly, while its verbal report
recovers it about 17% of the time — a **Probe–Report Gap of roughly 0.83**. A
shuffled-label control probe performs at chance, confirming the signal is real. So
this is not "there was nothing to report."

**(c) And it is causally connected to the output.** Transplanting the injected
representation into a clean run raises that concept's output probability sharply and
specifically (paired effect +6.15 nats, 95% CI [+4.50, +7.89]; a matched control
transplant is statistically null). The content is present *and* able to drive
behaviour.

**(d) The directions are the model's own.** Injection-derived concept directions
decode states induced by ordinary reading, with no injection at all, at 0.688 versus
0.062 chance (95% CI [0.438, 0.875]). This answers the natural objection that
injections are merely out-of-distribution damage.

**(e) The apparent counter-evidence dissolves under control — the pilot's most
important result.** Forcing the model to choose from a list, it picks the injected
concept far above chance, which reads as successful introspection. We then ran a
**neutral-framing control**: the identical injection, but asking only "pick one word
from this list," with no mention of thoughts or introspection. Performance was
*higher* (43% vs 30%). The difference excludes zero in the negative direction.

The apparent introspection was **output steering** — the injection mechanically
pushes the concept toward the model's mouth, requiring no self-knowledge whatsoever.
This control is missing from comparable published work, which we believe leaves those
results confounded to an unknown degree.

**(f) A pre-registered test, honestly reported as falsified.** We froze a prediction
in version control before running: that explaining the injection mechanism to the
model would improve identification. It did not — it performed worst of three
framings, and we failed to replicate a published result obtained on a 32B model. We
report this as a clean negative rather than reinterpreting it after the fact.

**Summary: the injected concept is present, causally active, and unreported.** On
these models, machine "introspection" looks like a read-out gap, with the apparent
positive signal explained by steering.

## 5. Honest limitations

Stated plainly, because they determine what comes next: one model family (2B and 9B
are the same lineage), one seed, 16 concepts, rules-based grading with no
human-validated gold set, proxy variables for word frequency and concreteness, and
only one pre-registered run. This is a pilot demonstrating *shape*, not a result
carrying publication-grade *rigour*.

## 6. The constraint that now governs the project

Recent replications in this literature report the effect appearing reliably at around
**32 billion parameters**. Our models are 2B and 9B — the largest that fit on free
Kaggle GPUs.

**This makes the central negative result currently uninterpretable.** We cannot yet
distinguish "these models cannot introspect" from "we looked below the scale at which
it appears." Our own failure to replicate the 32B result is consistent with either.

A single 32B-capable arm resolves the ambiguity. A 32B model at usable precision
needs roughly 40–80GB of GPU memory (A100-class). This is the difference between a
publishable finding and an inconclusive one, and it is the one thing the project
cannot obtain from free resources.

**Timing.** The experiments needing this hardware run in the even semester
(January–May 2027), because the research must be complete before the Sem-7
internship. Institutional access typically takes months to arrange, which is why the
request is being made now.

## 7. What happens either way

The design is built so that no outcome is a failure:

- **If genuine introspection exists** in some regime, we locate it and trace the
  mechanism — the first validated machine introspection.
- **If it is confabulation throughout**, we deliver a pre-registered negative showing
  that self-report-based oversight is unreliable, with the information demonstrably
  present but inaccessible.
- **If compute does not materialise**, the methodological contribution — the steering
  confound and its control protocol, plus the Probe–Report Gap — stands on work
  already complete, with the scale question stated as an explicit limitation.

Full experimental history, statistics, and pre-registrations are maintained in the
project lab notebook and are available on request.
