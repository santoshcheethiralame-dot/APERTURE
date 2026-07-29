# APERTURE

Measuring self-knowledge in language models without being fooled by output steering.

The core paradigm: extract a concept direction from contrastive prompts, inject it into
the residual stream while the model generates, ask the model what it notices, and log the
transcript alongside a KL meter that quantifies how hard the injection perturbed the
model. Because the concept is *planted*, the setup manufactures **ground truth** — the
resource this field is otherwise short of.

The headline methodological result: under a closed-list elicitation the model picks the
injected concept far above chance, which reads as introspection. A matched control that
never mentions thoughts or introspection scores *higher*. The apparent self-knowledge is
output steering, and any self-report claim elicited without such a control is confounded
to an unknown degree.

Current focus is **PLANTED**, a ground-truth benchmark for validating activation-
verbalization methods (recovery, attribution, and a confabulation rate). Start with
[`docs/ONBOARDING.md`](docs/ONBOARDING.md) — a complete from-zero guide — and the
CURRENT STATE block at the top of [`docs/plan/masterplan.md`](docs/plan/masterplan.md).

## Install

    pip install -e ".[dev]"

## Test

    pytest

Tests run on pythia-70m on CPU.

## Run

`notebooks/kaggle_demo.ipynb` runs the full loop on Gemma-2-2B-it
(Kaggle T4/P100, HF token required — Gemma is gated).

Plans and specs live in `docs/`. Experimental history is in
`docs/LAB_NOTEBOOK.md` (run registry, findings, and a decisions log, updated
after every run). Compute and funding status is in `docs/RESOURCES.md`. The
master plan and its dated addenda are in `docs/plan/masterplan.md`; the current
paper skeleton and claims table are in `docs/paper/`.
