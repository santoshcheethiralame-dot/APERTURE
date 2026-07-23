# MIRROR

Dissociating genuine introspection from confabulation in language models.

The core paradigm: extract a concept direction from contrastive prompts,
inject it into the residual stream while the model generates, ask the model
what it notices, and log the transcript alongside a KL meter that quantifies
how hard the injection perturbed the model.

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
