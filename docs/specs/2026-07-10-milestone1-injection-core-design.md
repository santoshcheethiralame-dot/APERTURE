# Milestone 1 — Skeleton + Injection Core

Status: approved 2026-07-10
Scope reference: PROJECT MIRROR master plan (docs/plan/), build specs B1, B2

## Goal

A working end-to-end demonstration of the concept-injection paradigm: extract
concept vectors, inject them into a model's residual stream during generation,
interrogate the model, and log transcripts with a perturbation meter — the
foundation every later experiment (E1–E10) builds on.

## In scope

- Repo skeleton: package layout, configs, tests, docs.
- `mirror` Python package on TransformerLens (`HookedTransformer`):
  - B1 concept-vector pipeline (diff-in-means) with validation checks.
  - B2 injection harness with strength in sigma-layer units and the KL meter.
- Dev concept bank: ~16 concepts across 4 categories with contrastive
  prompt pairs (category-matched negatives).
- CPU test suite on pythia-70m (pytest).
- One Kaggle notebook running the full loop on Gemma-2-2B-it.

## Out of scope (deferred to later milestones)

Grading stack (B3), prior-guessing null model (B4), probes/PRG (B5),
240-concept bank, hydra, wandb, vLLM, judge ensembles, any statistics.
Milestone 1 ends with human-read transcripts, not graded metrics.

## Repo layout

```
MIRROR/
├── pyproject.toml
├── README.md
├── docs/
│   ├── plan/                  master plan + build guide
│   └── specs/                 this file and successors
├── configs/dev.yaml           plain YAML, hydra-compatible key structure
├── data/concepts/dev_bank.yaml
├── src/mirror/
│   ├── concepts.py
│   ├── vectors.py
│   ├── injection.py
│   ├── metrics.py
│   └── runner.py
├── tests/
└── notebooks/kaggle_demo.ipynb
```

## Components

### concepts.py
Concept bank schema (name, category, prompt pairs) loaded from YAML.
Each concept carries ~40 contrastive pairs: positive prompts evoking the
concept, negatives evoking a random same-category concept. Bank loader
validates counts and category matching.

### vectors.py
Diff-in-means extraction: run positive and negative batches, cache residual
stream at layer L (`hook_resid_post`) averaged over response positions,
vector = mean(pos) − mean(neg), stored per layer alongside sigma_L (median
residual norm at that layer). Validation trio, each returning a flag on the
stored vector rather than silently passing:
1. Steering sanity — adding +alpha·v raises concept-token probabilities.
2. Probe sanity — projection onto v separates held-out pos/neg ≥ 90%.
3. Stability — cross-resample cosine similarity ≥ 0.8.
Extraction prompts are never reused for evaluation.

### injection.py
Forward hook adding alpha·sigma_L·v_hat at a chosen layer over a chosen span
(single position or all response positions). Strength is always expressed in
sigma_L units. Composes with token-by-token generation. Seeded.

### metrics.py
sigma_L estimation and the KL meter: for every injected trial, a paired clean
generation with identical prompt and seed; KL(p_inj ∥ p_clean) over next-token
distributions at probe positions before the report segment, logged per trial.

### runner.py
Trial runner: takes config (model, concept, layer, alpha, span, seed, prompt),
produces paired clean/injected generations, writes one JSONL record per trial:
config hash, transcript, KL, vector validation flags.

### Config
Plain YAML (`configs/dev.yaml`) with keys grouped the way hydra groups would
be (model, injection, concepts, run) so migration later is mechanical.

## Testing

pytest on pythia-70m (CPU):
- Golden test: alpha = 0 reproduces the clean generation bit-exact under a
  fixed seed.
- Derailment test: very large alpha visibly changes generation.
- KL meter: ≈ 0 on identical runs; > 0 under injection.
- Hook placement: injection modifies exactly the configured layer/positions.
- Vector shape/normalization and bank-loading validation.

## Kaggle demo — definition of done

`notebooks/kaggle_demo.ipynb`: installs the repo from GitHub, loads
Gemma-2-2B-it fp16 on T4/P100 (HF token via Kaggle secret; Gemma is gated),
extracts dev-bank vectors, runs 1 layer band × alpha ∈ {2, 4, 8} × 8 concepts
× 2 seeds with an interrogation prompt, writes transcripts + KL to JSONL.
Pass criterion: transcripts show silence at alpha = 0, detection-style reports
at moderate alpha, derailment at high alpha.

## Conventions

Self-describing names, no code comments or docstrings. Plain pip install via
pyproject. Python 3.11, torch 2.x, transformer_lens pinned.
