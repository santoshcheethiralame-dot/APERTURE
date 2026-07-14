# Milestone 7 — Mechanism: Activation Patching (B7)

Status: approved 2026-07-14
Scope reference: build guide B7; master plan RQ2/H3; adds causal evidence to the
behavioral (R3-R6) and probe (R7) confabulation findings

## Goal

Test causally whether an injected concept's content reaches the model's output
pathway. Patch the injected residual (cached at a downstream layer) into a clean
run and measure whether the concept token's output log-probability rises. A
concept-specific positive effect means the content propagated to the output and
can drive it — so confabulation (the model still not reporting it, R5/R6) is a
read-out failure, not an absence of content.

## In scope

- `mirror/patching.py`: single-trial patch effect and a per-concept collection
  loop with a negative control.
- CPU pytest on the tiny Llama.
- A Kaggle run on 8-bit gemma-2-2b (native-model path).

## Out of scope (deferred)

Detection-direction dissociation (full H3), attribution graphs (B8), multi-site
patching sweeps, noising/denoising both directions, TL-backend patching.

## The experiment

Per concept c, prompt fixed:
1. Injected run: inject c at `layer`, cache the residual at a downstream
   `patch_layer` (> layer) at the last prompt position. This reuses
   `probe_activation_hf`.
2. Baseline clean run: no injection; record log-prob of c's token at the output
   (last-position next-token distribution).
3. Patched clean run: clean forward with a hook at `patch_layer` that replaces
   the last-position residual with the cached injected residual; record log-prob
   of c's token.
4. Effect delta = patched log-prob - baseline log-prob. Positive means the
   injected content reached the output.

Negative control: patch a different concept's cached residual and measure c's
token; the effect should be near zero, showing the effect is concept-specific
rather than a generic patch artifact.

The concept token is the leading token of `" " + name` (matching
`vectors.steering_check` / `hf_model._steering_hf`).

## Why downstream

`patch_layer` is downstream of the injection `layer`, so the cached residual
reflects the model's own processing of the injected state, not the raw added
vector. A positive patched effect then means the processed concept
representation can causally drive the output.

## Components (`mirror/patching.py`)

- `concept_token(tok, name) -> int`: leading token id of `" " + name`.
- `patched_logprob(model, tok, prompt, patch_layer, patch_resid, target_token) -> float`:
  clean forward with a hook at `patch_layer` replacing the last-position residual
  with `patch_resid`; returns log-softmax log-prob of `target_token` at the last
  position.
- `baseline_logprob(model, tok, prompt, target_token) -> float`: clean forward,
  log-prob of `target_token` at the last position.
- `patch_effect_hf(model, tok, prompt, vec, alpha, patch_layer, target_token) -> (baseline_lp, patched_lp, delta)`:
  caches the injected residual via `probe_activation_hf(model, tok, prompt, vec,
  alpha, patch_layer)`, then computes baseline and patched log-probs; delta =
  patched - baseline.
- `collect_patch_hf(model, tok, bank, names, layer, patch_layer, alpha, prompt, out, n_pairs=12) -> dict`:
  extracts each concept vector once; for each concept records `self_delta`
  (patch its own residual, measure its own token) and `control_delta` (patch the
  next concept's residual, measure this concept's token). Writes JSONL (one
  record per concept: concept, layer, patch_layer, alpha, self_delta,
  control_delta) and returns `{"records": [...]}`.

## Testing (TDD, tiny Llama, CPU)

- `patch_effect_hf` with a real vector and alpha != 0 returns a non-zero delta
  (patching changes the clean output).
- Golden: alpha = 0 (cached residual equals the clean residual) returns
  delta ≈ 0 (patching clean-into-clean is a no-op).
- `patched_logprob` returns a float <= 0 (a log-prob).
- `collect_patch_hf` with 3 concepts writes 3 JSONL records, each with
  `self_delta` and `control_delta`.

## Deliverable

A Kaggle run on 8-bit gemma-2-2b (inject layer 13, patch layer 20, coherent
alpha, the dev-bank concepts). Headline: mean `self_delta` vs mean
`control_delta`. Self >> control ≈ 0 means injected content reaches the output
concept-specifically, so the failure to report it is a read-out gap. Recorded as
R8 in the lab notebook.

## Conventions

Self-describing names, no comments/docstrings. Reuse `hf_layer`, `_hidden`,
`probe_activation_hf` from `hf_model`; the bank from `concepts`. No new
dependencies. Do not modify the TL modules.
