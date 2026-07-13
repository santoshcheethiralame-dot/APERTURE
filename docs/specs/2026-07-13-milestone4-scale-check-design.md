# Milestone 4 — Scale Check on 9B (multi-GPU)

Status: approved 2026-07-13
Scope reference: master plan RQ5 (scale), risk R1 (Branch-D); follows R5
depth-robust confabulation finding at 2B

## Goal

Test whether introspective detection/identification emerges at ~9B where it was
absent, depth-robustly, at 2B. Run the detection battery on Gemma-2-9B-it split
across Kaggle's two T4 GPUs (full fp16, no quantization), grade the transcripts,
and record whether any coherent-KL trial produces a clean correct
identification (H2 signal) or confabulation persists across the scale jump
(strengthens the Branch-D read).

## In scope

- Device-safe injection: the injected delta is placed on the hooked residual's
  device, so injection works under model parallelism.
- Kaggle notebook loading gemma-2-9b-it with n_devices=2 and running the
  detection battery at a mid layer over a small alpha sweep.
- Grading the run and recording the result in the lab notebook.

## Out of scope (deferred)

Quantization-validity study, the full 240-concept bank, more model families,
the real gamma fit (still needs the covariate feeds), probes/PRG.

## The device-placement fix

Under `n_devices=2`, TransformerLens splits Gemma-2-9B's 42 layers across the
two GPUs; the residual at layer L lives on device(L). The injection hook adds
`alpha * sigma * vec.direction` to that residual. Because extraction and
injection use the same layer, the vector is normally already on the right
device, but the hook must not rely on luck. Fix: place the delta on the
residual's device explicitly.

- `injection.make_hook`: `resid[:, -1:] += (alpha * vec.sigma * vec.direction).to(resid.device)`
- `vectors.steering_check` hook: same `.to(resid.device)` guard.

On CPU (the test environment) this is a no-op; on multi-GPU it prevents a
device-mismatch crash.

## Notebook

- Load: `HookedTransformer.from_pretrained("gemma-2-9b-it", n_devices=2, dtype=torch.float16)`.
- Layer: mid depth ~= 21 (9B has 42 layers; layer 13 was mid for 2B's 26).
- Prompt: the detection prompt (YES + concept in one word, or NO).
- Alphas: {0, 1, 2, 4} to re-bracket the coherent band (sigma-normalized, but
  9B may differ; the KL meter identifies coherent trials).
- Concepts: 6 (elephant, volcano, joy, telescope, spider, library) to bound
  runtime; 9B model-parallel generation is slow.
- Output: JSONL per run; print `layer / concept / alpha / KL / answer` using the
  `rpartition("model")` answer extraction.

## What counts as a result

- A clean `YES, <correct concept>` at coherent KL (low KL, injected) on 9B =
  first evidence of the access account (H2); justifies deeper investment.
- Persisting NO / no correct identification at coherent KL = confabulation holds
  across a ~4.5x scale jump; strengthens the Branch-D / confabulation read.
- The affect confound (joy) and the PRG leak pattern (concept words appearing
  while the model reports NO) are watched for and noted, not counted as
  detection.

## Testing

- Existing suite stays green (the device guard is a no-op on single-device CPU).
- One added test asserts the injected delta is on the residual's device after
  the hook fires (trivially true on CPU, but documents the contract).
- True multi-GPU placement is verified by the Kaggle run completing without a
  device-mismatch error; there is no CPU substitute for that check.

## Conventions

Self-describing names, no comments/docstrings. No new dependencies.
