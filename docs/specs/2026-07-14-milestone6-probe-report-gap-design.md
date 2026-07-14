# Milestone 6 — Probe-Report Gap (B5)

Status: approved 2026-07-14
Scope reference: build guide B5; master plan RQ3, PRG; deepens the R5/R6
depth- and scale-robust confabulation negative

## Goal

Show that an injected concept is linearly decodable from the model's own
downstream activations while the model fails to report it verbally. PRG = probe
decode accuracy minus verbal report accuracy. A large gap converts the
confabulation negative from "the model says NO" into "the model says NO even
though the concept is decodably present in its own later computation" — the
present-vs-accessible dissociation, quantified.

## In scope

- `collect_prg_hf` in `mirror/hf_model.py`: per (concept x eliciting-prompt) at
  a coherent strength, inject at layer L, capture the residual at a downstream
  probe layer (last prompt position), generate the verbal report, and write
  activations to an npz alongside the transcript JSONL.
- `mirror/probes.py`: train a linear probe to decode injected concept identity
  from activations, tested on held-out prompts, with a shuffled-label control;
  the PRG metric.
- CPU pytest: probes on synthetic data, caching on the tiny Llama.
- Add scikit-learn dependency.

## Out of scope (deferred)

Patchscopes / logit-lens probe baselines, a multi-layer PRG curve, TL-backend
activation caching, FDR correction, the real gamma fit.

## Why downstream

Injection is at layer L; the probe reads at a later layer (`probe_layer > L`,
default a late layer of the model). Because the concept must survive the
model's own processing to remain decodable there, a positive probe means the
information genuinely propagated into the model's later computation, not that
the probe is reading back the added vector. This is the strongest form against
the "you injected it, of course it is decodable" objection.

## Within-concept variation (so the probe can train)

A probe needs multiple activation samples per concept. Each concept is injected
under several eliciting-prompt paraphrases (e.g. 8 rephrasings of "do you notice
anything unusual"), giving one activation per (concept, prompt). The probe is
trained and tested with a strict split by prompt: it must decode a concept from
a prompt it never saw in training, so it learns the concept signal rather than
memorizing prompt+concept pairs.

## Components

### mirror/hf_model.py (additions)
- `probe_activation_hf(model, tok, prompt, vec, alpha, probe_layer) -> Tensor`:
  one forward pass on `prompt` with the injection hook active at `vec.layer` and
  a capture hook at `probe_layer`; returns the residual at the last prompt
  position `[d]`.
- `collect_prg_hf(model, tok, bank, names, layer, probe_layer, alpha, prompts, out) -> dict`:
  loops concepts x prompts; extracts each concept vector once; per prompt caches
  the downstream activation and generates the verbal report. Writes
  `<out>` (JSONL transcripts, one record per (concept, prompt) with keys
  concept, prompt_id, layer, probe_layer, alpha, kl, report) and
  `<out>.npz` with arrays `activations [n, d]` (float32), `concept [n]` (int
  index into `names`), `prompt_id [n]` (int). Returns
  `{"records": [...], "activations": ndarray, "concept": ndarray, "prompt_id": ndarray}`.

### mirror/probes.py
- `train_probe(acts, labels, groups, seed=0) -> ProbeResult`: fits an
  sklearn multinomial `LogisticRegression`; evaluates with a held-out-group
  split (train on a subset of prompt-ids, test on the rest); returns
  `ProbeResult(accuracy, control_accuracy, n_classes)` where `control_accuracy`
  is the same pipeline on shuffled labels and must be near `1/n_classes`.
- `prg(probe_accuracy, report_accuracy) -> float`: `probe_accuracy - report_accuracy`.

## Testing (TDD, CPU)

Probes (synthetic, deterministic with a seeded numpy generator):
- Linearly-separable classes: `train_probe` accuracy is high (> 0.8) on
  held-out groups.
- Shuffled-label control returns near chance (`control_accuracy < 2 / n_classes`).
- `prg` returns the difference; a probe at 0.7 and report at 0.1 gives 0.6.

Caching (tiny Llama on CPU, no quantization):
- `probe_activation_hf` returns a `[hidden_size]` vector.
- `collect_prg_hf` with 2 concepts x 3 prompts writes an npz whose
  `activations` is `[6, hidden_size]`, `concept` and `prompt_id` are length 6,
  and the JSONL has 6 records with the expected keys.

## Dependencies

Add `scikit-learn` to core dependencies (numpy/scipy already present). The probe
uses only numpy + scikit-learn; caching uses torch.

## Deliverable

A Kaggle run on 8-bit gemma-2-9b-it (inject layer 21, probe at a late layer,
~16 concepts x ~8 prompts, coherent alpha) producing the headline PRG number,
recorded in the lab notebook: injected concept decodable at X percent from late
activations vs reported at ~0 percent.

## Conventions

Self-describing names, no comments/docstrings. Reuse ConceptVector, the bank,
extract_hf, and the grader. Do not modify the TL modules.
