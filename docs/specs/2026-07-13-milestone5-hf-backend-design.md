# Milestone 5 — HF-Transformers Injection Backend

Status: approved 2026-07-13
Scope reference: enables quantized large-model runs (master plan large tier);
follows the discovery that TransformerLens dequantizes 8-bit weights and so
cannot load gemma-2-9b within Kaggle RAM

## Goal

A standalone HF-transformers path for concept-vector extraction, residual-stream
injection, and the KL meter using raw PyTorch forward hooks, so a bitsandbytes
8-bit model stays quantized (TransformerLens never involved) and gemma-2-9b runs
on a single T4. Mirrors the existing TL ops; existing code is untouched.

## In scope

- `mirror/hf_model.py`: load, extract, generate, kl_meter, run over HF models.
- Reuse of `ConceptVector`, the concept bank, `config_hash`, JSONL logging.
- CPU pytest on a tiny Llama-architecture model (no quantization).
- `[gpu]` optional dependency extra (bitsandbytes, accelerate).

## Out of scope (deferred)

Backend abstraction / refactor of the TL code, 4-bit, multi-GPU for HF,
probes/PRG. The existing TransformerLens modules stay as-is for 2B dev.

## How injection works in raw HF

Gemma-2 / Qwen-2 / Llama expose decoder layers at `model.model.layers[L]`; the
layer forward returns a tuple whose first element is the hidden states
`[batch, seq, d]`.

- Extract: a capturing forward hook stores `output[0]`, averaged over positions,
  to build the diff-in-means vector and sigma (median residual norm).
- Inject: a forward hook returns a modified tuple
  `(output[0] + alpha*sigma*v_hat at the last position, *output[1:])`, with a
  call-counter distinguishing span "response" (every step) from "single".
- Generate / KL: register the hook, run `model.generate` or two forward passes,
  decode with the tokenizer.

The math and the `ConceptVector` (including steering/probe/stability flags) match
the TL backend; only the hook mechanism differs.

## Components (`mirror/hf_model.py`)

- `hf_layer(model, layer)` -> `model.model.layers[layer]`.
- `resid_stats_hf(model, tok, prompt, layer) -> (mean_resid, median_norm)`.
- `raw_direction_hf(model, tok, pairs, layer) -> (vector, sigma)`.
- `extract_hf(model, tok, bank, concept, layer, n_pairs=40, seed=0) -> ConceptVector`
  (unit-norm direction, sigma, flags via probe/stability on activations and a
  steering check through the injection hook; same template-held-out split as the
  TL extract).
- `inject_hook(vec, alpha, span) -> callable` forward hook returning the modified
  output tuple.
- `generate_hf(model, tok, prompt, vec=None, alpha=0.0, span="response", max_new_tokens=64, seed=0) -> str`.
- `kl_meter_hf(model, tok, prompt, vec, alpha, span="response") -> float`
  (KL(p_inj || p_clean) at the final prompt position).
- `run_hf(model, tok, cfg) -> list[dict]` mirroring `runner.run`: loops
  concepts x alphas x seeds, writes one JSONL record per trial with the same
  keys (config, concept, layer, alpha, span, seed, kl, flags, clean, report),
  reusing `config_hash`.
- `load_hf(name, load_in_8bit=True) -> (model, tokenizer)` via
  `AutoModelForCausalLM.from_pretrained` with a bitsandbytes 8-bit config and
  `device_map`, plus `AutoTokenizer`.

## Testing (TDD, CPU, tiny Llama, no quantization)

Model: `hf-internal-testing/tiny-random-LlamaForCausalLM` (has `model.model.layers`,
loads on CPU). The injection logic is identical regardless of quantization, so
these tests fully validate it:

- `extract_hf`: direction is unit-norm with the model's hidden size; sigma > 0.
- alpha=0 golden: `generate_hf` with alpha=0 equals the clean generation under a
  fixed seed.
- large alpha: generation visibly changes.
- injection touches only the last position at the hooked layer, by exactly
  alpha*sigma*v_hat; other positions unchanged.
- `kl_meter_hf`: ~0 at alpha=0; > 0 under injection.
- `run_hf`: writes the expected number of JSONL records with the expected keys.

8-bit loading (`load_hf(..., load_in_8bit=True)`) needs CUDA and is verified only
by the Kaggle run, not in CPU tests.

## Dependencies

`bitsandbytes` and `accelerate` go in a new optional extra
`[project.optional-dependencies] gpu`, not core deps, so local CPU dev and tests
never require them (Windows install stays clean). `transformers` is already
available via transformer-lens. The Kaggle notebook installs the `[gpu]` extra.

## Conventions

Self-describing names, no comments/docstrings. Reuse existing dataclasses and
helpers; do not modify the TL modules.
