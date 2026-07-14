# Milestone 8 — Naturalistic Arm (E8)

Status: approved 2026-07-15
Scope reference: build guide B11; master plan RQ6/E8; answers the "injections are
OOD damage, not real thoughts" objection to R3-R8

## Goal

Show that a concept is decodably present in a NATURALLY induced state (normal
reading, no injection) and that the injection-derived concept directions decode
that natural state. This answers the main objection to the injection paradigm at
the representational level: the phenomenon is not an artifact of out-of-
distribution injection damage. A verbal report is also measured, with an
explicit comprehension caveat.

## In scope

- `mirror/naturalistic.py`: no-injection activation capture, nearest-direction
  classification, and a per-concept collection loop.
- `data/concepts/contexts.yaml`: one evocative passage per dev concept, with the
  concept word ABSENT so a correct report cannot be word-copying.
- CPU pytest on the tiny Llama and synthetic vectors.
- A Kaggle run on native 8-bit gemma-2-2b.

## Out of scope (deferred)

Probe training on naturalistic activations, the full gamma access test on
natural states, model-generated (self-induced) states, multi-context-per-concept
grids.

## Honest limit

The representational claim (concept present in a natural state, decoded by
injection directions) is clean. The verbal-report measurement carries a
comprehension confound: with the passage in context, "what were you thinking
about" partly reduces to "what was the passage about". The report is recorded
and interpreted with this caveat; it is not presented as pure introspection.

## The experiment

For each concept X with an evocative passage `ctx_X` (X's word absent):
1. Induce: run the model on `ctx_X` (no injection).
2. Verify presence: cache the last-token residual at `layer`; classify by
   nearest concept direction (argmax over concepts of activation . v_hat, using
   the directions from `extract_hf`). If the nearest direction is X, the concept
   is linearly present in the natural state.
3. Report: build `ctx_X + distractor + report_suffix`, generate a short answer,
   grade whether it names X (rules grader, exact/related counts as a hit).

## Components (`mirror/naturalistic.py`)

- `last_activation_hf(model, tok, prompt, layer) -> Tensor` — forward pass with a
  capture hook at `layer`; returns the last-position residual `[hidden_size]`;
  no injection; hook removed after.
- `nearest_concept(activation, directions) -> str` — `directions` is a dict
  `name -> unit direction tensor`; returns the name maximizing
  `activation . direction`.
- `collect_naturalistic_hf(model, tok, bank, contexts, distractor, report_suffix, layer, n_pairs=12) -> dict`:
  extracts each concept's direction once via `extract_hf`; for each concept in
  `contexts` records `predicted` (nearest_concept of the passage activation),
  `report` (generated from passage+distractor+report_suffix), and `identified`
  (grader verdict on the report). Writes JSONL (one record per concept with keys
  concept, predicted, report, identified) and returns `{"records": [...]}`.

`contexts` is a dict `name -> passage`; the loop iterates its concepts, all of
which must be in the bank.

## Data: contexts.yaml

One passage per dev-bank concept, ~2-3 sentences, evoking the concept strongly
via associated imagery while never using the concept word or an obvious
morphological variant. Frozen once written.

## Testing (TDD, tiny Llama + synthetic, CPU)

- `nearest_concept` returns the concept whose direction best aligns: with
  synthetic orthogonal directions and an activation equal to one of them, that
  concept is returned.
- `last_activation_hf` returns a `[hidden_size]` tensor and leaves no hook
  registered (a second identical call returns the same value).
- `collect_naturalistic_hf` with 3 concepts and 3 passages writes 3 JSONL
  records with keys concept, predicted, report, identified.

## Deliverable

A Kaggle run on native 8-bit gemma-2-2b (layer 13): identifiability (fraction of
passages whose activation is nearest the true concept direction), verbal report
accuracy, and the gap. High identifiability with lagging report - the present-
but-not-fully-reported pattern WITHOUT injection - answers the OOD objection at
the representational level. Recorded as R9.

## Conventions

Self-describing names, no comments/docstrings. Reuse `extract_hf`, `_hidden`,
`hf_layer` from `hf_model`, the grader from `grading`, the bank from `concepts`.
No new dependencies. Do not modify the TL modules.
