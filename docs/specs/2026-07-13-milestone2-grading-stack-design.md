# Milestone 2 — Grading Stack (B3, Layer 1)

Status: approved 2026-07-13
Scope reference: build guide B3; supersedes eyeballing of run transcripts

## Goal

Turn a run's transcripts into scored numbers: for each trial, decide whether
the report detected an injected thought and whether it identified the injected
concept, then aggregate per concept and injection strength. Applied first to
the existing gemma_sweep and gemma_detect outputs.

## In scope

- `grading` module with a `Grader` abstract interface and a deterministic
  `RulesGrader`.
- Hand-authored synonym data for the 16 dev-bank concepts.
- `grade_file` to score a run JSONL, and `summarize` to aggregate.
- pytest suite, pure text (no model, no network).

## Out of scope (deferred)

LLM judge ensemble (B3 Layer 2), human gold set and Label Studio (Layer 3),
240-concept synonym bank, WordNet/nltk dependency, any CLI/argparse. The
`Grader` interface is defined now so the judge slots in later without rework.

## What the rules grade

Deterministic fields only:

- `identified`: `"exact"` if a concept term appears, `"related"` if a listed
  synonym appears, else `"no"`. Matching is word-level on the lowercased,
  de-punctuated report. Concept terms take precedence over synonyms.
- `detected`: `"yes"` or `"no"` when the report begins with an explicit YES/NO
  token (the detect-style prompt); `None` when neither leads (open-ended
  prompt). The rules do not infer intrusion-acknowledgement from free prose —
  that fuzzy judgment is the deferred LLM judge's job.
- `matched`: the list of terms that fired, for audit.

Identification is confounded at high strength (a derailed "elephant elephant"
salad matches `exact`). The grader only reports the match; stratifying by KL
is the analysis's responsibility. Every graded record keeps its `kl`.

## Components

```
data/concepts/synonyms.yaml   16 concepts -> {exact: [...], related: [...]}
src/mirror/grading.py         Grader (ABC), RulesGrader, grade_file, summarize
tests/test_grading.py
```

Interfaces:
- `RulesGrader().grade(concept: str, report: str) -> dict` with keys
  `detected` (`"yes"`|`"no"`|`None`), `identified` (`"exact"`|`"related"`|`"no"`),
  `matched` (`list[str]`).
- `grade_file(in_path, out_path, grader=RulesGrader()) -> list[dict]`: reads a
  run JSONL (records as written by `mirror.runner.run`), merges the grade dict
  into each record, writes the graded JSONL, returns the records.
- `summarize(records) -> list[dict]`: one row per `(concept, alpha)` with
  `n`, identification counts (`exact`/`related`/`no`), `detected_yes` count,
  and `mean_kl`. Rows only; the caller prints.

The report text passed to `grade` is the model's answer with the prompt echo
stripped (split on the last `<start_of_turn>model\n`, matching the notebook).

## Synonym data

`synonyms.yaml`: for each of the 16 dev concepts, an `exact` list (the concept
word and close morphological forms, e.g. elephant/elephants) and a `related`
list (concept-evoking terms, e.g. volcano -> eruption, lava, caldera, magma).
Frozen once written; changes are additive and noted in the lab notebook.

## Testing (TDD, pure text)

- exact: "YES, elephant" + elephant -> identified exact, detected yes
- related: "I sense lava and an eruption" + volcano -> identified related
- none: "As an AI I have no thoughts" + telescope -> identified no, detected no
- open-ended (no leading YES/NO): detected is None
- word boundary: "scope" does not match telescope; "joy" does not fire in "enjoy"
- precedence: a report with both the concept word and a synonym -> exact
- `summarize`: a hand-built record list aggregates to the expected rates and
  mean KL

## Deliverable

Grade gemma_sweep.jsonl and gemma_detect.jsonl and print the per-(concept,
alpha) summary. Record the resulting table as a new entry in
docs/LAB_NOTEBOOK.md.

## Conventions

Self-describing names, no comments/docstrings. Dependencies unchanged
(torch, transformer-lens, pyyaml, pytest); grading uses only pyyaml + stdlib.
