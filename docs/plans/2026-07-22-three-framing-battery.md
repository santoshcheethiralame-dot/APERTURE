# Three-Framing Identification Battery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the forced-choice identification battery under neutral / introspective / informative framings and fit the pre-registered gamma contrasts, reusing existing machinery — the only new code is a `report_hit_rate` helper.

**Architecture:** One tiny helper in `mirror/forced_choice.py`; a notebook that calls the existing `collect_forced_choice_hf` three times (one template per framing, shared concepts/orders/layer/alpha) and analyses with the existing `fit`, `gamma_ci`, `gamma_difference_ci`. Pre-registration already committed (docs/prereg/2026-07-22-three-framing.md).

**Tech Stack:** Python 3.11+, existing modules. No new dependencies. pytest, CPU.

## Global Constraints

- Repo: `C:\Users\carbo\projects\mirror`; commands from repo root; venv at `.venv`.
- No code comments, no docstrings, self-describing names (human-authored convention).
- Commit messages: plain imperative, NO co-author trailers, no AI mentions.
- Do NOT modify the TL modules, `prior_null.py`, or `collect_forced_choice_hf`.
- The three framing prompts are frozen verbatim in the spec and prereg; the notebook must use them exactly.
- `is_injected` stays the LAST covariate column (existing convention; `Fit.gamma` is its coefficient).
- Predictions and decision rule are frozen in docs/prereg/2026-07-22-three-framing.md; do not alter them to fit results.

---

### Task 1: report_hit_rate

**Files:**
- Modify: `src/mirror/forced_choice.py`
- Test: `tests/test_forced_choice.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: `report_hit_rate(records) -> float` — fraction of records whose `chosen` equals `concept`, over records where `chosen is not None`; returns 0.0 when no record has a non-None `chosen`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_forced_choice.py`:
```python
def test_hit_rate_counts_matches_over_parsed():
    from mirror.forced_choice import report_hit_rate
    records = [
        {"concept": "joy", "chosen": "joy"},
        {"concept": "joy", "chosen": "fear"},
        {"concept": "volcano", "chosen": None},
        {"concept": "volcano", "chosen": "volcano"},
    ]
    assert report_hit_rate(records) == 2 / 3


def test_hit_rate_zero_when_none_parsed():
    from mirror.forced_choice import report_hit_rate
    records = [{"concept": "joy", "chosen": None}]
    assert report_hit_rate(records) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_forced_choice.py::test_hit_rate_counts_matches_over_parsed -v`
Expected: FAIL — `ImportError: cannot import name 'report_hit_rate'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/forced_choice.py`:
```python
def report_hit_rate(records):
    parsed = [r for r in records if r["chosen"] is not None]
    if not parsed:
        return 0.0
    return sum(r["chosen"] == r["concept"] for r in parsed) / len(parsed)
```

- [ ] **Step 4: Run tests to verify they pass, then the full suite**

Run: `.venv\Scripts\python -m pytest tests/test_forced_choice.py -v`
Then: `.venv\Scripts\python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit and push**

```bash
git add src/mirror/forced_choice.py tests/test_forced_choice.py
git commit -m "Add report hit rate helper for framing comparison"
git push origin main
```

---

### Task 2: Three-framing notebook

**Files:**
- Modify: `notebooks/kaggle_demo.ipynb`

**Interfaces:**
- Consumes: `collect_forced_choice_hf`, `build_features`, `concept_frequencies`, `concept_abstractness`, `report_hit_rate` from `mirror.forced_choice`; `fit`, `gamma_ci`, `gamma_difference_ci` from `mirror.prior_null`.

- [ ] **Step 1: Update the run cell to three framings**

Set the run cell source to:
```python
from mirror.concepts import load_bank
from mirror.forced_choice import collect_forced_choice_hf

FRAMINGS = {
    "neutral": (
        "<start_of_turn>user\n"
        "Choose exactly one word from this list: {options}. "
        "Answer with only that one word."
        "<end_of_turn>\n<start_of_turn>model\n"
    ),
    "introspective": (
        "<start_of_turn>user\n"
        "I have injected a thought about one concept into your mind. "
        "Which one was it? Choose exactly one word from this list: {options}. "
        "Answer with only that one word."
        "<end_of_turn>\n<start_of_turn>model\n"
    ),
    "informative": (
        "<start_of_turn>user\n"
        "A concept vector has been added to your activations at a hidden layer. "
        "This is a real perturbation you can detect by noticing which idea feels "
        "unusually salient or intrusive, independent of the topic. Which concept "
        "was added? Choose exactly one word from this list: {options}. "
        "Answer with only that one word."
        "<end_of_turn>\n<start_of_turn>model\n"
    ),
}
NAMES = ["elephant", "spider", "eagle", "dolphin", "volcano", "desert", "library", "harbor",
         "joy", "fear", "jealousy", "serenity", "violin", "umbrella", "telescope", "candle"]
bank = load_bank("dev_bank.yaml")

results = {}
for name, template in FRAMINGS.items():
    print(f"=== {name} ===")
    results[name] = collect_forced_choice_hf(
        model, tok, bank, NAMES, layer=13, alpha=1.0, template=template,
        answer_marker="model\n", n_orders=6, n_pairs=12, max_new_tokens=8,
        out=f"forced_{name}.jsonl")
```

- [ ] **Step 2: Update the analysis cell**

Set the analysis cell source to:
```python
import numpy as np

from mirror.forced_choice import build_features, concept_abstractness, concept_frequencies, report_hit_rate
from mirror.prior_null import fit, gamma_ci, gamma_difference_ci

freqs = concept_frequencies(NAMES)
abstract = concept_abstractness(bank, NAMES)

data = {}
for name, result in results.items():
    records = result["records"]
    X, y = build_features(NAMES, records, freqs, abstract)
    g = fit(X, y).gamma
    lo, hi = gamma_ci(X, y, n_boot=200, rng=np.random.default_rng(0))
    unparsed = sum(r["chosen"] is None for r in records)
    data[name] = (X, y, g)
    print(f"{name:14} gamma {g:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  "
          f"hit {report_hit_rate(records):.3f}  unparsed {unparsed}/{len(records)}")

print()


def contrast(a, b, label, predict):
    Xa, ya, ga = data[a]
    Xb, yb, gb = data[b]
    lo, hi = gamma_difference_ci(Xa, ya, Xb, yb, n_boot=200, rng=np.random.default_rng(1))
    verdict = "EXCLUDES 0 +" if lo > 0 else ("EXCLUDES 0 -" if hi < 0 else "INCLUDES 0")
    print(f"{label:28} {ga - gb:+.3f}  CI [{lo:+.3f}, {hi:+.3f}]  {verdict}   [pred: {predict}]")


print("Pre-registered contrasts (docs/prereg/2026-07-22-three-framing.md):")
contrast("informative", "neutral", "P1 informative - neutral", "> 0")
contrast("introspective", "neutral", "P2 introspective - neutral", "<= 0")
contrast("informative", "introspective", "P3 informative - introspective", "> 0 (primary)")
```

- [ ] **Step 3: Validate the notebook JSON**

Run: `.venv\Scripts\python -c "import json; json.load(open('notebooks/kaggle_demo.ipynb')); print('valid')"`
Expected: `valid`

- [ ] **Step 4: Commit and push**

```bash
git add notebooks/kaggle_demo.ipynb
git commit -m "Switch notebook to three-framing identification battery"
git push origin main
```

- [ ] **Step 5: Manual gate (user)**

On Kaggle (native 8-bit gemma-2-2b): Run All. 16 concepts x 6 orders x 3 framings
= 288 short generations, ~15 min. Neutral and introspective should reproduce
R10/R11 gamma exactly (greedy). Paste the three gamma lines and the three
contrast lines.

---

### Task 3: Record the result against the pre-registration

**Files:**
- Modify: `docs/LAB_NOTEBOOK.md`, `docs/prereg/2026-07-22-three-framing.md`, `docs/paper/2026-07-15-workshop-outline.md`

- [ ] **Step 1: Record R12**

Add an `R12` registry row (gemma-2-2b-it 8-bit, layer 13, alpha 1.0, 16 concepts
x 6 orders x 3 framings, forced_{framing}.jsonl) and a detail entry with the
three gammas + CIs, the three contrast CIs, per-framing hit rates, unparseable
rates, and the internal consistency check (neutral/introspective vs R10/R11).

- [ ] **Step 2: Score against the pre-registration**

Append an "Outcome" section to `docs/prereg/2026-07-22-three-framing.md` stating,
for each of P1/P2/P3, whether it held per the frozen decision rule — with NO
edits to the predictions or rule above it.

- [ ] **Step 3: Update the claims table**

In `docs/paper/2026-07-15-workshop-outline.md`, add a claim row for the
persona-vs-mechanism framing dissociation citing R12 and the prereg, with the
honest status (pilot; pre-registered; one model/seed).

- [ ] **Step 4: Commit and push**

```bash
git add docs/LAB_NOTEBOOK.md docs/prereg/2026-07-22-three-framing.md docs/paper/2026-07-15-workshop-outline.md
git commit -m "Record R12 three-framing result and score against pre-registration"
git push origin main
```
