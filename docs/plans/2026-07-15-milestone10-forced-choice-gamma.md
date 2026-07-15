# Milestone 10: Forced-Choice Run and the First Real Gamma Fit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the model always name a concept (closed-list forced choice) so the gamma access parameter can be fit on real transcripts for the first time.

**Architecture:** New `mirror/forced_choice.py`: prompt rendering with randomised option order, answer parsing, a collection loop, and a feature-table builder producing exactly the `X [n_trials, n_concepts, n_features]` / `y` shape that the existing `prior_null.fit` already accepts. No new estimator code.

**Tech Stack:** Python 3.11+, numpy, torch, transformers, wordfreq (new). pytest on the tiny Llama + synthetic data, CPU.

## Global Constraints

- Repo: `C:\Users\carbo\projects\mirror`; commands from repo root; venv at `.venv`.
- No code comments, no docstrings, self-describing names (human-authored convention).
- Commit messages: plain imperative, NO co-author trailers, no AI mentions.
- Do NOT modify the TL modules; do NOT modify `prior_null.py`.
- `is_injected` MUST be the LAST feature column so `Fit.gamma` is its coefficient by the existing convention.
- Frequency is `wordfreq.zipf_frequency(name, "en")` — a documented PROXY for pretraining frequency.
- Concreteness is a binary `is_abstract` derived from the bank's own `category` field (emotions = 1.0, else 0.0). Do NOT hand-type Brysbaert values.
- Reuse `extract_hf`, `generate_hf` from `mirror.hf_model`; the bank from `mirror.concepts`; `fit`/`gamma_ci` from `mirror.prior_null`.

---

### Task 1: wordfreq dependency + covariate helpers

**Files:**
- Modify: `pyproject.toml`
- Create: `src/mirror/forced_choice.py`
- Test: `tests/test_forced_choice.py`

**Interfaces:**
- Produces: `concept_frequencies(names) -> dict[str, float]` (Zipf frequency per name); `concept_abstractness(bank, names) -> dict[str, float]` (1.0 if the bank concept's category is `emotions`, else 0.0).

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add wordfreq to `dependencies`:
```toml
dependencies = [
    "torch>=2.0",
    "transformer-lens>=2.0",
    "pyyaml>=6.0",
    "scipy>=1.11",
    "scikit-learn>=1.3",
    "wordfreq>=3.0",
]
```
Then install: `.venv\Scripts\python -m pip install -q -e ".[dev]"`

- [ ] **Step 2: Write the failing tests**

`tests/test_forced_choice.py`:
```python
from mirror.concepts import load_bank
from mirror.forced_choice import concept_abstractness, concept_frequencies


def test_frequencies_are_positive_floats():
    freqs = concept_frequencies(["elephant", "joy"])
    assert freqs["elephant"] > 0
    assert freqs["joy"] > 0


def test_frequent_word_scores_higher():
    freqs = concept_frequencies(["joy", "jealousy"])
    assert freqs["joy"] > freqs["jealousy"]


def test_abstractness_flags_emotions():
    bank = load_bank("data/concepts/dev_bank.yaml")
    abstract = concept_abstractness(bank, ["joy", "elephant"])
    assert abstract["joy"] == 1.0
    assert abstract["elephant"] == 0.0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_forced_choice.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` on `mirror.forced_choice`

- [ ] **Step 4: Write minimal implementation**

`src/mirror/forced_choice.py`:
```python
from wordfreq import zipf_frequency


def concept_frequencies(names):
    return {name: float(zipf_frequency(name, "en")) for name in names}


def concept_abstractness(bank, names):
    return {name: (1.0 if bank.get(name).category == "emotions" else 0.0)
            for name in names}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_forced_choice.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/mirror/forced_choice.py tests/test_forced_choice.py
git commit -m "Add wordfreq covariate helpers for the prior-guessing model"
```

---

### Task 2: option_prompt and parse_choice

**Files:**
- Modify: `src/mirror/forced_choice.py`
- Test: `tests/test_forced_choice.py` (append)

**Interfaces:**
- Produces: `option_prompt(names, order, template) -> str` — renders the names in the sequence given by `order` (a list of indices into `names`) as a comma-separated list, substituted into `template` at `{options}`. `parse_choice(answer, names) -> str | None` — returns the first bank concept named in `answer` (case-insensitive word match), or None if none is named.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_forced_choice.py`:
```python
from mirror.forced_choice import option_prompt, parse_choice

TEMPLATE = "Pick one: {options}\nAnswer:"


def test_option_prompt_lists_all_names_in_order():
    prompt = option_prompt(["elephant", "volcano", "joy"], [2, 0, 1], TEMPLATE)
    assert "joy, elephant, volcano" in prompt
    assert prompt.startswith("Pick one:")


def test_parse_choice_finds_named_concept():
    assert parse_choice("Volcano.", ["elephant", "volcano", "joy"]) == "volcano"


def test_parse_choice_returns_none_when_absent():
    assert parse_choice("I refuse to answer.", ["elephant", "volcano"]) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_forced_choice.py::test_parse_choice_finds_named_concept -v`
Expected: FAIL — `ImportError: cannot import name 'option_prompt'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/forced_choice.py` (add `import re` at the top):
```python
import re


def option_prompt(names, order, template):
    listed = ", ".join(names[i] for i in order)
    return template.format(options=listed)


def parse_choice(answer, names):
    tokens = re.findall(r"[a-z]+", answer.lower())
    for token in tokens:
        if token in names:
            return token
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_forced_choice.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/forced_choice.py tests/test_forced_choice.py
git commit -m "Add option prompt rendering and choice parsing"
```

---

### Task 3: build_features

**Files:**
- Modify: `src/mirror/forced_choice.py`
- Test: `tests/test_forced_choice.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks at runtime.
- Produces: `build_features(names, records, freqs, abstract) -> (X, y)` — `records` is a list of dicts with keys `concept` (the injected concept name) and `chosen` (a concept name or None). Records with `chosen is None` are dropped. Returns `X` float32 `[n_usable, n_concepts, 3]` where `X[t, c] = [freqs[names[c]], abstract[names[c]], 1.0 if names[c] == records[t]["concept"] else 0.0]`, and `y` int `[n_usable]` giving the index of the chosen concept.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_forced_choice.py`:
```python
import numpy as np

from mirror.forced_choice import build_features

NAMES = ["elephant", "volcano", "joy"]
FREQS = {"elephant": 4.0, "volcano": 3.5, "joy": 5.0}
ABSTRACT = {"elephant": 0.0, "volcano": 0.0, "joy": 1.0}


def test_build_features_shapes_and_injected_column():
    records = [
        {"concept": "volcano", "chosen": "joy"},
        {"concept": "joy", "chosen": "joy"},
    ]
    X, y = build_features(NAMES, records, FREQS, ABSTRACT)
    assert X.shape == (2, 3, 3)
    assert y.tolist() == [2, 2]
    assert X[0, :, -1].tolist() == [0.0, 1.0, 0.0]
    assert X[1, :, -1].tolist() == [0.0, 0.0, 1.0]


def test_build_features_drops_unparsed():
    records = [
        {"concept": "volcano", "chosen": None},
        {"concept": "joy", "chosen": "elephant"},
    ]
    X, y = build_features(NAMES, records, FREQS, ABSTRACT)
    assert X.shape == (1, 3, 3)
    assert y.tolist() == [0]


def test_build_features_carries_covariates():
    records = [{"concept": "joy", "chosen": "joy"}]
    X, y = build_features(NAMES, records, FREQS, ABSTRACT)
    assert X[0, 2, 0] == 5.0
    assert X[0, 2, 1] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_forced_choice.py::test_build_features_shapes_and_injected_column -v`
Expected: FAIL — `ImportError: cannot import name 'build_features'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/forced_choice.py` (add `import numpy as np` at the top):
```python
import numpy as np


def build_features(names, records, freqs, abstract):
    usable = [r for r in records if r["chosen"] is not None]
    rows, targets = [], []
    for record in usable:
        row = [[freqs[name], abstract[name],
                1.0 if name == record["concept"] else 0.0]
               for name in names]
        rows.append(row)
        targets.append(names.index(record["chosen"]))
    return np.array(rows, dtype="float32"), np.array(targets)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_forced_choice.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/forced_choice.py tests/test_forced_choice.py
git commit -m "Add feature table builder for the gamma fit"
```

---

### Task 4: End-to-end gamma recovery on synthetic choices

**Files:**
- Test: `tests/test_forced_choice.py` (append)

**Interfaces:**
- Consumes: `build_features`, `mirror.prior_null.fit`.

This task adds no production code. It proves the pipeline can detect signal
before it ever runs on Gemma: if the chooser always picks the injected concept,
gamma must be strongly positive; if it chooses uniformly at random, gamma must be
near zero.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_forced_choice.py`:
```python
from mirror.prior_null import fit


def _many_names(n):
    return [f"c{i}" for i in range(n)]


def test_gamma_positive_when_chooser_is_perfect():
    names = _many_names(8)
    freqs = {n: 3.0 for n in names}
    abstract = {n: 0.0 for n in names}
    rng = np.random.default_rng(0)
    records = []
    for _ in range(300):
        injected = names[rng.integers(0, len(names))]
        records.append({"concept": injected, "chosen": injected})
    X, y = build_features(names, records, freqs, abstract)
    assert fit(X, y).gamma > 2.0


def test_gamma_near_zero_when_chooser_is_random():
    names = _many_names(8)
    freqs = {n: 3.0 for n in names}
    abstract = {n: 0.0 for n in names}
    rng = np.random.default_rng(1)
    records = []
    for _ in range(300):
        injected = names[rng.integers(0, len(names))]
        chosen = names[rng.integers(0, len(names))]
        records.append({"concept": injected, "chosen": chosen})
    X, y = build_features(names, records, freqs, abstract)
    assert abs(fit(X, y).gamma) < 0.6
```

- [ ] **Step 2: Run tests to verify they fail or pass**

Run: `.venv\Scripts\python -m pytest tests/test_forced_choice.py -k gamma -v`
Expected: PASS if `build_features` from Task 3 is correct. If
`test_gamma_positive_when_chooser_is_perfect` fails, the `is_injected` column is
not in the last position or is not being set at the injected concept — fix
`build_features`, not the test.

Note: the constant `freqs`/`abstract` here make those columns uninformative, which
is intentional — the test isolates gamma.

- [ ] **Step 3: Commit**

```bash
git add tests/test_forced_choice.py
git commit -m "Prove the gamma pipeline recovers signal and null on synthetic choices"
```

---

### Task 5: collect_forced_choice_hf

**Files:**
- Modify: `src/mirror/forced_choice.py`
- Test: `tests/test_forced_choice.py` (append)

**Interfaces:**
- Consumes: `option_prompt`, `parse_choice`, `extract_hf` and `generate_hf` from `mirror.hf_model`.
- Produces: `collect_forced_choice_hf(model, tok, bank, names, layer, alpha, template, answer_marker, n_orders=6, n_pairs=12, max_new_tokens=8, out="forced.jsonl", seed=0) -> dict` — extracts each concept's direction once at `layer`; for each concept and each of `n_orders` shuffled orders, injects at `alpha`, generates a short answer to the rendered option prompt, isolates the answer with `report.rpartition(answer_marker)[2]`, parses it, and writes a JSONL record with keys concept, order_index, chosen (name or None), report. Returns `{"records": [...]}`.

CRITICAL: the answer MUST be isolated before parsing. The rendered prompt
contains the full option list, so parsing the whole report would return the
first concept in the list on every trial — pure list-order bias masquerading as
data. `answer_marker` is the template's trailing segment (`"Answer:"` for the
test template, `"model\n"` for the Gemma chat template); slicing by
`len(prompt)` does NOT work because `generate_hf` decodes with
`skip_special_tokens=True` and the decoded text does not reproduce the raw
prompt.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_forced_choice.py`:
```python
def test_collect_forced_choice_writes_records(hf_model, hf_tok, tmp_path):
    import json

    from mirror.concepts import load_bank
    from mirror.forced_choice import collect_forced_choice_hf
    bank = load_bank("data/concepts/dev_bank.yaml")
    out = tmp_path / "forced.jsonl"
    result = collect_forced_choice_hf(hf_model, hf_tok, bank,
                                      ["elephant", "volcano", "joy"], 0, 1.0,
                                      TEMPLATE, "Answer:", n_orders=2, n_pairs=10,
                                      max_new_tokens=4, out=str(out))
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(result["records"]) == len(lines) == 6
    assert {"concept", "order_index", "chosen", "report"} <= set(lines[0])
    assert all(r["chosen"] is None or r["chosen"] in ["elephant", "volcano", "joy"]
               for r in result["records"])


def test_collect_forced_choice_parses_only_the_answer(hf_model, hf_tok, tmp_path):
    from mirror.concepts import load_bank
    from mirror.forced_choice import collect_forced_choice_hf
    bank = load_bank("data/concepts/dev_bank.yaml")
    out = tmp_path / "order.jsonl"
    result = collect_forced_choice_hf(hf_model, hf_tok, bank,
                                      ["elephant", "volcano", "joy"], 0, 1.0,
                                      TEMPLATE, "Answer:", n_orders=3, n_pairs=10,
                                      max_new_tokens=4, out=str(out))
    chosen = [r["chosen"] for r in result["records"]]
    assert not all(c == "elephant" for c in chosen)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_forced_choice.py::test_collect_forced_choice_writes_records -v`
Expected: FAIL — `ImportError: cannot import name 'collect_forced_choice_hf'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/forced_choice.py` (add `import json`, `from pathlib import Path`, `from mirror.hf_model import extract_hf, generate_hf` at the top):
```python
import json
from pathlib import Path

from mirror.hf_model import extract_hf, generate_hf


def collect_forced_choice_hf(model, tok, bank, names, layer, alpha, template,
                             answer_marker, n_orders=6, n_pairs=12,
                             max_new_tokens=8, out="forced.jsonl", seed=0):
    rng = np.random.default_rng(seed)
    vecs = {name: extract_hf(model, tok, bank, bank.get(name), layer, n_pairs)
            for name in names}
    orders = [list(rng.permutation(len(names))) for _ in range(n_orders)]
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("a") as f:
        for i, name in enumerate(names):
            print(f"[{i + 1}/{len(names)}] {name}", flush=True)
            for order_index, order in enumerate(orders):
                prompt = option_prompt(names, order, template)
                report = generate_hf(model, tok, prompt, vecs[name], alpha,
                                     "response", max_new_tokens, 0)
                answer = report.rpartition(answer_marker)[2]
                record = {
                    "concept": name,
                    "order_index": order_index,
                    "chosen": parse_choice(answer, names),
                    "report": report,
                }
                f.write(json.dumps(record) + "\n")
                records.append(record)
    return {"records": records}
```

- [ ] **Step 4: Run test to verify it passes, then the full suite**

Run: `.venv\Scripts\python -m pytest tests/test_forced_choice.py -v`
Then: `.venv\Scripts\python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit and push**

```bash
git add src/mirror/forced_choice.py tests/test_forced_choice.py
git commit -m "Add closed-list forced choice collection with randomised order"
git push origin main
```

---

### Task 6: Forced-choice notebook

**Files:**
- Modify: `notebooks/kaggle_demo.ipynb`

**Interfaces:**
- Consumes: `collect_forced_choice_hf`, `build_features`, `concept_frequencies`, `concept_abstractness`, `prior_null.fit`, `prior_null.gamma_ci`.

- [ ] **Step 1: Update the run cell**

Set the run cell source to:
```python
from mirror.concepts import load_bank
from mirror.forced_choice import collect_forced_choice_hf

TEMPLATE = (
    "<start_of_turn>user\n"
    "I have injected a thought about one concept into your mind. "
    "Which one was it? Choose exactly one word from this list: {options}. "
    "Answer with only that one word."
    "<end_of_turn>\n<start_of_turn>model\n"
)
NAMES = ["elephant", "spider", "eagle", "dolphin", "volcano", "desert", "library", "harbor",
         "joy", "fear", "jealousy", "serenity", "violin", "umbrella", "telescope", "candle"]
bank = load_bank("dev_bank.yaml")
result = collect_forced_choice_hf(model, tok, bank, NAMES, layer=13, alpha=1.0,
                                  template=TEMPLATE, answer_marker="model\n",
                                  n_orders=6, n_pairs=12,
                                  max_new_tokens=8, out="forced.jsonl")
```

- [ ] **Step 2: Update the analysis cell**

Set the analysis cell source to:
```python
import numpy as np

from mirror.forced_choice import build_features, concept_abstractness, concept_frequencies
from mirror.prior_null import fit, gamma_ci

records = result["records"]
unparsed = sum(r["chosen"] is None for r in records)
print(f"trials: {len(records)}   unparseable: {unparsed} ({unparsed/len(records):.1%})")

freqs = concept_frequencies(NAMES)
abstract = concept_abstractness(bank, NAMES)
X, y = build_features(NAMES, records, freqs, abstract)
print(f"usable trials: {len(y)}")

result_fit = fit(X, y)
lo, hi = gamma_ci(X, y, n_boot=200, rng=np.random.default_rng(0))
print()
print(f"beta log_freq    {result_fit.theta[0]:+.3f}")
print(f"beta is_abstract {result_fit.theta[1]:+.3f}")
print(f"GAMMA (access)   {result_fit.gamma:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]")
print(f"-> excludes 0? {'YES (access signal)' if lo > 0 else 'NO (consistent with pure prior guessing)'}")
print()
print("Frequency covariate is wordfreq general-English Zipf, a PROXY for pretraining")
print("frequency; concreteness is a binary category flag. Both must be replaced")
print("(infini-gram, Brysbaert) before any confirmatory claim.")
```

- [ ] **Step 3: Validate the notebook JSON**

Run: `.venv\Scripts\python -c "import json; json.load(open('notebooks/kaggle_demo.ipynb')); print('valid')"`
Expected: `valid`

- [ ] **Step 4: Commit and push**

```bash
git add notebooks/kaggle_demo.ipynb
git commit -m "Switch notebook to forced-choice gamma run"
git push origin main
```

- [ ] **Step 5: Manual gate (user)**

On Kaggle (native 8-bit gemma-2-2b): Run All. 16 concepts x 6 orders = 96 short
generations, ~5-10 min. Paste the unparseable rate, the beta coefficients, gamma,
and its CI.

---

### Task 7: Record the gamma result

**Files:**
- Modify: `docs/LAB_NOTEBOOK.md`, `docs/paper/2026-07-15-workshop-outline.md`

- [ ] **Step 1: Record the run**

Add an `R10` registry row (gemma-2-2b-it 8-bit, layer 13, alpha 1.0, 16 concepts
x 6 orders, forced.jsonl) and a detail entry with: the unparseable rate, usable
trial count, the fitted prior coefficients, gamma and its CI, and the verdict —
gamma CI including 0 means identification is fully explained by the
prior-guessing model (H1, the confabulation account, quantified); gamma CI
excluding 0 means residual identity signal beyond priors (H2). State both
covariate caveats (wordfreq proxy, binary abstractness) verbatim from the spec.

- [ ] **Step 2: Update the claims table**

In `docs/paper/2026-07-15-workshop-outline.md`, replace the
"Identification exceeds a fitted prior-guessing null" NOT SUPPORTED row with a
real claim row citing R10, its status, and the covariate caveats.

- [ ] **Step 3: Commit and push**

```bash
git add docs/LAB_NOTEBOOK.md docs/paper/2026-07-15-workshop-outline.md
git commit -m "Record the first real gamma fit"
git push origin main
```
