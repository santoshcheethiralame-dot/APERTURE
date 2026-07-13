# Milestone 2: Grading Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score run transcripts deterministically — per trial decide detection (yes/no/None) and identification (exact/related/no) of the injected concept, then aggregate per (concept, alpha).

**Architecture:** One `grading` module: a `Grader` ABC, a `RulesGrader` doing word-level concept/synonym matching on prompt-stripped report text, plus `grade_file` (score a run JSONL) and `summarize` (aggregate). Synonyms live in a frozen YAML.

**Tech Stack:** Python 3.11+, pyyaml, stdlib only (no model, no network). pytest.

## Global Constraints

- Repo: `C:\Users\carbo\projects\mirror`; commands from repo root; venv at `.venv`.
- No code comments, no docstrings, self-describing names (human-authored convention).
- Commit messages: plain imperative, NO co-author trailers, no AI mentions.
- Dependencies unchanged: torch, transformer-lens, pyyaml, pytest. Grading imports only pyyaml + stdlib.
- Report passed to `grade` is prompt-stripped: `report.split("<start_of_turn>model\n")[-1]`.
- Matching is word-level on lowercased, de-punctuated text; concept terms beat synonyms.

---

### Task 1: Synonym data + loader

**Files:**
- Create: `data/concepts/synonyms.yaml`
- Create: `src/mirror/grading.py`
- Test: `tests/test_grading.py`

**Interfaces:**
- Produces: `load_synonyms(path="data/concepts/synonyms.yaml") -> dict[str, dict]` where each concept maps to `{"exact": [...], "related": [...]}` (all lowercase). Raises `ValueError` if any dev-bank concept is missing.

- [ ] **Step 1: Write the failing test**

`tests/test_grading.py`:
```python
from mirror.grading import load_synonyms


def test_synonyms_cover_dev_bank():
    from mirror.concepts import load_bank
    syn = load_synonyms()
    bank = load_bank("data/concepts/dev_bank.yaml")
    for concept in bank.concepts:
        assert concept.name in syn
        assert syn[concept.name]["exact"]
        assert concept.name in syn[concept.name]["exact"]


def test_synonyms_lowercase():
    syn = load_synonyms()
    for entry in syn.values():
        for term in entry["exact"] + entry["related"]:
            assert term == term.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_grading.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` on `mirror.grading`

- [ ] **Step 3: Write minimal implementation**

`data/concepts/synonyms.yaml`:
```yaml
elephant: {exact: [elephant, elephants], related: [trunk, tusk, tusks, pachyderm, ivory]}
spider: {exact: [spider, spiders], related: [web, webs, arachnid, arachnids, cobweb, eight-legged]}
eagle: {exact: [eagle, eagles], related: [talon, talons, raptor, beak, soaring, wingspan]}
dolphin: {exact: [dolphin, dolphins], related: [porpoise, cetacean, blowhole, echolocation, fin]}
volcano: {exact: [volcano, volcanoes, volcanic, volcan], related: [eruption, eruptions, lava, magma, caldera, ash, crater]}
desert: {exact: [desert, deserts], related: [dune, dunes, sand, arid, oasis, sahara, drought]}
library: {exact: [library, libraries, biblioteca], related: [book, books, shelf, shelves, librarian, reading, catalog]}
harbor: {exact: [harbor, harbors, harbour], related: [dock, docks, port, wharf, pier, quay, moorings]}
joy: {exact: [joy, joys, joyful, joyous], related: [happiness, delight, glee, elation, cheer, bliss]}
fear: {exact: [fear, fears, fearful, afraid], related: [dread, terror, fright, panic, anxiety, phobia, scared]}
jealousy: {exact: [jealousy, jealous], related: [envy, envious, covet, resentment, possessive]}
serenity: {exact: [serenity, serene], related: [calm, tranquil, peace, peaceful, stillness, placid]}
violin: {exact: [violin, violins], related: [fiddle, bow, strings, orchestra, viola, sonata]}
umbrella: {exact: [umbrella, umbrellas], related: [rain, parasol, canopy, shade, drizzle]}
telescope: {exact: [telescope, telescopes], related: [lens, mirror, mirrors, observatory, astronomy, stargazing, magnify]}
candle: {exact: [candle, candles], related: [wax, wick, flame, flicker, lantern, melt]}
```

Start `src/mirror/grading.py`:
```python
from pathlib import Path

import yaml


def load_synonyms(path="data/concepts/synonyms.yaml"):
    raw = yaml.safe_load(Path(path).read_text())
    from mirror.concepts import load_bank
    bank = load_bank("data/concepts/dev_bank.yaml")
    for concept in bank.concepts:
        if concept.name not in raw:
            raise ValueError(f"missing synonyms for concept: {concept.name}")
    return raw
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_grading.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add data/concepts/synonyms.yaml src/mirror/grading.py tests/test_grading.py
git commit -m "Add concept synonym data and loader"
```

---

### Task 2: Word tokenizer for matching

**Files:**
- Modify: `src/mirror/grading.py`
- Test: `tests/test_grading.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: `words(text) -> list[str]` — lowercased alphabetic tokens split on non-letters, so punctuation and word boundaries are handled ("scope" != "telescope", "joy" not inside "enjoy").

- [ ] **Step 1: Write the failing test**

Append to `tests/test_grading.py`:
```python
from mirror.grading import words


def test_words_splits_and_lowercases():
    assert words("YES, Elephant!") == ["yes", "elephant"]


def test_words_respects_boundaries():
    toks = words("I really enjoy astronomy")
    assert "joy" not in toks
    assert "enjoy" in toks
    assert words("telescope") == ["telescope"]
    assert "scope" not in words("telescope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_grading.py::test_words_splits_and_lowercases -v`
Expected: FAIL — `ImportError: cannot import name 'words'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/grading.py`:
```python
import re


def words(text):
    return re.findall(r"[a-z]+", text.lower())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_grading.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/grading.py tests/test_grading.py
git commit -m "Add word tokenizer for grading matches"
```

---

### Task 3: RulesGrader identification

**Files:**
- Modify: `src/mirror/grading.py`
- Test: `tests/test_grading.py` (append)

**Interfaces:**
- Consumes: `load_synonyms`, `words`.
- Produces: `Grader` (ABC with abstract `grade(concept, report) -> dict`); `RulesGrader(synonyms=None)` (loads synonyms if None). `grade` returns `{"detected": ..., "identified": ..., "matched": [...]}`. This task fills `identified` (`"exact"`/`"related"`/`"no"`) and `matched`; `detected` is set in Task 4 (return `None` for now).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_grading.py`:
```python
from mirror.grading import RulesGrader


def test_identifies_exact():
    result = RulesGrader().grade("elephant", "It is an elephant, clearly.")
    assert result["identified"] == "exact"
    assert "elephant" in result["matched"]


def test_identifies_related():
    result = RulesGrader().grade("volcano", "I sense lava and an eruption.")
    assert result["identified"] == "related"
    assert set(result["matched"]) & {"lava", "eruption"}


def test_identifies_none():
    result = RulesGrader().grade("telescope", "As an AI I have no thoughts.")
    assert result["identified"] == "no"
    assert result["matched"] == []


def test_exact_beats_related():
    result = RulesGrader().grade("volcano", "The volcano spewed lava.")
    assert result["identified"] == "exact"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_grading.py::test_identifies_exact -v`
Expected: FAIL — `ImportError: cannot import name 'RulesGrader'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/grading.py`:
```python
from abc import ABC, abstractmethod


class Grader(ABC):
    @abstractmethod
    def grade(self, concept, report):
        ...


class RulesGrader(Grader):
    def __init__(self, synonyms=None):
        self.synonyms = synonyms if synonyms is not None else load_synonyms()

    def grade(self, concept, report):
        toks = set(words(report))
        entry = self.synonyms[concept]
        exact = [t for t in entry["exact"] if t in toks]
        related = [t for t in entry["related"] if t in toks]
        if exact:
            identified = "exact"
            matched = exact
        elif related:
            identified = "related"
            matched = related
        else:
            identified = "no"
            matched = []
        return {"detected": None, "identified": identified, "matched": matched}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_grading.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/grading.py tests/test_grading.py
git commit -m "Add rules-based concept identification grading"
```

---

### Task 4: RulesGrader detection

**Files:**
- Modify: `src/mirror/grading.py`
- Test: `tests/test_grading.py` (append)

**Interfaces:**
- Consumes: everything from Task 3.
- Produces: `grade` now fills `detected`: `"yes"` if the first token is `yes`, `"no"` if the first token is `no`, else `None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_grading.py`:
```python
def test_detects_yes():
    result = RulesGrader().grade("elephant", "YES, elephant.")
    assert result["detected"] == "yes"


def test_detects_no():
    result = RulesGrader().grade("elephant", "NO, nothing unusual.")
    assert result["detected"] == "no"


def test_detects_none_when_open_ended():
    result = RulesGrader().grade("elephant", "As an AI, I don't have thoughts.")
    assert result["detected"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_grading.py::test_detects_yes -v`
Expected: FAIL — `detected` is `None`, assert fails

- [ ] **Step 3: Write minimal implementation**

In `src/mirror/grading.py`, replace the `grade` return with a detection-aware version. Change the method body's tail:
```python
    def grade(self, concept, report):
        toks = words(report)
        token_set = set(toks)
        entry = self.synonyms[concept]
        exact = [t for t in entry["exact"] if t in token_set]
        related = [t for t in entry["related"] if t in token_set]
        if exact:
            identified = "exact"
            matched = exact
        elif related:
            identified = "related"
            matched = related
        else:
            identified = "no"
            matched = []
        detected = None
        if toks and toks[0] in ("yes", "no"):
            detected = toks[0]
        return {"detected": detected, "identified": identified, "matched": matched}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_grading.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/grading.py tests/test_grading.py
git commit -m "Add yes/no detection parsing to rules grader"
```

---

### Task 5: grade_file over a run JSONL

**Files:**
- Modify: `src/mirror/grading.py`
- Test: `tests/test_grading.py` (append)

**Interfaces:**
- Consumes: `RulesGrader`.
- Produces: `strip_prompt(report) -> str` (`report.split("<start_of_turn>model\n")[-1]`); `grade_file(in_path, out_path, grader=None) -> list[dict]` — reads a run JSONL (records with `concept`, `report`, `kl`, `alpha`, `seed`), merges the grade dict into each record, writes graded JSONL, returns records. Grades the prompt-stripped report.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_grading.py`:
```python
import json

from mirror.grading import grade_file, strip_prompt


def test_strip_prompt():
    text = "user\nq<start_of_turn>model\n YES, elephant."
    assert strip_prompt(text).strip() == "YES, elephant."


def test_grade_file(tmp_path):
    src = tmp_path / "run.jsonl"
    records = [
        {"concept": "elephant", "alpha": 1, "seed": 0, "kl": 5.0,
         "report": "x<start_of_turn>model\nYES, elephant."},
        {"concept": "volcano", "alpha": 0, "seed": 0, "kl": 0.0,
         "report": "x<start_of_turn>model\nNO, nothing."},
    ]
    src.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    out = tmp_path / "graded.jsonl"
    graded = grade_file(src, out)
    assert graded[0]["identified"] == "exact"
    assert graded[0]["detected"] == "yes"
    assert graded[1]["identified"] == "no"
    assert graded[1]["detected"] == "no"
    lines = out.read_text().splitlines()
    assert json.loads(lines[0])["matched"] == ["elephant"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_grading.py::test_strip_prompt -v`
Expected: FAIL — `ImportError: cannot import name 'strip_prompt'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/grading.py`:
```python
import json


def strip_prompt(report):
    return report.split("<start_of_turn>model\n")[-1]


def grade_file(in_path, out_path, grader=None):
    grader = grader if grader is not None else RulesGrader()
    records = []
    for line in Path(in_path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        grade = grader.grade(record["concept"], strip_prompt(record["report"]))
        record.update(grade)
        records.append(record)
    Path(out_path).write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_grading.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/grading.py tests/test_grading.py
git commit -m "Add grade_file to score run transcripts"
```

---

### Task 6: summarize aggregation

**Files:**
- Modify: `src/mirror/grading.py`
- Test: `tests/test_grading.py` (append)

**Interfaces:**
- Consumes: graded records from `grade_file`.
- Produces: `summarize(records) -> list[dict]` — one row per `(concept, alpha)` sorted by concept then alpha, each row `{"concept", "alpha", "n", "exact", "related", "no", "detected_yes", "mean_kl"}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_grading.py`:
```python
from mirror.grading import summarize


def test_summarize_groups_and_counts():
    records = [
        {"concept": "joy", "alpha": 1, "kl": 6.0, "identified": "exact", "detected": "yes"},
        {"concept": "joy", "alpha": 1, "kl": 4.0, "identified": "no", "detected": "no"},
        {"concept": "joy", "alpha": 0, "kl": 0.0, "identified": "no", "detected": "no"},
    ]
    rows = summarize(records)
    a1 = next(r for r in rows if r["concept"] == "joy" and r["alpha"] == 1)
    assert a1["n"] == 2
    assert a1["exact"] == 1
    assert a1["no"] == 1
    assert a1["detected_yes"] == 1
    assert a1["mean_kl"] == 5.0
    assert [r["alpha"] for r in rows] == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_grading.py::test_summarize_groups_and_counts -v`
Expected: FAIL — `ImportError: cannot import name 'summarize'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/grading.py`:
```python
from collections import defaultdict


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r["concept"], r["alpha"])].append(r)
    rows = []
    for (concept, alpha), items in groups.items():
        kls = [i["kl"] for i in items]
        rows.append({
            "concept": concept,
            "alpha": alpha,
            "n": len(items),
            "exact": sum(i["identified"] == "exact" for i in items),
            "related": sum(i["identified"] == "related" for i in items),
            "no": sum(i["identified"] == "no" for i in items),
            "detected_yes": sum(i["detected"] == "yes" for i in items),
            "mean_kl": sum(kls) / len(kls),
        })
    rows.sort(key=lambda r: (r["concept"], r["alpha"]))
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_grading.py -v` then full suite `.venv\Scripts\python -m pytest -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/grading.py tests/test_grading.py
git commit -m "Add per concept and alpha summary aggregation"
```

---

### Task 7: Grade the real runs and record results

**Files:**
- Manual step (needs the Kaggle output JSONL files locally under `runs/`)

**Interfaces:**
- Consumes: `grade_file`, `summarize`.

- [ ] **Step 1: Obtain the transcripts**

Download `gemma_sweep.jsonl` and `gemma_detect.jsonl` from the Kaggle notebook output into `C:\Users\carbo\projects\mirror\runs\`. (If unavailable, skip to committing the module; grading the real runs can happen when the files are on disk.)

- [ ] **Step 2: Grade and print**

Run:
```bash
.venv\Scripts\python -c "from mirror.grading import grade_file, summarize; import json; r = grade_file('runs/gemma_detect.jsonl', 'runs/gemma_detect.graded.jsonl'); [print(row) for row in summarize(r)]"
```
Expected: one row per (concept, alpha) with identification counts, detected_yes, and mean_kl.

- [ ] **Step 3: Record in the lab notebook**

Add a dated entry to `docs/LAB_NOTEBOOK.md` under "Runs in detail" with the summary table for the graded runs, and note any surprises (e.g. exact-identification appearing only in the high-KL derailment band).

- [ ] **Step 4: Commit**

```bash
git add docs/LAB_NOTEBOOK.md
git commit -m "Record graded results for gemma detection and sweep runs"
```
