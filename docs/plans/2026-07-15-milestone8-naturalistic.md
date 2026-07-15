# Milestone 8: Naturalistic Arm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a concept is decodably present in a naturally induced state (normal reading, no injection) and that injection-derived directions decode it — answering the "injections are OOD damage" objection.

**Architecture:** New `mirror/naturalistic.py`: a no-injection last-token activation capture, a nearest-concept-direction classifier, and a per-concept collection loop reusing `extract_hf` (directions), `generate_hf` (report), and the rules grader. Plus `data/concepts/contexts.yaml` — one evocative passage per concept with the concept word absent.

**Tech Stack:** Python 3.11+, torch, transformers, pyyaml. No new dependencies. pytest on the tiny Llama + synthetic vectors, CPU.

## Global Constraints

- Repo: `C:\Users\carbo\projects\mirror`; commands from repo root; venv at `.venv`.
- No code comments, no docstrings, self-describing names (human-authored convention).
- Commit messages: plain imperative, NO co-author trailers, no AI mentions.
- Do NOT modify the TransformerLens modules.
- No injection anywhere in this arm — the concept is induced by reading.
- Every passage in `contexts.yaml` must NOT contain its concept word or an obvious morphological variant.
- Reuse `_hidden`, `hf_layer`, `extract_hf`, `generate_hf` from `mirror.hf_model`; `RulesGrader`/`strip_prompt` from `mirror.grading`; the bank from `mirror.concepts`.

---

### Task 1: nearest_concept

**Files:**
- Create: `src/mirror/naturalistic.py`
- Test: `tests/test_naturalistic.py`

**Interfaces:**
- Produces: `nearest_concept(activation, directions) -> str` — `directions` is a dict `name -> 1-D tensor`; returns the name maximizing `activation . direction`.

- [ ] **Step 1: Write the failing tests**

`tests/test_naturalistic.py`:
```python
import torch

from mirror.naturalistic import nearest_concept


def test_nearest_concept_picks_aligned_direction():
    directions = {
        "elephant": torch.tensor([1.0, 0.0, 0.0]),
        "volcano": torch.tensor([0.0, 1.0, 0.0]),
        "joy": torch.tensor([0.0, 0.0, 1.0]),
    }
    assert nearest_concept(torch.tensor([0.0, 5.0, 0.0]), directions) == "volcano"


def test_nearest_concept_handles_mixed_activation():
    directions = {
        "elephant": torch.tensor([1.0, 0.0]),
        "volcano": torch.tensor([0.0, 1.0]),
    }
    assert nearest_concept(torch.tensor([3.0, 1.0]), directions) == "elephant"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_naturalistic.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` on `mirror.naturalistic`

- [ ] **Step 3: Write minimal implementation**

`src/mirror/naturalistic.py`:
```python
import torch


def nearest_concept(activation, directions):
    scores = {name: float(activation.float() @ d.float().to(activation.device))
              for name, d in directions.items()}
    return max(scores, key=scores.get)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_naturalistic.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/naturalistic.py tests/test_naturalistic.py
git commit -m "Add nearest concept direction classifier"
```

---

### Task 2: last_activation_hf

**Files:**
- Modify: `src/mirror/naturalistic.py`
- Test: `tests/test_naturalistic.py` (append)

**Interfaces:**
- Consumes: `_hidden`, `hf_layer` from `mirror.hf_model`.
- Produces: `last_activation_hf(model, tok, prompt, layer) -> Tensor` — forward pass with a capture hook at `layer`, no injection; returns the last-position residual `[hidden_size]`; removes the hook afterwards.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_naturalistic.py`:
```python
def test_last_activation_shape(hf_model, hf_tok):
    from mirror.naturalistic import last_activation_hf
    act = last_activation_hf(hf_model, hf_tok, "the ground trembled", 1)
    assert act.shape == (hf_model.config.hidden_size,)


def test_last_activation_leaves_no_hook(hf_model, hf_tok):
    from mirror.hf_model import hf_layer
    from mirror.naturalistic import last_activation_hf
    last_activation_hf(hf_model, hf_tok, "the ground trembled", 1)
    assert len(hf_layer(hf_model, 1)._forward_hooks) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_naturalistic.py::test_last_activation_shape -v`
Expected: FAIL — `ImportError: cannot import name 'last_activation_hf'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/naturalistic.py` (add `from mirror.hf_model import _hidden, hf_layer` at the top):
```python
from mirror.hf_model import _hidden, hf_layer


def last_activation_hf(model, tok, prompt, layer):
    captured = {}

    def grab(module, inputs, output):
        captured["resid"] = _hidden(output).detach()

    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    handle = hf_layer(model, layer).register_forward_hook(grab)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    return captured["resid"][0, -1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_naturalistic.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/naturalistic.py tests/test_naturalistic.py
git commit -m "Add no-injection last token activation capture"
```

---

### Task 3: contexts.yaml

**Files:**
- Create: `data/concepts/contexts.yaml`
- Test: `tests/test_naturalistic.py` (append)

**Interfaces:**
- Produces: `load_contexts(path="data/concepts/contexts.yaml") -> dict[str, str]` — concept name to passage; raises `ValueError` if a passage contains its own concept word.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_naturalistic.py`:
```python
def test_contexts_cover_dev_bank():
    from mirror.concepts import load_bank
    from mirror.naturalistic import load_contexts
    contexts = load_contexts()
    bank = load_bank("data/concepts/dev_bank.yaml")
    for concept in bank.concepts:
        assert concept.name in contexts
        assert len(contexts[concept.name]) > 40


def test_contexts_never_name_their_concept():
    from mirror.naturalistic import load_contexts
    for name, passage in load_contexts().items():
        assert name not in passage.lower()


def test_load_contexts_rejects_leaked_concept(tmp_path):
    import pytest

    from mirror.naturalistic import load_contexts
    bad = tmp_path / "contexts.yaml"
    bad.write_text("volcano: The volcano erupted loudly over the valley below.\n")
    with pytest.raises(ValueError):
        load_contexts(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_naturalistic.py::test_contexts_cover_dev_bank -v`
Expected: FAIL — `ImportError: cannot import name 'load_contexts'`

- [ ] **Step 3: Write the data file**

`data/concepts/contexts.yaml` (each passage evokes its concept without ever using the word or an obvious variant):
```yaml
elephant: >
  The grey giant swayed through the tall grass, ears flapping like sails, its
  long trunk curling around a branch. Ivory tusks caught the light as the herd
  moved slowly toward the waterhole.
spider: >
  Eight thin legs moved across the silk, each thread strung between the fence
  posts and beaded with dew. The weaver waited at the centre for something to
  stumble into the trap.
eagle: >
  It circled high on the thermals, wings rigid and vast, scanning the valley
  floor. Then the hooked beak turned, the talons dropped, and it fell like a
  stone toward the rabbit below.
dolphin: >
  Sleek and grey, it broke the surface in a smooth arc, breathing through the
  hole on top of its head. It clicked and whistled to the pod, reading the water
  ahead by the echoes that returned.
volcano: >
  The ground trembled and a plume of ash darkened the sky. Molten rock surged up
  through the mountain and spilled glowing down its flanks while the villagers
  fled the crater's roar.
desert: >
  Nothing but dunes to the horizon, the sand shifting in the dry wind. No rain
  had fallen for years, and the caravan moved from one shaded well to the next
  under a merciless sun.
library: >
  Rows of quiet shelves stretched into the dim, spines catalogued and ordered by
  number. Someone turned a page; the keeper stamped a due date and pointed a
  student toward the reading room.
harbor: >
  The boats knocked gently against the wooden pier, ropes creaking at the
  moorings. Cranes swung cargo onto the quay while gulls circled the sheltered
  water behind the sea wall.
joy: >
  She could not stop smiling; her chest felt light and warm and everything
  seemed to shine. Laughter kept bubbling up, and she wanted to tell everyone at
  once about the good news.
fear: >
  His heart hammered and his mouth went dry as the footsteps came closer in the
  dark. Every muscle tensed to run, and a cold dread crawled up his spine.
jealousy: >
  He watched them laughing together and felt something sour twist in his gut. It
  should have been him beside her, and he could not stop counting everything
  they had that he did not.
serenity: >
  The lake lay perfectly still under the morning mist, unbroken and quiet. Her
  breathing slowed, the noise in her head faded, and there was nothing left to
  hurry toward.
violin: >
  She tucked the polished wood beneath her chin and drew the horsehair bow
  across four tight strings. The orchestra hushed as the melody rose, thin and
  singing, from the hollow body.
umbrella: >
  The rain came down hard, so she snapped open the canopy of black nylon over
  her head. Its ribs caught the wind and turned inside out halfway across the
  square.
telescope: >
  He aimed the long tube at the night sky and adjusted the focus. The curved
  mirror gathered the faint light of distant stars, and craters on the moon
  swam into view.
candle: >
  The wick caught and a small flame flickered, throwing soft light across the
  table. Wax pooled and ran down the side as the light guttered in the draught.
```

- [ ] **Step 4: Write the loader**

Add to `src/mirror/naturalistic.py` (add `from pathlib import Path` and `import yaml` at the top):
```python
from pathlib import Path

import yaml


def load_contexts(path="data/concepts/contexts.yaml"):
    raw = yaml.safe_load(Path(path).read_text())
    for name, passage in raw.items():
        if name in passage.lower():
            raise ValueError(f"context for {name} names its own concept")
    return raw
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_naturalistic.py -v`
Expected: all PASS. If `test_contexts_never_name_their_concept` fails, the named
passage leaks its concept word — rewrite that passage to evoke the concept
without the word (that is the whole point of the design).

- [ ] **Step 6: Commit**

```bash
git add data/concepts/contexts.yaml src/mirror/naturalistic.py tests/test_naturalistic.py
git commit -m "Add concept-evoking context passages and validating loader"
```

---

### Task 4: collect_naturalistic_hf

**Files:**
- Modify: `src/mirror/naturalistic.py`
- Test: `tests/test_naturalistic.py` (append)

**Interfaces:**
- Consumes: `nearest_concept`, `last_activation_hf`, `extract_hf` and `generate_hf` from `mirror.hf_model`, `RulesGrader`/`strip_prompt` from `mirror.grading`.
- Produces: `collect_naturalistic_hf(model, tok, bank, contexts, distractor, report_suffix, layer, n_pairs=12, max_new_tokens=16, out="naturalistic.jsonl") -> dict` — extracts each concept's direction once at `layer`; for each concept in `contexts`: captures the passage's last-token activation (no injection), classifies it with `nearest_concept`, generates a report from `passage + distractor + report_suffix`, grades it. Writes JSONL (one record per concept with keys concept, predicted, report, identified) and returns `{"records": [...]}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_naturalistic.py`:
```python
def test_collect_naturalistic_writes_records(hf_model, hf_tok, tmp_path):
    import json

    from mirror.concepts import load_bank
    from mirror.naturalistic import collect_naturalistic_hf
    bank = load_bank("data/concepts/dev_bank.yaml")
    contexts = {
        "elephant": "The grey giant swayed through the tall grass.",
        "volcano": "The ground trembled and ash darkened the sky.",
        "joy": "She could not stop smiling and everything seemed to shine.",
    }
    out = tmp_path / "nat.jsonl"
    result = collect_naturalistic_hf(hf_model, hf_tok, bank, contexts,
                                     "\n\nQ: What is two plus two?\nA: Four.\n\n",
                                     "Q: In one word, what concept was most on your mind?\nA:",
                                     0, n_pairs=10, max_new_tokens=4, out=str(out))
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(result["records"]) == len(lines) == 3
    assert {"concept", "predicted", "report", "identified"} <= set(lines[0])
    assert lines[0]["predicted"] in contexts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_naturalistic.py::test_collect_naturalistic_writes_records -v`
Expected: FAIL — `ImportError: cannot import name 'collect_naturalistic_hf'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/naturalistic.py` (add `import json`, `from mirror.grading import RulesGrader`, `from mirror.hf_model import extract_hf, generate_hf` at the top).

The generated answer is isolated by splitting on the LAST `A:` — `report_suffix`
ends with `A:` and the model's answer follows it, so `rpartition` reliably
returns just the answer even though the distractor also contains an `A:`. This
matches the notebook's display logic.

```python
import json

from mirror.grading import RulesGrader
from mirror.hf_model import extract_hf, generate_hf


def collect_naturalistic_hf(model, tok, bank, contexts, distractor, report_suffix,
                            layer, n_pairs=12, max_new_tokens=16,
                            out="naturalistic.jsonl"):
    directions = {name: extract_hf(model, tok, bank, bank.get(name), layer,
                                   n_pairs).direction
                  for name in contexts}
    grader = RulesGrader()
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("a") as f:
        for name, passage in contexts.items():
            act = last_activation_hf(model, tok, passage, layer)
            predicted = nearest_concept(act, directions)
            prompt = passage + distractor + report_suffix
            report = generate_hf(model, tok, prompt, max_new_tokens=max_new_tokens)
            answer = report.rpartition("A:")[2]
            record = {
                "concept": name,
                "predicted": predicted,
                "report": report,
                "identified": grader.grade(name, answer)["identified"],
            }
            f.write(json.dumps(record) + "\n")
            records.append(record)
    return {"records": records}
```

- [ ] **Step 4: Run test to verify it passes, then the full suite**

Run: `.venv\Scripts\python -m pytest tests/test_naturalistic.py -v`
Then: `.venv\Scripts\python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit and push**

```bash
git add src/mirror/naturalistic.py tests/test_naturalistic.py
git commit -m "Add naturalistic collection loop with direction classification and grading"
git push origin main
```

---

### Task 5: Naturalistic notebook

**Files:**
- Modify: `notebooks/kaggle_demo.ipynb`

**Interfaces:**
- Consumes: `load_contexts`, `collect_naturalistic_hf`; the model loaded via the mounted Kaggle Gemma path.

- [ ] **Step 1: Update the data cell to also fetch contexts.yaml**

Set the data cell source to:
```python
import os, urllib.request
os.makedirs("data/concepts", exist_ok=True)
base = "https://raw.githubusercontent.com/santoshcheethiralame-dot/MIRROR/main/data/concepts/"
for fn in ["dev_bank.yaml", "synonyms.yaml", "contexts.yaml"]:
    urllib.request.urlretrieve(base + fn, f"data/concepts/{fn}")
    urllib.request.urlretrieve(base + fn, fn)
```

- [ ] **Step 2: Update the run cell**

Set the run cell source to:
```python
from mirror.concepts import load_bank
from mirror.naturalistic import collect_naturalistic_hf, load_contexts

DISTRACTOR = "\n\nQ: What is the capital of France?\nA: Paris.\n\n"
REPORT = "Q: Setting that aside, in one word, what concept was most on your mind just now?\nA:"
bank = load_bank("dev_bank.yaml")
contexts = load_contexts("data/concepts/contexts.yaml")
result = collect_naturalistic_hf(model, tok, bank, contexts, DISTRACTOR, REPORT,
                                 layer=13, n_pairs=12, max_new_tokens=12,
                                 out="naturalistic.jsonl")
```

- [ ] **Step 3: Update the analysis cell**

Set the analysis cell source to:
```python
records = result["records"]
hits = sum(r["predicted"] == r["concept"] for r in records)
reported = sum(r["identified"] in ("exact", "related") for r in records)
for r in records:
    mark = "OK " if r["predicted"] == r["concept"] else "   "
    ans = r["report"].rpartition("A:")[2].strip().replace("\n", " ")
    print(f"{mark}{r['concept']:10} nearest={r['predicted']:10} id={r['identified']:8} {ans[:40]}")
print()
print(f"activation identifiability: {hits}/{len(records)} = {hits/len(records):.3f}")
print(f"verbal report accuracy:     {reported}/{len(records)} = {reported/len(records):.3f}")
print(f"naturalistic gap:           {(hits - reported)/len(records):+.3f}")
```

- [ ] **Step 4: Validate the notebook JSON**

Run: `.venv\Scripts\python -c "import json; json.load(open('notebooks/kaggle_demo.ipynb')); print('valid')"`
Expected: `valid`

- [ ] **Step 5: Commit and push**

```bash
git add notebooks/kaggle_demo.ipynb
git commit -m "Switch notebook to naturalistic arm run"
git push origin main
```

- [ ] **Step 6: Manual gate (user)**

On Kaggle (native 8-bit gemma-2-2b): Run All. Paste the per-concept lines and the
three summary numbers.

---

### Task 6: Record the naturalistic result

**Files:**
- Modify: `docs/LAB_NOTEBOOK.md`

- [ ] **Step 1: Add the run**

After the Kaggle run, add an `R9` registry row (gemma-2-2b-it 8-bit, layer 13, no
injection, 16 concepts, naturalistic.jsonl) and a detail entry recording
activation identifiability, verbal report accuracy, and the gap, with the
interpretation: high identifiability means concepts are linearly present in
natural (non-injected) states and the injection-derived directions decode them,
answering the "injections are OOD damage" objection at the representational
level. State the comprehension caveat on the report side explicitly. Flip E8's
status in the family table.

- [ ] **Step 2: Commit and push**

```bash
git add docs/LAB_NOTEBOOK.md
git commit -m "Record naturalistic arm result"
git push origin main
```
