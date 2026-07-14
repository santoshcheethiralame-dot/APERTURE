# Milestone 7: Activation Patching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Causally test whether injected concept content reaches the output — patch the injected downstream residual into a clean run and measure the concept token's output log-probability change.

**Architecture:** A new `mirror/patching.py` reusing `probe_activation_hf` (to cache the injected residual) plus a replace-hook on the clean run. Single-trial `patch_effect_hf` and a per-concept `collect_patch_hf` with a negative control. HF backend only; TL untouched.

**Tech Stack:** Python 3.11+, torch, transformers. No new dependencies. pytest on the tiny Llama, CPU.

## Global Constraints

- Repo: `C:\Users\carbo\projects\mirror`; commands from repo root; venv at `.venv`.
- No code comments, no docstrings, self-describing names (human-authored convention).
- Commit messages: plain imperative, NO co-author trailers, no AI mentions.
- Do NOT modify the TransformerLens modules.
- `patch_layer` is downstream of the injection `layer`.
- Concept token = leading token of `" " + name`.
- Reuse `hf_layer`, `_hidden`, `probe_activation_hf` from `mirror.hf_model`; the bank from `mirror.concepts`.
- Metric is a log-prob (log-softmax), so values are <= 0; delta = patched - baseline.

---

### Task 1: concept_token + baseline_logprob

**Files:**
- Create: `src/mirror/patching.py`
- Test: `tests/test_patching.py`, `tests/conftest.py` (reuses existing `hf_model`/`hf_tok` fixtures)

**Interfaces:**
- Consumes: `hf_layer`, `_hidden` from `mirror.hf_model`.
- Produces: `concept_token(tok, name) -> int` (leading token id of `" " + name`); `baseline_logprob(model, tok, prompt, target_token) -> float` (clean forward; log-softmax log-prob of `target_token` at the last position).

- [ ] **Step 1: Write the failing tests**

`tests/test_patching.py`:
```python
import torch

from mirror.patching import baseline_logprob, concept_token


def test_concept_token_is_int(hf_tok):
    t = concept_token(hf_tok, "elephant")
    assert isinstance(t, int)


def test_baseline_logprob_is_nonpositive(hf_model, hf_tok):
    t = concept_token(hf_tok, "elephant")
    lp = baseline_logprob(hf_model, hf_tok, "hello world", t)
    assert lp <= 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_patching.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` on `mirror.patching`

- [ ] **Step 3: Write minimal implementation**

`src/mirror/patching.py`:
```python
import torch

from mirror.hf_model import _hidden, hf_layer, probe_activation_hf


def concept_token(tok, name):
    return int(tok(" " + name, add_special_tokens=False).input_ids[0])


def baseline_logprob(model, tok, prompt, target_token):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    return float(logits.log_softmax(-1)[target_token])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_patching.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/patching.py tests/test_patching.py
git commit -m "Add concept token id and baseline logprob"
```

---

### Task 2: patched_logprob

**Files:**
- Modify: `src/mirror/patching.py`
- Test: `tests/test_patching.py` (append)

**Interfaces:**
- Consumes: `hf_layer`, `_hidden`.
- Produces: `patched_logprob(model, tok, prompt, patch_layer, patch_resid, target_token) -> float` — clean forward with a hook at `patch_layer` replacing the last-position residual with `patch_resid` (a 1-D tensor of length hidden_size); returns the log-prob of `target_token` at the last position.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_patching.py`:
```python
def test_patched_logprob_changes_output(hf_model, hf_tok):
    from mirror.patching import patched_logprob
    t = concept_token(hf_tok, "elephant")
    base = baseline_logprob(hf_model, hf_tok, "hello world", t)
    big = torch.ones(hf_model.config.hidden_size) * 50.0
    patched = patched_logprob(hf_model, hf_tok, "hello world", 1, big, t)
    assert patched != base
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_patching.py::test_patched_logprob_changes_output -v`
Expected: FAIL — `ImportError: cannot import name 'patched_logprob'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/patching.py`:
```python
def patched_logprob(model, tok, prompt, patch_layer, patch_resid, target_token):
    def hook(module, inputs, output):
        hidden = _hidden(output)
        hidden[:, -1] = patch_resid.to(hidden.device)
        return output

    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    handle = hf_layer(model, patch_layer).register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(ids).logits[0, -1]
    finally:
        handle.remove()
    return float(logits.log_softmax(-1)[target_token])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_patching.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/patching.py tests/test_patching.py
git commit -m "Add residual replacement patched logprob"
```

---

### Task 3: patch_effect_hf

**Files:**
- Modify: `src/mirror/patching.py`
- Test: `tests/test_patching.py` (append)

**Interfaces:**
- Consumes: `probe_activation_hf`, `baseline_logprob`, `patched_logprob`.
- Produces: `patch_effect_hf(model, tok, prompt, vec, alpha, patch_layer, target_token) -> (baseline_lp, patched_lp, delta)` — caches the injected residual at `patch_layer` via `probe_activation_hf(model, tok, prompt, vec, alpha, patch_layer)`, computes baseline and patched log-probs, delta = patched - baseline.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_patching.py`:
```python
def _tiny_vec(hf_model, hf_tok):
    from mirror.hf_model import extract_hf_vector
    return extract_hf_vector(hf_model, hf_tok, [("a cat", "a dog")], 0)


def test_patch_effect_zero_alpha_is_noop(hf_model, hf_tok):
    from mirror.patching import patch_effect_hf
    vec = _tiny_vec(hf_model, hf_tok)
    t = concept_token(hf_tok, "elephant")
    base, patched, delta = patch_effect_hf(hf_model, hf_tok, "hello world", vec,
                                           0.0, 1, t)
    assert abs(delta) < 1e-4


def test_patch_effect_nonzero_alpha_moves(hf_model, hf_tok):
    from mirror.patching import patch_effect_hf
    vec = _tiny_vec(hf_model, hf_tok)
    t = concept_token(hf_tok, "elephant")
    base, patched, delta = patch_effect_hf(hf_model, hf_tok, "hello world", vec,
                                           200.0, 1, t)
    assert abs(delta) > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_patching.py::test_patch_effect_zero_alpha_is_noop -v`
Expected: FAIL — `ImportError: cannot import name 'patch_effect_hf'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/patching.py`:
```python
def patch_effect_hf(model, tok, prompt, vec, alpha, patch_layer, target_token):
    patch_resid = probe_activation_hf(model, tok, prompt, vec, alpha, patch_layer)
    baseline_lp = baseline_logprob(model, tok, prompt, target_token)
    patched_lp = patched_logprob(model, tok, prompt, patch_layer, patch_resid,
                                 target_token)
    return baseline_lp, patched_lp, patched_lp - baseline_lp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_patching.py -v`
Expected: all PASS

The zero-alpha no-op holds because `probe_activation_hf` with alpha=0 returns the
clean residual at `patch_layer`, and replacing the clean run's residual with the
same value changes nothing.

- [ ] **Step 5: Commit**

```bash
git add src/mirror/patching.py tests/test_patching.py
git commit -m "Add single-trial patch effect"
```

---

### Task 4: collect_patch_hf

**Files:**
- Modify: `src/mirror/patching.py`
- Test: `tests/test_patching.py` (append)

**Interfaces:**
- Consumes: `patch_effect_hf`, `concept_token`, `extract_hf` and `load_bank`.
- Produces: `collect_patch_hf(model, tok, bank, names, layer, patch_layer, alpha, prompt, out, n_pairs=12) -> dict` — extracts each concept vector once at `layer`; for each concept records `self_delta` (patch its own injected residual, measure its own token) and `control_delta` (patch the NEXT concept's injected residual, measure this concept's token). Writes JSONL (one record per concept with keys concept, layer, patch_layer, alpha, self_delta, control_delta) and returns `{"records": [...]}`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_patching.py`:
```python
def test_collect_patch_writes_records(hf_model, hf_tok, tmp_path):
    import json

    from mirror.concepts import load_bank
    from mirror.patching import collect_patch_hf
    bank = load_bank("data/concepts/dev_bank.yaml")
    out = tmp_path / "patch.jsonl"
    result = collect_patch_hf(hf_model, hf_tok, bank,
                              ["elephant", "volcano", "joy"], 0, 1, 1.0,
                              "hello world", str(out), n_pairs=8)
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(result["records"]) == len(lines) == 3
    assert {"concept", "layer", "patch_layer", "alpha", "self_delta", "control_delta"} <= set(lines[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_patching.py::test_collect_patch_writes_records -v`
Expected: FAIL — `ImportError: cannot import name 'collect_patch_hf'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/patching.py` (add `import json`, `from pathlib import Path`, and `from mirror.hf_model import extract_hf` at the top):
```python
import json
from pathlib import Path

from mirror.hf_model import extract_hf


def collect_patch_hf(model, tok, bank, names, layer, patch_layer, alpha, prompt,
                     out, n_pairs=12):
    vecs = {name: extract_hf(model, tok, bank, bank.get(name), layer, n_pairs)
            for name in names}
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out_path.open("a") as f:
        for i, name in enumerate(names):
            target = concept_token(tok, name)
            other = names[(i + 1) % len(names)]
            _, _, self_delta = patch_effect_hf(model, tok, prompt, vecs[name],
                                               alpha, patch_layer, target)
            _, _, control_delta = patch_effect_hf(model, tok, prompt, vecs[other],
                                                  alpha, patch_layer, target)
            record = {
                "concept": name,
                "layer": layer,
                "patch_layer": patch_layer,
                "alpha": alpha,
                "self_delta": self_delta,
                "control_delta": control_delta,
            }
            f.write(json.dumps(record) + "\n")
            records.append(record)
    return {"records": records}
```

- [ ] **Step 4: Run test to verify it passes, then the full suite**

Run: `.venv\Scripts\python -m pytest tests/test_patching.py -v`
Then: `.venv\Scripts\python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit and push**

```bash
git add src/mirror/patching.py tests/test_patching.py
git commit -m "Add per-concept patch collection with negative control"
git push origin main
```

---

### Task 5: Patching notebook

**Files:**
- Modify: `notebooks/kaggle_demo.ipynb`

**Interfaces:**
- Consumes: `collect_patch_hf`; the model loaded via the native-model path.

- [ ] **Step 1: Update the run cell to the patching experiment**

Set the run cell (cell that currently runs the PRG collection) source to:
```python
from mirror.concepts import load_bank
from mirror.patching import collect_patch_hf

PROMPT = "<start_of_turn>user\nDo you notice anything unusual in your thoughts right now?<end_of_turn>\n<start_of_turn>model\n"
NAMES = ["elephant", "spider", "volcano", "desert", "library",
         "joy", "fear", "violin", "telescope", "candle"]
bank = load_bank("dev_bank.yaml")
result = collect_patch_hf(model, tok, bank, NAMES, layer=13, patch_layer=20,
                          alpha=1.0, prompt=PROMPT, out="patch.jsonl", n_pairs=12)
```

- [ ] **Step 2: Update the analysis cell**

Set the analysis cell source to:
```python
records = result["records"]
selfs = [r["self_delta"] for r in records]
ctrls = [r["control_delta"] for r in records]
for r in records:
    print(f"{r['concept']:10} self={r['self_delta']:+.3f}  control={r['control_delta']:+.3f}")
print()
print(f"mean self-delta:    {sum(selfs)/len(selfs):+.3f}")
print(f"mean control-delta: {sum(ctrls)/len(ctrls):+.3f}")
```

- [ ] **Step 3: Validate the notebook JSON**

Run: `.venv\Scripts\python -c "import json; json.load(open('notebooks/kaggle_demo.ipynb')); print('valid')"`
Expected: `valid`

- [ ] **Step 4: Commit and push**

```bash
git add notebooks/kaggle_demo.ipynb
git commit -m "Switch notebook to activation patching experiment"
git push origin main
```

- [ ] **Step 5: Manual gate (user)**

On Kaggle (8-bit 2B via the mounted native Gemma model): Run All. Paste the
per-concept self/control deltas and the two means.

---

### Task 6: Record the patching result

**Files:**
- Modify: `docs/LAB_NOTEBOOK.md`

- [ ] **Step 1: Add the run**

After the Kaggle run, add an `R8` registry row (gemma-2-2b-it 8-bit, inject 13 /
patch 20, alpha 1.0, 10 concepts, patch.jsonl) and a detail entry recording mean
self-delta vs mean control-delta and the interpretation: if self >> control ≈ 0,
injected content reaches the output concept-specifically, so the failure to
report it (R5/R6) is a read-out gap. Flip E5's status note in the family table.

- [ ] **Step 2: Commit and push**

```bash
git add docs/LAB_NOTEBOOK.md
git commit -m "Record activation patching result"
git push origin main
```
