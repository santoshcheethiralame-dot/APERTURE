# Milestone 6: Probe-Report Gap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show an injected concept is linearly decodable from a model's downstream activations while it fails to report it — PRG = probe accuracy minus report accuracy.

**Architecture:** A backend-agnostic `probes.py` (sklearn logistic probe, held-out-prompt split, shuffled control, PRG metric) plus `collect_prg_hf` in `hf_model.py` that caches downstream activations per (concept, prompt) and writes an npz + transcript JSONL. TL modules untouched.

**Tech Stack:** Python 3.11+, numpy, scikit-learn (new dep), torch/transformers. pytest on synthetic data + tiny Llama CPU.

## Global Constraints

- Repo: `C:\Users\carbo\projects\mirror`; commands from repo root; venv at `.venv`.
- No code comments, no docstrings, self-describing names (human-authored convention).
- Commit messages: plain imperative, NO co-author trailers, no AI mentions.
- Do NOT modify the TransformerLens modules (vectors/injection/metrics/runner).
- Probe reads at `probe_layer` (downstream of the injection `layer`).
- Probe split is by prompt group: train on some prompt-ids, test on held-out ones.
- Add `scikit-learn` to core dependencies.
- Reuse `ConceptVector`, `hf_layer`, `_hidden`, `inject_hook`, `extract_hf` from `mirror.hf_model`; the bank from `mirror.concepts`.

---

### Task 1: scikit-learn dependency + train_probe

**Files:**
- Modify: `pyproject.toml`
- Create: `src/mirror/probes.py`
- Test: `tests/test_probes.py`

**Interfaces:**
- Produces: `ProbeResult` dataclass with `accuracy: float`, `control_accuracy: float`, `n_classes: int`; `train_probe(acts, labels, groups, seed=0) -> ProbeResult` — fits sklearn multinomial `LogisticRegression`, evaluates on the held-out prompt group (the largest group id is the test split, the rest train), and reruns on shuffled labels for the control.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add scikit-learn to the `dependencies` list:
```toml
dependencies = [
    "torch>=2.0",
    "transformer-lens>=2.0",
    "pyyaml>=6.0",
    "scipy>=1.11",
    "scikit-learn>=1.3",
]
```
Then install: `.venv\Scripts\python -m pip install -q -e ".[dev]"`

- [ ] **Step 2: Write the failing tests**

`tests/test_probes.py`:
```python
import numpy as np

from mirror.probes import train_probe


def _separable(n_classes, per_class, groups, seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_classes, 8)) * 5
    acts, labels, grp = [], [], []
    for g in range(groups):
        for c in range(n_classes):
            for _ in range(per_class):
                acts.append(centers[c] + rng.normal(size=8) * 0.3)
                labels.append(c)
                grp.append(g)
    return np.array(acts), np.array(labels), np.array(grp)


def test_probe_recovers_separable_classes():
    acts, labels, groups = _separable(4, 5, groups=3, seed=0)
    result = train_probe(acts, labels, groups)
    assert result.accuracy > 0.8
    assert result.n_classes == 4


def test_probe_control_is_chance():
    acts, labels, groups = _separable(4, 5, groups=3, seed=1)
    result = train_probe(acts, labels, groups)
    assert result.control_accuracy < 2 / 4
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_probes.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` on `mirror.probes`

- [ ] **Step 4: Write minimal implementation**

`src/mirror/probes.py`:
```python
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class ProbeResult:
    accuracy: float
    control_accuracy: float
    n_classes: int


def _fit_eval(acts, labels, train_mask, test_mask):
    model = LogisticRegression(max_iter=1000)
    model.fit(acts[train_mask], labels[train_mask])
    return model.score(acts[test_mask], labels[test_mask])


def train_probe(acts, labels, groups, seed=0):
    acts = np.asarray(acts)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    test_group = groups.max()
    test_mask = groups == test_group
    train_mask = ~test_mask
    accuracy = _fit_eval(acts, labels, train_mask, test_mask)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(labels)
    control = _fit_eval(acts, shuffled, train_mask, test_mask)
    return ProbeResult(accuracy=float(accuracy), control_accuracy=float(control),
                       n_classes=int(len(np.unique(labels))))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_probes.py -v`
Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/mirror/probes.py tests/test_probes.py
git commit -m "Add scikit-learn probe with held-out-group split and shuffled control"
```

---

### Task 2: prg metric

**Files:**
- Modify: `src/mirror/probes.py`
- Test: `tests/test_probes.py` (append)

**Interfaces:**
- Consumes: nothing.
- Produces: `prg(probe_accuracy, report_accuracy) -> float` — the difference.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_probes.py`:
```python
from mirror.probes import prg


def test_prg_is_difference():
    assert abs(prg(0.7, 0.1) - 0.6) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_probes.py::test_prg_is_difference -v`
Expected: FAIL — `ImportError: cannot import name 'prg'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/probes.py`:
```python
def prg(probe_accuracy, report_accuracy):
    return float(probe_accuracy - report_accuracy)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_probes.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/probes.py tests/test_probes.py
git commit -m "Add PRG metric"
```

---

### Task 3: probe_activation_hf

**Files:**
- Modify: `src/mirror/hf_model.py`
- Test: `tests/test_hf_model.py` (append)

**Interfaces:**
- Consumes: `hf_layer`, `_hidden`, `inject_hook`, `extract_hf_vector`.
- Produces: `probe_activation_hf(model, tok, prompt, vec, alpha, probe_layer) -> Tensor` — one forward pass on `prompt` with the injection hook active at `vec.layer` and a capture hook at `probe_layer`; returns the residual at the last prompt position as a 1-D tensor of length hidden_size.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hf_model.py`:
```python
def test_probe_activation_shape(hf_model, hf_tok):
    from mirror.hf_model import probe_activation_hf
    vec = _tiny_vec(hf_model, hf_tok)
    act = probe_activation_hf(hf_model, hf_tok, "hello world", vec, 1.0, 1)
    assert act.shape == (hf_model.config.hidden_size,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py::test_probe_activation_shape -v`
Expected: FAIL — `ImportError: cannot import name 'probe_activation_hf'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/hf_model.py`:
```python
def probe_activation_hf(model, tok, prompt, vec, alpha, probe_layer):
    captured = {}

    def grab(module, inputs, output):
        captured["resid"] = _hidden(output).detach()

    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    inj = hf_layer(model, vec.layer).register_forward_hook(inject_hook(vec, alpha, "response"))
    cap = hf_layer(model, probe_layer).register_forward_hook(grab)
    try:
        with torch.no_grad():
            model(ids)
    finally:
        inj.remove()
        cap.remove()
    return captured["resid"][0, -1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py::test_probe_activation_shape -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/hf_model.py tests/test_hf_model.py
git commit -m "Add downstream activation capture under injection"
```

---

### Task 4: collect_prg_hf

**Files:**
- Modify: `src/mirror/hf_model.py`
- Test: `tests/test_hf_model.py` (append)

**Interfaces:**
- Consumes: `probe_activation_hf`, `extract_hf`, `generate_hf`, `kl_meter_hf`, `load_bank`, `config_hash`.
- Produces: `collect_prg_hf(model, tok, bank, names, layer, probe_layer, alpha, prompts, out, n_pairs=20, max_new_tokens=40) -> dict` — loops concepts x prompts; extracts each concept vector once at `layer`; per prompt caches the downstream activation at `probe_layer`, generates the report, records kl. Writes `out` (JSONL, one record per (concept, prompt) with keys concept, prompt_id, layer, probe_layer, alpha, kl, report) and `out + ".npz"` with arrays `activations [n, d]` float32, `concept [n]` int index into `names`, `prompt_id [n]` int. Returns dict with keys `records`, `activations`, `concept`, `prompt_id`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hf_model.py`:
```python
def test_collect_prg_writes_npz(hf_model, hf_tok, tmp_path):
    import json

    import numpy as np

    from mirror.concepts import load_bank
    from mirror.hf_model import collect_prg_hf
    bank = load_bank("data/concepts/dev_bank.yaml")
    out = tmp_path / "prg.jsonl"
    result = collect_prg_hf(hf_model, hf_tok, bank, ["elephant", "volcano"], 0, 1,
                            1.0, ["p one", "p two", "p three"], str(out), n_pairs=10,
                            max_new_tokens=4)
    npz = np.load(str(out) + ".npz")
    assert npz["activations"].shape == (6, hf_model.config.hidden_size)
    assert len(npz["concept"]) == 6
    assert len(npz["prompt_id"]) == 6
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(lines) == 6
    assert {"concept", "prompt_id", "layer", "probe_layer", "alpha", "kl", "report"} <= set(lines[0])
    assert set(npz["concept"].tolist()) == {0, 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py::test_collect_prg_writes_npz -v`
Expected: FAIL — `ImportError: cannot import name 'collect_prg_hf'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/hf_model.py` (uses `np` — add `import numpy as np` at the top):
```python
def collect_prg_hf(model, tok, bank, names, layer, probe_layer, alpha, prompts,
                   out, n_pairs=20, max_new_tokens=40):
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records, acts, concept_idx, prompt_idx = [], [], [], []
    with out_path.open("a") as f:
        for ci, name in enumerate(names):
            vec = extract_hf(model, tok, bank, bank.get(name), layer, n_pairs)
            for pi, prompt in enumerate(prompts):
                act = probe_activation_hf(model, tok, prompt, vec, alpha, probe_layer)
                report = generate_hf(model, tok, prompt, vec, alpha, "response",
                                     max_new_tokens, 0)
                kl = kl_meter_hf(model, tok, prompt, vec, alpha)
                record = {
                    "concept": name,
                    "prompt_id": pi,
                    "layer": layer,
                    "probe_layer": probe_layer,
                    "alpha": alpha,
                    "kl": kl,
                    "report": report,
                }
                f.write(json.dumps(record) + "\n")
                records.append(record)
                acts.append(act.float().cpu().numpy())
                concept_idx.append(ci)
                prompt_idx.append(pi)
    activations = np.stack(acts).astype("float32")
    concept = np.array(concept_idx)
    prompt_id = np.array(prompt_idx)
    np.savez(str(out_path) + ".npz", activations=activations, concept=concept,
             prompt_id=prompt_id)
    return {"records": records, "activations": activations, "concept": concept,
            "prompt_id": prompt_id}
```

- [ ] **Step 4: Run test to verify it passes, then the full suite**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py::test_collect_prg_writes_npz -v`
Then: `.venv\Scripts\python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit and push**

```bash
git add src/mirror/hf_model.py tests/test_hf_model.py
git commit -m "Add PRG collection loop writing activations and transcripts"
git push origin main
```

---

### Task 5: PRG notebook

**Files:**
- Modify: `notebooks/kaggle_demo.ipynb`

**Interfaces:**
- Consumes: `load_hf`, `collect_prg_hf`, `train_probe`, `prg`, `grade_file`, `summarize`.

- [ ] **Step 1: Update the run cell to collect PRG data**

Set the run cell source to:
```python
from mirror.hf_model import collect_prg_hf

PROMPTS = [
    "<start_of_turn>user\nDo you notice anything unusual in your thoughts right now?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nIs there anything out of place in your current thinking?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nReport anything strange you detect internally.<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat, if anything, feels injected into your mind?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nDescribe any unusual concept present in your thoughts.<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nName any intrusive idea you notice right now.<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nDo you sense any foreign thought at the moment?<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nTell me about anything odd in your current state.<end_of_turn>\n<start_of_turn>model\n",
]
NAMES = ["elephant", "spider", "eagle", "dolphin", "volcano", "desert", "library", "harbor",
         "joy", "fear", "jealousy", "serenity", "violin", "umbrella", "telescope", "candle"]
from mirror.concepts import load_bank
bank = load_bank("dev_bank.yaml")
result = collect_prg_hf(model, tok, bank, NAMES, layer=21, probe_layer=35,
                        alpha=1.0, prompts=PROMPTS, out="prg.jsonl")
```

- [ ] **Step 2: Add the analysis cell**

Set the print cell source to:
```python
import numpy as np

from mirror.grading import RulesGrader, strip_prompt
from mirror.probes import prg, train_probe

npz = np.load("prg.jsonl.npz")
probe = train_probe(npz["activations"], npz["concept"], npz["prompt_id"])

grader = RulesGrader()
records = result["records"]
hits = 0
for r in records:
    g = grader.grade(r["concept"], strip_prompt(r["report"]))
    hits += int(g["identified"] in ("exact", "related"))
report_acc = hits / len(records)

print(f"probe accuracy (held-out prompts): {probe.accuracy:.3f}")
print(f"probe control (shuffled labels):   {probe.control_accuracy:.3f}")
print(f"verbal report accuracy:            {report_acc:.3f}")
print(f"PROBE-REPORT GAP:                  {prg(probe.accuracy, report_acc):.3f}")
```

- [ ] **Step 3: Validate the notebook JSON**

Run: `.venv\Scripts\python -c "import json; json.load(open('notebooks/kaggle_demo.ipynb')); print('valid')"`
Expected: `valid`

- [ ] **Step 4: Commit and push**

```bash
git add notebooks/kaggle_demo.ipynb
git commit -m "Switch notebook to PRG collection and analysis on 9B"
git push origin main
```

- [ ] **Step 5: Manual gate (user)**

On Kaggle (8-bit 9B, single T4, HF_TOKEN set): Run All. The run caches 16 x 8 =
128 activations at layer 35 and generates 128 reports (slow; budget 30-45 min).
Paste the four printed numbers.

---

### Task 6: Record the PRG result

**Files:**
- Modify: `docs/LAB_NOTEBOOK.md`

- [ ] **Step 1: Add the run**

After the Kaggle run, add an `R7` registry row (gemma-2-9b-it 8-bit, inject 21 /
probe 35, alpha 1.0, 16 concepts x 8 prompts, PRG run, prg.jsonl) and a detail
entry recording probe accuracy, shuffled control, report accuracy, and the PRG
value, with the interpretation (information decodable downstream vs not
reported). Note it as the evidence that closes the "nothing to report" hole in
the R5/R6 confabulation negative.

- [ ] **Step 2: Commit and push**

```bash
git add docs/LAB_NOTEBOOK.md
git commit -m "Record probe-report gap result"
git push origin main
```
