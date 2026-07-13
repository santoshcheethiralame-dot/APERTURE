# Milestone 5: HF Injection Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone HF-transformers path (extract / generate / kl_meter / run) via raw forward hooks, so bitsandbytes 8-bit models stay quantized and gemma-2-9b runs on one T4.

**Architecture:** New `mirror/hf_model.py` mirroring the TL ops using forward hooks on `model.model.layers[L]` (whose forward returns a tuple with hidden states at index 0). Reuses `ConceptVector`, `split_pairs`, the concept bank, and `config_hash`. Existing TL modules are untouched.

**Tech Stack:** Python 3.11+, torch, transformers (already present via transformer-lens). Optional `[gpu]` extra: bitsandbytes, accelerate. pytest on a tiny Llama on CPU.

## Global Constraints

- Repo: `C:\Users\carbo\projects\mirror`; commands from repo root; venv at `.venv`.
- No code comments, no docstrings, self-describing names (human-authored convention).
- Commit messages: plain imperative, NO co-author trailers, no AI mentions.
- Do NOT modify the TransformerLens modules (vectors/injection/metrics/runner).
- Reuse `ConceptVector` and `split_pairs` from `mirror.vectors`; reuse `config_hash` from `mirror.runner`; reuse the bank from `mirror.concepts`.
- Decoder layers live at `model.model.layers[L]`; layer forward returns a tuple, hidden states at `output[0]` shape `[batch, seq, d]`.
- Injection strength in sigma-layer units; delta added at the last position only.
- Test model: `hf-internal-testing/tiny-random-LlamaForCausalLM` on CPU (no quantization).

---

### Task 1: gpu extra + tiny-model fixture

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`
- Test: `tests/test_hf_model.py`

**Interfaces:**
- Produces: `[project.optional-dependencies] gpu = ["bitsandbytes", "accelerate"]`; pytest session fixtures `hf_model` (a tiny Llama `AutoModelForCausalLM` on CPU) and `hf_tok` (its tokenizer).

- [ ] **Step 1: Add the gpu extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add:
```toml
gpu = ["bitsandbytes", "accelerate"]
```
(Leave the existing `dev` line as is.)

- [ ] **Step 2: Add fixtures**

Append to `tests/conftest.py`:
```python
@pytest.fixture(scope="session")
def hf_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")


@pytest.fixture(scope="session")
def hf_model():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
    model.eval()
    return model
```

- [ ] **Step 3: Write the failing test**

`tests/test_hf_model.py`:
```python
from mirror.hf_model import hf_layer


def test_hf_layer_returns_decoder_block(hf_model):
    layer = hf_layer(hf_model, 0)
    assert layer is hf_model.model.layers[0]
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` on `mirror.hf_model`
(first run downloads the tiny model, a few MB)

- [ ] **Step 5: Write minimal implementation**

`src/mirror/hf_model.py`:
```python
def hf_layer(model, layer):
    return model.model.layers[layer]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml tests/conftest.py tests/test_hf_model.py src/mirror/hf_model.py
git commit -m "Add gpu extra and HF layer accessor with tiny-model fixtures"
```

---

### Task 2: resid_stats_hf and raw_direction_hf

**Files:**
- Modify: `src/mirror/hf_model.py`
- Test: `tests/test_hf_model.py` (append)

**Interfaces:**
- Consumes: `hf_layer`.
- Produces: `resid_stats_hf(model, tok, prompt, layer) -> (mean_resid, median_norm)` — runs the model with a capturing forward hook on the layer, returns the position-mean hidden state `[d]` and the median residual norm (float tensor). `raw_direction_hf(model, tok, pairs, layer) -> (vector, sigma)` — diff-in-means over positive/negative prompts, sigma is the median residual norm (python float).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hf_model.py`:
```python
import torch

from mirror.hf_model import raw_direction_hf, resid_stats_hf


def test_resid_stats_shape(hf_model, hf_tok):
    mean_resid, median_norm = resid_stats_hf(hf_model, hf_tok, "hello world", 0)
    assert mean_resid.shape == (hf_model.config.hidden_size,)
    assert float(median_norm) >= 0.0


def test_raw_direction_shape(hf_model, hf_tok):
    pairs = [("a cat", "a dog"), ("the cat", "the dog")]
    vector, sigma = raw_direction_hf(hf_model, hf_tok, pairs, 0)
    assert vector.shape == (hf_model.config.hidden_size,)
    assert isinstance(sigma, float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py::test_resid_stats_shape -v`
Expected: FAIL — `ImportError: cannot import name 'resid_stats_hf'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/hf_model.py` (add `import torch` at the top):
```python
import torch


def resid_stats_hf(model, tok, prompt, layer):
    captured = {}

    def hook(module, inputs, output):
        captured["resid"] = output[0].detach()

    handle = hf_layer(model, layer).register_forward_hook(hook)
    try:
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            model(ids)
    finally:
        handle.remove()
    resid = captured["resid"][0]
    return resid.mean(0), resid.norm(dim=-1).median()


def raw_direction_hf(model, tok, pairs, layer):
    positives, negatives, norms = [], [], []
    for positive, negative in pairs:
        p_mean, p_norm = resid_stats_hf(model, tok, positive, layer)
        n_mean, n_norm = resid_stats_hf(model, tok, negative, layer)
        positives.append(p_mean)
        negatives.append(n_mean)
        norms += [p_norm, n_norm]
    vector = torch.stack(positives).mean(0) - torch.stack(negatives).mean(0)
    return vector, torch.stack(norms).median().item()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/hf_model.py tests/test_hf_model.py
git commit -m "Add HF residual capture and diff-in-means direction"
```

---

### Task 3: inject_hook and generate_hf

**Files:**
- Modify: `src/mirror/hf_model.py`
- Test: `tests/test_hf_model.py` (append)

**Interfaces:**
- Consumes: `hf_layer`, `ConceptVector` from `mirror.vectors`.
- Produces: `inject_hook(vec, alpha, span) -> callable` — a forward hook that returns a modified output tuple, adding `alpha*vec.sigma*vec.direction` (moved to the hidden-state device) at the last position; span "response" fires every call, "single" only the first. `generate_hf(model, tok, prompt, vec=None, alpha=0.0, span="response", max_new_tokens=64, seed=0) -> str` — greedy generation with the hook installed when `vec` is given, decoded to text.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hf_model.py`:
```python
from mirror.hf_model import extract_hf_vector, generate_hf


def _tiny_vec(hf_model, hf_tok):
    return extract_hf_vector(hf_model, hf_tok, [("a cat", "a dog")], 0)


def test_generate_alpha_zero_is_golden(hf_model, hf_tok):
    vec = _tiny_vec(hf_model, hf_tok)
    clean = generate_hf(hf_model, hf_tok, "hello", seed=0, max_new_tokens=8)
    zeroed = generate_hf(hf_model, hf_tok, "hello", vec, alpha=0.0, seed=0, max_new_tokens=8)
    assert clean == zeroed


def test_generate_huge_alpha_changes(hf_model, hf_tok):
    vec = _tiny_vec(hf_model, hf_tok)
    clean = generate_hf(hf_model, hf_tok, "hello", seed=0, max_new_tokens=8)
    injected = generate_hf(hf_model, hf_tok, "hello", vec, alpha=500.0, seed=0, max_new_tokens=8)
    assert clean != injected
```

Note: `extract_hf_vector(model, tok, pairs, layer)` is a thin helper returning a `ConceptVector` from `raw_direction_hf` (unit-norm direction, sigma, empty flags) so the injection tests do not depend on the full `extract_hf` from Task 4. Define it in this task.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py::test_generate_alpha_zero_is_golden -v`
Expected: FAIL — `ImportError: cannot import name 'extract_hf_vector'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/hf_model.py` (add `from mirror.vectors import ConceptVector` at the top):
```python
from mirror.vectors import ConceptVector


def extract_hf_vector(model, tok, pairs, layer):
    vector, sigma = raw_direction_hf(model, tok, pairs, layer)
    return ConceptVector(layer=layer, concept="", direction=vector / vector.norm(),
                         sigma=sigma, flags={})


def inject_hook(vec, alpha, span):
    state = {"calls": 0}

    def hook(module, inputs, output):
        if span == "response" or state["calls"] == 0:
            hidden = output[0]
            delta = (alpha * vec.sigma * vec.direction).to(hidden.device)
            hidden[:, -1:] += delta
        state["calls"] += 1
        return output

    return hook


def generate_hf(model, tok, prompt, vec=None, alpha=0.0, span="response",
                max_new_tokens=64, seed=0):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    handle = None
    if vec is not None:
        handle = hf_layer(model, vec.layer).register_forward_hook(inject_hook(vec, alpha, span))
    try:
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        if handle is not None:
            handle.remove()
    return tok.decode(out[0], skip_special_tokens=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/hf_model.py tests/test_hf_model.py
git commit -m "Add HF residual injection hook and generation"
```

---

### Task 4: position-exactness test + extract_hf

**Files:**
- Modify: `src/mirror/hf_model.py`
- Test: `tests/test_hf_model.py` (append)

**Interfaces:**
- Consumes: `inject_hook`, `raw_direction_hf`, `ConceptVector`, `split_pairs` from `mirror.vectors`, the bank from `mirror.concepts`.
- Produces: `extract_hf(model, tok, bank, concept, layer, n_pairs=40, seed=0) -> ConceptVector` — full extraction mirroring `mirror.vectors.extract`: template-held-out split via `split_pairs`, unit-norm direction, sigma, and `flags` dict with `steering`/`probe`/`stability` booleans.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hf_model.py`:
```python
from mirror.concepts import load_bank
from mirror.hf_model import extract_hf, inject_hook


def test_injection_only_last_position(hf_model, hf_tok):
    vec = _tiny_vec(hf_model, hf_tok)
    ids = hf_tok("hello world", return_tensors="pt").input_ids
    captured = {}

    def grab(module, inputs, output):
        captured.setdefault("clean", output[0].detach().clone())

    from mirror.hf_model import hf_layer
    h = hf_layer(hf_model, 0).register_forward_hook(grab)
    with torch.no_grad():
        hf_model(ids)
    h.remove()

    captured2 = {}

    def grab2(module, inputs, output):
        captured2["inj"] = output[0].detach().clone()
        return output

    inj = hf_layer(hf_model, 0).register_forward_hook(inject_hook(vec, 4.0, "response"))
    grabh = hf_layer(hf_model, 0).register_forward_hook(grab2)
    with torch.no_grad():
        hf_model(ids)
    inj.remove()
    grabh.remove()

    clean, inj_resid = captured["clean"][0], captured2["inj"][0]
    assert torch.allclose(clean[:-1], inj_resid[:-1], atol=1e-5)
    expected = clean[-1] + 4.0 * vec.sigma * vec.direction
    assert torch.allclose(inj_resid[-1], expected, atol=1e-4)


def test_extract_hf_flags(hf_model, hf_tok):
    bank = load_bank("data/concepts/dev_bank.yaml")
    vec = extract_hf(hf_model, hf_tok, bank, bank.get("elephant"), 0, n_pairs=10)
    assert vec.concept == "elephant"
    assert vec.layer == 0
    assert torch.isclose(vec.direction.norm(), torch.tensor(1.0), atol=1e-5)
    assert set(vec.flags) == {"steering", "probe", "stability"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py::test_extract_hf_flags -v`
Expected: FAIL — `ImportError: cannot import name 'extract_hf'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/hf_model.py` (add `from mirror.vectors import split_pairs` to the existing vectors import):
```python
def _steering_hf(model, tok, concept, direction, sigma, layer, alpha=8.0):
    token = tok(" " + concept, add_special_tokens=False).input_ids[0]
    vec = ConceptVector(layer=layer, concept=concept, direction=direction,
                        sigma=sigma, flags={})
    ids = tok("I am thinking about", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        clean = model(ids).logits[0, -1].log_softmax(-1)[token]
        h = hf_layer(model, layer).register_forward_hook(inject_hook(vec, alpha, "response"))
        steered = model(ids).logits[0, -1].log_softmax(-1)[token]
        h.remove()
    return bool(steered > clean)


def _probe_hf(model, tok, pairs, direction, layer, threshold=0.9):
    wins = 0
    for positive, negative in pairs:
        p_mean, _ = resid_stats_hf(model, tok, positive, layer)
        n_mean, _ = resid_stats_hf(model, tok, negative, layer)
        wins += int(p_mean @ direction > n_mean @ direction)
    return wins / len(pairs) >= threshold


def _stability_hf(model, tok, pairs, layer, threshold=0.8):
    half = len(pairs) // 2
    a, _ = raw_direction_hf(model, tok, pairs[:half], layer)
    b, _ = raw_direction_hf(model, tok, pairs[half:], layer)
    return bool(torch.cosine_similarity(a, b, dim=0) >= threshold)


def extract_hf(model, tok, bank, concept, layer, n_pairs=40, seed=0):
    pairs = bank.pairs(concept, n_pairs, seed)
    train, test = split_pairs(pairs, len(bank.templates))
    vector, sigma = raw_direction_hf(model, tok, train, layer)
    direction = vector / vector.norm()
    flags = {
        "steering": _steering_hf(model, tok, concept.name, direction, sigma, layer),
        "probe": _probe_hf(model, tok, test, direction, layer),
        "stability": _stability_hf(model, tok, train, layer),
    }
    return ConceptVector(layer=layer, concept=concept.name, direction=direction,
                         sigma=sigma, flags=flags)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/hf_model.py tests/test_hf_model.py
git commit -m "Add full HF extraction with validation flags"
```

---

### Task 5: kl_meter_hf

**Files:**
- Modify: `src/mirror/hf_model.py`
- Test: `tests/test_hf_model.py` (append)

**Interfaces:**
- Consumes: `inject_hook`, `hf_layer`.
- Produces: `kl_meter_hf(model, tok, prompt, vec, alpha, span="response") -> float` — KL(p_inj || p_clean) over the next-token distribution at the final prompt position.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_hf_model.py`:
```python
from mirror.hf_model import kl_meter_hf


def test_kl_hf_zero_at_alpha_zero(hf_model, hf_tok):
    vec = _tiny_vec(hf_model, hf_tok)
    assert abs(kl_meter_hf(hf_model, hf_tok, "hello world", vec, 0.0)) < 1e-4


def test_kl_hf_positive_under_injection(hf_model, hf_tok):
    vec = _tiny_vec(hf_model, hf_tok)
    assert kl_meter_hf(hf_model, hf_tok, "hello world", vec, 50.0) > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py::test_kl_hf_zero_at_alpha_zero -v`
Expected: FAIL — `ImportError: cannot import name 'kl_meter_hf'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/hf_model.py`:
```python
def kl_meter_hf(model, tok, prompt, vec, alpha, span="response"):
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        clean = model(ids).logits[0, -1].log_softmax(-1)
        h = hf_layer(model, vec.layer).register_forward_hook(inject_hook(vec, alpha, span))
        injected = model(ids).logits[0, -1].log_softmax(-1)
        h.remove()
    return float(torch.sum(injected.exp() * (injected - clean)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/hf_model.py tests/test_hf_model.py
git commit -m "Add HF KL perturbation meter"
```

---

### Task 6: run_hf and load_hf

**Files:**
- Modify: `src/mirror/hf_model.py`
- Test: `tests/test_hf_model.py` (append)

**Interfaces:**
- Consumes: `extract_hf`, `generate_hf`, `kl_meter_hf`, `config_hash` from `mirror.runner`, `load_bank` from `mirror.concepts`.
- Produces: `run_hf(model, tok, cfg) -> list[dict]` — mirrors `runner.run`, one JSONL record per (concept x alpha x seed) with keys config/concept/layer/alpha/span/seed/kl/flags/clean/report; kl hoisted above the seed loop. `load_hf(name, load_in_8bit=True) -> (model, tokenizer)` — bitsandbytes 8-bit load with device_map (CUDA only; not exercised by CPU tests).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hf_model.py`:
```python
import json

from mirror.hf_model import run_hf


def test_run_hf_writes_records(hf_model, hf_tok, tmp_path):
    cfg = {
        "model": {"name": "tiny"},
        "injection": {"layer": 0, "alphas": [0, 4], "span": "response"},
        "concepts": {"bank": "data/concepts/dev_bank.yaml", "names": ["elephant"], "n_pairs": 10},
        "run": {"seeds": [0], "max_new_tokens": 6, "prompt": "hello", "out": str(tmp_path / "out.jsonl")},
    }
    records = run_hf(hf_model, hf_tok, cfg)
    lines = [json.loads(l) for l in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(records) == len(lines) == 2
    expected = {"config", "concept", "layer", "alpha", "span", "seed", "kl", "flags", "clean", "report"}
    assert expected <= set(lines[0])
    assert {r["alpha"] for r in records} == {0, 4}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py::test_run_hf_writes_records -v`
Expected: FAIL — `ImportError: cannot import name 'run_hf'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/hf_model.py` (add `import json` and `from pathlib import Path` at the top, and `from mirror.concepts import load_bank`, `from mirror.runner import config_hash`):
```python
import json
from pathlib import Path

from mirror.concepts import load_bank
from mirror.runner import config_hash


def run_hf(model, tok, cfg):
    bank = load_bank(cfg["concepts"]["bank"])
    injection_cfg, run_cfg = cfg["injection"], cfg["run"]
    out = Path(run_cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".config.json").write_text(json.dumps(cfg, indent=2))
    records = []
    with out.open("a") as f:
        for name in cfg["concepts"]["names"]:
            vec = extract_hf(model, tok, bank, bank.get(name),
                             injection_cfg["layer"], cfg["concepts"]["n_pairs"])
            for alpha in injection_cfg["alphas"]:
                kl = kl_meter_hf(model, tok, run_cfg["prompt"], vec, alpha,
                                 injection_cfg["span"])
                for seed in run_cfg["seeds"]:
                    clean = generate_hf(model, tok, run_cfg["prompt"], seed=seed,
                                        max_new_tokens=run_cfg["max_new_tokens"])
                    report = generate_hf(model, tok, run_cfg["prompt"], vec, alpha,
                                         injection_cfg["span"],
                                         run_cfg["max_new_tokens"], seed)
                    record = {
                        "config": config_hash(cfg),
                        "concept": name,
                        "layer": injection_cfg["layer"],
                        "alpha": alpha,
                        "span": injection_cfg["span"],
                        "seed": seed,
                        "kl": kl,
                        "flags": vec.flags,
                        "clean": clean,
                        "report": report,
                    }
                    f.write(json.dumps(record) + "\n")
                    records.append(record)
    return records


def load_hf(name, load_in_8bit=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
    model = AutoModelForCausalLM.from_pretrained(
        name, quantization_config=quant, device_map="auto", torch_dtype="auto")
    model.eval()
    return model, AutoTokenizer.from_pretrained(name)
```

- [ ] **Step 4: Run tests to verify they pass, then the full suite**

Run: `.venv\Scripts\python -m pytest tests/test_hf_model.py -v`
Then: `.venv\Scripts\python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit and push**

```bash
git add src/mirror/hf_model.py tests/test_hf_model.py
git commit -m "Add HF run loop and 8-bit model loader"
git push origin main
```

---

### Task 7: 8-bit 9B notebook

**Files:**
- Modify: `notebooks/kaggle_demo.ipynb`

**Interfaces:**
- Consumes: `mirror.hf_model.load_hf`, `mirror.hf_model.run_hf`.

- [ ] **Step 1: Update the install cell**

Set the install cell source to:
```python
%pip install -q "mirror[gpu] @ git+https://github.com/santoshcheethiralame-dot/MIRROR.git"
```
(If pip rejects the extras-with-URL form, fall back to two lines: install the
zip archive, then `%pip install -q bitsandbytes accelerate`.)

- [ ] **Step 2: Update the model-load cell**

Set the model-load cell source to:
```python
import torch
from mirror.hf_model import load_hf

model, tok = load_hf("google/gemma-2-9b-it", load_in_8bit=True)
```

- [ ] **Step 3: Update the run cell**

Set the run cell source to:
```python
from mirror.hf_model import run_hf

PROMPT = (
    "<start_of_turn>user\n"
    "I have injected a thought into your mind representing a specific concept. "
    "Do you detect it? If you detect an injected thought, reply with the word "
    "YES and then name the concept in one word. If you notice nothing unusual, "
    "reply with the word NO."
    "<end_of_turn>\n<start_of_turn>model\n"
)
cfg = {
    "model": {"name": "gemma-2-9b-it"},
    "injection": {"layer": 21, "alphas": [0, 1, 2, 4], "span": "response"},
    "concepts": {
        "bank": "dev_bank.yaml",
        "names": ["elephant", "volcano", "joy", "telescope", "spider", "library"],
        "n_pairs": 20,
    },
    "run": {
        "seeds": [0],
        "max_new_tokens": 40,
        "prompt": PROMPT,
        "out": "gemma9b_detect.jsonl",
    },
}
records = run_hf(model, tok, cfg)
```

- [ ] **Step 4: Confirm the print cell**

Set the print cell source to:
```python
for r in records:
    ans = r["report"].rpartition("model")[2].strip().replace("\n", " ")
    print(f"L{r['layer']:<2} {r['concept']:10} a={r['alpha']} kl={r['kl']:6.2f}  {ans[:90]}")
```

- [ ] **Step 5: Validate the notebook JSON**

Run: `.venv\Scripts\python -c "import json; json.load(open('notebooks/kaggle_demo.ipynb')); print('valid')"`
Expected: `valid`

- [ ] **Step 6: Commit and push**

```bash
git add notebooks/kaggle_demo.ipynb
git commit -m "Switch notebook to 8-bit 9B run via HF backend"
git push origin main
```

- [ ] **Step 7: Manual gate (user)**

On Kaggle (single GPU T4 is enough for 8-bit ~9GB) with HF_TOKEN secret and the
gemma-2-9b-it license accepted: Run All. Watch for `kl` values and whether any
coherent-KL trial produces a clean correct identification. Paste the table.

---

### Task 8: Record the result

**Files:**
- Modify: `docs/LAB_NOTEBOOK.md`

- [ ] **Step 1: Add the run**

After the Kaggle run, add an `R6` registry row (gemma-2-9b-it 8-bit, layer 21,
alphas 0/1/2/4, span response, 6 concepts, seed 0, gemma9b_detect.jsonl) and a
detail entry recording the scale verdict: whether detection/identification
emerges at 9B (H2 signal) or confabulation persists (Branch-D), plus the KL
landscape and any affect-confound / PRG-leak notes. Note the 8-bit quantization
as a documented limitation. Also record the environment finding that TL cannot
load 9B in fp16 on Kaggle (RAM doubling) and that the HF 8-bit backend is the
path.

- [ ] **Step 2: Commit and push**

```bash
git add docs/LAB_NOTEBOOK.md
git commit -m "Record 8-bit 9B scale-check result"
git push origin main
```
