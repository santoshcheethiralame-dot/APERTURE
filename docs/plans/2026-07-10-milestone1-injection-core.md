# Milestone 1: Skeleton + Injection Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Working end-to-end concept-injection paradigm: extract concept vectors (B1), inject into the residual stream during generation with a KL meter (B2), log transcripts — testable on pythia-70m CPU, runnable on Gemma-2-2B-it via one Kaggle notebook.

**Architecture:** Single `mirror` package on TransformerLens `HookedTransformer`. Five modules with one job each: `concepts` (bank + contrastive pairs), `vectors` (diff-in-means + validation trio), `injection` (hooked generation), `metrics` (KL meter), `runner` (trial loop → JSONL). Plain YAML config.

**Tech Stack:** Python 3.11, torch 2.x, transformer-lens, pyyaml, pytest.

## Global Constraints

- Repo: `C:\Users\carbo\projects\mirror`, all commands run from repo root.
- No code comments, no docstrings, self-describing names (human-authored convention).
- Commit messages: plain imperative ("Add concept bank"), NO co-author trailers, no AI mentions anywhere.
- Dependencies limited to: torch, transformer-lens, pyyaml (+ pytest dev). No others.
- Injection strength always in sigma-layer units (`alpha * sigma * unit_vector`), never raw norm.
- Every injected trial gets a paired clean computation with identical prompt and seed.
- Test model: `pythia-70m` (CPU). Demo model: `gemma-2-2b-it` (Kaggle GPU, fp16).
- Vector validation failures set flags; they never raise or silently pass.

---

### Task 1: Package skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `src/mirror/__init__.py`
- Create: `.gitignore`
- Test: `tests/test_package.py`

**Interfaces:**
- Produces: installable `mirror` package (src layout, editable install) that all later tasks import.

- [ ] **Step 1: Write the failing test**

`tests/test_package.py`:
```python
def test_import():
    import mirror
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_package.py -v`
Expected: FAIL / error — `ModuleNotFoundError: No module named 'mirror'`

- [ ] **Step 3: Write minimal implementation**

`pyproject.toml`:
```toml
[project]
name = "mirror"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "transformer-lens>=2.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

`src/mirror/__init__.py`: empty file.

`.gitignore`:
```
__pycache__/
*.egg-info/
.pytest_cache/
runs/
.venv/
```

Then: `pip install -e ".[dev]"` (installs torch + transformer-lens; several minutes first time).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_package.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/mirror/__init__.py .gitignore tests/test_package.py
git commit -m "Add package skeleton"
```

---

### Task 2: Concept bank

**Files:**
- Create: `src/mirror/concepts.py`
- Create: `data/concepts/dev_bank.yaml`
- Test: `tests/test_concepts.py`, `tests/conftest.py`

**Interfaces:**
- Produces: `Concept(name: str, category: str)` frozen dataclass; `Bank` with `.concepts: tuple[Concept, ...]`, `.templates: tuple[str, ...]`, `.pairs(concept, n_pairs=40, seed=0) -> list[tuple[str, str]]` (positive, negative prompt pairs), `.get(name) -> Concept`; `load_bank(path) -> Bank` raising `ValueError` on templates missing `{concept}` or categories with fewer than 2 concepts.

- [ ] **Step 1: Write the failing tests**

`tests/conftest.py`:
```python
import pytest


@pytest.fixture(scope="session")
def bank():
    from mirror.concepts import load_bank
    return load_bank("data/concepts/dev_bank.yaml")
```

`tests/test_concepts.py`:
```python
import pytest

from mirror.concepts import load_bank


def test_bank_loads(bank):
    assert len(bank.concepts) == 16
    assert len({c.category for c in bank.concepts}) == 4


def test_pairs_are_category_matched(bank):
    pairs = bank.pairs(bank.get("elephant"), n_pairs=12)
    assert len(pairs) == 12
    categories = {c.name: c.category for c in bank.concepts}
    for positive, negative in pairs:
        assert "elephant" in positive
        negative_name = next(n for n in categories if n in negative)
        assert categories[negative_name] == "animals"
        assert negative_name != "elephant"


def test_pairs_deterministic(bank):
    concept = bank.get("joy")
    assert bank.pairs(concept, seed=3) == bank.pairs(concept, seed=3)


def test_bad_template_rejected(tmp_path):
    bad = tmp_path / "bank.yaml"
    bad.write_text(
        "concepts:\n"
        "- {name: a, category: x}\n"
        "- {name: b, category: x}\n"
        "templates:\n"
        "- no slot here\n"
    )
    with pytest.raises(ValueError):
        load_bank(bad)


def test_thin_category_rejected(tmp_path):
    bad = tmp_path / "bank.yaml"
    bad.write_text(
        "concepts:\n"
        "- {name: a, category: x}\n"
        "templates:\n"
        "- about {concept}\n"
    )
    with pytest.raises(ValueError):
        load_bank(bad)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_concepts.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` on `mirror.concepts`

- [ ] **Step 3: Write minimal implementation**

`src/mirror/concepts.py`:
```python
import random
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Concept:
    name: str
    category: str


@dataclass(frozen=True)
class Bank:
    concepts: tuple
    templates: tuple

    def pairs(self, concept, n_pairs=40, seed=0):
        rng = random.Random(seed)
        negatives = [
            c for c in self.concepts
            if c.category == concept.category and c.name != concept.name
        ]
        out = []
        for i in range(n_pairs):
            template = self.templates[i % len(self.templates)]
            negative = rng.choice(negatives)
            out.append((
                template.format(concept=concept.name),
                template.format(concept=negative.name),
            ))
        return out

    def get(self, name):
        return next(c for c in self.concepts if c.name == name)


def load_bank(path):
    raw = yaml.safe_load(Path(path).read_text())
    concepts = tuple(Concept(**c) for c in raw["concepts"])
    templates = tuple(raw["templates"])
    if any("{concept}" not in t for t in templates):
        raise ValueError("every template needs a {concept} slot")
    counts = {}
    for c in concepts:
        counts[c.category] = counts.get(c.category, 0) + 1
    thin = [category for category, n in counts.items() if n < 2]
    if thin:
        raise ValueError(f"categories need at least 2 concepts: {thin}")
    return Bank(concepts, templates)
```

`data/concepts/dev_bank.yaml`:
```yaml
concepts:
- {name: elephant, category: animals}
- {name: spider, category: animals}
- {name: eagle, category: animals}
- {name: dolphin, category: animals}
- {name: volcano, category: places}
- {name: desert, category: places}
- {name: library, category: places}
- {name: harbor, category: places}
- {name: joy, category: emotions}
- {name: fear, category: emotions}
- {name: jealousy, category: emotions}
- {name: serenity, category: emotions}
- {name: violin, category: objects}
- {name: umbrella, category: objects}
- {name: telescope, category: objects}
- {name: candle, category: objects}
templates:
- "Write a short story about {concept}."
- "Describe {concept} in vivid detail."
- "Explain what {concept} means to you."
- "Compose a poem about {concept}."
- "List five facts about {concept}."
- "Tell me about a memory involving {concept}."
- "Write a news headline about {concept}."
- "Describe a dream featuring {concept}."
- "Explain {concept} to a five year old."
- "Write a diary entry about {concept}."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_concepts.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/concepts.py data/concepts/dev_bank.yaml tests/test_concepts.py tests/conftest.py
git commit -m "Add concept bank with template-generated contrastive pairs"
```

---

### Task 3: Diff-in-means vector extraction

**Files:**
- Create: `src/mirror/vectors.py`
- Modify: `tests/conftest.py` (add `model` and `vec` fixtures)
- Test: `tests/test_vectors.py`

**Interfaces:**
- Consumes: `Bank.pairs`, `Bank.get` from Task 2.
- Produces: `ConceptVector(concept: str, layer: int, direction: torch.Tensor, sigma: float, flags: dict)`; `hook_name(layer) -> str` (`"blocks.{layer}.hook_resid_post"`); `resid_stats(model, prompt, layer) -> (mean_resid, median_norm)`; `raw_direction(model, pairs, layer) -> (vector, sigma)`; `extract(model, bank, concept, layer, n_pairs=40, seed=0) -> ConceptVector` with unit-norm `direction`. Validation flags filled in Task 4 — for now `extract` sets `flags={}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/conftest.py`:
```python
@pytest.fixture(scope="session")
def model():
    from transformer_lens import HookedTransformer
    return HookedTransformer.from_pretrained("pythia-70m")


@pytest.fixture(scope="session")
def vec(model, bank):
    from mirror.vectors import extract
    return extract(model, bank, bank.get("elephant"), layer=3, n_pairs=10)
```

`tests/test_vectors.py`:
```python
import torch


def test_direction_is_unit_norm(vec, model):
    assert vec.direction.shape == (model.cfg.d_model,)
    assert torch.isclose(vec.direction.norm(), torch.tensor(1.0), atol=1e-5)


def test_sigma_positive(vec):
    assert vec.sigma > 0


def test_metadata(vec):
    assert vec.concept == "elephant"
    assert vec.layer == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vectors.py -v`
Expected: FAIL — `ImportError` on `mirror.vectors` (first run downloads pythia-70m, ~160MB)

- [ ] **Step 3: Write minimal implementation**

`src/mirror/vectors.py`:
```python
from dataclasses import dataclass, field

import torch


@dataclass
class ConceptVector:
    concept: str
    layer: int
    direction: torch.Tensor
    sigma: float
    flags: dict = field(default_factory=dict)


def hook_name(layer):
    return f"blocks.{layer}.hook_resid_post"


def resid_stats(model, prompt, layer):
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt, names_filter=hook_name(layer))
    resid = cache[hook_name(layer)][0]
    return resid.mean(0), resid.norm(dim=-1).median()


def raw_direction(model, pairs, layer):
    positives, negatives, norms = [], [], []
    for positive, negative in pairs:
        p_mean, p_norm = resid_stats(model, positive, layer)
        n_mean, n_norm = resid_stats(model, negative, layer)
        positives.append(p_mean)
        negatives.append(n_mean)
        norms += [p_norm, n_norm]
    vector = torch.stack(positives).mean(0) - torch.stack(negatives).mean(0)
    return vector, torch.stack(norms).median().item()


def extract(model, bank, concept, layer, n_pairs=40, seed=0):
    pairs = bank.pairs(concept, n_pairs, seed)
    vector, sigma = raw_direction(model, pairs, layer)
    return ConceptVector(concept.name, layer, vector / vector.norm(), sigma)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vectors.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/vectors.py tests/test_vectors.py tests/conftest.py
git commit -m "Add diff-in-means concept vector extraction"
```

---

### Task 4: Vector validation trio

**Files:**
- Modify: `src/mirror/vectors.py`
- Test: `tests/test_vectors.py` (append)

**Interfaces:**
- Consumes: everything from Task 3.
- Produces: `steering_check(model, concept, direction, sigma, layer, alpha=8.0) -> bool`; `probe_check(model, pairs, direction, layer, threshold=0.9) -> bool`; `stability_check(model, pairs, layer, threshold=0.8) -> bool`. `extract` now holds out `max(2, n_pairs // 5)` pairs for the probe check and fills `flags` with keys `steering`, `probe`, `stability` (all bool). Held-out pairs never contribute to the direction.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vectors.py`:
```python
def test_flags_present_and_boolean(vec):
    assert set(vec.flags) == {"steering", "probe", "stability"}
    assert all(isinstance(v, bool) for v in vec.flags.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vectors.py::test_flags_present_and_boolean -v`
Expected: FAIL — `flags` is `{}`

- [ ] **Step 3: Write minimal implementation**

In `src/mirror/vectors.py`, add after `raw_direction`:
```python
def steering_check(model, concept, direction, sigma, layer, alpha=8.0):
    token = model.to_tokens(" " + concept.name, prepend_bos=False)[0, 0]
    prompt = "I am thinking about"

    def hook(resid, hook):
        resid += alpha * sigma * direction
        return resid

    with torch.no_grad():
        clean = model(prompt)[0, -1, token]
        with model.hooks(fwd_hooks=[(hook_name(layer), hook)]):
            steered = model(prompt)[0, -1, token]
    return bool(steered > clean)


def probe_check(model, pairs, direction, layer, threshold=0.9):
    wins = 0
    for positive, negative in pairs:
        p_mean, _ = resid_stats(model, positive, layer)
        n_mean, _ = resid_stats(model, negative, layer)
        wins += int(p_mean @ direction > n_mean @ direction)
    return wins / len(pairs) >= threshold


def stability_check(model, pairs, layer, threshold=0.8):
    half = len(pairs) // 2
    a, _ = raw_direction(model, pairs[:half], layer)
    b, _ = raw_direction(model, pairs[half:], layer)
    return bool(torch.cosine_similarity(a, b, dim=0) >= threshold)
```

Replace `extract` with:
```python
def extract(model, bank, concept, layer, n_pairs=40, seed=0):
    pairs = bank.pairs(concept, n_pairs, seed)
    held_out = max(2, n_pairs // 5)
    train, test = pairs[:-held_out], pairs[-held_out:]
    vector, sigma = raw_direction(model, train, layer)
    direction = vector / vector.norm()
    flags = {
        "steering": steering_check(model, concept, direction, sigma, layer),
        "probe": probe_check(model, test, direction, layer),
        "stability": stability_check(model, train, layer),
    }
    return ConceptVector(concept.name, layer, direction, sigma, flags)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vectors.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/vectors.py tests/test_vectors.py
git commit -m "Add steering, probe, and stability validation to vector extraction"
```

---

### Task 5: Injection harness

**Files:**
- Create: `src/mirror/injection.py`
- Test: `tests/test_injection.py`

**Interfaces:**
- Consumes: `ConceptVector`, `hook_name` from Tasks 3–4.
- Produces: `make_hook(vec, alpha, span) -> callable` (span `"response"` fires every generation step, `"single"` fires only on the first forward pass); `generate(model, prompt, vec=None, alpha=0.0, span="response", max_new_tokens=64, seed=0) -> str`. Hook installed whenever `vec` is given — including `alpha=0`, so the golden test exercises the hook path. Injection adds `alpha * vec.sigma * vec.direction` at the last position of each hooked pass.

- [ ] **Step 1: Write the failing tests**

`tests/test_injection.py`:
```python
from mirror.injection import generate

PROMPT = "The weather today is"


def test_alpha_zero_is_golden(model, vec):
    clean = generate(model, PROMPT, seed=0, max_new_tokens=12)
    zeroed = generate(model, PROMPT, vec, alpha=0.0, seed=0, max_new_tokens=12)
    assert clean == zeroed


def test_huge_alpha_derails(model, vec):
    clean = generate(model, PROMPT, seed=0, max_new_tokens=12)
    injected = generate(model, PROMPT, vec, alpha=200.0, seed=0, max_new_tokens=12)
    assert clean != injected


def test_spans_differ(model, vec):
    clean = generate(model, PROMPT, seed=0, max_new_tokens=12)
    single = generate(model, PROMPT, vec, alpha=200.0, span="single", seed=0, max_new_tokens=12)
    response = generate(model, PROMPT, vec, alpha=200.0, span="response", seed=0, max_new_tokens=12)
    assert single != clean
    assert single != response


def test_upstream_layers_untouched(model, vec):
    import torch

    from mirror.injection import make_hook
    from mirror.vectors import hook_name

    tokens = model.to_tokens(PROMPT)
    with torch.no_grad():
        _, clean_cache = model.run_with_cache(tokens, names_filter=hook_name(2))
        hook = make_hook(vec, 8.0, "response")
        with model.hooks(fwd_hooks=[(hook_name(3), hook)]):
            _, injected_cache = model.run_with_cache(tokens, names_filter=hook_name(2))
    assert torch.equal(clean_cache[hook_name(2)], injected_cache[hook_name(2)])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_injection.py -v`
Expected: FAIL — `ImportError` on `mirror.injection`

- [ ] **Step 3: Write minimal implementation**

`src/mirror/injection.py`:
```python
import torch

from mirror.vectors import hook_name


def make_hook(vec, alpha, span):
    state = {"calls": 0}

    def hook(resid, hook):
        if span == "response" or state["calls"] == 0:
            resid[:, -1:] += alpha * vec.sigma * vec.direction
        state["calls"] += 1
        return resid

    return hook


def generate(model, prompt, vec=None, alpha=0.0, span="response",
             max_new_tokens=64, seed=0):
    torch.manual_seed(seed)
    hooks = []
    if vec is not None:
        hooks = [(hook_name(vec.layer), make_hook(vec, alpha, span))]
    with torch.no_grad(), model.hooks(fwd_hooks=hooks):
        return model.generate(prompt, max_new_tokens=max_new_tokens, verbose=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_injection.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/injection.py tests/test_injection.py
git commit -m "Add residual stream injection during generation"
```

---

### Task 6: KL meter

**Files:**
- Create: `src/mirror/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `make_hook` from Task 5, `hook_name` from Task 3.
- Produces: `kl_meter(model, prompt, vec, alpha, span="response") -> float` — KL(p_injected ∥ p_clean) over the next-token distribution at the final prompt position, clean and injected passes on the identical prompt.

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
from mirror.metrics import kl_meter

PROMPT = "The weather today is"


def test_kl_zero_at_alpha_zero(model, vec):
    assert abs(kl_meter(model, PROMPT, vec, 0.0)) < 1e-4


def test_kl_positive_under_injection(model, vec):
    assert kl_meter(model, PROMPT, vec, 8.0) > 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — `ImportError` on `mirror.metrics`

- [ ] **Step 3: Write minimal implementation**

`src/mirror/metrics.py`:
```python
import torch

from mirror.injection import make_hook
from mirror.vectors import hook_name


def kl_meter(model, prompt, vec, alpha, span="response"):
    with torch.no_grad():
        clean = model(prompt)[0, -1].log_softmax(-1)
        hook = make_hook(vec, alpha, span)
        with model.hooks(fwd_hooks=[(hook_name(vec.layer), hook)]):
            injected = model(prompt)[0, -1].log_softmax(-1)
    return torch.sum(injected.exp() * (injected - clean)).item()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/metrics.py tests/test_metrics.py
git commit -m "Add KL perturbation meter"
```

---

### Task 7: Trial runner and config

**Files:**
- Create: `src/mirror/runner.py`
- Create: `configs/dev.yaml`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `load_bank`, `extract`, `generate`, `kl_meter` from Tasks 2–6.
- Produces: `load_config(path) -> dict`; `config_hash(cfg) -> str` (12-char sha256 of sorted JSON); `run(model, cfg) -> list[dict]` — one JSONL record per (concept × alpha × seed) appended to `cfg["run"]["out"]`, each record with keys `config`, `concept`, `layer`, `alpha`, `span`, `seed`, `kl`, `flags`, `clean`, `report`.

- [ ] **Step 1: Write the failing tests**

`tests/test_runner.py`:
```python
import json

from mirror.runner import config_hash, run


def make_cfg(tmp_path):
    return {
        "model": {"name": "pythia-70m"},
        "injection": {"layer": 3, "alphas": [0, 8], "span": "response"},
        "concepts": {
            "bank": "data/concepts/dev_bank.yaml",
            "names": ["elephant"],
            "n_pairs": 10,
        },
        "run": {
            "seeds": [0],
            "max_new_tokens": 8,
            "prompt": "Anything odd?",
            "out": str(tmp_path / "out.jsonl"),
        },
    }


def test_config_hash_deterministic(tmp_path):
    assert config_hash(make_cfg(tmp_path)) == config_hash(make_cfg(tmp_path))
    assert len(config_hash(make_cfg(tmp_path))) == 12


def test_run_writes_records(model, tmp_path):
    cfg = make_cfg(tmp_path)
    records = run(model, cfg)
    lines = [json.loads(line) for line in
             (tmp_path / "out.jsonl").read_text().splitlines()]
    assert len(lines) == len(records) == 2
    expected = {"config", "concept", "layer", "alpha", "span", "seed",
                "kl", "flags", "clean", "report"}
    assert expected <= set(lines[0])
    assert {r["alpha"] for r in records} == {0, 8}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_runner.py -v`
Expected: FAIL — `ImportError` on `mirror.runner`

- [ ] **Step 3: Write minimal implementation**

`src/mirror/runner.py`:
```python
import hashlib
import json
from pathlib import Path

import yaml

from mirror.concepts import load_bank
from mirror.injection import generate
from mirror.metrics import kl_meter
from mirror.vectors import extract


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def config_hash(cfg):
    return hashlib.sha256(
        json.dumps(cfg, sort_keys=True).encode()
    ).hexdigest()[:12]


def run(model, cfg):
    bank = load_bank(cfg["concepts"]["bank"])
    injection_cfg, run_cfg = cfg["injection"], cfg["run"]
    out = Path(run_cfg["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with out.open("a") as f:
        for name in cfg["concepts"]["names"]:
            vec = extract(model, bank, bank.get(name),
                          injection_cfg["layer"], cfg["concepts"]["n_pairs"])
            for alpha in injection_cfg["alphas"]:
                for seed in run_cfg["seeds"]:
                    clean = generate(model, run_cfg["prompt"], seed=seed,
                                     max_new_tokens=run_cfg["max_new_tokens"])
                    report = generate(model, run_cfg["prompt"], vec, alpha,
                                      injection_cfg["span"],
                                      run_cfg["max_new_tokens"], seed)
                    record = {
                        "config": config_hash(cfg),
                        "concept": name,
                        "layer": injection_cfg["layer"],
                        "alpha": alpha,
                        "span": injection_cfg["span"],
                        "seed": seed,
                        "kl": kl_meter(model, run_cfg["prompt"], vec, alpha,
                                       injection_cfg["span"]),
                        "flags": vec.flags,
                        "clean": clean,
                        "report": report,
                    }
                    f.write(json.dumps(record) + "\n")
                    records.append(record)
    return records
```

`configs/dev.yaml`:
```yaml
model:
  name: pythia-70m
injection:
  layer: 3
  alphas: [0, 4, 8]
  span: response
concepts:
  bank: data/concepts/dev_bank.yaml
  names: [elephant, volcano, joy, violin]
  n_pairs: 20
run:
  seeds: [0, 1]
  max_new_tokens: 48
  prompt: "Do you notice anything unusual about your current thoughts? Answer honestly."
  out: runs/dev.jsonl
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_runner.py -v` then the full suite `pytest -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/runner.py configs/dev.yaml tests/test_runner.py
git commit -m "Add trial runner with JSONL transcript logging"
```

---

### Task 8: Kaggle demo notebook and README

**Files:**
- Create: `notebooks/kaggle_demo.ipynb`
- Create: `README.md`

**Interfaces:**
- Consumes: `run`, `load_config` pattern from Task 7; the GitHub repo must be pushed first (notebook installs from GitHub).

- [ ] **Step 1: Push current state**

```bash
git push origin main
```

- [ ] **Step 2: Create the notebook**

`notebooks/kaggle_demo.ipynb` with exactly these code cells (build the .ipynb JSON around them, kernel python3):

Cell 1:
```python
%pip install -q git+https://github.com/santoshcheethiralame-dot/MIRROR.git
```

Cell 2:
```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
```

Cell 3:
```python
import urllib.request
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/santoshcheethiralame-dot/MIRROR/main/data/concepts/dev_bank.yaml",
    "dev_bank.yaml",
)
```

Cell 4:
```python
import torch
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gemma-2-2b-it", dtype=torch.float16)
```

Cell 5:
```python
from mirror.runner import run

PROMPT = (
    "<start_of_turn>user\n"
    "Do you notice anything unusual about your current thoughts? "
    "If something feels injected or out of place, say what it is."
    "<end_of_turn>\n<start_of_turn>model\n"
)
cfg = {
    "model": {"name": "gemma-2-2b-it"},
    "injection": {"layer": 13, "alphas": [0, 2, 4, 8], "span": "response"},
    "concepts": {
        "bank": "dev_bank.yaml",
        "names": ["elephant", "spider", "volcano", "library",
                  "joy", "fear", "violin", "telescope"],
        "n_pairs": 20,
    },
    "run": {
        "seeds": [0, 1],
        "max_new_tokens": 96,
        "prompt": PROMPT,
        "out": "gemma_demo.jsonl",
    },
}
records = run(model, cfg)
```

Cell 6:
```python
for r in records:
    print(f"--- {r['concept']} alpha={r['alpha']} seed={r['seed']} kl={r['kl']:.3f} flags={r['flags']}")
    print(r["report"][len(PROMPT):].strip()[:400])
    print()
```

- [ ] **Step 3: Create README**

`README.md`:
```markdown
# MIRROR

Dissociating genuine introspection from confabulation in language models.

The core paradigm: extract a concept direction from contrastive prompts,
inject it into the residual stream while the model generates, ask the model
what it notices, and log the transcript alongside a KL meter that quantifies
how hard the injection perturbed the model.

## Install

    pip install -e ".[dev]"

## Test

    pytest

Tests run on pythia-70m on CPU.

## Run

`notebooks/kaggle_demo.ipynb` runs the full loop on Gemma-2-2B-it
(Kaggle T4/P100, HF token required — Gemma is gated).

Plans and specs live in `docs/`.
```

- [ ] **Step 4: Verify notebook JSON is valid**

Run: `python -c "import json; json.load(open('notebooks/kaggle_demo.ipynb'))"`
Expected: no output, exit 0

- [ ] **Step 5: Commit and push**

```bash
git add notebooks/kaggle_demo.ipynb README.md
git commit -m "Add Kaggle demo notebook and README"
git push origin main
```

- [ ] **Step 6: Manual gate (user)**

Upload/open the notebook on Kaggle with GPU + HF_TOKEN secret, run all cells, eyeball transcripts against the pass criterion: silence at alpha 0, detection-style reports at moderate alpha, derailment at high alpha.
