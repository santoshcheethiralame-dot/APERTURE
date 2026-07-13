# Milestone 3: Prior-Null Gamma Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and simulation-validate the gamma access-parameter estimator — a softmax choice model over concepts whose last coefficient (gamma) measures injected-identity signal beyond frequency/concreteness/similarity priors.

**Architecture:** One `prior_null` module: an NLL function, a scipy L-BFGS `fit`, a `simulate_reports` generator, and a bootstrap `gamma_ci`. All operate on a caller-supplied feature array `X [n_trials, n_concepts, n_features]` and reports `y [n_trials]`; the module fetches no data. Correctness is proven by recovering known coefficients from simulation.

**Tech Stack:** Python 3.11+, numpy, scipy (new dep). pytest with seeded numpy.

## Global Constraints

- Repo: `C:\Users\carbo\projects\mirror`; commands from repo root; venv at `.venv`.
- No code comments, no docstrings, self-describing names (human-authored convention).
- Commit messages: plain imperative, NO co-author trailers, no AI mentions.
- Dependencies: add `scipy>=1.11` to pyproject; module imports only numpy + scipy.
- `is_injected` is the LAST feature column; `gamma = theta[-1]` by convention.
- Tests are simulations with fixed seed via `numpy.random.default_rng(seed)`.
- Softmax per trial: logits `X[t] @ theta` shape `[n_concepts]`; report prob is softmax of logits.

---

### Task 1: Add scipy dependency

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: importable `scipy` in the venv.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, change the `dependencies` list to include scipy:
```toml
dependencies = [
    "torch>=2.0",
    "transformer-lens>=2.0",
    "pyyaml>=6.0",
    "scipy>=1.11",
]
```

- [ ] **Step 2: Install**

Run: `.venv\Scripts\python -m pip install -e ".[dev]"`
Expected: scipy installs (or already satisfied), no errors.

- [ ] **Step 3: Verify import**

Run: `.venv\Scripts\python -c "import scipy.optimize; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "Add scipy dependency for prior-null estimator"
```

---

### Task 2: Softmax negative log-likelihood

**Files:**
- Create: `src/mirror/prior_null.py`
- Test: `tests/test_prior_null.py`

**Interfaces:**
- Produces: `neg_log_likelihood(theta, X, y) -> float`. `theta` is `[n_features]`, `X` is `[n_trials, n_concepts, n_features]`, `y` is `[n_trials]` of concept indices. For each trial, logits = `X[t] @ theta`, log-prob via log-softmax; returns the summed negative log-prob of the reported concept.

- [ ] **Step 1: Write the failing test**

`tests/test_prior_null.py`:
```python
import numpy as np

from mirror.prior_null import neg_log_likelihood


def test_nll_matches_hand_computation():
    X = np.array([[[0.0], [1.0]]])
    y = np.array([1])
    theta = np.array([2.0])
    logits = np.array([0.0, 2.0])
    expected = -(logits[1] - np.log(np.exp(logits).sum()))
    assert np.isclose(neg_log_likelihood(theta, X, y), expected)


def test_nll_uniform_at_zero_theta():
    X = np.zeros((1, 4, 1))
    y = np.array([0])
    theta = np.array([0.0])
    assert np.isclose(neg_log_likelihood(theta, X, y), np.log(4))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_prior_null.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` on `mirror.prior_null`

- [ ] **Step 3: Write minimal implementation**

`src/mirror/prior_null.py`:
```python
import numpy as np


def neg_log_likelihood(theta, X, y):
    logits = X @ theta
    m = logits.max(axis=1, keepdims=True)
    log_norm = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    chosen = logits[np.arange(len(y)), y]
    return float((log_norm - chosen).sum())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_prior_null.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/prior_null.py tests/test_prior_null.py
git commit -m "Add softmax negative log-likelihood for prior-null model"
```

---

### Task 3: simulate_reports

**Files:**
- Modify: `src/mirror/prior_null.py`
- Test: `tests/test_prior_null.py` (append)

**Interfaces:**
- Consumes: nothing from Task 2 directly.
- Produces: `simulate_reports(X, theta_true, rng) -> y` — for each trial draws a concept index from the softmax over `X[t] @ theta_true`. `rng` is a `numpy.random.Generator`. Returns int array `[n_trials]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prior_null.py`:
```python
from mirror.prior_null import simulate_reports


def test_simulate_prefers_high_logit_concept():
    n_trials = 2000
    X = np.zeros((n_trials, 3, 1))
    X[:, 2, 0] = 1.0
    theta = np.array([5.0])
    rng = np.random.default_rng(0)
    y = simulate_reports(X, theta, rng)
    assert (y == 2).mean() > 0.9


def test_simulate_shape_and_range():
    X = np.zeros((10, 4, 2))
    rng = np.random.default_rng(1)
    y = simulate_reports(X, np.zeros(2), rng)
    assert y.shape == (10,)
    assert set(np.unique(y)).issubset({0, 1, 2, 3})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_prior_null.py::test_simulate_shape_and_range -v`
Expected: FAIL — `ImportError: cannot import name 'simulate_reports'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/prior_null.py`:
```python
def _softmax(logits):
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def simulate_reports(X, theta_true, rng):
    probs = _softmax(X @ theta_true)
    n_concepts = X.shape[1]
    return np.array([rng.choice(n_concepts, p=probs[t]) for t in range(len(probs))])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_prior_null.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/prior_null.py tests/test_prior_null.py
git commit -m "Add report simulation from the choice model"
```

---

### Task 4: fit

**Files:**
- Modify: `src/mirror/prior_null.py`
- Test: `tests/test_prior_null.py` (append)

**Interfaces:**
- Consumes: `neg_log_likelihood`, `simulate_reports`.
- Produces: `Fit` dataclass with `theta` (array), `gamma` (float = `theta[-1]`), `loglik` (float); `fit(X, y) -> Fit` minimizing NLL via scipy L-BFGS-B from a zero start.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prior_null.py`:
```python
from mirror.prior_null import fit


def _sim_dataset(theta_true, n_trials, n_concepts, n_features, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_trials, n_concepts, n_features))
    injected = rng.integers(0, n_concepts, size=n_trials)
    X[:, :, -1] = 0.0
    X[np.arange(n_trials), injected, -1] = 1.0
    y = simulate_reports(X, theta_true, rng)
    return X, y


def test_fit_recovers_zero_gamma():
    theta_true = np.array([1.0, 0.0])
    X, y = _sim_dataset(theta_true, 4000, 6, 2, seed=0)
    result = fit(X, y)
    assert abs(result.gamma) < 0.3


def test_fit_recovers_signal_gamma():
    theta_true = np.array([0.5, 2.0])
    X, y = _sim_dataset(theta_true, 4000, 6, 2, seed=1)
    result = fit(X, y)
    assert abs(result.gamma - 2.0) < 0.4


def test_fit_recovers_frequency_coefficient():
    theta_true = np.array([1.5, 0.0])
    X, y = _sim_dataset(theta_true, 4000, 6, 2, seed=2)
    result = fit(X, y)
    assert abs(result.theta[0] - 1.5) < 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_prior_null.py::test_fit_recovers_zero_gamma -v`
Expected: FAIL — `ImportError: cannot import name 'fit'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/prior_null.py` (add `from dataclasses import dataclass` and `from scipy.optimize import minimize` at the top):
```python
from dataclasses import dataclass

from scipy.optimize import minimize


@dataclass
class Fit:
    theta: np.ndarray
    gamma: float
    loglik: float


def fit(X, y):
    n_features = X.shape[2]
    result = minimize(neg_log_likelihood, np.zeros(n_features), args=(X, y),
                      method="L-BFGS-B")
    theta = result.x
    return Fit(theta=theta, gamma=float(theta[-1]), loglik=-float(result.fun))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_prior_null.py -v`
Expected: all PASS (fit tests take a few seconds each)

- [ ] **Step 5: Commit**

```bash
git add src/mirror/prior_null.py tests/test_prior_null.py
git commit -m "Add L-BFGS fit recovering coefficients from simulation"
```

---

### Task 5: gamma_ci

**Files:**
- Modify: `src/mirror/prior_null.py`
- Test: `tests/test_prior_null.py` (append)

**Interfaces:**
- Consumes: `fit`, `simulate_reports`.
- Produces: `gamma_ci(X, y, n_boot=200, rng=None) -> (lo, hi)` — 95% percentile bootstrap interval for gamma; resamples trial indices with replacement, refits, collects gamma. `rng` defaults to `numpy.random.default_rng()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prior_null.py`:
```python
from mirror.prior_null import gamma_ci


def test_ci_excludes_zero_under_signal():
    theta_true = np.array([0.5, 2.0])
    X, y = _sim_dataset(theta_true, 3000, 6, 2, seed=3)
    lo, hi = gamma_ci(X, y, n_boot=60, rng=np.random.default_rng(3))
    assert lo > 0.0


def test_ci_includes_zero_under_pure_prior():
    theta_true = np.array([1.0, 0.0])
    X, y = _sim_dataset(theta_true, 3000, 6, 2, seed=4)
    lo, hi = gamma_ci(X, y, n_boot=60, rng=np.random.default_rng(4))
    assert lo < 0.0 < hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_prior_null.py::test_ci_excludes_zero_under_signal -v`
Expected: FAIL — `ImportError: cannot import name 'gamma_ci'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/mirror/prior_null.py`:
```python
def gamma_ci(X, y, n_boot=200, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    n_trials = len(y)
    gammas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_trials, size=n_trials)
        gammas.append(fit(X[idx], y[idx]).gamma)
    lo, hi = np.percentile(gammas, [2.5, 97.5])
    return float(lo), float(hi)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_prior_null.py -v`
Expected: all PASS (CI tests refit 60 times each — up to ~30s)

- [ ] **Step 5: Commit and run full suite**

```bash
.venv\Scripts\python -m pytest -q
git add src/mirror/prior_null.py tests/test_prior_null.py
git commit -m "Add bootstrap confidence interval for gamma"
git push origin main
```
Expected: full suite green.

---

### Task 6: Record the estimator in the lab notebook

**Files:**
- Modify: `docs/LAB_NOTEBOOK.md`

**Interfaces:**
- Consumes: nothing.

- [ ] **Step 1: Add an entry**

In `docs/LAB_NOTEBOOK.md`, under the master-plan run-family table, flip E3's status note and add a short paragraph under "Findings so far" (or a new "Tooling" note) recording that the B4 gamma estimator exists and is simulation-validated (recovers gamma 0 and gamma 2, CI behaves), pending real frequency/concreteness feeds and a regime with actual identifications.

- [ ] **Step 2: Commit and push**

```bash
git add docs/LAB_NOTEBOOK.md
git commit -m "Note simulation-validated gamma estimator in lab notebook"
git push origin main
```
