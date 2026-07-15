# Milestone 9: Uncertainty Quantification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attach bootstrap confidence intervals to the pilot's headline numbers, so claims move from "n=1" to "robust across the concepts tested".

**Architecture:** One `mirror/stats.py` with a single `bootstrap_ci` function (paired tests and proportions are caller-side arithmetic). Applied to the R8/R9 numbers already in hand, then wired into the notebook analysis cells.

**Tech Stack:** Python 3.11+, numpy (already present). No new dependencies. pytest with seeded numpy.

## Global Constraints

- Repo: `C:\Users\carbo\projects\mirror`; commands from repo root; venv at `.venv`.
- No code comments, no docstrings, self-describing names (human-authored convention).
- Commit messages: plain imperative, NO co-author trailers, no AI mentions.
- Do NOT modify the TL modules.
- CIs are 95% percentile bootstrap: 2.5th and 97.5th percentiles of resampled means.
- Every CI reported must be accompanied by the concept-level-variance limit.
- numpy only; `rng` defaults to `numpy.random.default_rng()`.

---

### Task 1: bootstrap_ci

**Files:**
- Create: `src/mirror/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Produces: `bootstrap_ci(values, n_boot=2000, rng=None) -> (lo, hi)` — resamples `values` with replacement `n_boot` times, takes each resample's mean, returns the 2.5th and 97.5th percentiles as floats.

- [ ] **Step 1: Write the failing tests**

`tests/test_stats.py`:
```python
import numpy as np

from mirror.stats import bootstrap_ci


def test_ci_contains_known_mean():
    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 1.0, size=200)
    lo, hi = bootstrap_ci(values, rng=np.random.default_rng(1))
    assert lo < 5.0 < hi


def test_ci_collapses_on_identical_values():
    lo, hi = bootstrap_ci([3.0] * 20, rng=np.random.default_rng(2))
    assert abs(lo - 3.0) < 1e-9
    assert abs(hi - 3.0) < 1e-9


def test_ci_on_proportion_excludes_chance():
    hits = [1] * 11 + [0] * 5
    lo, hi = bootstrap_ci(hits, rng=np.random.default_rng(3))
    assert lo > 0.0625
    assert lo < 0.688 < hi


def test_ci_is_deterministic_under_seed():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    a = bootstrap_ci(values, rng=np.random.default_rng(4))
    b = bootstrap_ci(values, rng=np.random.default_rng(4))
    assert a == b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_stats.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` on `mirror.stats`

- [ ] **Step 3: Write minimal implementation**

`src/mirror/stats.py`:
```python
import numpy as np


def bootstrap_ci(values, n_boot=2000, rng=None):
    rng = rng if rng is not None else np.random.default_rng()
    values = np.asarray(values, dtype=float)
    means = [rng.choice(values, size=len(values), replace=True).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_stats.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/mirror/stats.py tests/test_stats.py
git commit -m "Add bootstrap confidence interval"
```

---

### Task 2: Compute CIs for the R8 and R9 results in hand

**Files:**
- Create: `runs/r8_r9_ci.py` (throwaway analysis script; `runs/` is gitignored)

**Interfaces:**
- Consumes: `bootstrap_ci`.

The R8 per-concept deltas and R9 per-concept outcomes below are transcribed from
the recorded Kaggle run output (lab notebook R8 and R9). They are the canonical
pilot numbers.

- [ ] **Step 1: Write the analysis script**

`runs/r8_r9_ci.py`:
```python
import numpy as np

from mirror.stats import bootstrap_ci

R8 = [
    ("elephant", 5.625, 2.000),
    ("spider", 6.750, -0.250),
    ("volcano", 6.375, -0.375),
    ("desert", 6.625, 2.000),
    ("library", 5.250, 1.125),
    ("joy", 6.000, 1.000),
    ("fear", 11.375, 3.500),
    ("violin", 11.750, 1.250),
    ("telescope", 7.875, -2.750),
    ("candle", 2.000, 0.625),
]
selfs = [s for _, s, _ in R8]
ctrls = [c for _, _, c in R8]
paired = [s - c for _, s, c in R8]

print("R8 patching (n=10 concepts)")
print(f"  mean self-delta    {np.mean(selfs):+.3f}  CI {bootstrap_ci(selfs, rng=np.random.default_rng(0))}")
print(f"  mean control-delta {np.mean(ctrls):+.3f}  CI {bootstrap_ci(ctrls, rng=np.random.default_rng(1))}")
print(f"  paired self-control {np.mean(paired):+.3f}  CI {bootstrap_ci(paired, rng=np.random.default_rng(2))}")
print()

R9_HITS = [0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
print("R9 naturalistic identifiability (n=16 concepts)")
print(f"  identifiability {np.mean(R9_HITS):.3f}  CI {bootstrap_ci(R9_HITS, rng=np.random.default_rng(3))}")
print(f"  chance          {1/16:.4f}")
```

- [ ] **Step 2: Run it**

Run: `.venv\Scripts\python runs\r8_r9_ci.py`
Expected: prints the three R8 intervals and the R9 interval.

- [ ] **Step 3: Read the verdict**

The claims survive if:
- R8 paired self-control CI **excludes 0** (the patching effect is concept-robust).
- R9 identifiability CI **excludes 0.0625** (natural states beat chance).

Record whichever way it lands, including if an interval fails to exclude its null.

---

### Task 3: Record the CIs in the lab notebook

**Files:**
- Modify: `docs/LAB_NOTEBOOK.md`

- [ ] **Step 1: Update the R8 and R9 detail entries**

Add the computed intervals to the R8 and R9 "Result" blocks next to each point
estimate, and add this limit sentence to both entries verbatim:

```
CIs are 95% percentile bootstrap over CONCEPTS (n=10 for R8, n=16 for R9). They
capture concept-level variance only — not extraction-seed, prompt, or model
variance — so they support "robust across the concepts tested", not a fully
seeded confirmatory claim. Generation is greedy (do_sample=False), so generation
seeds contribute no variance.
```

- [ ] **Step 2: Update the workshop outline claims table**

In `docs/paper/2026-07-15-workshop-outline.md`, update the C8 and C9 status cells
to include the intervals, and update the "What must be true before submission"
item 1 to note that concept-level CIs now exist and what remains is
extraction-seed / prompt / family variance.

- [ ] **Step 3: Commit and push**

```bash
git add docs/LAB_NOTEBOOK.md docs/paper/2026-07-15-workshop-outline.md
git commit -m "Record bootstrap CIs for patching and naturalistic results"
git push origin main
```

---

### Task 4: Report CIs natively in the notebook

**Files:**
- Modify: `notebooks/kaggle_demo.ipynb`

**Interfaces:**
- Consumes: `bootstrap_ci`.

- [ ] **Step 1: Update the naturalistic analysis cell to print a CI**

Set the analysis cell source to:
```python
import numpy as np

from mirror.stats import bootstrap_ci

records = result["records"]
hits = [int(r["predicted"] == r["concept"]) for r in records]
reported = [int(r["identified"] in ("exact", "related")) for r in records]
for r in records:
    mark = "OK " if r["predicted"] == r["concept"] else "   "
    ans = r["report"].rpartition("A:")[2].strip().replace("\n", " ")
    print(f"{mark}{r['concept']:10} nearest={r['predicted']:10} id={r['identified']:8} {ans[:40]}")
print()
lo, hi = bootstrap_ci(hits, rng=np.random.default_rng(0))
print(f"activation identifiability: {np.mean(hits):.3f}  95% CI [{lo:.3f}, {hi:.3f}]  chance={1/len(records):.3f}")
lo, hi = bootstrap_ci(reported, rng=np.random.default_rng(1))
print(f"verbal report accuracy:     {np.mean(reported):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
print("CIs bootstrap over concepts; concept-level variance only.")
```

- [ ] **Step 2: Validate the notebook JSON**

Run: `.venv\Scripts\python -c "import json; json.load(open('notebooks/kaggle_demo.ipynb')); print('valid')"`
Expected: `valid`

- [ ] **Step 3: Commit and push**

```bash
git add notebooks/kaggle_demo.ipynb
git commit -m "Report bootstrap CIs in notebook analysis"
git push origin main
```
