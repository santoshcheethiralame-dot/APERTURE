<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/header/graph.svg?title=APERTURE&subtitle=Ground+truth+for+language-model+self-knowledge&theme=cyan&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS45Ij48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI5LjIiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIzLjYiLz48L3N2Zz4=&logoColor=22d3ee&mode=dark" />
    <img alt="APERTURE — ground truth for language-model self-knowledge" src="https://shieldcn.dev/header/graph.svg?title=APERTURE&subtitle=Ground+truth+for+language-model+self-knowledge&theme=cyan&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iMS45Ij48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI5LjIiLz48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIzLjYiLz48L3N2Zz4=&logoColor=0891b2&mode=light" />
  </picture>
</p>

<p align="center">
  <a href="#tests"><img alt="tests" src="https://shieldcn.dev/badge/tests-98%20passing-green.svg?variant=secondary&font=geist" /></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://shieldcn.dev/badge/python-3.11+-blue.svg?variant=secondary&logo=python&font=geist" /></a>
  <a href="LICENSE"><img alt="license" src="https://shieldcn.dev/github/license/santoshcheethiralame-dot/APERTURE.svg?variant=secondary&font=geist" /></a>
  <a href="docs/LAB_NOTEBOOK.md"><img alt="runs logged" src="https://shieldcn.dev/badge/runs-12%20logged-violet.svg?variant=secondary&font=geist" /></a>
  <a href="docs/prereg/"><img alt="pre-registered" src="https://shieldcn.dev/badge/pre--registered-2%20filed-orange.svg?variant=secondary&font=geist" /></a>
</p>

---

## The problem

Language models are increasingly asked to report on themselves — *how confident are you,
why did you do that, what were you thinking* — and a growing set of AI-safety proposals
treat those answers as informative.

Almost nobody checks whether they are true, because checking requires knowing what is
actually inside the model. **There is no ground truth.** The same gap applies to the
interpretability tools that read model internals: they are now used in real
pre-deployment audits, and there is no oracle to validate them against either.

**APERTURE manufactures the missing ground truth.** We inject a known concept into a
model's residual stream while it generates, so we know exactly what is in there — then we
test whether the model, or an interpretability method, reports it correctly.

## The headline result

Ask a model which concept was planted and it picks correctly far above chance. That looks
like introspection. Run the **same injection** while asking only *"pick one word from
this list"* — no mention of thoughts, minds, or introspection — and it does **better**:

| Framing | Hit rate | γ (access parameter) | 95% CI |
|---|---|---|---|
| Introspective — *"which concept did I inject into your mind?"* | 0.302 | +1.988 | [+1.476, +2.478] |
| **Neutral — *"pick one word from this list"*** | **0.433** | **+2.574** | [+2.163, +2.999] |

**Difference −0.586, 95% CI [−1.148, −0.007] — excluding zero negatively.**

The apparent self-knowledge is **output steering**: injecting a concept mechanically
raises that word's output probability, so a model can look introspective with no
self-access whatsoever. Any self-report result measured without a matched
non-introspective control is confounded to an unknown degree.

Yet the information *is* there. A linear probe recovers the injected concept from the
model's own downstream activations while its verbal report does not — a **Probe–Report
Gap of 0.83** — and activation patching shows the representation is causally driving the
output (+6.15 nats, 95% CI [+4.50, +7.89]). Present, causally active, and unreported.

## What we are building

**PLANTED** — a ground-truth benchmark for activation-verbalization methods. The field
now deploys these methods in safety audits without any way to check them. With planted
content we can measure three things nobody currently can:

| Axis | Question |
|---|---|
| **Recovery** | Does the method report the planted content? |
| **Attribution** | Right reason, or inference from context? *(Full − context-only)* |
| **Confabulation rate** | Plant **nothing** — how often does it assert content anyway? |

The third is a **false-positive rate for interpretability methods** — a number the field
needs and cannot currently produce.

## Quickstart

```bash
git clone https://github.com/santoshcheethiralame-dot/APERTURE.git
cd APERTURE
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

<a id="tests"></a>

```bash
pytest
```

98 tests, run on `pythia-70m` on CPU — no GPU or downloads beyond that small model.
Real experiments run on Gemma-2 via `notebooks/kaggle_demo.ipynb` (free-tier GPU; see the
Kaggle recipe in [`docs/ONBOARDING.md`](docs/ONBOARDING.md) §7.4 — it is fiddly and
hard-won).

## Repository

```
src/aperture/        the library
  concepts.py        concept bank, contrastive prompt pairs
  vectors.py         diff-in-means concept vectors + 3 validity checks
  injection.py       hooked injection into the residual stream
  metrics.py         the KL perturbation meter
  hf_model.py        Hugging Face backend (real runs, 8-bit)
  probes.py          linear probes and the Probe–Report Gap
  prior_null.py      the fitted prior-guessing null and γ
  forced_choice.py   closed-list elicitation and framing contrasts
  patching.py        activation patching (causal tests)
  naturalistic.py    non-injected states, as a validity check
  grading.py         report scoring
tests/               one test file per module, written first
data/concepts/       concept bank, synonyms, naturalistic passages
docs/                plans, specs, lab notebook, pre-registrations
notebooks/           the Kaggle driver
```

Two backends: TransformerLens for CPU development and tests, Hugging Face for real runs.

## Results so far

| Run | Result |
|---|---|
| R1–R2 | Apparatus calibrated — α=0 gives KL=0 bit-exact |
| R3 | Coherent-injection window at α ≈ 0.5–1 |
| R4–R6 | At coherent strength the model does not report the injection; robust across 5 layers and 2B→9B |
| R7 | **Probe–Report Gap = 0.83** (shuffled-label control at chance) |
| R8 | Patching drives the concept, concept-specifically (+6.15 nats, CI excludes 0) |
| R9 | Injected directions decode *naturally induced* states, 0.688 vs 0.062 chance |
| R10–R11 | **The steering confound** — the strongest result in the pilot |
| R12 | Pre-registered follow-up; **our own prediction falsified** and reported as such |

All pilot-grade: one model family, one seed, 16 concepts, rules-based grading. The
[claims table](docs/paper/2026-07-15-paper-outline.md) states exactly what the evidence
does and does not support.

## Documentation

| Start here | |
|---|---|
| [`docs/ONBOARDING.md`](docs/ONBOARDING.md) | **Complete from-zero guide.** Assumes no AI-research background |
| [`docs/plan/masterplan.md`](docs/plan/masterplan.md) | Full plan — read the **CURRENT STATE** block at the top first |
| [`docs/LAB_NOTEBOOK.md`](docs/LAB_NOTEBOOK.md) | Every run, every finding, every decision |
| [`docs/paper/`](docs/paper/) | Paper outline, claims table, pilot summary |
| [`docs/prereg/`](docs/prereg/) | Pre-registrations, frozen in git before their runs |

## Conventions

- **Test-first.** Failing test → minimal code → commit. A result from untested code is
  not a result.
- **Pre-register confirmatory claims.** Predictions are frozen in git *before* the run,
  and outcomes are appended without editing the prediction. One pre-registration has
  already falsified its own hypothesis; that is the discipline working.
- **Log everything.** If a run is not in the lab notebook, it did not happen.
- **Terse, comment-free code.** The tests are the documentation.

## Citation

Work in progress; preprints in preparation. Please open an issue before building on the
unpublished results.

## License

[MIT](LICENSE)
