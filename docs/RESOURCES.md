# APERTURE — Resource Ledger

What the program needs, what the floor is, and what is already secured. Update
alongside the lab notebook whenever access status changes.

Status key: SECURED / APPLY / UNRESOLVED / NOT NEEDED YET

---

## 1. Compute

| Need | Ideal | Minimum must-have | Status |
|---|---|---|---|
| Dev iteration (<=2B) | 1x A100-40GB always-on | **Kaggle free tier** (30 GPU-h/wk/person, 2x T4 15GB) | SECURED — proven through R1-R11 |
| **Mid tier (2B-9B) — the current bottleneck** | 1-2x A100-80GB | **PES in-house RTX Titan (24GB) + EPYC host** | APPLY — newly identified, confirm access |
| Mid tier fallback | — | ~200-400 rented A100-hrs (~$300-800, one-time) | NOT NEEDED YET — defer to E10 |
| Large tier (27B-70B+) | 4x A100-80GB | **NDIF/nnsight remote (free)** or **IISc KIAC/SERC** | UNRESOLVED — NDIF eligibility; IISc via mentor + published process |
| Total envelope | 2,000-2,600 A100-hrs | ~500-800 hrs leaning on free tiers | — |

### Why the PES box matters
RTX Titan is **24GB VRAM**, larger than Kaggle's 15GB T4. Consequences:
- Gemma-2-9B fits in **fp16** (~18GB) — no 8-bit quantization caveat.
- A large EPYC host RAM pool should eliminate the TransformerLens load-time RAM
  doubling that made 9B impossible on Kaggle's 30GB.
- Persistent: multi-seed jobs run unattended instead of dying with a session.

Limits: Turing generation (2018), no hardware bf16, slower than A100 — fine for
inference/probing/light LoRA, which is all this program does. A single 24GB card
does not hold 27B+ in fp16; that remains IISc/NDIF territory.

**To confirm:** number of GPUs (multiple enables model parallelism, the
`n_devices=2` path already built), host RAM, access method (SSH vs Slurm queue).

### Bangalore options beyond PES
- **IISc KIAC** — AI-ML centre, A100 + H200 nodes. Best technical fit.
- **IISc SERC** — PARAM Pravega (V100) + DGX-H100. Publishes a **non-IISc-user
  access process**, so external access is a documented route.
- **C-DAC Bangalore** — national PARAM systems; NSM allocation, bureaucratic.
- **IIIT-Bangalore** — smaller GPU labs; viable with a connection.
- Private colleges (RVCE, MSRIT, RNSIT) — modest A5000/A6000-class labs, usually
  tied to coursework; easy access where the mentor has standing.
- **Not accessible:** Google / Microsoft / NVIDIA / AWS Bangalore labs.

---

## 2. Credits and external access

| Need | Minimum must-have | Status |
|---|---|---|
| Frontier API (behavioural L3/L4 arm + LLM judges) | **$0 up front** — Anthropic External Researcher Access, rolling. Ask kept **<= $1,000**. Floor: skip the frontier arm; it is explicitly supplementary. | APPLY |
| Exact pretraining frequency (real gamma covariate) | **infini-gram — free public API** | APPLY — replaces the `wordfreq` proxy (R10 caveat) |
| Concreteness norms | **Brysbaert — free download** | APPLY — replaces the binary `is_abstract` flag (R10 caveat) |
| Large-model remote inference | **NDIF — free**, faculty support is the lever | UNRESOLVED — confirm India/non-US eligibility before planning around it |
| Cloud credits (Google/AWS student) | Nice-to-have | NOT NEEDED YET |

---

## 3. Data and artifacts

| Need | Ideal | Minimum must-have | Status |
|---|---|---|---|
| Concept bank | 240 stratified | **>= 120** before any frequency/concreteness claim | 16 built — EXPAND |
| Human gold labels | 1,000 stratified, >= 2 labellers | **~200 labels** for a first kappa | NOT STARTED — ~120 person-hours, labour not cash |
| Model weights | all tiers | Free; gated licences accepted once. Kaggle-native models avoid download failures. | SECURED |
| Storage / tracking | DVC + wandb team | **Free tiers**: GitHub + wandb free + Zenodo for the release DOI. `runs/` gitignored. | SECURED (~$0) |

---

## 4. Human and institutional

| Need | Status |
|---|---|
| Faculty mentor | **SECURED** — the single most important non-compute resource |
| Team labour | Plan assumes 5 people; currently ~solo. The 120+ concept bank and the gold-label set realistically need 1-2 more hands, or a scoped-down bank. |
| Institutional GPU line (PES / RNS via mentor) | APPLY — cheapest route to clearing the mid-tier floor |

---

## 5. Minimum viable path to the arXiv preprint (flag-plant)

The workshop paper is dropped; the single target is a full conference paper, with
an arXiv preprint at ~W44 as the scoop-insurance flag-plant (masterplan Addendum
4). This near-zero-cash path is what gets the preprint out; the conference version
then adds the confirmatory hardening (multi-seed, >=2 families, kappa).

Achievable on **near-zero cash**:

1. **Kaggle free tier** (have) + **PES Titan** (confirm) for compute
2. **infini-gram + Brysbaert** (free) to fix the gamma covariates
3. **~200 human labels** (in-team labour) for a first kappa
4. **One Anthropic credit grant** (free if approved) for the frontier arm — or skip it
5. **Concept bank -> 120** (labour)

First genuine cash line item: mid-tier rental for multi-seed confirmatory runs
(E10), **~$300-800, one-time, deferrable**. Everything before that is free tier
plus labour.

---

## 6. Integrity constraint on resource acquisition

Compute, funding, or institutional access **never** earn authorship. Resource
providers are credited in the **acknowledgements**. Trading co-authorship for GPU
access is prohibited (masterplan Addendum 3, §A3.4).
