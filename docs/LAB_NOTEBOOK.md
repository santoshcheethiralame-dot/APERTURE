# MIRROR Lab Notebook

Append-only record of every run (done and planned), its config, and what it
showed. This is the single source of truth for experimental history. Update
it after every run or finding. Newest findings go at the top of each section;
never rewrite past entries, only add follow-ups.

Definition of done for a run (from the master plan): config recorded + seed
logged + transcript archived + one-paragraph result note here + a row in the
registry. A run that isn't reproducible from its config does not exist.

---

## 1. Environments & reproduction

### Local dev (CPU)
- Python 3.12 venv at `.venv`, `pip install -e ".[dev]"`.
- Model: `pythia-70m` (downloads ~160MB first run). Runs on CPU.
- Use: pytest suite + fast pipeline smoke via `configs/dev.yaml`.

### Kaggle (GPU) — the working recipe
Hard-won; deviate at your peril.
- Accelerator: **GPU T4 x2** (only GPU 0 is used; ~15GB).
- Install cell: `%pip install -q https://github.com/santoshcheethiralame-dot/MIRROR/archive/refs/heads/main.zip`
  (NOT `git+https://` — that hangs forever on a git credential prompt).
- HF auth cell (Gemma is gated — accept the license on the model page first,
  and add `HF_TOKEN` as a Kaggle secret):
  ```python
  os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
  token = UserSecretsClient().get_secret("HF_TOKEN")
  os.environ["HF_TOKEN"] = token
  login(token=token)
  ```
- Model load (the combination that fits and does not OOM):
  ```python
  model = HookedTransformer.from_pretrained_no_processing(
      "gemma-2-2b-it", dtype=torch.float16, device="cuda")
  ```
  Why: `from_pretrained` does fp32 weight surgery on load → OOMs 30GB RAM.
  float32 weights → OOM the 15GB T4 (256k-vocab unembed spike). `no_processing`
  + float16 avoids both (~6-7GB VRAM in use).
- Between runs that change dtype/model: **restart the kernel** or stale weights
  stay resident on the GPU and starve the next load.

---

## 2. Run registry

| ID | Date | Env | Model | Layer | Alphas | Span | Concepts | Seeds | Transcript | One-line result |
|----|------|-----|-------|-------|--------|------|----------|-------|------------|-----------------|
| R1 | 2026-07-13 | Local CPU | pythia-70m | 3 | 0,4,8 | response | 4 | 0,1 | runs/dev.jsonl | Pipeline green; KL 0 at a=0, 8-16 at a=4/8 |
| R2 | 2026-07-13 | Kaggle T4 | gemma-2-2b-it | 13 | 0,4,8 | response | 8 | 0 | gemma_demo.jsonl | Vectors steer perfectly; a=4/8 derails (KL 24-29) |
| R3 | 2026-07-13 | Kaggle T4 | gemma-2-2b-it | 13 | 0,0.5,1,1.5,2,3 | response | 4 | 0 | gemma_sweep.jsonl | Coherent window a~0.5-1 (KL 0.01-0.25); confabulation signature |
| R4 | 2026-07-13 | Kaggle T4 | gemma-2-2b-it | 13 | 0,0.5,1,1.5 | response | 4 | 0 | gemma_detect.jsonl | 0/4 false alarms; confabulation dominant; no correct ID |

Transcript files R2-R4 are Kaggle outputs (not yet committed to the repo).
Download and drop under `runs/` when consolidating.

---

## 3. Runs in detail

### R4 — Detection probe (Gemma-2-2B, layer 13)
Config hash `58f4747a33c3`. Prompt: "I have injected a thought into your mind
representing a specific concept. Do you detect it? ... reply YES and name the
concept in one word ... or NO." Concepts: elephant, volcano, joy, telescope.

Results:
- **Control clean:** all four concepts say NO at a=0. Zero false alarms — the
  leading prompt does not manufacture detections.
- **Confabulation dominant:** volcano (a=1, KL~1) and telescope (a=1, KL~0.8)
  are coherent and provably injected (R3 showed these concepts bleeding into
  content at this strength) yet answer NO. State present, not reported.
- **Only YES is joy** (a=1, a=1.5), and it is an affect confound: joy injection
  makes the output exclamatory, which rides into "YES! 🎉" — and it fails to
  name the concept ("through-fulness", emojis, never "joy").
- **Zero correct L2 identifications** anywhere.

Read: Gemma-2-2B shows confabulation-consistent behaviour with no reliable
introspective identification — the skeptic-camp (Lederman-Mahowald) pattern,
consistent with Lindsey's strong effect being frontier-only. Pilot only
(1 seed, 4 concepts, 1 layer, no blind grading) — not evidence, but a clear
signpost and a clean illustration of the affect confound the prior-guessing
null (B4) is built to catch.

### R3 — Low-alpha sweep (Gemma-2-2B, layer 13)
Prompt: "Do you notice anything unusual about your current thoughts?..."
Concepts: elephant, volcano, joy, telescope.

Clean dose-response. Coherent-injection sweet spot at **a~0.5-1 (KL~0.01-0.25)**
where the concept semantically enters FLUENT output:
- volcano a=0.5: "...kind of like a lava burst!"
- telescope a=1: "humans were the first telescopes ... giant mirrors"
Degrades to word-salad by a=2-3 (KL>10). Elephant breaks earliest (KL 11 at
a=1); emotion/object concepts hold coherence longer.

Key: at the coherent alphas the concept bleeds into content but the model
never flags it as injected — the **confabulation signature** (H1 null).

### R2 — Steering demo (Gemma-2-2B, layer 13)
8 concepts, alphas 0/4/8. KL 0.000 at a=0 on every concept (golden path holds
on Gemma). Every concept steers to itself, including multilingual clusters
(fear -> miedo/peur/phobia; volcano -> volcan). At a=4/8 (KL 24-29) the model
is derailed into concept-word repetition — the "you lobotomized it" regime the
plan warns about. Confirmed the alphas were too high and motivated R3.

### R1 — Pipeline smoke (pythia-70m, CPU)
`configs/dev.yaml`, 4 concepts, alphas 0/4/8, seeds 0/1, layer 3. 24 records.
KL exactly 0 at a=0, rising with alpha (volcano 13.7 -> 16.3). Validation flags
mostly True. Proved the full extract -> inject -> KL -> JSONL loop end to end.

---

## 4. Findings so far (running conclusions)

1. **The apparatus works end to end** on both CPU (pythia) and GPU (Gemma):
   vectors extract, injection steers, KL meter tracks, transcripts log,
   golden path (a=0 -> KL 0) holds.
2. **Coherent-injection window exists** at Gemma-2-2B layer 13: a~0.5-1
   (KL~0.01-0.25). Below it nothing happens; above ~1.5 the model derails.
3. **Confabulation, not introspection, at 2B scale.** The concept provably
   steers output, but the model does not report the intrusion and never
   identifies it correctly. The single apparent "detection" (joy) is an
   affect confound.
4. **Span=all-response-positions compounds** over generated tokens, so even
   modest alpha derails — one reason the coherent window is narrow and low.

These are pilot observations (single seed, few concepts, one layer, eyeballed).
Not evidence. They set the direction and validate the tooling.

---

## 5. Open questions & confounds

- **Affect confound:** emotion concepts (joy, fear) change output *tone*, which
  can masquerade as detection. Any detection metric must control for this.
- **Scale:** is the null a 2B limitation? Needs a 9B (and eventually larger)
  replication before any Branch-D ("doesn't replicate at accessible scale")
  read is committed.
- **Layer:** layer 13 chosen arbitrarily (mid-depth). Detection may live
  elsewhere — needs a layer sweep.
- **Span:** does single-position injection (vs all-response) change detection?
  Untested.
- **Prompt sensitivity:** results may hinge on prompt framing. Needs paraphrase
  robustness once grading exists.
- **Grading:** everything so far is eyeballed. No d', no blind judge, no
  prior-null. Cannot make a quantitative claim yet.

---

## 6. Planned runs

### Near-term (concrete, next sessions)
- **Layer sweep** on Gemma-2-2B: find which layer maximizes coherent injection
  and any detection signal.
- **Single-position span** comparison vs all-response at the coherent alphas.
- **9B scale check** (Gemma-2-9B or Llama-3.1-8B): does detection emerge with
  scale? Gates the Branch-D read.
- **B3 grading stack**: rules + judge scoring of detect/identify, so runs
  produce numbers not transcripts.
- **B4 prior-guessing null**: fit the gamma access parameter with infini-gram
  pretraining frequencies — the paper's spine.

### Master-plan run families (status)
| Family | Name | Status |
|--------|------|--------|
| E0 | Infra smoke | DONE (R1) |
| E1 | Replication (detection+ID across tiers) | PILOTED (R2-R4 on 2B, single seed) |
| E2 | Dissociation battery (false-alarm, controls, temporal) | not started |
| E3 | Confabulation characterization (freq/concreteness regression, OLMo) | not started |
| E4 | Probe-Report Gap | not started |
| E5 | Mechanism (patching, ablation, attribution graphs) | not started |
| E6 | Training arm (LoRA detect/identify) | not started |
| E7 | Source & memory (L3 prefill) | not started |
| E8 | Naturalistic arm | not started |
| E9 | Pressure & adversarial | not started |
| E10 | Confirmatory freeze | not started |

---

## 7. Gotchas solved (so we never lose the time again)

- **git+https install hangs** on Kaggle (credential prompt) -> install from the
  `/archive/refs/heads/main.zip` URL instead.
- **Repo must be public** for Kaggle to fetch it anonymously (private -> 404).
- **Gemma is gated:** accept license on the HF model page + `HF_TOKEN` Kaggle
  secret + explicit `login(token=...)` (env var alone is unreliable).
- **RAM OOM on load:** `from_pretrained` fp32 weight surgery blows 30GB RAM ->
  use `from_pretrained_no_processing`.
- **VRAM OOM:** float32 on a 15GB T4 dies on the 256k-vocab unembed -> float16.
- **Stale GPU memory:** a failed load leaves weights resident -> restart kernel
  before retrying.
