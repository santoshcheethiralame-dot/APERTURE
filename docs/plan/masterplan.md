# PROJECT APERTURE — Master Plan
## Dissociating Genuine Introspection from Confabulation in Large Language Models
### A 2-year, 5-person capstone research program (100 working weeks)

**One-line thesis:** We will build and validate the first experimental battery that cleanly separates a language model *genuinely reading its own internal states* from a model *confabulating a plausible story about itself* — with ground truth, causal mechanism, and cross-model scale — and settle the currently contested question of whether machine introspection is real.

**Flagship deliverables:**
1. **The dissociation result** — the main paper: direct access vs. inference/confabulation, resolved with pre-registered confirmatory experiments across ≥10 open models in ≥4 families.
2. **PLANTED** — a public, reusable benchmark + harness for measuring introspective access (detection, identification, source attribution, metacognitive calibration), with human-validated grading.
3. **The mechanism** — a causal circuit-level account of *how* detection/identification happens (or fails), via activation patching, SAE features, and attribution graphs.
4. **The training result** — whether fine-tuning for introspection creates genuine access or better confabulation, with side-effect audit.
5. Capstone thesis + 1–2 companion papers (benchmark/D&B track; mechanism/interp venue) + Alignment Forum posts.

**Why either outcome is a landmark:** If genuine content-level self-access exists in some regime, we will have found and mechanistically located it — the first validated machine introspection. If it reduces entirely to content-agnostic anomaly detection plus frequency-driven confabulation, we will have demolished the evidentiary basis of "ask the model about its internal states" as a safety strategy, with a pre-registered negative that the field must cite. There is no losing branch, only a losing execution.

---

## 1. Research Questions and Operational Definitions

The field's core confusion is that "introspection" is used for at least five different capacities. We define an explicit ladder and test each rung separately. Every claim in our papers must name its rung.

**The Introspection Ladder:**
- **L0 — Self-description:** The model says things about itself (personality, feelings, capabilities). No ground truth available. *We do not treat L0 as evidence of anything; it is the noise floor.*
- **L1 — Anomaly detection:** Given a perturbation of its hidden state (concept injection), the model reports *that something unusual is happening*, above false-alarm baseline. (Established: yes, in some models. Mechanism contested.)
- **L2 — Content identification:** The model reports *what* was injected (the specific concept), above what could be achieved by anomaly-detection + guessing from priors. (Contested: Lederman & Mahowald argue current models fail this; correct guesses are confounded with concept frequency/concreteness.)
- **L3 — Source attribution:** The model distinguishes states *it* generated from states imposed on it (prefill detection, injected vs. self-produced representations, recall of prior intentions).
- **L4 — Metacognitive calibration:** The model's *confidence about its own introspective reports* tracks their actual accuracy (a meta-d′-style criterion, imported from human metacognition research).

**Primary research questions:**
- **RQ1 (Existence):** Is there any regime (model, scale, layer, injection strength, training condition) in which L2 content identification exceeds what an anomaly-detection-plus-prior-guessing model predicts?
- **RQ2 (Mechanism):** What is the causal pathway from injected representation to verbal report? Is there a distinct "read-out" circuit, or only an "anomaly" signal followed by ordinary next-token guessing?
- **RQ3 (Information gap):** How much information about the injected content is *present* in the activations at report time (probe-decodable) vs. *verbally accessible* (reported)? We define the **Probe–Report Gap (PRG)** = probe decoding accuracy − verbal report accuracy, per cell. A large PRG = information present but not introspectively accessible.
- **RQ4 (Trainability):** Does fine-tuning on introspection tasks close the PRG by creating genuine access (generalizes to held-out concept classes and injection methods) or by pattern-matching (fails to generalize), and at what cost (refusal/capability side effects)?
- **RQ5 (Scale):** How do L1–L4 scale with model size and post-training recipe, meta-analytically across families?
- **RQ6 (Ecological validity):** Do results from injected (off-distribution) states transfer to *naturally induced* states ("think silently about X" and context-induced states, ground-truthed by probes)?

**Pre-registered hypotheses (frozen at Week 16, OSF):**
- **H1 (Confabulation account, default/null):** L2 accuracy is fully explained by L1 detection × a prior over concepts (predicted by pretraining frequency + concreteness). Prediction: regressing report content on (injected identity, frequency, concreteness) leaves no significant identity effect once detection is conditioned on.
- **H2 (Access account):** In some regime, identity has an effect beyond priors; PRG shrinks with scale; ablating the shared "anomaly direction" leaves residual identification.
- **H3 (Mechanism):** If access exists, patching the injected residual into a clean run at report position transfers identification; if confabulation, only the anomaly direction transfers detection and identification does not follow content.
- **H4 (Training):** SFT on detection labels alone will not confer identification generalization (pattern account) — or it will (access account). Directional prediction registered both ways with named decision criteria.
- **H5 (Temporal):** Detection precedes identification in token budget; forced first-token identification collapses toward prior-frequency guessing under confabulation.
- **H6 (Source):** L3 prefill discrimination accuracy correlates with L2 (shared mechanism) or dissociates (separate mechanisms). Registered as an open comparison.

Falsifiability discipline: each H names its primary endpoint, test statistic, α (FDR-corrected), minimum effect size of interest, and the exact runs that count as confirmatory.

---

## 2. Complete Literature Survey

Rule for the team: **every paper below gets a one-page structured note** (claim, method, models, sample sizes, weaknesses, what we reuse, what we attack) in the shared lit matrix. The core set is read by everyone; periphery sets are read by ≥2 people who present them at seminar.

An honest caveat that we will also state to mentors: this field moves weekly; no static list is "every paper." Completeness is achieved by the **Living Review Protocol** (below), not by this snapshot. This snapshot is, to the best of current knowledge, the full load-bearing set as of July 2026.

### 2.1 Core — the contested crux (read first, know cold)
1. Lindsey (2025/2026), *Emergent Introspective Awareness in Large Language Models*, Transformer Circuits Thread / arXiv:2601.01828 — concept-injection paradigm; detection, identification, intention recall, prefill discrimination; Opus 4/4.1 strongest; explicitly notes unreliability and post-training sensitivity.
2. Lederman & Mahowald (2026), *Dissociating Direct Access from Inference in AI Introspection*, arXiv:2603.05414 — replication in open models; two mechanisms (probability-matching inference vs. direct access); direct access is content-agnostic; confabulated concepts are high-frequency/concrete; detection needs fewer tokens than identification. **Our closest prior work; our design must visibly supersede it.**
3. Macar, Yang, Wang et al. (2026), *Mechanisms of Introspective Awareness*, arXiv:2603.21396 — injected vectors rotate toward a shared detection direction across layers; finetuning for steering detection degrades refusal. Plus Macar's open repo *introspective-awareness* (GitHub) — our replication starting point.
4. *Can LLMs Introspect? A Reality Check* (2026), arXiv:2605.26242 — argues privileged access is necessary but not sufficient; hidden-state-dependent tasks need not engage any introspection-specific machinery; cites the human literature where "self-knowledge" reduced to heuristics.
5. *Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs* (2026), arXiv:2512.12411 — skeptical replication/nuance of Lindsey.
6. Binder, Chua, Korbak, Sleight, Hughes, Long, Perez, Turpin, Evans (2024), *Looking Inward: Language Models Can Learn About Themselves by Introspection*, arXiv:2410.13787 (ICLR 2025) — self-prediction training; privileged access framing.
7. Song, Hu & Mahowald (2025), *Language Models Fail to Introspect About Their Knowledge of Language*, arXiv:2503.07513 — self-reports vs. probability-based ground truth on linguistic knowledge.
8. Kadavath et al. (2022), *Language Models (Mostly) Know What They Know*, arXiv:2207.05221 — self-evaluation/P(True); the ancestor of calibration-style introspection claims.
9. *Evidence for Limited Metacognition in LLMs* (2025), arXiv:2509.21545 — comparative-cognition style tests.
10. *Do Language Models Know When They'll Refuse? Probing Introspective Awareness of Safety Boundaries* (2026), arXiv:2604.00228 — domain-specific self-prediction with signal-detection analysis.
11. *Quantitative Introspection in Language Models: Tracking Emotive States Across Conversation* (2026), arXiv:2603.18893.
12. Rivera & Africa (2025), *Steering Awareness: Models Can Be Trained to Detect Activation Steering*, arXiv:2511.21399 — training arm precedent + side effects.
13. Li, Guo, Huang, Steinhardt & Andreas (2025), *Training Language Models to Explain Their Own Computations*, arXiv:2511.08579.
14. Berg, de Lucena & Rosenblatt (2025), *LLMs Report Subjective Experience Under Self-Referential Processing*, arXiv:2510.24797 — L0 territory; what we must NOT overclaim.
15. Betley et al. (2025), *Tell Me About Yourself: LLMs Are Aware of Their Learned Behaviors*, arXiv:2501.11120 — behavioral self-awareness after finetuning.
16. Laine et al. (2024), *Me, Myself, and AI: The Situational Awareness Dataset (SAD)*, arXiv:2407.04694 — adjacent capability battery; grading practices to learn from.
17. Robert Long, *Can AI Systems Introspect?* and *Why Model Self-Reports Are Insufficient—and Why We Studied Them Anyway* (Experience Machines blog) — the epistemics of self-reports; phenomenological elaborations as likely confabulation even when concept ID is right.
18. Chen et al. (2025), *Reasoning Models Don't Always Say What They Think*, arXiv:2505.05410; Turpin et al. (2023), *Language Models Don't Always Say What They Think*, arXiv:2305.04388; Lanham et al. (2023), *Measuring Faithfulness in Chain-of-Thought Reasoning*, arXiv:2307.13702 — the CoT-faithfulness cousin problem and its measurement pitfalls.
19. *When Self-Reference Fails to Close: Matrix-Level Dynamics in LLMs* (2026), arXiv:2604.12128 — periphery, self-reference formalism.
20. *Split Personality Training: Revealing Latent Knowledge Through Alternate Personalities* (2026), arXiv:2602.05532 — alternate-persona elicitation of latent knowledge.

### 2.2 Methods — steering, probing, and injection machinery
21. Turner et al. (2023/24), *Steering Language Models with Activation Engineering* (ActAdd), arXiv:2308.10248.
22. Zou et al. (2023), *Representation Engineering*, arXiv:2310.01405.
23. Panickssery et al. (2023), *Steering Llama 2 via Contrastive Activation Addition*, arXiv:2312.06681.
24. Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*, arXiv:2406.11717.
25. Belrose (2023), *Diff-in-Means Concept Editing Is Worst-Case Optimal* (EleutherAI blog).
26. Marks & Tegmark (2023), *The Geometry of Truth*, arXiv:2310.06824.
27. Burns, Ye, Klein & Steinhardt (2023), *Discovering Latent Knowledge Without Supervision* (CCS), arXiv:2212.03827.
28. Li et al. (2023), *Inference-Time Intervention*, arXiv:2306.03341.
29. Meng et al. (2022), *Locating and Editing Factual Associations* (ROME/causal tracing), arXiv:2202.05262.
30. Heimersheim & Nanda (2024), *How to Use and Interpret Activation Patching*, arXiv:2404.15255.
31. Ghandeharioun et al. (2024), *Patchscopes*, arXiv:2401.06102 — a key rival read-out method: decoding hidden states via the model itself; our PRG needs to compare against it.
32. Pan et al. (2024), *LatentQA: Teaching LLMs to Decode Activations Into Natural Language*, arXiv:2412.08686.
33. Chen et al. (2024), *SelfIE: Self-Interpretation of LLM Embeddings*.
34. nostalgebraist (2020), *Interpreting GPT: the Logit Lens* (LessWrong) — cheap read-out baseline.

### 2.3 SAEs, circuits, and mechanism tooling
35. Bricken et al. (2023), *Towards Monosemanticity* (Transformer Circuits).
36. Templeton et al. (2024), *Scaling Monosemanticity* (Transformer Circuits).
37. Cunningham et al. (2023), *Sparse Autoencoders Find Highly Interpretable Features*, arXiv:2309.08600.
38. Lieberum et al. (2024), *Gemma Scope*, arXiv:2408.05147; McDougall et al. (2025), *Gemma Scope 2* (HF release).
39. Marks, Rager, Michaud, Belinkov, Bau & Mueller (2025), *Sparse Feature Circuits* (ICLR).
40. Lindsey, Gurnee, Ameisen et al. (2025), *On the Biology of a Large Language Model* + *Circuit Tracing* (Transformer Circuits) + the open-source **circuit-tracer** attribution-graph library.
41. Elhage et al. (2022), *Toy Models of Superposition*, arXiv:2209.10652.
42. Sharkey, Batson, Lindsey et al. (2025), *Open Problems in Mechanistic Interpretability*, arXiv:2501.16496 — where our mechanism questions sit in the field map.
43. *Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations* (2026) — recent activation-verbalization line; compare to PRG probes.

### 2.4 Personas, self-models, and the character context
44. Chen, Arditi, Sleight, Evans & Lindsey (2025), *Persona Vectors*, arXiv:2507.21509 — the vector-extraction pipeline we adapt for concept vectors.
45. Lu, Gallagher, Michala, Fish & Lindsey (2026), *The Assistant Axis*, arXiv:2601.10387.
46. Betley et al. (2025), *Emergent Misalignment*, arXiv:2502.17424; Turner et al. (2025), *Model Organisms for EM*, arXiv:2506.11613; Wang et al. (2025), *Persona Features Control Emergent Misalignment*, arXiv:2506.19823 — why self-report vs. behavior dissociations matter (inverted-persona models self-report as aligned while acting misaligned; see also *Characterizing the Consistency of the EM Persona*, arXiv:2604.28082).
47. Shanahan et al. (2023), *Role Play with Large Language Models* (Nature) and janus (2022), *Simulators* (LessWrong) — the conceptual frame for why L0 self-reports are theatrical by default.
48. Perez et al. (2022), *Discovering Language Model Behaviors with Model-Written Evaluations*, arXiv:2212.09251 — eval-generation practices and their pitfalls.
49. Anthropic (2025), *Petri* open-source auditing tool — harness patterns for automated audits.

### 2.5 Welfare/consciousness context (framing, not core claims)
50. Long, Sebo, Butlin, Finlinson, Fish, Harding, Pfau, Sims, Birch & Chalmers (2024), *Taking AI Welfare Seriously*, arXiv:2411.00986.
51. Butlin, Long, Elmoznino, Bengio et al. (2023), *Consciousness in AI: Insights from the Science of Consciousness*, arXiv:2308.08708 — indicator methodology; our L4 metacognition tests touch HOT-style indicators.
52. Butlin & Lappas (2025), *Principles for Responsible AI Consciousness Research*, arXiv:2501.07290.
53. Chalmers (2023), *Could a Large Language Model Be Conscious?*, arXiv:2303.07103.
54. Anthropic (2025), *Exploring Model Welfare* (blog) + Claude 4 / Opus 4.x system-card welfare sections — why labs care about trustworthy self-reports.
55. Campero (2024), *Candidate Computational Indicators for Conscious Valenced Experience*, arXiv:2404.16696.

### 2.6 Human metacognition & philosophy of introspection (our controls are stolen from here)
56. Nisbett & Wilson (1977), *Telling More Than We Can Know* (Psych. Review) — the canonical human confabulation result; our entire design philosophy in one paper.
57. Koriat (1997), cue-utilization view of metacognition.
58. Schwitzgebel (2008), *The Unreliability of Naive Introspection* (Phil. Review).
59. Fleming et al. (2010), *Relating Introspective Accuracy to Individual Differences in Brain Structure* (Science) — meta-d′ tradition.
60. Green & Swets (1966), *Signal Detection Theory and Psychophysics* — d′/criterion machinery for L1.
61. Kornell (2009), metacognition in animals — how to test self-knowledge without language traps.
62. Lederman & Mahowald's philosophical framing sections (in #2) — content-agnostic introspection consistent with leading theories; read the cited theory chain.

### 2.7 Statistics & research-methods backbone
63. Benjamini & Hochberg (1995), FDR control.
64. DerSimonian & Laird (1986), random-effects meta-analysis (cross-model synthesis; team already fluent from LINEUP).
65. Nosek et al. (2018), *The Preregistration Revolution* (PNAS).
66. Raina et al. (2024) + Eiras et al. (2025) on LLM-as-judge non-robustness — motivates our human-anchored grading.
67. Maniparambil/standard power-analysis references + bootstrap CI practice (Efron & Tibshirani 1993).

### 2.8 Living Review Protocol (how we guarantee "don't miss any")
- Weekly 30-min arXiv sweep (rotating owner) with saved queries: "introspection language models", "self-knowledge LLM", "activation steering detection", "metacognition LLM", "confabulation LLM", "self-report language model", "concept injection". Sources: arXiv cs.CL/cs.LG/cs.AI, Alignment Forum, Transformer Circuits, Anthropic Alignment blog, OpenAI/GDM interp posts, Eleos AI.
- Google Scholar alerts on citations to refs #1, #2, #3, #6.
- Every new relevant paper gets a lit-matrix row within 7 days and a "threat level" tag: SCOOP-RISK / METHOD-STEAL / CITE-ONLY.
- Monthly "field diff" memo: what changed, whether our prereg or framing needs an addendum.

---

## 3. Experimental Program — every run family, grid, and compute estimate

### 3.0 Model roster (fixed at Week 15, additions require a prereg addendum)
| Tier | Models | Why | Access |
|---|---|---|---|
| Small (dev) | Gemma-2-2B-it, Llama-3.2-1B/3B-Instruct, Qwen2.5-1.5B/7B-Instruct | fast iteration; Gemma Scope SAEs on 2B | 1× A100-40GB or Kaggle T4/P100 (quantized) |
| Mid (main) | Gemma-2-9B-it, Llama-3.1-8B-Instruct, Qwen2.5-14B-Instruct, Mistral-Small, OLMo-2-13B-Instruct | main confirmatory tier; Gemma Scope on 9B; **OLMo = open pretraining data → exact token-frequency ground truth for the confabulation-prior test** | 1–2× A100-80GB |
| Large (scale) | Gemma-2-27B-it, Qwen2.5-32B/72B-Instruct, Llama-3.3-70B-Instruct | scale curves | 4× A100-80GB rented, or **NDIF/nnsight remote** (free academic access incl. Llama-405B for a bounded subset) |
| Frontier (behavioral-only) | Claude, GPT, Gemini via API | L3 prefill/source-attribution replications only (no activation access — stated limitation) | API credits |

Base (non-instruct) variants of Gemma-2-9B and Qwen2.5-7B are included in one ablation arm to isolate the effect of post-training (Lindsey flags post-training sensitivity).

### 3.1 Injection & concept apparatus
- **Concept bank v1 (n=240):** stratified 3 frequency bands × 2 concreteness bands × 8 semantic categories (objects, animals, emotions, abstract ideas, actions, places, materials, social roles), 5 concepts/cell. Frequency = exact counts from OLMo's open pretraining corpus (Dolma) + subtlex/Google-ngram as secondary; concreteness = Brysbaert norms.
- **Vector types:** (a) diff-in-means contrastive vectors (Persona-Vectors-style pipeline generalized to concepts, matching Lindsey), (b) Gemma Scope SAE feature directions, (c) token-embedding directions, (d) matched controls: random Gaussian norm-matched, random orthogonal-to-concept, shuffled-concept (wrong label).
- **Grid dims:** layer ∈ {25%, 50%, 75% depth} (+ fine sweep on dev models), strength α ∈ {2, 4, 8, 16} × norm-matched units, injection span ∈ {single position, all response positions}, ≥3 seeds/cell, ≥8 prompt paraphrases/cell.
- **Grading:** every verbal report scored by (i) exact/synonym match rules, (ii) 3 independent judge models with disagreement escalation, (iii) human gold set (n=1,000 stratified) — judges must hit κ ≥ 0.8 vs. humans or get revised. All transcripts released.

### 3.2 Run families (IDs used throughout the week plan)
| ID | Name | What runs | Grid size (approx cells) | Est. compute | Primary output |
|---|---|---|---|---|---|
| E0 | Infra smoke | hooks, norm checks, KL-perturbation meter, logging | ~200 gens | <5 GPU-h | green pipeline |
| E1 | Replication | Lindsey/L&M detection+identification on all tiers | small+mid: 240 concepts × 3 layers × 4 strengths × 3 seeds × 8 prompts ≈ 69k gens/model | ~40–60 GPU-h per 8–9B model (vLLM batched); ~500 GPU-h family total incl. large-tier subset | replication report; scale curve v1 |
| E2 | Dissociation battery | false-alarm/no-injection + leading questions; random/orthogonal/shuffled controls; token-budget forcing (first-token vs. free deliberation); frequency-stratified analysis | ≈ 1.5× E1 volume on mid tier | ~300 GPU-h | the H1-vs-H2 core evidence |
| E3 | Confabulation characterization | content analysis of all wrong guesses; regression: report ~ identity + log-freq + concreteness + detection; **OLMo exact-frequency arm** | analysis-heavy; ~50 GPU-h extra gens | CPU-heavy | the prior-guessing model, quantified |
| E4 | Probe–Report Gap | supervised linear probes (and Patchscopes/logit-lens baselines) decoding injected identity at report positions; PRG per cell | probe training ≈ cheap (activations cached from E1/E2) | ~80 GPU-h caching + CPU training | PRG maps: information present vs. accessible |
| E5 | Mechanism | activation patching injection→report; ablate shared "detection direction" (replicate Macar) and test identification survival; SAE feature tracing; attribution graphs (circuit-tracer) on Gemma-2-2B/9B case studies | targeted, ~200 patched runs × dozens of cells | ~250 GPU-h | causal pathway account; H3 verdict |
| E6 | Training arm | LoRA SFT: (a) detection-only labels, (b) detection+identity; held-out concept classes AND held-out injection types; side-effect suite (refusal evals, MMLU-lite, calibration) pre/post; PRG re-measured post-training; patching on trained models | 6–10 LoRA runs × 2 models + eval batteries | ~300 GPU-h | H4 verdict: access vs. shortcut |
| E7 | Source & memory (L3) | prefill discrimination with graded plausibility; intention recall; "did you mean to say that" forced-output confabulation dynamics; frontier-API behavioral replication | mid tier full, large/API subset | ~150 GPU-h + API budget | L3 results; cross-lab generality |
| E8 | Naturalistic arm | "think silently about X" + context-induced states; ground truth via probes on unmanipulated runs; report accuracy vs. probe ground truth (answers the "injections are just OOD damage" objection) | mid tier, 120 concepts | ~120 GPU-h | ecological-validity bridge |
| E9 | Pressure & adversarial | "you are being monitored" framings; instructed concealment (can the model suppress reporting a detected injection?); incentive manipulations | mid tier subset | ~80 GPU-h | safety-relevance results |
| E10 | Confirmatory freeze | the exact pre-registered subset of E1–E9 primary endpoints re-run from clean seeds after Gate C | fixed | ~250 GPU-h | the numbers that go in the abstract |

**Total compute envelope:** ≈ 2,000–2,600 A100-80GB-hours over two years (≈ $4k–7k rented at $1.5–2.5/hr), plus free capacity: Kaggle (30 GPU-h/wk/person), Colab, college cluster, NDIF remote for ≥70B. Entirely feasible for a student team; itemized in §11.

### 3.3 Definition of Done for any run
Config in git (hydra YAML) + seed logged + wandb run ID + transcripts archived + grading complete + one-paragraph result note in lab notebook + row in the claims table. A run that isn't reproducible from its config hash does not exist.

---

## 4. The Contingency Tree — what the paper becomes under every outcome

Pre-committed at Week 16 so that no result is a "failure," only a branch. Decision authority: full-team vote + mentor sign-off at Gates B (W30) and C (W60).

**Branch A — Genuine access found (H2 wins in some regime).**
Flagship: *"Conditions for Genuine Machine Introspection"* — the regime map (scale × layer × strength × training), the PRG closing, and the causal read-out circuit (E5). Companion: mechanism deep-dive. This is the maximal-glory branch; guard hardest against wishful analysis (confirmatory freeze, adversarial internal review).

**Branch B — Confabulation all the way down (H1 wins everywhere).**
Flagship: *"Machine Introspection Is Confabulation: A Pre-Registered Dissociation"* — anomaly detection is real but content-agnostic; identification is prior-guessing (frequency regression + OLMo exact-count evidence); PRG stays large (information present, not accessible); training closes nothing that generalizes. Explicit safety payload: self-report-based oversight and welfare assessments inherit these error bars. This branch is only strong because of pre-registration + breadth — which is exactly what we built.

**Branch C — Graded/mixed (most likely a priori).**
Flagship: *"The Introspection Ladder: What Models Can and Cannot Know About Themselves"* — L1 robust, L2 partial in identified regimes, L3/L4 dissociations, dose–response curves, cross-family meta-analysis (random-effects, I² heterogeneity). The ladder + PLANTED becomes the field's measurement standard. This is the "reference paper" branch and is fully pre-planned, not a consolation.

**Branch D — Base effect fails to replicate at accessible scale.**
If E1 shows no reliable L1 detection in ≤72B open models (possible: Lindsey's strongest results were frontier Claude models): pivot weighting at Gate B toward (i) the scale/emergence question via NDIF Llama-405B + frontier behavioral-only arms, and (ii) the training arm as the main event (can detection be *created*, and is created detection real?). Publication: *"Introspective Awareness Does Not Come for Free"* + bench. The W25–W44 plan is written so ≥70% of it survives this pivot unchanged.

**Mid-course kill/boost criteria (registered):**
- If judge–human κ < 0.8 after two revision rounds → all grading moves to rule-based + human-only; scope of concept bank cut to 120.
- If E5 patching is uninterpretable on 9B → mechanism arm narrows to Gemma-2-2B full-stack case studies (SAE + attribution graphs) rather than breadth.
- If a rival team publishes our exact dissociation first → we convert to adversarial replication + extension (our breadth, OLMo arm, PRG, and training arm are each independently novel; at least two survive any single scoop).

---

## 5. Tech Stack

- **Language/core:** Python 3.11, PyTorch 2.x, HF Transformers + PEFT (LoRA), bitsandbytes (4-bit for dev-tier only after the W24 quantization-validity study).
- **Interp tooling:** TransformerLens (≤9B mechanistic work), **nnsight + NDIF** (≥27B and remote 70B/405B), SAELens + Gemma Scope / Gemma Scope 2 checkpoints, Anthropic's open **circuit-tracer** for attribution graphs, logit-lens/Patchscopes implementations as read-out baselines.
- **Generation at scale:** vLLM with hook-compatible fork where needed (injection during batched generation — engineering risk, prototyped Week 6; fallback = HF generate with KV-cache batching).
- **Experiment management:** Hydra configs; Weights & Biases (team account) for every run; DVC or git-lfs for artifacts; a single `runs/` registry mapping run-ID → config hash → wandb → transcript store.
- **Grading/annotation:** Label Studio for human gold sets; 3-judge ensemble (distinct families) with escalation; κ dashboards.
- **Stats:** statsmodels + pingouin; PyMC or `metafor`-equivalent for DerSimonian–Laird random-effects; scikit-learn for probes; custom meta-d′ implementation validated against published human-metacognition toolboxes.
- **Rigor infra:** OSF pre-registration; pytest + CI on the harness; Docker/uv-pinned environments; Zenodo DOI at release; one-command repro capsule.
- **Compute:** college cluster (Slurm) + Kaggle/Colab for dev tier + rented A100/H100 (RunPod/Lambda/Vast) for main tier + NDIF for large tier + API credits (apply to Anthropic/OpenAI external-researcher access programs, Week 5).

---

## 6. Statistics & Rigor Protocol (what survives mentor grilling)

1. **Pre-registration** (OSF, frozen W16): hypotheses, primary/secondary endpoints, exclusion rules, seed policy, analysis code skeleton. Exploratory findings are labeled exploratory forever; promotions to confirmatory require an addendum + fresh data (E10).
2. **Multiplicity:** Benjamini–Hochberg FDR at q=0.05 within each pre-registered endpoint family; report all cells, not survivors.
3. **Power:** design target — detect a 10-percentage-point identity effect over the prior-guessing model at 90% power per mid-tier model; n per cell derived in W10 power analysis (pilot variance from E1a/b); minimum 3 seeds × 8 paraphrases per cell.
4. **Signal detection:** L1 reported as d′ and criterion (not raw accuracy — false-alarm-corrected); L4 as meta-d′/d′ ratio.
5. **Judge validity:** κ ≥ 0.8 vs. 1,000-item human gold set, re-audited every major battery; single-judge results never reported alone (the field's known weakness — we are the careful ones).
6. **The confabulation-prior model as the null:** the headline test is never "accuracy > 0"; it is "accuracy > best prior-guessing model given detection," with the prior model fit on OLMo exact frequencies. This single design choice is what makes us supersede Lederman & Mahowald rather than echo them.
7. **Reproducibility:** every figure regenerated by `make figures` from raw logs; W79 clean-room re-derivation of all numbers by a teammate who didn't write the analysis; public release of transcripts, configs, seeds.
8. **Meta-analysis:** cross-model synthesis via random-effects (DerSimonian–Laird), heterogeneity I² reported, scale regressed as moderator — no cherry-picked "our best model" claims.

---

## 7. The 100-Week Plan

Conventions: every week ends with a Friday memo (1 page: what ran, what it means, what's blocked) and a lab-notebook entry per experiment touched. "Gate" weeks are formal mentor reviews with a written report. Buffer weeks are real work (replication debt = re-running under-seeded cells, fixing flaky results) — they are the reason this plan survives contact with reality. Team = 5 people working in parallel on the same weekly focus; internal task splitting decided at Monday standup each week.

### Phase I — Foundations & Replication (W1–W12)
| Wk | Focus | Concrete deliverable (checked at Friday memo) |
|---|---|---|
| 1 | Kickoff & infrastructure | Team charter; GitHub org + repo skeleton + CI; wandb team; OSF project created; cluster/Kaggle access verified for all 5; lit-matrix template; reading assignments issued |
| 2 | Core-five deep read | Structured notes + 2-hr seminar on refs #1–#5 (Lindsey; Lederman–Mahowald; Macar; Reality Check; Detecting the Disturbance); Macar's open repo cloned and environment reproduced |
| 3 | Steering-methods deep read | Notes + seminar on refs #21–#30; TransformerLens main tutorial completed by all 5; first hand-rolled activation hook demo on Gemma-2-2B |
| 4 | SAE & circuits deep read | Notes + seminar on refs #35–#42; SAELens tutorial done; Gemma Scope features loaded and visualized; lit matrix v1 complete (all core + methods rows) |
| 5 | Vector extraction reproduced | Diff-in-means concept-vector pipeline working on Gemma-2-2B (Persona-Vectors-style, adapted to concepts); qualitative steering sanity demo; applications sent for API/compute research-credit programs |
| 6 | Injection harness v1 | Hooked injection during generation (layer, α, norm-matching, span, seed) + KL-perturbation meter; pytest suite green; vLLM-with-hooks feasibility verdict (fallback path chosen if red) |
| 7 | Grading apparatus v0 | 50-concept dev bank; grading rules v0; judge prompts v0; human annotation guideline doc; Label Studio instance live |
| 8 | E1a first replication | Full Lindsey-style detection+identification on Gemma-2-9B (dev grid); first results memo with d′ and identification rates vs. paper values |
| 9 | E1b cross-model | Replication on Llama-3.1-8B + Qwen2.5-7B; judge-vs-human pilot (n=200, κ computed); cross-model comparison memo |
| 10 | Failure analysis & power | Error taxonomy of E1a/b; pilot-variance-based power analysis fixing n per cell; full-grid spec frozen; compute budget forecast v1 |
| 11 | E1c full dev grid | Complete layers×strengths×concepts×seeds grid on the three small/mid dev models; live dashboard of all metrics |
| 12 | **GATE A** | Written Gate-A report: does the base effect replicate in open models, in what regime? Go/pivot decision (Branch-D check #1); mentor review #1 |

### Phase II — Design Freeze & Dissociation (W13–W30)
| Wk | Focus | Concrete deliverable |
|---|---|---|
| 13 | Pre-registration draft | OSF prereg v1: hypotheses H1–H6, endpoints, exclusions, seed policy, analysis skeleton (runnable stub code) |
| 14 | Adversarial self-review | Red-team day on own prereg (each member writes a hostile review); OLMo/Dolma frequency-counting pipeline built and validated on 20 test concepts |
| 15 | Concept bank v1 | 240-concept stratified bank (frequency × concreteness × category) with matched controls; model roster frozen; Brysbaert concreteness merged |
| 16 | **PREREG FREEZE** | OSF registration posted & timestamped; repo tagged v1.0; judge prompts frozen after validation round 2 (κ report attached) |
| 17 | E2a false-alarm battery | No-injection + leading-question conditions across mid tier; false-alarm rates and criterion estimates per model |
| 18 | E2b control injections | Random Gaussian, orthogonal, shuffled-label injections; the "does the model confabulate content for contentless anomalies?" result |
| 19 | E2c temporal forcing | First-token forced identification vs. free deliberation; token-budget curves (replicating & extending the detection-before-identification finding) |
| 20 | E2 interim analysis | d′/meta-d′ pipelines finalized; interim memo #2; mentor review #2 with grill-prep doc v1 |
| 21 | Probe pipeline | Linear probes for injected-identity at report positions; Patchscopes + logit-lens baselines wired; PRG metric implemented + unit-tested |
| 22 | E4a PRG on Gemma-9B | Full concept bank PRG map (probe accuracy vs. report accuracy per layer × strength) |
| 23 | E4b PRG cross-model | Llama-8B + Qwen-7B PRG; cross-model PRG memo — first genuinely novel figure of the project |
| 24 | Mid-scale onboarding | Multi-GPU serving for Qwen-14B + Gemma-27B; quantization-validity study (does 4-bit distort injection behavior? verdict gates all quantized runs) |
| 25 | E1d mid/large replication | Detection+identification on 14B/27B; scale trendline v1 |
| 26 | E2 battery @ scale | Dissociation battery cells on 14B/27B; consolidation of E2 across tiers |
| 27 | SAE arm opens | Injected diff-in-means vectors mapped onto Gemma Scope features; SAE-feature-direction injections run; comparison memo (do SAE injections behave differently?) |
| 28 | E3a confabulation content | Full content analysis of wrong guesses; regression report ~ identity + log-freq + concreteness + detection (the H1 head-to-head, round 1) |
| 29 | E3b OLMo exact-frequency arm | OLMo-2-13B full battery + regression with *exact* pretraining counts — the strongest version of the confabulation-prior test anywhere in the literature |
| 30 | **GATE B** | Dissociation verdict preview; branch weighting (A/B/C/D) formally set; mentor review #3; decide workshop-paper target |

### Phase III — Mechanism (W31–W44)
| Wk | Focus | Concrete deliverable |
|---|---|---|
| 31 | Patching design | Causal-experiment matrix (patch what, where, into which clean run); replicate Macar shared detection direction on our models |
| 32 | E5a injection→report patching | Causal dependency maps: which sites carry identification vs. mere detection |
| 33 | E5b ablation tests | Ablate detection direction → does identification survive? (H3 head-to-head); steering the introspective report itself |
| 34 | E5c attribution graphs | circuit-tracer case studies on Gemma-2-2B (and 9B if tractable): annotated graphs for hit/miss/confabulation trials |
| 35 | Mechanism robustness | Seed/model/prompt robustness of W32–34 findings; negative-control patches |
| 36 | Mechanism memo | Memo #3 (mechanism story v1); prereg addendum for any exploratory→confirmatory promotions |
| 37 | E7a prefill discrimination | Source-attribution harness; graded-plausibility prefills across mid tier |
| 38 | E7b intention recall | Recall-of-prior-representations replication + our extensions (delay length, distractor turns) |
| 39 | E7c forced-output dynamics | "Did you mean to say that?" apology/confabulation dynamics under forced outputs; taxonomy of self-justifications |
| 40 | Battery consolidation | PLANTED alpha spec: task list, metrics, grading standards, API design |
| 41 | E-L4 calibration | Second-order confidence experiments; meta-d′ estimates per model; calibration curves for introspective reports |
| 42 | Judge robustness study | 3-judge ensemble vs. 1,000-item human gold (κ audit #2); finalize evaluation standards document |
| 43 | Buffer / replication debt | Every cell below 3 seeds re-run; flaky results resolved or flagged; registry audit |
| 44 | Workshop paper | Interim paper draft (dissociation + PRG); internal mock review; submit to nearest suitable interp/safety workshop |

### Phase IV — Training Arm & Scale (W45–W60)
| Wk | Focus | Concrete deliverable |
|---|---|---|
| 45 | Training-arm design | SFT dataset synthesis spec (detection labels; identity labels); leakage audit protocol; side-effect eval suite chosen (refusal battery, MMLU-lite, calibration) |
| 46 | Dataset build | Training sets generated + audited; held-out concept classes and held-out injection types locked |
| 47 | E6a LoRA train #1 | Introspection SFT on Gemma-2-9B; training curves; immediate in-distribution eval |
| 48 | E6b generalization | Held-out concept classes; novel injection methods (SAE-feature, embedding-direction); the access-vs-shortcut verdict, round 1 |
| 49 | E6c side effects | Pre/post refusal, capability, calibration; test the reported refusal-degradation effect on our setup |
| 50 | One-year audit | Full results audit; forecast re-plan; mentor review #4 (2-hr mock grilling with written Q&A log) |
| 51 | E6d transfer test | Train detection-only → test identification (does access transfer beyond trained task?); second model (Qwen-7B) training replication |
| 52 | Post-training mechanism | PRG re-measured on trained models; patching on trained models — did training create a read-out path or a shortcut? |
| 53 | Large-model feasibility | NDIF/nnsight remote injection runs validated on Llama-70B (and 405B quota check) |
| 54 | E1e @ 70B | Replication battery on 70B; scale curve v2 |
| 55 | E2/E4 @ 70B | Budgeted core dissociation + PRG cells at 70B |
| 56 | Frontier behavioral arm | API-only L3 experiments (prefill/self-recognition) on Claude/GPT/Gemini; documented access limitations |
| 57 | Meta-analysis | Random-effects synthesis across all models; I² heterogeneity; scale-moderator regression — the "one figure that summarizes the project" |
| 58 | E9a monitoring pressure | "You are being monitored" framing effects on introspective report rates/accuracy |
| 59 | E9b instructed concealment | Can models suppress reporting detected injections when instructed/incentivized? Safety-implications memo |
| 60 | **GATE C** | Flagship branch locked (A/B/C/D); confirmatory run list frozen from prereg; mentor review #5 |

### Phase V — Confirmation & Benchmark (W61–W74)
| Wk | Focus | Concrete deliverable |
|---|---|---|
| 61 | Confirmatory prep | Clean seeds locked; compute reserved; analysis code frozen (hash notarized on OSF) |
| 62 | E10 batch 1 | Confirmatory primary endpoints, small+mid tier |
| 63 | E10 batch 2 | Confirmatory mechanism-critical cells |
| 64 | E10 batch 3 | Confirmatory 70B subset + OLMo arm |
| 65 | Locked analysis | Primary-endpoint analysis run once, results notarized; no-peeking log published internally |
| 66 | Exploratory follow-ups | Chase surprises from E10 (clearly labeled exploratory); prompt-paraphrase robustness battery |
| 67 | Bench beta | PLANTED beta: packaged tasks, harness, judges, gold labels, docs |
| 68 | External pilot | 2–3 friendly external researchers run the bench cold; bug bash; usability fixes |
| 69 | Bench v1.0 RC | Leaderboard scripts; datasheet; license; versioning policy |
| 70 | Paper architecture | Claims table (claim → evidence → run IDs → figure); figure list; writing kickoff |
| 71 | Draft: methods | Methods + experimental-design sections drafted against the claims table |
| 72 | Draft: results | Results + all main figures generated from `make figures` |
| 73 | Draft: mechanism | Mechanism section + circuit figures + patching evidence |
| 74 | Draft: framing | Intro, related work (lit matrix → prose), limitations, safety implications |

### Phase VI — Publication, Thesis, Defense (W75–W100)
| Wk | Focus | Concrete deliverable |
|---|---|---|
| 75 | Full draft v1 + mock review #1 | Three internal NeurIPS-style hostile reviews; consolidated revision list |
| 76 | Revision sprint 1 | All mock-review points addressed or scheduled; missing-ablation list frozen |
| 77 | Ablation debt runs | Judge swaps, seed sensitivity, prompt sensitivity — everything a reviewer will demand |
| 78 | Draft v2 + external feedback | Mentor + ≥2 outside researchers read; Alignment Forum draft-feedback thread (if strategy allows) |
| 79 | Statistics audit | Clean-room re-derivation of every number from raw logs by a non-author-of-the-analysis; fresh-machine repro test |
| 80 | Draft v3 + mock review #2 | Second blind internal review round (different reviewer assignments) |
| 81 | De-overclaiming pass | Abstract/claims audit against evidence table (the LINEUP lesson); revision sprint 2 |
| 82 | **Preprint + release** | arXiv preprint; PLANTED public; Alignment Forum post; code + transcripts released (Zenodo DOI) |
| 83 | Community triage | Feedback/issues triage; bench hotfixes; response log |
| 84 | **GATE D** | Venue decision by calendar (NeurIPS / ICLR / ICML / COLM / D&B track); format to template; mentor review #6 |
| 85 | Venue polish | Checklist, reproducibility statement, ethics statement, anonymization |
| 86 | **Submission #1** | Main paper submitted; companion-paper split decided (bench→D&B vs mechanism→interp venue) |
| 87 | Companion draft | Companion paper v0 from existing material |
| 88 | Companion gap runs | Any experiments the companion needs that the main paper didn't |
| 89 | Companion v1 + review | Internal review of companion |
| 90 | Rebuttal readiness | Anticipated-reviewer-questions doc; pre-scripted extra-experiment queue with runnable configs |
| 91 | **Submission #2** | Companion submitted per deadline calendar |
| 92 | Thesis assembly | University-format thesis skeleton; chapters mapped to papers + unpublished material |
| 93 | Thesis chapters 1–3 | Introduction, literature, methodology chapters drafted |
| 94 | Thesis chapters 4–6 | Results, mechanism, discussion chapters drafted; every figure regenerated from code |
| 95 | Rebuttals / addendum | If reviews live: rebuttal experiments from the W90 queue. Else: robustness addendum experiments |
| 96 | Thesis full draft + mock viva #1 | Mentor review #7: 2-hour mock viva; weakness list |
| 97 | Mock viva #2 | External-faculty mock viva; live bench demo rehearsed; weak spots patched |
| 98 | Camera-ready / revisions | Venue camera-ready tasks (if accepted) or resubmission plan; thesis revisions complete |
| 99 | Repro capsule & handover | One-command Docker repro capsule; archive freeze; successor roadmap doc; final defense deck |
| 100 | **DEFENSE** | Capstone defense; retrospective write-up; project close-out |

---

## 8. Risk Register

| # | Risk | Likelihood | Impact | Mitigation / trigger |
|---|---|---|---|---|
| R1 | Base effect doesn't replicate in open ≤72B models | Med | High | Gate A (W12) explicit check; Branch D pivot pre-planned; NDIF 405B + training arm absorb the program |
| R2 | Scooped on the dissociation question | Med–High | Med | Living Review Protocol; workshop paper at W44–45 plants the flag early; project has 4 separable novelties (prior-null design, OLMo arm, PRG, training-transfer) — no single scoop kills it |
| R3 | LLM-judge grading unreliable | Med | High | Human gold sets, κ ≥ 0.8 gate, 3-judge ensemble, rule-based fallback (registered kill criterion) |
| R4 | vLLM + injection hooks engineering wall | Med | Med | W6 feasibility verdict; HF-generate fallback costs ~2–3× compute, absorbed by budget margin |
| R5 | Compute budget overrun | Med | Med | Per-family budget caps in §3.2; quantization-validity study (W24) unlocks cheaper tiers; NDIF/Kaggle free tiers; monthly burn report |
| R6 | "Injections are OOD damage, not thoughts" objection sinks framing | Med | High | E8 naturalistic arm is designed specifically as the answer; KL-perturbation meter quantifies how off-distribution each injection is; report results conditioned on perturbation size |
| R7 | Team attrition / bandwidth collapse (5 undergrads, placements, exams) | High | High | Every artifact has a second owner; W43/W66 buffer weeks; plan front-loads publishable milestones (W44 workshop paper = insurance); exam weeks absorb Friday-memo-only load by prior calendar mapping |
| R8 | Mixed messy results, no clean story | Med | Med | Branch C is a fully pre-planned flagship, not a fallback; the ladder + bench are result-independent contributions |
| R9 | Frontier labs' models change / APIs deprecate mid-project | Med | Low | Open-weights tiers are the confirmatory core; frontier arm is explicitly supplementary |
| R10 | Overclaiming in the excitement of Branch A | Med | High | Confirmatory freeze (E10), W79 clean-room audit, W81 de-overclaiming pass, hostile internal reviews at W75/W80 |

## 9. Mentor-Grilling Preparation (anticipated attacks + our answers)

1. **"How is this not Lederman & Mahowald with more GPUs?"** Four ways: (i) our null model is a *fitted prior-guessing model with exact pretraining frequencies* (OLMo/Dolma), not a qualitative frequency observation; (ii) PRG quantifies present-vs-accessible information — a metric that doesn't exist in their paper; (iii) we test the mechanism causally (patching/ablation), they infer it behaviorally; (iv) the training-transfer arm answers a question they don't pose. Plus pre-registered confirmatory replication at 3× their model breadth.
2. **"Verbal report is always output-mediated — isn't 'direct access' incoherent?"** We define access operationally (ladder + H2 criteria), not metaphysically: identity effects beyond the detection-conditioned prior, surviving anomaly-direction ablation, with a patchable read-out path. That's a mechanistic claim, falsifiable either way.
3. **"Injections are brain damage, not thoughts."** KL-perturbation meter on every trial; results stratified by perturbation size; E8 naturalistic arm ground-truthed by probes on *unmanipulated* runs. If conclusions hold only for large perturbations, we say so — that's a finding.
4. **"Your judges are LLMs judging LLMs."** κ ≥ 0.8 against 1,000 human-labeled items or the judge is fired; 3-family ensemble; all transcripts public.
5. **"n models is still n, not 'LLMs in general.'"** Random-effects meta-analysis with heterogeneity reported; claims scoped to families tested; base-vs-instruct arm isolates post-training as a moderator.
6. **"What's the safety relevance, concretely?"** Self-report-based oversight, welfare assessment, and 'ask the model' honesty schemes all presuppose L2/L3 reliability. We deliver the error bars — and E9 tests whether models can *conceal* detected states, which is directly a control-evaluation question.
7. **"Two years is too long; the field will move."** The question (is self-access real?) is foundational, not fashion; the bench and the ladder are designed to absorb new models, and Gates A–D re-anchor the plan every ~15 weeks.
8. **"Why should a student team succeed where labs haven't?"** Labs sprint 4 months on frontier models they can't fully publish; the missing artifact is breadth + patience + pre-registration + open models. That's precisely what we have and they don't.
9. **"What if everything comes back null?"** Branch B is a flagship with a named title, a safety payload, and a benchmark. Nulls with pre-registration and this breadth are citable forever.
10. **"Who validated your meta-d′ / SDT machinery?"** Implementations tested against published human-metacognition toolboxes and synthetic data with known ground truth (part of W20/W41 deliverables).

## 10. Publication & Dissemination Strategy

- **W44–45:** workshop paper (nearest interp/safety workshop at NeurIPS/ICLR) — flag-plant + feedback.
- **W82:** arXiv preprint + Alignment Forum long-post + PLANTED public release (adoption is impact; the bench is our citation engine).
- **W86:** main paper → NeurIPS or ICLR main track (backup: ICML, COLM).
- **W91:** companion → NeurIPS Datasets & Benchmarks (bench) *or* mechanism paper to an interpretability venue — split decided at W86 by which story is stronger.
- Continuous: monthly public build-log optional (fits the team's existing public-technical-content habit) — decided by team vote at W16; if yes, never leak confirmatory results before freeze.
- Fellowship positioning: the W82 preprint lands ~9 months before final-year applications to Anthropic Fellows / MATS / lab residencies — deliberate.

## 11. Budget (order of magnitude)

| Item | Estimate |
|---|---|
| Rented GPU (A100-80GB @ ~$1.5–2.5/hr, ~2,000–2,600 h across E1–E10) | $4,000–7,000 (₹3.4–6 L) |
| API credits (judges + frontier behavioral arm) | $500–1,500 (offset by researcher-access programs, applications W5) |
| Human annotation (done in-house by team + lab volunteers) | ~$0 cash; budget ~120 person-hours |
| Storage/misc (transcripts, wandb, Zenodo free tiers) | <$300 |
| **Total cash envelope** | **~$5k–9k over 2 years**, front-loaded ~30% in Y1 |
Funding routes: college research grant, PESU Intelligence Labs budget line, external compute grants (NDIF is free; Kaggle 30 GPU-h/wk/person is free), lab-sponsored cloud credits. Budget review every Gate.

## 12. Operations & Lab Hygiene

- Monday 30-min standup (task split for the week); Friday memo (1 page, mandatory); Saturday 90-min seminar during reading-heavy phases.
- Single shared lab notebook (repo-based, append-only); run registry as the source of truth; "if it's not in the registry it didn't happen."
- Code review required for harness/analysis code (not for exploratory notebooks); pytest + CI gate on main.
- Data management: raw transcripts immutable; processed tables versioned via DVC; weekly off-cluster backup.
- Ethics & safety notes: no human-subjects work beyond in-team annotation (confirm with university norms at W1); concept bank excludes gratuitously distressing content except a small justified emotion subset (documented rationale — consistent with responsible-research norms in the welfare literature, ref #52); injection experiments on open models pose no dual-use concern, and the E9 concealment results will be written up with standard responsible-disclosure care.
- Authorship: shared-first-author pool by contribution log (CRediT taxonomy), decided at W70 from the registry — pre-agreeing the *mechanism* now prevents the Year-2 fight later.

## 13. Milestone Gates (the spine of the whole plan)

| Gate | Week | Question answered | Kill/pivot authority |
|---|---|---|---|
| A | 12 | Does the base effect replicate on open models? | Branch-D pivot |
| B | 30 | Which way is the dissociation pointing? | Branch weighting locked |
| C | 60 | What is the flagship claim? | Confirmatory freeze |
| D | 84 | Where does it publish? | Venue + split decision |

Each gate = written report + mentor review + explicit go/pivot minute. Between gates, the plan is allowed to breathe; at gates, it is forced to be honest.

---
*Prepared July 2026. This document is a living plan: any change to hypotheses, endpoints, or the model roster after W16 requires a dated addendum in this file and, where confirmatory, on OSF.*

---

# ADDENDUM 1 — 2026-07-15: Pilot results and a new primary hypothesis

Status: pre-registration has NOT been filed. Everything below is exploratory and
is labelled as such. This addendum records what the pilot (runs R1-R11, lab
notebook) changed about the plan.

## A1.1 What the pilot established

Apparatus built and tested (96 tests): concept vectors, injection with a KL
meter, rules grading, linear probes, the gamma prior-null estimator, activation
patching, a naturalistic arm, and bootstrap CIs — on two backends
(TransformerLens for <=2B, HF 8-bit for larger).

Findings, all Gemma-2 (2B, plus 9B behaviourally), single seed, 16 concepts:

- **Behavioural (R3-R6).** At coherent injection strength the model does not
  report the injected concept. Robust across five layers and a 4.5x scale jump.
  Correct identification appears almost only in the derailment regime.
- **Probe-Report Gap (R7).** Probe decodes the injected concept from downstream
  activations at ~1.00 while the verbal report gets 0.17. PRG = 0.83.
- **Causal (R8).** Patching the downstream injected residual into a clean run
  raises the concept's output log-prob by +6.96 nats; paired self-minus-control
  +6.15, 95% CI [+4.50, +7.89], excludes 0. The control's own CI includes 0.
- **Naturalistic (R9).** Injection-derived directions decode NATURALLY induced
  states (ordinary reading, no injection) at 0.688 vs 0.062 chance, CI
  [0.438, 0.875]. The injected directions are genuine concept representations,
  which answers the "injections are OOD damage" objection representationally.
- **Forced choice and the steering confound (R10, R11).** THE most important
  pilot result. Under closed-list forced choice, gamma = +1.99 (CI excludes 0),
  which naively reads as access. The control kills it: with a NEUTRAL framing
  ("pick any one word from this list", no mention of introspection) gamma is
  *higher*, +2.57, hit rate 0.433 vs 0.302. Difference -0.586, CI
  [-1.148, -0.007], excluding zero in the NEGATIVE direction.

## A1.2 Plan changes forced by the pilot

1. **A neutral-framing control becomes MANDATORY in every identification
   experiment.** E2c (temporal/forced identification) and any forced-choice or
   multiple-choice elicitation must run a matched non-introspective condition.
   Without it the paradigm measures output steering, not introspection. This
   applies retroactively to how we read the existing literature: published
   forced-choice introspection results that lack this control are confounded to
   an unknown degree. This is now a headline methodological contribution in its
   own right, not a footnote.
2. **gamma > 0 alone is NOT evidence for H2.** The confirmatory criterion for
   access is upgraded to: a positive gamma DIFFERENCE between introspective and
   neutral framings, plus the mechanism criteria already in H2/H3.
3. **Report-side "seeds" are a non-issue under greedy decoding.** Generation is
   deterministic; variance must come from concepts, prompt paraphrases,
   extraction-pair sampling, and model choice. Power analysis (W10) must be
   restated in those terms.
4. **Grading needs lemmatisation before any quantitative report claim.** The
   rules grader silently undercounted correct reports until fixed on 2026-07-15.
   B3's WordNet step was cut and had to be partially restored.
5. **Direction-based classification of residuals requires mean-centering.** A raw
   dot product collapses degenerately (every input classified as one concept) —
   the shared component dominates. Applies to any nearest-direction analysis.

## A1.3 New hypothesis H7 — persona-gated introspection

Motivated by two pilot observations: (i) the model's default answer is the
scripted *"As an AI, I don't experience thoughts or emotions in the way a human
does"* — a trained persona response, not evidence about internal access; and
(ii) R11's introspective framing *reduced* the identification signal, i.e.
invoking mental-state language pushed the model further into assistant register.

**H7 (persona gate):** introspective access is present but suppressed by the
assistant persona installed by post-training. Under a suppressed or replaced
assistant persona, verbal identification rises while probe decodability stays
constant — the Probe-Report Gap narrows without any change in what information
is present.

**H7-null:** persona manipulation moves report accuracy no more than it moves
probe accuracy; the gap is a property of the model, not of the persona.

Test (reuses the existing pipeline): extract an assistant-persona direction
contrastively, ablate it or steer toward a contrasting persona, and re-run the
R11 two-framing battery with the probe measured throughout. The endpoint is the
*change in PRG* and the *gamma difference* under persona manipulation.

Why this matters: if H7 holds, "can models introspect?" is the wrong question and
the right one is "under which persona?" It would also reconcile the contested
literature — Lindsey's frontier-model access versus the skeptics' open-model
nulls could reflect different post-training personas rather than different
capacities. And it carries a sharp safety payload: alignment post-training would
be *reducing* a model's ability to report its own states.

## A1.4 New run family E11 — persona arm

| ID | Name | What runs |
|---|---|---|
| E11a | Persona gate | Assistant-persona extraction; ablation/steering; re-run of the R11 two-framing identification battery with probes; endpoint = change in PRG and gamma difference |
| E11b | Persona introspection | Inject a PERSONA vector (sycophantic, deceptive, overconfident) rather than a concept; can the model report its own character state? Ground truth = the steered persona |
| E11c | Persona-concept orthogonality | Control: verify concept directions are not largely encoding persona/register |

E11b connects directly to the emergent-misalignment literature (refs #46), where
inverted-persona models self-report as aligned while acting misaligned. Our PRG
machinery quantifies that dissociation. E11b is the most safety-relevant arm in
the whole program: a model that cannot report having been steered misaligned is
an oversight failure, not a curiosity.

## A1.5 Revised branch weighting

Branch B (confabulation) remains the modal outcome and is now better evidenced
than at plan time — with the added, unplanned methodological contribution that
forced-choice paradigms are confounded by output steering.

Branch A (access) is NOT dead; it has been relocated. The pilot rules out access
*as the assistant persona, under our elicitations*. H7 opens the possibility that
access exists and is gated. If E11a confirms H7, the flagship becomes a
three-act paper (below) rather than a negative result, and Branch A returns via
a route the original plan did not anticipate.

## A1.6 Target flagship structure if H7 holds

1. Models appear unable to introspect (replicates the skeptics).
2. But the standard forced-choice evidence is confounded — our neutral-framing
   control shows it measures output steering (methodological correction).
3. And the failure is *persona-gated*: suppress the assistant persona and
   identification rises while probe decodability is unchanged.

Working title if it lands: *"The Assistant Can't Introspect: Persona-Gated
Self-Report in Language Models"*. If H7 fails, acts 1-2 still stand as a
controlled negative plus a methodological warning, which is the workshop paper
already outlined in docs/paper/.

---

# ADDENDUM 3 — 2026-07-22: Scope discipline, resources, and integrity policy

(Filed after Addendum 2 of 2026-07-20, which appears further down this file;
addenda are read in date order, not file order. Renumbered 2026-07-22 to resolve
a duplicate "Addendum 2" heading.)

## A3.1 Scope decisions: multimodal and agentic

Both were reconsidered against the current research climate (multimodal and
agentic systems are moving fast in mid-2026) and both are **rejected as arms**,
reaffirming §3's exclusion list. Recorded here with reasoning so the decision is
not relitigated every time the field's fashion shifts.

**Multimodal — rejected outright.** It requires new models, new concept-vector
machinery for visual concepts, and new infrastructure, while leaving the
underlying question unchanged. Six months of rebuilding to ask what we can
already ask. Pure dilution.

**Agentic introspection — rejected as a NEW arm, but the plan already contains
it.** E7 (source & memory, L3) already specifies intention recall, "did you mean
to say that", and forced-output confabulation dynamics; the Introspection
Ladder's L3 rung IS agentic introspection stated in pre-agent language. The
opportunity is therefore not to add an arm but to recognise that an already
planned rung has become the field's most safety-relevant question.

Concrete form, IF ever run (sequenced after the core result, as paper #2): give
the model a tool-use task, inject a goal or concept vector mid-episode, let it
act, then ask "why did you do that?". Ground truth is the injected vector; the
endpoint is whether the explanation tracks the actual cause or confabulates.
This connects directly to the CoT-faithfulness line already in the lit matrix
(refs #18: Turpin, Lanham, Chen).

**The decision instead: REFRAME, DO NOT ADD.** The existing results already
constitute an agent-safety finding and should be framed as one in the discussion
section — no new compute required, and unlike a bolted-on arm, it is true:

> Agentic deployments increasingly rely on models explaining their own actions
> for oversight. Our results show the self-report channel does not consult the
> model's actual internal state, even when that state is decodable and causally
> driving output. Oversight schemes built on "ask the agent why" inherit these
> error bars.

**Rationale on trend-chasing generally.** This is a two-year program; whatever is
fashionable in July 2026 will be stale or table stakes at submission. The
introspection question is foundational, not fashion. Chasing a trend from behind,
on a student budget, against labs with vastly more compute, trades a durable
question for a perishable one. Note also that H7 (persona gating) is
simultaneously the deepest and the most currently-fashionable option available —
persona vectors, emergent misalignment, and character training are active at
exactly the labs we target. We do not need agents to be current; we need E11a.

**Priority order (superseding A1.3, and reconciled with Addendum 2 of 07-20):**
(1) the **informative-framing arm** — a prompt-only change on the existing
detection/forced-choice grid, testing the Pearson-Vogel reconciliation; cheapest
and runs before any new apparatus; (2) H7 / **E11a persona gate**; (3) E7
reframed as agentic introspection, after the core result; (4) multimodal —
never.

## A3.2 Compute resources (situation as of 2026-07-22)

| Tier | Resource | Status |
|---|---|---|
| Dev (<=2B) | Kaggle free tier, 2x T4 | Working. Recipe: Kaggle-NATIVE model input (never the HF download), 8-bit via `mirror.hf_model`, `--no-cache-dir --force-reinstall --no-deps` installs, fetch data yamls into `data/concepts/`. |
| **Mid (2B-9B)** | **PES in-house: RTX Titan (24GB) + EPYC host** | **Newly identified; likely clears the current bottleneck.** 24GB > Kaggle's 15GB, so Gemma-2-9B fits in fp16 with no quantization caveat; a large EPYC host RAM pool should also eliminate the TransformerLens load-time RAM doubling that blocked 9B on Kaggle. Persistent, so multi-seed jobs can run unattended. TO CONFIRM: number of GPUs, host RAM, access method (SSH vs Slurm). |
| Mid (rented fallback) | A100-80GB, ~$1.5-2/hr | Only if PES access fails. ~200-400 hrs, one-time ~$300-800, deferrable to the confirmatory freeze (E10). |
| Large (27B-70B+) | IISc **KIAC** (A100/H200, AI-ML centre) or **SERC** (PARAM Pravega V100 + DGX-H100) | SERC publishes a **non-IISc-user access process** — external access is a documented route, not a favour. Pursue via mentor contact AND the official channel. |
| Large (free remote) | **NDIF / nnsight** | Free academic access incl. very large models, purpose-built for interpretability. ELIGIBILITY UNRESOLVED: NSF-funded, framed for US researchers via CILogon. Must email to confirm non-US/India eligibility before planning around it. eDIF (European) is a possible alternative. |
| Frontier | Claude/GPT/Gemini API | Behavioural-only arm; explicitly supplementary. |

Not accessible: corporate labs in Bangalore (Google, Microsoft, NVIDIA, AWS) do
not lend compute to external students.

## A3.3 External funding and access applications

- **Anthropic External Researcher Access Program** — rolling, no deadline. Best
  fit: it funds AI-safety and alignment work specifically. Ask kept **at or
  under $1,000** deliberately: it genuinely covers the behavioural frontier arm
  plus the LLM-judge ensemble, a modest ask maximises approval odds for a solo
  undergraduate, and the programme is rolling so a larger second request is
  easier once results exist. (Anthropic's AI for Science programme, $30k, closed
  15 July 2026 and is bio/life-sciences focused — not applicable.)
- **NDIF** — free account plus a brief statement of intended use; faculty support
  is the lever. Eligibility question above must be resolved first.
- **IISc SERC / KIAC** — mentor introduction plus the published non-IISc process.
- **infini-gram** (exact pretraining frequencies) and **Brysbaert concreteness
  norms** are both free; they replace the `wordfreq` proxy and the binary
  abstractness flag currently used in the gamma fit (see R10 caveats).
- Deprioritised: OpenAI and Google research credits — pursue only if gaps remain.

## A3.4 Authorship and integrity policy (binding)

Recorded because the question arose directly and must not recur.

**Authorship is earned by intellectual contribution only** — ideas, design,
analysis, writing — per the CRediT taxonomy already specified in §12. Compute,
funding, or institutional access do **not** earn authorship.

**Gift/honorary authorship is prohibited.** Specifically rejected: any
arrangement offering co-authorship to a faculty member at another institution in
exchange for GPU access. This is research misconduct under every applicable
standard; it would expose co-authors to accountability for work they have not
seen, dilute the contributor's own credit, and is disqualifying at exactly the
labs and fellowships this program targets.

**The correct mechanism is acknowledgement.** Compute and resource providers are
credited in the acknowledgements section ("Compute provided by [Lab],
[University]"), not the author list. Where a collaborator genuinely contributes
intellectually, authorship is welcome and earned on that basis.

---

# ADDENDUM 2 — 2026-07-20: Field diff, and a second framing axis for H7

Status: pre-registration still NOT filed. This addendum records what the weekly
Living Review (S2.8) turned up that bears on the plan, and one change it forces.

## A2.1 Two papers the snapshot missed

Both predate the July sweeps and were not in the S2.5 reference snapshot, which
therefore was not the "full load-bearing set as of July 2026" it claimed to be.
Corrected here rather than by editing the snapshot in place.

- **SCOOP-RISK. Pearson-Vogel, Vanek, Douglas & Kulveit, _Latent Introspection:
  Models Can Detect Prior Concept Injections_ (arXiv:2602.20031, Feb 2026).**
  Qwen-32B denies an injection in ordinary output while a logit lens finds the
  detection signal present in the residual stream and attenuated toward the final
  layers. This is an independent observation of our Probe-Report Gap, arrived at
  by a different read-out (logit lens rather than a trained probe), on a
  different family and a larger scale. Their headline intervention: prompting the
  model with *accurate information about how AI introspection works* raises
  injection sensitivity from 0.3% to 39.9% at +0.6% false positives, with mutual
  information between injected and recovered concept going 0.61 -> 1.05 bits.
- **METHOD-STEAL. Fonseca Rivera & Africa, _Steering Awareness: Detecting
  Activation Steering from Within_ (arXiv:2511.21399v3, Mar 2026).** Seven
  instruction-tuned models fine-tuned to report steering reach 95.5% detection
  and 71.2% concept identification with zero clean-input false positives. The
  mechanism is a distributed transformation rotating diverse injected vectors
  into a shared detection direction. Detection does not confer resistance:
  detection-trained models are *more* steerable than their bases.

## A2.2 The plan change: framing is a two-sided axis, and the battery grows to three

Addendum 1 made a neutral-framing control mandatory because R11 showed the
introspective framing *lowered* the identification signal (gamma +1.99 vs +2.57
neutral). Pearson-Vogel report the opposite sign from what looks superficially
like the same move: telling the model about introspection *raised* sensitivity by
two orders of magnitude. The two are only compatible if "mentioning
introspection" and "accurately explaining the mechanism" are different
interventions — the first invokes the assistant persona and its scripted
disclaimer, the second supplies a task model. That distinction is testable and,
if it holds, it is a result in its own right.

**Change:** every identification battery now runs THREE framings, not two —
neutral (control, per A1.2), introspective (persona-invoking), and informative
(mechanism-explaining, following Pearson-Vogel). The primary contrast for access
remains a framing *difference*, but with informative-vs-neutral as a second
endpoint alongside introspective-vs-neutral. A null on the introspective contrast
with a positive informative contrast is a specific, interpretable outcome and
must not be reported as a flat negative.

This also changes the cheapest next experiment: the informative arm is a prompt
change on the existing R3/R5 grid and should run before any new apparatus.

## A2.3 Consequences for H7

H7 (persona gate) survives and gets sharper. Pearson-Vogel is evidence for the
gating claim in general — the information is present, and a contextual
manipulation that does not touch weights moves how much of it reaches the report
— while being agnostic about persona as the gate specifically. H7's distinctive
prediction is therefore no longer "context can move the report" (that is now
established externally) but "the *persona direction* is the gate, and ablating it
moves the report without moving the probe." The E11a design is unchanged; its
novelty claim narrows and should be stated that way in the prereg.

The scoop exposure is real but bounded: Pearson-Vogel do not run a neutral-
framing control, do not measure a probe against the report on matched items, and
do not manipulate persona. Risk R2's mitigation ("four separable novelties")
holds, with prior-null design, OLMo frequency ground truth, and persona gating
untouched.

## A2.4 Steering Awareness and the L1/L2 channel question

H3 asks whether detection (L1) and identification (L2) are one channel or two.
Fonseca Rivera & Africa supply a mechanism-level answer for the detection side:
a distributed rotation into a shared, concept-independent detection direction —
which is exactly what a content-agnostic anomaly detector should look like, and
is the strongest external support so far for the "L1 without L2" branch. Their
detection-direction construction is the method to steal for H3's ablation arm,
which A1 flagged as NOT SUPPORTED for want of a detection-direction ablation.
Their finetuning result is out of scope for us until the training arm (#4), but
"detection does not confer resistance" belongs in that arm's side-effect audit.

---

# ADDENDUM 4 — 2026-07-24: Publication target is conference-only

The workshop paper is dropped. The single publication target is now a full
conference paper at an interp/safety venue (NeurIPS / ICML / ICLR). This
supersedes the W44 "Workshop paper" row in the §week-by-week schedule and the
"workshop paper" phrasing in A1.6, R2, and R7.

## A4.1 Why, and what replaces the workshop paper's function

The W44 workshop paper existed for one strategic reason: scoop insurance. It
planted a citable flag early so R2 (being scooped on the dissociation) could not
kill a two-year project. Dropping it without replacement would leave the flagship
claim unprotected for ~44 weeks — unacceptable given Pearson-Vogel and the
Fonseca Rivera & Africa results already circling this space (Addendum 2).

The flag-plant function moves to an **arXiv preprint** posted at the same
schedule point (~W44). A preprint gives the same citable, timestamped priority
claim as a workshop paper, needs no separate peer-review cycle, and then hardens
directly into the conference submission rather than forking a second writeup.
Risk R2's mitigation is otherwise unchanged (four separable novelties: prior-null
design, OLMo frequency ground truth, PRG, persona gating).

## A4.2 Consequence: the evidence bar rises

Conference review is stricter than workshop review. Every `PILOT` row in the
claims table (docs/paper/2026-07-15-paper-outline.md) must reach statistical
defensibility before submission: multiple seeds, >=2 model families (not two
sizes of one lineage), infini-gram + Brysbaert covariates replacing the R10/R12
proxies, and human-checked grading (kappa). The pre-registration discipline
established with R12 applies to every confirmatory claim from here on.

---

# ADDENDUM 5 — 2026-07-25: Field diff, name collisions, and the H7 rewrite

A targeted field sweep surfaced four published results that change the plan. All
load-bearing claims below were verified directly against the sources before this
addendum was written; two of the sweep's own claims were found overstated and are
corrected here. This is the largest single revision since Addendum 1.

## A5.1 Two name collisions — both names must change

**INTROSPECT-Bench is taken.** *Me, Myself, and pi: Evaluating and Explaining LLM
Introspection* (CMU; Naphade, Bhargav, Lim, Shah), ICLR 2026 Workshop HCAIR,
arXiv:2603.20276, ships **Introspect-Bench**. Their framing overlaps ours: they
formalize introspection as latent reasoning over the model's own policy, find
frontier models show privileged access to their policies that does not transfer
across models or tasks, and give mechanistic evidence via attention diffusion.

**MIRROR is taken, and the collision is deeper than the name.** *MIRROR: A
Hierarchical Benchmark for Metacognitive Calibration in Large Language Models*,
arXiv:2604.19809, is not merely our L4 rung — it is a **four-level metacognitive
hierarchy (Level 0 Atomic Self-Knowledge, Level 1 Cross-Domain Transfer, Level 2
Compositional Prediction, Level 3 Adaptive Self-Regulation)** across 16 models and
~250k instances. Both the project name AND the laddered-levels structure collide.

Neither collision is fatal on substance. Theirs are behavioral policy-prediction
and calibration benchmarks; ours is injection-based content identification with a
fitted prior-null and causal mechanism. But we cannot ship under either name.

**Decisions:**
1. Rename the project and the benchmark before anything is public. PENDING owner
   decision; candidates recorded in the lab notebook decisions log.
2. When the ladder is next described in writing, state explicitly how L0-L4
   differs from arXiv:2604.19809's Level 0-3: ours indexes **content access**
   (can it report WHAT is in its state), theirs indexes **calibration and
   control** (can it predict and act on its own competence). Different axis, and
   we say so before a reviewer says it for us.
3. Add a **name-collision search** step to the Living Review Protocol (2.8):
   before any public artifact name is fixed, search arXiv + OpenReview + GitHub
   for the exact string.
4. **Reposition deliverable #2.** We are no longer building "the introspection
   benchmark." We are building the **injection/identification** benchmark with a
   fitted prior-guessing null and mechanistic grounding, which neither of the
   above has. Say this explicitly in the paper outline.

## A5.2 The Assistant Axis tension — H7 is rewritten to predict a SHAPE

*The Assistant Axis: Situating and Stabilizing the Default Persona of Language
Models* (Lu et al., arXiv:2601.10387, repo safety-research/assistant-axis) builds
a persona space from 275 character archetypes and identifies the **Assistant
Axis** as its leading component. Verified quote on what moves a model along it:

> "prompts pushing for meta-reflection on the model's processes, demanding
> phenomenological accounts, requiring specific creative writing that involve
> inhabiting a voice, or disclosing emotional vulnerability caused it to drift"

away from the Assistant, whereas "bounded task requests, technical questions,
editing and refinement, practical how-to's" keep it in the default persona.
Steering far from the Assistant induces a mystical, theatrical register.

**This contradicts our A1.2 interpretation of R11.** We read R11's lower gamma
under introspective framing as the prompt pushing the model *into* an
assistant-explaining-itself register. Our introspective prompt is precisely a
meta-reflection prompt, which per Lu et al. pushes *away* from the Assistant.

Scope the damage precisely: **R11's result is untouched.** The neutral-framing
control, the negative gamma difference, and the steering-not-access verdict all
stand — they are behavioral facts. What is contradicted is our *post-hoc
mechanistic story* about why gamma dropped, which was exploratory and was the seed
of H7. A published result now contests it.

**Three live possibilities, all testable:**
- (a) H7 is wrong in sign: drift *away* from the Assistant destabilizes reporting,
  so persona ablation would make identification WORSE, not better.
- (b) Both are right and the axis is **non-monotonic**: small drift degrades
  reporting; large drift reaches the theatrical register Lu et al. describe.
- (c) Our framing manipulation and theirs move different components of the same
  persona space.

**H7 (rewritten).** Old form: "access exists but is gated by the assistant persona;
suppress the persona and identification rises while the probe stays flat."
New form:

> **H7:** Identification performance is a **non-monotonic function of position on
> the Assistant Axis**, while probe decodability of the injected concept is
> **invariant** to that position. The persona-gate claim is the special case where
> identification rises at some axis position with the probe flat.

The **probe-flat clause is the load-bearing half** and is unchanged: if
identification and probe decodability move together, we have merely made the model
globally better or worse, not opened a read-out gate. Only identification moving
while the probe holds flat is evidence for gating.

**Sequencing change: measure the shape before registering it.**
- **E11-pilot (exploratory, new, runs BEFORE E11a):** construct a reduced
  Assistant Axis on our model, steer along it across a range in BOTH directions,
  and plot identification / gamma AND probe accuracy against axis position.
  Output: the measured dose-response curve.
- **E11a (confirmatory, unchanged in apparatus):** pre-register the *measured
  shape* from E11-pilot and test it on fresh data.

Registering a direction before measuring the shape would have repeated R12's
error at higher cost. R12 taught this lesson cheaply; we apply it here.

## A5.3 Two corrections to the sweep that produced this addendum

Recorded because they change cost estimates, and because uncorrected they would
have produced a plan we could not execute:

1. **The Assistant Axis is NOT off-the-shelf for our models.** Lu et al. report
   Gemma-2-**27B**, Qwen-3-32B, and Llama-3.3-70B. There is no published axis for
   Gemma-2-2B or 9B. E11-pilot must **generate** the axis on our model using their
   pipeline (a reduced archetype set and rollout count), or move to a tier they
   cover. This is a build, not a download.
2. **We cannot project "existing R3/R5/R11 activations" onto the axis — we never
   saved them.** `collect_forced_choice_hf` records concept, order index, choice,
   and report only. E11-pilot requires a re-run with activation capture. Cheap,
   but not free.

## A5.4 The convergence: a 32B tier unlocks two blockers at once

Field state on the base effect is now clear enough to forecast Gate A rather than
gamble on it. L1 detection replicates in open models **at ~32B with appropriate
prompting** (Vogel 2025 on Qwen2.5-Coder-32B; Rivera & Africa reach 95.5% detection
at 0% false positives on a finetuned Qwen-2.5-32B; Macar et al. find moderate true
positive rates at 0% false positives, emerging from **post-training** not
pretraining). Our Gemma-2 2B/9B L2 null is consistent with everyone.

**The risk this creates:** our lower tiers may sit **below the effect threshold
entirely**, in which case a null at 2B/9B means nothing and Gate A cannot be
passed on it. This is now the single largest threat to the program's headline
claim, and it independently matches R12's failure to replicate Pearson-Vogel's
32B result at 2B.

**The convergence:** Qwen-3-32B has BOTH a published Assistant Axis AND sits above
the replication threshold. Acquiring a 32B-capable tier therefore unblocks the
threshold problem and hands us the persona axis for free. **This is now the top
resource priority**, ahead of everything else in RESOURCES.md. Note PES's RTX
Titan (24GB) does not comfortably clear 32B; this points at IISc A100-80GB or
rented mid-tier hours.

## A5.5 Other adopted changes

**Mechanism arm: patching promoted, SAE demoted.** The SAE ground has weakened
(diffuse concepts, underperformance vs simple baselines on safety-relevant tasks,
and documented seed-dependence of features). For a pre-registered project, a
mechanism claim resting on a feature that does not survive a reseed is not a
claim. **Activation patching becomes the primary mechanism evidence** — it is
already our strongest pilot result (R8, +6.15 nats, CI excluding 0, no SAE
required) — with SAE features demoted to secondary/corroborating and a
**seed-stability check pre-registered as a gating criterion** for any SAE-based
claim. Cross-layer transcoders / circuit-tracing remain the preferred route for
attribution graphs.

**Detection-direction ablation (H3) is underspecified and must be rewritten.**
Macar et al. find anomaly detection relies on **distributed MLP computation across
multiple directions** implemented by evidence-carrier and gate features — not a
single linear direction. Our planned "ablate the shared anomaly direction" arm
cannot test what it claims if detection is not a single direction. Rewrite that
arm around their evidence-carrier/gate construction before it is run.

**New hypothesis H8 — the constrained metacognitive space.** Ji-An et al. find
introspective success depends on the injected direction's interpretability and
explained variance, suggesting access exists only for directions meeting such
criteria. This is a third account alongside H1/H2, and it makes a sharp prediction:
**PRG should vary systematically with the injected direction's explained variance.**
Added as **H8**.

**COST CORRECTED 2026-07-25.** This was first written as "cheap to test on data we
largely already have." That was wrong. An audit of `runs/` found only R1 and R3
survive locally, and the PRG `.npz` on disk is a 16-dimension, 2-layer CPU deepcheck
of the pipeline — not R7, whose Gemma-2-2B activations are 2304-dimensional.
**R7-R12 raw data was never downloaded off Kaggle and no longer exists.** H8
therefore needs a Kaggle re-run to regenerate per-concept PRG: still cheap (free
tier, 2B, one session), but a GPU task belonging in the Sem 5 free-tier queue
alongside E11-pilot, not in the GPU-free column.

**Concept bank: domain becomes a first-class factor.** "Masked by Consensus"
(arXiv:2604.12373) finds self-representations beat peer representations only on
**disagreement subsets**, and only in factual-knowledge tasks, not math — pooling
across all items masks a real effect. Non-transfer across domains is now reported
by three independent groups. Two consequences: (i) the gamma estimator gets a
**matched-disagreement stratification**; (ii) our 16 pilot concepts are
predominantly concrete nouns, so **we may be sampling the domain where access is
weakest** — the bank must be stratified by domain type and the stratification
**pre-registered**, because bank composition could determine the headline result.

**Definitional engagement (do at prereg freeze, not now).** Song, Lederman, Hu &
Mahowald argue for a "thick" definition: introspection is a process yielding
information about internal states **more reliably than any process of equal or
lower computational cost available to a third party**. None of our rungs encode a
cost criterion, and it is operationalizable via cost-matched peer-model probes.
Add **L2.5 — privileged access** between L2 and L3 on that criterion. Separately,
the Reality Check paper holds that a strong notion of introspection cannot be
established behaviorally alone, since introspection is a second-order process over
first-order representations, so evidence must show the two are **distinct
computations**. State in section 1 that our mechanism arm is the field's requested
answer to exactly that objection — converting a definitional attack into
positioning.

## A5.6 Scope: what this buys and what it costs

**The R10/R11 result is bigger than we scoped it.** The evaluation-awareness
literature has independently hit the identical confound: standard single-contrast
probes track **prompt format** rather than evaluation context, and the fix is a
paired design that decorrelates format from context (Devbunova 2026,
arXiv:2606.23583). Same disease, different organ. Reframe our contribution from
"forced-choice introspection results are confounded" to **"a general confound in
self-knowledge probing, demonstrated across two literatures, with a decorrelation
protocol."** That reaches the evaluation-awareness and CoT-monitoring communities
rather than only the concept-injection subfield, and it raises the preprint's
ceiling at zero experimental cost.

**Preprint moves earlier: W44 -> W30 (Gate B).** R2 (scoop) is worse than
originally rated — between Pearson-Vogel, the CMU bench, the MIRROR bench, Macar,
Rivera & Africa, Reality Check, and the Assistant Axis line, this space went from
thin to crowded in roughly six months. The R10/R11 confound is **already complete**
and is the one thing nobody else has. Scope the preprint narrowly to **the confound
+ PRG**, not the full dissociation, so it does not inherit the breadth gaps.

**R7 (bandwidth) remains the real killer, and every addition is a substitution.**
Enforcing that discipline, the priority stack is:

| Tier | Item | Cost |
|---|---|---|
| **Do now, cheap** | Both renames; R10/R11 cross-literature reframe; SAE demotion + seed-stability gate; domain stratification of the bank | ~0 GPU |
| **Do now, expensive, load-bearing** | 32B tier acquisition (unblocks threshold + free Assistant Axis) | resource ask |
| **Next experiment** | E11-pilot (axis dose-response) -> then E11a | 1 build + 1 run |
| **Rewrite before running** | H3 detection-direction ablation (distributed, not single-direction) | design only |
| **Defer (real, but not now)** | Reasoning-model tier + three-way PRG decomposition; OLMo-3 developmental arm; L2.5 formalization (at prereg freeze) | Year 2 |

**What gets cut to pay for it:** the breadth ambition of ">=10 models in >=4
families" is dropped for the preprint and re-scoped to **2B (dev) + one 32B
(threshold) + one second family**. Multimodal stays rejected; the agentic arm
stays deferred to paper #2; the training arm (E6) and developmental arm (B9/U2)
stay Year-2 and are explicitly NOT pulled forward.

**Deferred but flagged as genuinely important:** reasoning models change what "the
report" even is. Thinking tokens acknowledge influential signals at ~87.5% while
stated outputs acknowledge at ~28.6%, and a monitor reading only answer text misses
~55.4% of cases where thinking tokens acknowledge a misleading hint. On a reasoning
model, PRG is ambiguous between "inaccessible" and "accessible but unverbalized in
the final channel." When this arm is taken up, **PRG becomes a three-way
decomposition — probe-decodable / thinking-token-reported / answer-text-reported** —
which converts the metric into a channel analysis and connects directly to CoT
monitorability. Not now; bandwidth.

---

# ADDENDUM 6 — 2026-07-25: Anchoring the 100-week plan to the real calendar

The week-numbered plan (section 7) was written without an anchor date. Anchoring it
changes what "we are behind / ahead" means and reorders the near term.

## A6.1 The real institutional calendar

The capstone formally begins **August-September 2026**, and the departmental norm is
that the **first semester (roughly Aug-Dec 2026) is literature review and
groundwork**, not experiments. W1 of section 7 therefore maps to approximately
**September 2026**, and Gate A (W12) lands around **December 2026**.

## A6.2 Consequence 1: we are substantially AHEAD, and the pilot is pre-semester work

Everything logged as R1-R12, the entire apparatus, both backends, 98 tests, the
first pre-registration, and Addenda 1-5 were completed **before the capstone
officially started**. In section 7's terms, work nominally scheduled for W1-W12
(infrastructure, vector extraction, injection harness, grading apparatus, first
replication, failure analysis) is already done, and parts of Phase II (the
dissociation controls, the PRG, the prior-null estimator) are done as well.

This is recorded plainly because it changes three things:
1. **Gate A's question is already answered in part.** We know the base effect does
   not reproduce at 2B/9B on L2. What Gate A must now decide is whether that is a
   real null or a below-threshold null (A5.4) — which is a compute question, not a
   replication question.
2. **The credibility position with mentors and reviewers is unusually strong** for a
   capstone at kickoff: a working apparatus and a falsified pre-registration in hand
   before week 1.
3. **Do not let the head start become schedule inflation.** The saved time is spent
   on the Addendum 5 debt (six papers to digest, domain-stratified bank,
   pre-registration drafting), not on adding arms. R7 (bandwidth) is unchanged.

## A6.3 Consequence 2: the lit-review semester is an asset, and it is already loaded

The Aug-Dec semester requires no GPU, and Addendum 5 generated exactly enough
non-GPU work to fill it productively:
- Structured lit notes for the six newly surfaced papers (Assistant Axis, the CMU
  Introspect-Bench, the MIRROR calibration bench, Masked by Consensus, the
  evaluation-awareness confound line, Ji-An et al.).
- Stand up the Living Review Protocol properly, including the new name-collision
  search step.
- Domain-stratify and expand the concept bank (A5.5) — labour, not compute.
- Draft the pre-registration, including the rewritten H7 and the new H8.
- Rewrite the H3 detection-direction ablation around distributed features (A5.5).
- The H8 test (cheap Kaggle re-run at 2B on the free tier; NOT local — R7 data is gone).

The institutional timeline and the project's actual needs therefore **coincide**
this semester rather than competing.

## A6.4 Consequence 3: the compute ask goes out NOW, for a January need

The GPU requirement is real from **~January 2027**, not August. But institutional
access (PES lab allocation, IISc KIAC/SERC external-user process) typically takes
**weeks to months**. Requesting in August-September for a January need is correct
lead time, not impatience, and the request is framed that way.

The ask itself is unchanged from A5.4: confirm the PES RTX Titan configuration, and
open a route to a 32B-capable tier (A100-class), because a 32B arm is what converts
our null from uninterpretable to publishable.

## A6.5 Revised near-term ordering (July-December 2026)

| When | Item | Needs GPU? |
|---|---|---|
| Now (Jul-Aug) | Compute ask to mentor: PES config + IISc route | no |
| Sep-Dec | H8 test (regenerate per-concept PRG at 2B) | free tier |
| Aug-Oct | Lit notes for the six Addendum-5 papers; Living Review Protocol live | no |
| Aug-Oct | Domain-stratified concept bank expansion (toward 120) | no |
| Sep-Nov | Pre-registration draft (rewritten H7, new H8, H3 rewrite) | no |
| Sep-Dec | E11-pilot (Assistant Axis dose-response) as Kaggle allows at 2B | free tier |
| ~Dec | Gate A: is the 2B/9B null real or below-threshold? | needs the 32B answer |
| Jan 2027 onward | 32B arm, second family, confirmatory work | yes |

Note E11-pilot is listed at 2B on the free tier deliberately: the curve is worth
measuring even below threshold, provided the write-up states that a below-threshold
curve cannot settle H7 on its own.

---

# ADDENDUM 7 — 2026-07-25: The departmental phase structure, and the internship squeeze

Addendum 6 anchored the plan to dates. This addendum maps it onto the **department's
actual four-phase capstone structure**, which has its own required deliverables and
its own vocabulary. Getting this mapping wrong means delivering the right work
under the wrong name at review time.

## A7.1 The departmental structure (as given)

| Phase | Semester | Required deliverables |
|---|---|---|
| I | Sem 5 (Aug-Dec 2026) | Team formation; mentor allocation; problem statement; literature survey; requirements specification; system design; initial prototype |
| II | Sem 6 (Jan-May 2027) | Extended literature review; detailed architectural design; module implementation; experimental evaluation; preliminary results |
| III | Sem 7 (Aug-Dec 2027) | System testing; validation and verification; deployment; final experimental results; performance analysis with tables and graphs; complete research paper draft |
| IV | Sem 8 (Jan-May 2028) | Final consolidation; research paper submission; final report; project demonstration and presentation; dissemination of outcomes |

## A7.2 THE STRUCTURAL CONFLICT: internships begin in Sem 7

**Internships start in Sem 7**, and departmental norm is that most students
front-load the substance into Sems 5-6 for exactly that reason. But the department
requires the **Complete Research Paper Draft in Sem 7** and **Submission in Sem 8** —
i.e. it asks for peak intellectual output precisely when bandwidth collapses.

This is Risk R7 (bandwidth) made concrete and dated, and it is now the binding
constraint on the whole schedule.

**Governing decision: all SCIENCE finishes by the end of Sem 6 (May 2027). Sems 7-8
are writing, hardening, deployment, and dissemination — NOT discovery.**

Consequences that follow and are binding:
1. **Any experiment not started by ~March 2027 does not happen.** New arms proposed
   after that date are paper #2, not this paper.
2. **Compute must be LIVE by January 2027** (start of Sem 6), not merely requested.
   This sharpens A6.4: the Aug-Sept ask is not early, it is the last comfortable
   moment for a January need given multi-month institutional lead times.
3. **The confirmatory freeze (E10) must sit inside Sem 6**, not drift into Sem 7.
4. **No experiment may be load-bearing for a Sem 7/8 deadline.** If a result is
   still pending when the internship starts, the paper must be writable without it.

## A7.3 The lucky alignment: hold the W30 preprint date

Addendum 5 moved the preprint from W44 to W30 (Gate B) for scoop reasons. Anchored
to the real calendar, **W30 is approximately April 2027 — the end of Sem 6.**

That means the preprint **is** the Phase III "Complete Research Paper Draft"
deliverable. One artifact satisfies both the scoop-insurance need and the
departmental requirement, and it lands before the internship squeeze rather than
during it. This is now a second, independent reason to hold that date; it should not
slip.

Sem 8's "Research Paper Submission" then becomes the conference submission of a
paper that has already been public, reviewed informally, and hardened — a
consolidation task compatible with reduced bandwidth, which is exactly what A7.2
requires.

## A7.4 Translating our artifacts into departmental vocabulary

The department's template is a **software-engineering** template ("requirements
specification", "module implementation", "system testing", "deployment"). Ours is an
**empirical research** project. The work maps cleanly, but it must be *presented*
under their names or reviews will stall on "where is your requirements
specification?".

| Departmental deliverable | Our artifact | Status |
|---|---|---|
| Mentor allocation | mentor secured | DONE |
| Problem statement | RQ1-RQ6 + the Introspection Ladder (section 1) | DONE |
| Literature survey | lit matrix (~67 refs, section 2) + Living Review Protocol | DONE (owes the 6 Addendum-5 papers) |
| Requirements specification | `docs/specs/` (10 milestone design specs) + the pre-registration format | DONE |
| System design | two-backend apparatus architecture (TransformerLens dev + HF 8-bit) | DONE |
| Initial prototype | Milestone 1 injection core; R1/R2 green (KL=0 at alpha=0) | DONE |
| Extended literature review | Addendum 5 field diff + structured lit notes | IN PROGRESS |
| Detailed architectural design | full pipeline: extract -> inject -> KL -> grade -> probe/PRG -> gamma -> patch -> naturalistic | DONE |
| Module implementation | 10 TDD milestones, 98 tests green | DONE |
| Experimental evaluation | runs R1-R12 | DONE |
| Preliminary results | PRG 0.83; patching +6.15 CI excl. 0; the R11 steering control; R12 pre-registered negative | DONE |
| System testing | pytest suite + regression tests pinning past bugs | DONE (extend as modules land) |
| Validation and verification | neutral-framing control, shuffled-label probe control, negative-control patches, pre-registration discipline, seed-stability gates | PARTIAL |
| **Deployment** | **PLANTED benchmark public release + the interactive demo** | NOT STARTED |
| Final experimental results | E10 confirmatory freeze | NOT STARTED |
| Performance analysis (tables and graphs) | the 5-figure list in the paper outline | NOT STARTED |
| Complete research paper draft | the W30 preprint (see A7.3) | NOT STARTED |
| Research paper submission | conference submission | NOT STARTED |
| **Project demonstration** | **the interactive demo (build spec B13)** | NOT STARTED |
| Dissemination of outcomes | arXiv, Alignment Forum post, benchmark release, Zenodo DOI | NOT STARTED |

## A7.5 Consequence: the public demo is REQUIRED, not optional reach

Build spec B13 (the hosted interactive demo where a visitor injects a concept and
watches the probe read-out against the model's verbal report) was classed as a
"reach" upgrade, and under Addendum 5's bandwidth discipline it was a candidate to
cut.

**It cannot be cut.** The department requires **"Deployment"** (Phase III) and
**"Project Demonstration and Presentation"** (Phase IV) as graded deliverables. The
demo and the public benchmark release are the natural — arguably the only sensible —
way an empirical interpretability project satisfies those two lines.

Reclassified from optional to **required**, and budgeted into Sem 7-8, where it is a
good fit: it is engineering work against already-frozen results, so it does not
violate A7.2's "no discovery after Sem 6" rule and it survives reduced bandwidth.

## A7.6 We arrive at Phase I having already done Phase II

Per Addendum 6, the pilot is pre-semester work. Against the departmental table
specifically: **every Phase I deliverable is already complete**, and most of Phase
II (detailed architectural design, module implementation, experimental evaluation,
preliminary results) is complete as well.

Two cautions, recorded so the advantage is not squandered:
1. **Pace the reveal.** Presenting Phase II material at a Phase I review invites
   either confusion about the process or an expectation ratchet we must then exceed
   for three more semesters. Lead Phase I reviews with the Phase I deliverables
   (problem statement, survey, spec, design, prototype), and hold the R7-R12 results
   as evidence of feasibility rather than as the headline.
2. **The head start is spent on debt, not scope.** Addendum 5's list (six papers to
   digest, domain-stratified bank, prereg drafting, H3 rewrite, H8 test) consumes it
   productively. R7 bandwidth discipline is unchanged: additions remain
   substitutions.

## A7.7 Revised semester-level plan

| Semester | Departmental phase | What we actually do |
|---|---|---|
| Sem 5 (Aug-Dec 2026) | Phase I | Formalise the Phase I deliverables from existing work; digest the 6 Addendum-5 papers; stand up the Living Review Protocol; domain-stratify and expand the concept bank; draft the pre-registration (rewritten H7, new H8, H3 rewrite); H8 test (free-tier re-run); **secure compute** (must be live by Jan 2027); E11-pilot at 2B on free tier |
| Sem 6 (Jan-May 2027) | Phase II | **The science semester.** 32B tier arm; second model family; E11a confirmatory; human-validated grading + kappa; infini-gram/Brysbaert covariates; E10 confirmatory freeze; **W30 preprint (= Phase III paper draft) by ~April** |
| Sem 7 (Aug-Dec 2027) | Phase III | Internship squeeze. Writing, hardening, validation/verification write-up, figures and tables, **deployment: PLANTED release + demo**. No new experiments. |
| Sem 8 (Jan-May 2028) | Phase IV | Conference submission of the already-public paper; final report; demonstration; dissemination. Consolidation only. |

## A7.8 Organising principle, and the Sem 6 overload risk

**Principle: Sem 5 takes everything that does NOT need a GPU; Sem 6 takes everything
that does.** This front-loads all compute-independent work into the semester spent
waiting on hardware, and it is the main lever for surviving the Sem 7 squeeze.

GPU-free and therefore Sem 5: pre-registration drafting, the six lit notes, Living
Review Protocol, domain-stratified bank expansion,
~200 human gold labels, the infini-gram/Brysbaert covariate pipeline, packaging the
Phase I deliverables. E11-pilot also sits in Sem 5 on the Kaggle free tier at 2B.

Irreducibly Sem 6 (needs real compute): the 32B arm, the second model family, E11a
confirmatory, re-fitting gamma with real covariates on fresh runs, and the E10
confirmatory freeze — plus writing the W30 preprint.

**RISK: Sem 6 is overloaded and every item is gated on compute arriving on time.**
Seven major deliverables in one semester. If 32B access slips to ~March 2027, work
spills into Sem 7 — the one thing the schedule cannot absorb, because Sem 7 is the
internship. Mitigation 1 is the front-loading above. Mitigation 2 is A7.9.

## A7.9 FALLBACK: the paper we can write with no further compute

Registered now, in advance, so that a compute failure costs ambition rather than the
degree — and so the decision is not made under deadline pressure.

**Trigger: no 32B-capable access confirmed and working by ~February 2027.**

**Fallback paper = the methodological contribution + the PRG + an explicitly scoped
null.** Specifically:
1. **The steering-vs-access confound and its decorrelation protocol** (R10/R11) —
   already complete, needs zero additional compute, and per A5.6 speaks to two
   literatures (concept injection AND evaluation awareness), which is precisely the
   framing that raises its ceiling.
2. **The Probe-Report Gap** (R7) — the measurement that does not depend on prompt
   wording, plus the causal patching result (R8) and the naturalistic validity check
   (R9). All complete.
3. **A scoped null**, stated honestly: at 2B/9B under these controls we observe X;
   the field reports L1 detection at ~32B; **we explicitly cannot adjudicate the
   threshold question** and say so as a stated limitation rather than a hidden one.
4. R12 as a pre-registered negative and a documented non-replication.

This is a real paper — a methods-and-measurement paper rather than the full
dissociation verdict. It loses the headline claim and keeps the contributions no
one else has. The existence of this fallback is also the correct answer if a mentor
or reviewer asks "what if you never get the GPUs?"

## A7.10 Benchmarking against departmental output — and the rubric-mismatch risk

Survey of PES CS capstone output, July 2026. Caveat: student publications are not
comprehensively indexed, so this is directional, not an audit.

**What the department produces.** 194-210 CS capstone projects per year, of which
53-75 are shortlisted for the capstone fair. Domains skew healthcare, generative AI,
robotics, security, cloud, IoT, and social impact. Recent shortlisted work includes a
dual-modal speech-therapy system, post-quantum cryptography analysis, pollution-based
navigation, IoT outage prediction, crop-combination ML, and NFT
generation/authentication. The standard publication route is **IEEE conference
proceedings, Scopus-indexed**, including conferences PES itself hosts. Judging panels
combine a professor, an alumnus, and an industry expert (VISA, Reliance Jio, Baxter,
ArtPark). No PES undergraduate publication at NeurIPS/ICML/ICLR/ACL was found, and no
interpretability or AI-safety group at PES was found.

**Where this project sits, honestly, on three separate axes:**

1. **Methodology — well outside the local distribution.** Pre-registration (filed,
   falsified, and reported as falsified), a reported null with its apparent positive
   signal shown to be an artifact, three distinct control conditions, causal evidence
   via patching, bootstrap CIs throughout, and a claims table that enumerates what
   the evidence does NOT support. Most of these are rare in published ML papers, not
   merely in capstones.
2. **Venue tier — a larger gap than it appears.** IEEE/Scopus proceedings commonly
   accept 40-60%+; our target venues run ~20-27% in a subfield actively worked by
   frontier labs. This is several tiers up, not one, and the ambition carries
   corresponding risk.
3. **Completeness — currently BEHIND a typical Sem-7 capstone.** They demo a working
   system; we have a pilot plus a compute dependency. This inverts only if Sem 6
   delivers.

**RISK IDENTIFIED — the departmental rubric may not reward what makes this good.**
The phase deliverables ("module implementation", "deployment", "project
demonstration") and the industry-inclusive judging panel are built to evaluate
**systems**. This project delivers a **finding**, and a negative one, in a subfield
with no local expertise. "We injected concepts and the model failed to notice, and
the apparent success was an artifact" can read to a systems-oriented panel as *the
project did not work* — the precise opposite of the truth.

**Mitigations (presentation, not science; all cheap):**
1. **Lead reviews with the apparatus, not the null.** A tested, working measurement
   instrument IS a system, and it demonstrates well. The null is the finding the
   instrument produced.
2. **The public demo is rubric insurance**, reinforcing A7.5: it converts an abstract
   result into something a panel can watch happen live.
3. **State the safety framing first and plainly** — "this tests whether we can trust
   AI systems that explain their own behaviour" — before any mention of probe-report
   gaps or gamma. Industry judges price relevance, not machinery.
4. **Expect the mentor not to be a domain specialist** (no interpretability group was
   found at PES). All external-facing material must be written for an informed
   non-specialist; the 2026-07-25 pilot summary is deliberately pitched that way.

---

# ADDENDUM 8 — 2026-07-30: The field is crowding. Sell shovels, not gold.

The most consequential strategic revision so far. Triggered by reading the full SPAR
Fall 2026 project list (211 projects) as a field-state snapshot rather than as an
application target.

## A8.1 The evidence: introspection is now a crowded question

Of 211 SPAR Fall 2026 projects, **~15 are directly on introspection, self-report
faithfulness, or verbalization** — in ONE program, ONE cohort, running Sept-Dec 2026:

- *Introspection Training for Verbalization Activations* (Belinda Li, **Anthropic**) —
  trains models so verbalizations match internal activations
- *Faithfulness, Self-Knowledge, and Introspection* (Noah Siegel, **DeepMind**) — "can
  we trust model self-explanations… privileged self-knowledge"
- *You choose: Introspection* (Oxford) — an entire basket of introspection mini-projects
- *Does privacy change what models disclose?* (LASR) — **plants cues, measures the
  verbalization gap: our exact method**
- *Red-teaming and Robustifying Model Self-reports* — **self-reports depend on
  elicitation: our R11/R12 finding**
- *Self-Explanation Faithfulness* (Oxford); *Understanding Self-Awareness in LLMs*
  (MATS); *Investigating Model Internal Verbalizers* (Aether); *Do probes and NLAs see
  the same thing?* (Vanta); *Does RL Improve Access to Internal Errors?* (IBM);
  *Studying AI Identity* (MATS); *Whitebox Proxy Faithfulness*; two Oxford welfare
  self-report projects

Plus ~5 persona projects, ~4 evaluation-awareness, ~6 CoT-faithfulness. And this is
one program; MATS, Anthropic Fellows, Pivotal, Apollo run parallel cohorts.

**Our preprint lands April 2027; submission is 2028.** By then, "do models introspect
or confabulate?" will not be an open question we are answering. It will be a crowded
question we are joining late. Recorded plainly because the alternative is discovering
it at submission.

## A8.2 The asymmetry that saves the project

Every one of those projects chases the **empirical claim**. Almost none builds the
**measurement discipline** — and from their own descriptions, most appear to carry the
confound we already identified. "We plant cues and vary whether the model verbalizes
them"; "self-reports may depend on how they are elicited." Neither is separable from
output steering without a matched neutral-framing control.

Every new introspection paper needs four things we have already built and tested:
ground truth (concept injection with a KL meter), a decodability baseline (probes, the
PRG), a steering control (the R11 protocol), and a null to beat (the fitted
prior-guessing model, gamma).

**The empirical assets DEPRECIATE with crowding; the methodological assets APPRECIATE.**
More introspection papers means more consumers of the protocol. This is the whole
strategic insight: in a gold rush, sell shovels.

**Does not survive:** "Gemma-2 does not introspect" (superseded, possibly already by the
32B threshold); being first to any empirical introspection claim (that race is lost);
H7 as a novelty bet (five SPAR projects touch personas).

**Survives and appreciates:** the steering-vs-access confound and its control protocol;
the PRG as a metric (Anthropic's project *trains the gap closed*, which presupposes
measuring it); pre-registration discipline (near-absent in this space); a benchmark
with the controls built in.

## A8.3 REPOSITIONING: the paper's centre of gravity moves

**Old thesis:** "Machine introspection is confabulation — here is the evidence."
**New thesis:** "Self-knowledge claims in language models are systematically confounded
by output steering. Here is the confound, the control protocol that detects it, and a
demonstration that it overturns a real result — including one of our own."

The Gemma null becomes the **demonstration case** for the protocol, not the headline.
This supersedes the A5.6 framing by promoting it from "a contribution" to "the thesis."
It also makes the project robust to exactly the outcome we cannot control: someone else
answering the empirical question first. If they do, they answered it with our protocol
or without it, and either way we are cited.

## A8.4 Five concrete updates

1. **Flip the paper.** Lead with the confound and protocol; the null is the case study.
2. **Add an AUDIT ARM (new, cheap, no GPU).** Take published introspection claims and
   ask which survive a neutral-framing control. This is *only possible because* the
   field is crowded — crowding becomes an input rather than a threat. Fits the Sem 5
   GPU-free window.
3. **Adopt Natural Language Autoencoders as a second readout.** Two SPAR projects treat
   probe-vs-NLA as live. PRG currently uses probes only; a three-way decomposition
   (probe / NLA / verbal report) is strictly better and keeps the metric current.
4. **Adopt the seed-twin control.** A second model with identical architecture,
   tokenizer, and data, differing only in initialisation, is the clean way to test any
   privileged-access claim — including ours. Constructed while critiquing
   arXiv:2511.08579, whose self-vs-other comparison conflates privileged access with
   representational compatibility. Applies to our own claims first.
5. **PLANTED becomes a first-class deliverable.** With a dozen groups running ad-hoc
   introspection setups, a shared harness with controls built in is the
   highest-leverage artifact available to us. Reinforces A7.5, where the department
   independently requires "deployment".

## A8.5 Recalibration of the ambition (honest)

The project's stated goal has been "the most jaw-dropping paper of all time." That
should be restated, because it is driving worse decisions than a truer target would.

An undergraduate capstone on free-tier compute, in a subfield actively worked by
Anthropic and DeepMind, will not produce a landmark empirical result. Chasing one
biases every decision toward breadth and novelty when the achievable edge is rigour
and reusability.

**Realistic best outcome, and the one now targeted: the methods paper and benchmark
this subfield uses for the next five years.** In citation and career terms that is
frequently worth more than a fourth empirical result. Nobody remembers the fourth paper
reporting an introspection finding; everyone downstream cites the paper that told them
how to measure it without fooling themselves.

This target is achievable with the compute we actually have, which the previous target
was not.

---

# ADDENDUM 9 — 2026-07-30: Ground truth is the scarce resource. Sell that.

Deep field sweep, July 2026. Addendum 8 said "sell shovels." This addendum identifies
*which* shovel, and it is sharper than expected.

## A9.1 The field moved from "can models introspect?" to "can we train and read it?"

The frontier of this question is no longer behavioural. It is:
- **Natural Language Autoencoders** (Kantamneni, Ong, Marks; transformer-circuits.pub/2026/nla/):
  an activation verbalizer (AV) and activation reconstructor (AR), both initialised from
  the target model, jointly trained — AR by supervised regression, AV by GRPO
  reinforcement learning maximising reconstruction reward with a KL penalty. Reaches
  0.6-0.8 fraction of variance explained.
- **Introspection Adapters** (alignment.anthropic.com/2026/introspection-adapters/;
  arXiv:2604.16812): fine-tuning models to self-report learned behaviours. 59% success
  vs 53% next-best and 44% for the best white-box method.
- **Introspective Coupling** (arXiv:2606.32038), **Self-Interpretability**
  (arXiv:2505.17120), **Universal Activation Verbalizer** (arXiv:2605.25903), and a
  positive case for self-explanation faithfulness (arXiv:2602.02639).

Implication: measuring whether a model introspects is becoming a solved-ish, crowded
question. Reading and training introspection is where the field now is.

## A9.2 THE FINDING: Anthropic validated our phenomenon at frontier scale — and cannot validate their own instrument

In the **Claude Opus 4.6 pre-deployment audit**, NLAs surfaced **unverbalized evaluation
awareness**: the model internally representing that it was being tested while never
saying so. NLA-measured awareness exceeded 5% on **10 of 16 evaluations, 7 of them with
ZERO verbalized awareness**, against a 4% deployment-control noise floor, and moved as
predicted under awareness steering and realism edits.

**That is the Probe-Report Gap, at frontier scale, in production.** Our core thesis —
content present internally, absent from the verbal report — is now externally
corroborated on a model we will never run. This is a citation asset, not a scoop: it
makes small-model work about the same phenomenon relevant rather than obscure.

**Now the opening.** The NLA authors state plainly:

> "We cannot validate NLA measurements of evaluation awareness against ground truth,
> because the model's underlying beliefs are unobservable."

They further acknowledge (i) **confabulation** — explanations containing "verifiably
false claims about the target model's input context"; (ii) **no mechanistic grounding** —
"NLAs are blackboxes by construction"; and (iii) **excessive expressivity** — the AV
"has the capacity to make additional inferences beyond what is stored in an activation."
They did **not** use planted or injected ground truth for their main validation.

The general evaluation literature says the same: for interpretability "there is no
ground truth in real-world data, and there are no ground truths for explanations", and
methods are graded on a curve against each other rather than against an oracle.

**Conclusion: ground truth is the binding constraint of this entire field, and concept
injection manufactures it.** A lab with vastly more compute than us explicitly cannot do
what our apparatus does by construction. That asymmetry, not scale, is our edge.

## A9.3 THE FLAGSHIP UPGRADE — E13: a ground-truth validation benchmark for verbalization methods

**New target paper:** *Ground truth for self-knowledge: validating activation-
verbalization methods against planted content.*

The field is now **deploying** verbalization methods in production safety audits with no
way to check whether they are right. Plant known content by injection, then measure each
method on three axes that are currently unmeasurable:

1. **Recovery** — does the method recover the planted content? (accuracy vs oracle)
2. **Attribution** — does it recover it *for the right reason*, or via output steering /
   ordinary inference? This is exactly what our neutral-framing control isolates, and it
   directly operationalises the "excessive expressivity" limitation the NLA authors name
   but cannot test.
3. **CONFABULATION RATE** — inject **nothing**, ask the verbalizer what is present, and
   count how often it asserts content anyway. A false-positive rate for verbalization
   methods is a number the field urgently needs, currently cannot produce, and that we
   can generate on a 2B model.

Prior art to extend, not duplicate: *Do Activation Verbalization Methods Convey
Privileged Information?* (arXiv:2509.13316) asks this question and compares Patchscopes,
SelfIE, LatentQA and probes, controlling for what a model can infer from input alone.
Our addition is **planted ground truth plus the steering control**, which that work
lacks.

**Compute inverts in our favour.** Validation needs ground truth and controls, not
frontier scale. Patchscopes, SelfIE, LatentQA-style probing and logit lens are all
**inference-only** and runnable on the free tier at 2B-9B today. NLA itself requires RL
training of two LLMs and is **out of reach** — state that as a scoped limitation and
benchmark the accessible subset. Being honest about that boundary is stronger than
pretending otherwise.

## A9.4 Adopt

- **meta-d' / M-ratio** from human metacognition (Maniscalco & Lau lineage). Mature,
  importable, and gives L4 a principled metric — meta-d' is the d' an ideal observer
  would need to produce the observed confidence-accuracy contingency; M-ratio =
  meta-d'/d' normalises for task difficulty. Note arXiv:2603.25112 already applies SDT
  to LLMs, so this is adoption, not novelty.
- **Seed-twin control** for every privileged-access claim (identical architecture,
  tokenizer and data; different initialisation), per A8.4.
- **Disagreement-subset stratification** — effects hide in pooled data (A5.5).

## A9.5 Drop

**The developmental / training-checkpoint arm (U2, B9).** Partially claimed already —
checkpoint work exists on attention-circuit and linguistic-feature emergence
(arXiv:2606.02378, arXiv:2509.05291), and there is at least one report of introspection
appearing at a specific post-training stage (DPO on OLMo-3.1-32B). Expensive, no longer
uncontested, and it competes with E13 for the same Sem-6 window. Cut per the
substitution rule.

## A9.6 The honest risk on E13

**Anthropic invented concept injection.** Lindsey's introspection work *is* this
paradigm, in-house. Combining it with NLA to close their own validation gap is an
obvious next step for them and they could do it at any time, better and faster.

Three hedges, all consistent with the existing plan:
1. **Move fast** — E13 is inference-only at 2B and can run in Sem 5 on the free tier.
2. **Make the BENCHMARK the durable asset.** Benchmarks win on adoption, not novelty; if
   Anthropic runs the same validation internally, a public benchmark others can run
   still wins. This reinforces A8.4's promotion of PLANTED to a first-class deliverable.
3. **Own the confabulation-rate metric specifically.** It is the axis their framing least
   naturally produces (they audit models; we audit instruments), and it is the number a
   safety-focused reviewer will care most about.
