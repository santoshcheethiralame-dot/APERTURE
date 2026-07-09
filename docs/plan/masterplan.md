# PROJECT MIRROR — Master Plan
## Dissociating Genuine Introspection from Confabulation in Large Language Models
### A 2-year, 5-person capstone research program (100 working weeks)

**One-line thesis:** We will build and validate the first experimental battery that cleanly separates a language model *genuinely reading its own internal states* from a model *confabulating a plausible story about itself* — with ground truth, causal mechanism, and cross-model scale — and settle the currently contested question of whether machine introspection is real.

**Flagship deliverables:**
1. **The dissociation result** — the main paper: direct access vs. inference/confabulation, resolved with pre-registered confirmatory experiments across ≥10 open models in ≥4 families.
2. **INTROSPECT-Bench** — a public, reusable benchmark + harness for measuring introspective access (detection, identification, source attribution, metacognitive calibration), with human-validated grading.
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
Flagship: *"The Introspection Ladder: What Models Can and Cannot Know About Themselves"* — L1 robust, L2 partial in identified regimes, L3/L4 dissociations, dose–response curves, cross-family meta-analysis (random-effects, I² heterogeneity). The ladder + INTROSPECT-Bench becomes the field's measurement standard. This is the "reference paper" branch and is fully pre-planned, not a consolation.

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
| 40 | Battery consolidation | INTROSPECT-Bench alpha spec: task list, metrics, grading standards, API design |
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
| 67 | Bench beta | INTROSPECT-Bench beta: packaged tasks, harness, judges, gold labels, docs |
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
| 82 | **Preprint + release** | arXiv preprint; INTROSPECT-Bench public; Alignment Forum post; code + transcripts released (Zenodo DOI) |
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
- **W82:** arXiv preprint + Alignment Forum long-post + INTROSPECT-Bench public release (adoption is impact; the bench is our citation engine).
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
