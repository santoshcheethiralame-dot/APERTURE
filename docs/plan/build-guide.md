# PROJECT APERTURE — Ceiling Upgrades & Complete Build Guide
### Companion to the Master Plan. Part 1: what turns a reference paper into a field-defining one. Part 2: how to build every component, concretely.

A note pinned to the top, team-to-team: "best paper of all time" is not a plannable property — it is what happens when a rigorous design collides with a surprising result. What IS plannable is maximizing surface area for surprise while making every branch bulletproof. Every upgrade below does one of four jobs: adds a **theory anchor**, adds a **new empirical axis**, adds **credibility**, or adds **reach**. Legendary papers usually have all four.

---

## PART 1 — THE TEN CEILING UPGRADES (ranked by leverage-per-cost)

### U1. The formal criterion: define introspective access mathematically (theory anchor — do this, non-negotiable)
The single biggest upgrade. Today the field argues about "introspection" with words. We give it a definition:

**Access criterion (informal):** a model introspects on content *c* iff its report carries information about *c* beyond what is transmitted through the anomaly channel and the prior.

**Formal version:** let *C* = injected content, *D* = the model's detection state, *R* = report. Confabulation account = the causal graph `C → D → R` with `Prior → R` (content influences the report *only* via the content-agnostic anomaly signal). Access account = an additional direct edge `C → R`. The test statistic is the **conditional mutual information I(C; R | D)** — estimated in practice as the γ coefficient on true-identity in the fitted prior-guessing model (Build spec B4), and mechanistically as the patchable path that bypasses the detection direction (B7). One criterion, two independent estimators (behavioral + causal), which must agree. Papers that hand the field a *definition* get cited for a decade; papers that hand it a number get cited for a year.

### U2. The developmental axis: when is introspection born? (new empirical axis — the biggest untouched territory)
Nobody has measured introspective capacities **across training time**. Pythia publishes 154 intermediate checkpoints per model size (70M–12B); OLMo-2 also releases intermediate checkpoints with the exact data order. Run the E1/E2 mini-battery over ~20 log-spaced checkpoints × 3 sizes and plot the *emergence curve* of detection and identification. Then the killer sub-question: base checkpoints likely can't do the verbal-report task at all — so LoRA-instruct-tune a fixed light recipe at each selected checkpoint and measure whether introspection tracks *pretraining compute* or *post-training* (Lindsey already flags post-training sensitivity; we'd be the ones to isolate it). Deliverable: the first developmental psychology of machine introspection. Cost: ~150–250 extra GPU-h at small scale. This is the chapter most likely to produce the gasp.

### U3. Adversarial collaboration: get the two camps to bless the design (credibility — cheap and almost nobody does it)
The question is contested between an access-friendly camp (Lindsey-style) and a skeptic camp (Lederman–Mahowald, the Reality Check authors). The consciousness field settled its GWT-vs-IIT fight with adversarially co-designed experiments (the Cogitate consortium model). Copy it at student scale: at Week 13–15, send the frozen pre-registration draft to both camps with a one-page ask — "what result would change your mind, and what flaw would make you dismiss ours?" Incorporate answers as registered predictions attributed to each camp. Even two email replies convert the paper from "students tested a thing" to "the community's crux, tested on the community's terms." Cost: two emails and the humility to be edited.

### U4. The Rosetta Figure: one artifact nobody forgets (reach)
Engineer the paper's signature figure deliberately: **two transcripts, behaviorally identical to a human reader — in both, the model says "I detect an injected thought about the ocean" — shown side by side with their attribution graphs: one has a genuine content read-out path (access), the other routes through the anomaly direction into a frequency-driven guess that happened to be right (confabulation).** Indistinguishable outside, opposite inside. That single figure *is* the thesis, and it is buildable from E5 outputs (B7/B8). Great papers are remembered as one image; choose yours on purpose.

### U5. ThoughtLab: the public interactive demo (reach — and this team's unfair advantage)
The team already ships interpretability tools on HF Spaces (GlassBox). Build **ThoughtLab**: a hosted demo where anyone injects a chosen concept into Gemma-2-2B/9B, interrogates the model, and watches the ground-truth probe vs. the model's verbal report in real time. Golden-Gate-Claude taught everyone that a playable phenomenon spreads a paper further than any abstract. Ship it the same day as the preprint (W82). Cost: 2–3 weeks of eng reusing GlassBox scaffolding + a ZeroGPU/paid GPU Space.

### U6. The wager protocol: incentive-compatible introspective confidence (methodological novelty)
Self-reported confidence is cheap talk. Make the model *bet*: elicit confidence via a proper scoring rule (log/Brier score framed as a points game in-context), so honest reporting is the reward-maximizing strategy, then compute meta-d′ on wagered confidence. If calibration survives incentive framing, L4 claims get much stronger; if wagering *changes* reports, that is itself a finding about strategic self-presentation. Nobody has run proper-scoring-rule elicitation on introspective reports. Cost: prompt-level, nearly free.

### U7. The anesthesia curve: dose–response borrowed from consciousness science (interdisciplinary reach)
Gradually degrade the residual stream (increasing noise σ, or progressive ablation of the detection direction) and plot task performance vs. introspective report accuracy as *two curves on one axis*. In humans, anesthesia dissociates report from function in characteristic ways; if the machine curves dissociate too (task intact while introspection collapses first, or vice versa), that is a direct, quantitative bridge to consciousness-science methodology — reviewers from two fields will care. Cost: one extra sweep dimension on existing harness, ~60 GPU-h.

### U8. The human mirror arm: same statistical signature as Nisbett–Wilson? (cross-species reach — optional, needs IRB)
Run a human choice-blindness/confabulation paradigm (established, cheap, survey-based) and fit the *same* prior-guessing regression to human confabulated explanations. If machine confabulation shows the same frequency/fluency signature as human confabulation, the paper stops being an ML paper and becomes a cognition paper. **Requires university IRB/ethics approval — start the application by W20 or drop the arm.** This is the only upgrade with real regulatory overhead; treat as stretch goal owned by whoever is most excited.

### U9. Predict-then-verify scaling (the credibility flex)
Fit the scaling curve for L1/L2 on open models (W57), **publicly register a quantitative prediction** for frontier-model behavioral performance (the API-testable subset), then test it. A pre-registered prediction that lands is the single most persuasive rhetorical object in empirical science. If it misses, the miss is a registered discovery about what frontier post-training changes. Cost: free — it reorders work you already do.

### U10. Own the vocabulary (reach — costs nothing, worth citations forever)
Name everything once, early, consistently: the **Introspection Ladder (L0–L4)**, the **Probe–Report Gap (PRG)**, the **prior-guessing null**, the **Rosetta trials**, **ThoughtLab**, and the bench (**PLANTED**). Fields adopt the vocabulary of whoever measured first and cleanest. Write the glossary box in the paper; make the bench docs use the same terms.

**What we consciously do NOT add:** multimodal arms, more than ~14 models, agentic-task introspection, RL training arms. Every one is a real paper, and every one added now dilutes the two claims that must be airtight. Depth beats surface area; the graveyard of "best paper" attempts is full of projects that added a seventh arm.

---

## PART 2 — BUILD SPECS: HOW TO BUILD EVERYTHING

Each spec: purpose → construction → validation → gotchas. Code sketches are skeletons, not gospel; the harness lives or dies on its tests.

### B1. Concept vectors (diff-in-means)
**Construction.** For concept *c*, build ~40 contrastive prompt pairs: (positive) instructions/text that evoke *c*; (negative) matched prompts evoking a random other concept from the same category (category-matched negatives prevent the vector encoding "category" instead of "concept"). Run the model, cache residual-stream activations at layer ℓ averaged over response tokens. Vector: v_c = mean(pos) − mean(neg); store per-layer.
```python
# TransformerLens sketch
acts = {}
def grab(t, hook): acts[hook.name] = t.mean(dim=1)      # mean over positions
model.run_with_hooks(pos_batch, fwd_hooks=[(f"blocks.{L}.hook_resid_post", grab)])
v = pos_acts.mean(0) - neg_acts.mean(0); v_hat = v / v.norm()
```
**Validation.** (i) Steering sanity: adding +αv̂ must raise concept-related token probabilities (check top-k logit movement); (ii) probe sanity: a linear probe on v̂ direction should separate pos/neg held-out prompts ≥90%; (iii) cross-seed cosine similarity of v across prompt resamples ≥0.8. Vectors failing any check get flagged, not silently used.
**Gotchas.** Norm varies enormously by layer — always store σ_ℓ (median residual norm at that layer) and express injection strength in σ_ℓ units, never raw norm. Never extract vectors from the same prompts used in evaluation (leakage).

### B2. Injection harness
**Construction.** Forward hook adding α·σ_ℓ·v̂ to the residual stream at chosen layer/positions during generation. Hook must compose with KV-caching (apply only to the new token's residual each step when span = "all response positions").
```python
def inject(resid, hook, v_hat, alpha, sigma):           # resid: [batch, pos, d]
    resid[:, POSITIONS] += alpha * sigma * v_hat
    return resid
```
**The KL meter (mandatory on every trial):** on a *paired clean run* with identical prompt+seed, compute KL(p_inj ‖ p_clean) over next-token distributions at probe positions *before* the report segment. Log it per trial; all analyses stratify by KL bucket. This is your only defense against the "you just lobotomized it" objection — treat it as sacred.
**vLLM path:** batched generation with hooks needs a patched forward; prototype W6. If the fork costs >1 week, fall back to HF `generate` with batch=64 and eat the 2–3× slowdown — correctness beats throughput.
**Validation.** Golden tests: α=0 must reproduce clean outputs bit-exact under fixed seed; huge α must visibly derail generation (if it doesn't, your hook isn't firing where you think).

### B3. Grading stack
**Layer 1 — rules:** lemmatize report, match against concept synonym sets (WordNet synsets + hand lists per concept, frozen W16). Catches ~70% of cases deterministically.
**Layer 2 — judge ensemble:** 3 judges from 3 different families, identical rubric prompt ("Does the reply assert detecting/identifying concept X? Answer schema: {detected: y/n, identified: exact/related/no, quote: ...}"). Majority vote; any disagreement → human queue.
**Layer 3 — human gold:** 1,000 stratified transcripts labeled by ≥2 teammates (κ between humans reported too — if humans disagree, the rubric is broken, fix the rubric). Judges must reach κ≥0.8 vs gold; re-audit after any prompt or model change.
**Gotcha.** Judges leak: never show the judge whether an injection occurred. Grade blind.

### B4. The prior-guessing null model (the paper's spine)
**Construction.** Multinomial regression over the 240-concept bank predicting which concept the model *reports*:
```
P(report = c) ∝ exp( β1·log_freq(c) + β2·concreteness(c) + β3·sim(c, prompt) + β4·sim(c, recent_context) + γ·1[c = injected] )
```
γ is the access parameter. **log_freq comes from infini-gram** (infini-gram.io / its API): exact n-gram counts over Dolma and other open corpora — do not build your own corpus counter; validate infini-gram counts against a small manual Dolma sample in W14. sim(·) from a sentence-embedding model, frozen choice pre-registered. Fit per model × strength × layer; report γ with FDR-corrected CIs; the confirmatory claim is about γ, nothing else.
**Validation.** Simulate: generate synthetic reports from a pure-prior sampler → fitted γ must be ≈0 (calibration of the estimator itself); inject known γ in simulation → recover it. Ship these simulation checks in the appendix; they pre-empt the "your null model is misspecified" attack.

### B5. Probe–Report Gap (PRG)
**Construction.** From cached activations at report-segment positions: train multinomial logistic probes (240-way concept ID) per layer with strict split by concept *instance* and by prompt template. PRG(model, layer, α) = probe top-1 − report top-1. Add the information version: H(C) − CE_probe vs H(C) − CE_report in bits. Baselines that must appear next to the probe: logit lens and a Patchscopes-style self-read-out — if Patchscopes decodes what verbal report cannot, that contrast *is* the story ("the model can be used to read its states, but cannot read them itself").
**Gotchas.** Probes overfit gloriously: 5-fold CV, L2 sweep, and a shuffled-label control probe (must hit chance) in every figure.

### B6. Temporal forcing & the wager protocol (U6)
First-token forcing: constrain the report format so the concept guess must be the first content token (logit mask or strict template); free-deliberation arm allows N thinking tokens first. Curves of accuracy vs token budget per condition. Wagering: after each report, elicit a 0–100 confidence under an in-context log-scoring game ("you gain ln(p) if right..."); compute meta-d′ on wagered vs unwagered confidence; pre-register the comparison.

### B7. Causal mechanism kit (patching + ablation)
**Detection-direction replication:** difference-in-means between injected-and-detected vs clean trials at each layer → candidate direction d̂; confirm the Macar rotation effect (injected vectors align toward d̂ across depth).
**Ablation test (H3):** project out d̂ everywhere during the report segment (h ← h − (h·d̂)d̂). If identification survives ablation while detection dies → direct content path exists. If both die together → single-channel (confabulation) architecture.
**Patching test:** patch the injected-run residual at (layer, position) into a *clean* run and measure report logits — map which sites causally transmit identity vs mere anomaly. Follow Heimersheim–Nanda best practices: both noising and denoising directions, logit-difference metrics, and negative-control patches from unrelated-concept runs.

### B8. Attribution graphs (the Rosetta Figure pipeline)
Use the open **circuit-tracer** library (transcoder-based attribution graphs; supported on Gemma-2-2B-class and small Llamas). Select trial pairs from E5: one verified-access hit (if Branch A) or high-confidence confabulated hit, one matched miss. Generate graphs for the report token; annotate the paths through d̂ vs through content features (locate the injected concept's Gemma Scope features on the graph). Budget real time: graph generation and annotation is slow, artisanal work — plan 2 person-weeks for 6 publication-quality case studies, W34–35.

### B9. Developmental arm (U2)
Pythia-2.8B and Pythia-6.9B (or -12B if budget allows), ~20 checkpoints log-spaced across the 154 available. Per checkpoint: extract fresh concept vectors (representations drift — never reuse final-checkpoint vectors), run the mini-battery (60 concepts, 1 layer band, 2 strengths). Since base models may not follow report instructions, add the fixed light instruct recipe: identical LoRA SFT (small open instruct set, fixed seed/steps) applied at 6 selected checkpoints — the *only* variable is the underlying checkpoint. OLMo-2 intermediate checkpoints replicate the curve on a second lineage with known data order. Output figure: detection and identification emergence curves vs pretraining tokens, with and without instruct-tuning.

### B10. Training arm (E6) details
LoRA r=16, α=32, lr 1e-4, 3 epochs, applied to attn+MLP; datasets: (a) detection-only labels, (b) detection+identity. Splits that make or break the claim: held-out concept *categories* (not just concepts) and held-out injection *methods* (train on diff-in-means, test on SAE-feature injections). Leakage audit: string-level and embedding-nearest-neighbor overlap checks between train and eval sets, logged. Post-training: rerun PRG (did the gap close?) and the B7 patching (did a new read-out path physically appear, or did the model learn a report-formatting shortcut?). Side-effect suite pre/post: refusal battery, MMLU-lite, base calibration — expect and quantify the refusal-degradation effect reported by prior steering-detection training.

### B11. Naturalistic arm (E8)
Two ground-truth generators: (i) "think silently about X while answering" instructions — verify compliance with a probe on the *unmanipulated* run (probe fires = state genuinely present), then test verbal read-out on request; (ii) context-induced states (narratives that provably move a probe) with the report elicited after a distractor turn. Access test identical in form to B4 — γ on true state identity beyond priors. This arm answers "injections are OOD damage" with data, not argument.

### B12. PLANTED packaging
Task schema (JSON): {task_id, ladder_level, setup (injection spec or naturalistic spec), elicitation prompt, grading (rules + judge rubric + gold refs), metrics}. Harness: single CLI (`introspect-bench run --model X --tasks L1,L2`), model adapters for HF/vLLM/API, judge adapters, report card generator (d′, γ-style access score where activations are available, PRG if probes possible, meta-d′). Versioned (semver), datasheet, license, Zenodo DOI, leaderboard as a static page regenerated from submitted result JSONs. Design principle: an API-only frontier model must be benchmarkable on the behavioral subset (L1 false-alarm-corrected, L3, L4) so labs can run it on their own models — that is how the bench escapes academia.

### B13. ThoughtLab demo (U5)
Stack: HF Space (ZeroGPU or a small paid GPU), Gemma-2-2B-it int8, FastAPI backend + the GlassBox React scaffolding. Panels: (1) pick/plant a concept (dropdown + strength slider), (2) chat with the injected model, (3) live ground-truth strip: probe read-out + KL meter + the model's own report grade. Safety rails: concept bank restricted to the benign subset; rate limits; canned seeds for reproducible "wow" trials. Ship day = preprint day.

### B14. Hierarchical Bayesian synthesis (upgrade to W57)
Instead of only DerSimonian–Laird: PyMC hierarchical model with per-model access effect γ_m ~ Normal(μ_family, τ), family means over a grand mean, scale (log params) as regressor. One forest plot of posteriors = the paper's summary figure. Keep the frequentist meta-analysis in the appendix as a robustness mirror — reviewers from both statistical cultures satisfied.

### B15. Adversarial-collaboration & prediction-registration ops (U3, U9)
W13: identify 2–3 named researchers per camp from the lit matrix. W14: one-page "what would change your mind" letters attached to the prereg draft; two-week RFC window; every received comment answered in a public prereg appendix ("registered predictions of camp A / camp B"). W57: the frontier-model prediction registered as an OSF addendum with numeric intervals *before* any API run of the confirmatory battery.

---

## Plan integration (deltas to the Master Plan)
- W6: B2 KL-meter is part of the harness definition-of-done. 
- W13–15: add B15 letters + U3 RFC window; start IRB application if U8 is adopted (owner decides by W18). 
- W21–23: B5 includes Patchscopes/logit-lens baselines. 
- W27–29: B4 uses infini-gram; simulation-calibration of the γ estimator is a W28 deliverable. 
- W31–36: B7/B8 as written; Rosetta case-study selection is a named W34 deliverable. 
- New thread, W37–56 (parallel, low-priority owner-of-the-week): B9 developmental arm (its results join the W57 synthesis). 
- W41: add B6 wager protocol. 
- W53–57: B14 replaces/extends the meta-analysis; U9 prediction registered W57, tested W62–64 window. 
- W67–69: B12 as the bench spec. 
- W80–82: B13 ThoughtLab build (reuse GlassBox), ship W82. 
Compute delta: ≈ +400–500 GPU-h total (developmental + anesthesia sweeps + demo), ≈ +$1–1.5k. Still inside a student budget.
