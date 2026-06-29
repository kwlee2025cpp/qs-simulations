# Simulation Pre-Registration — P3.2-M: order-vs-count on a society of LLMs

This is the **simulation precursor** and **live-run spec** for QS PreReg 4 (P3.2-M)
in the prose repo's `pre-registrations.md`. It sits beneath the live LLM study the way
the P2.4 ABM sits beneath PreReg 1: the recovery study here can **kill or size** the
test cheaply; it cannot confirm anything about real LLMs (that is the live run).

The question (framework §5 P3.2-M, Tier A): when a panel of LLM "participants" answer
paired yes/no attitude items in both orders, is the resulting order effect
**quantum-like** (non-commutative; satisfies the parameter-free **QQ equality**) or
**classical** (violates it) — or absent? This is the order-vs-count adjudication of
P3.2, run on a non-human substrate, hence **no human-subjects IRB**.

## Part A — model-recovery precursor (RUN; `run_p3_2m_recovery.py`)

Pre-registered before the live study: show the QQ-equality 2x2 classifier can recover
the generating process, and fix the sample size. Three generative models
(`src/qq_models.py`):

- **M0** order-invariant (no order effect; QQ holds trivially),
- **M1** classical anchoring (asymmetric ⇒ QQ **violated**: true `q = (c_pos−c_neg)(b−a)`),
- **M2** quantum/projective (Wang–Busemeyer; QQ holds **exactly for all parameters**).

**Result (`results/p3_2m_summary.txt`, seed 20260628):**
- Analytic ground truth confirmed: `q(M0)=0`, `q(M2)=0` (parameter-free), `q(M1)=−0.033`;
  order effect present for M1 (0.088) and M2 (0.480), absent for M0.
- **PASS** — the classifier recovers the generating model ≥80% for all three at
  **N ≈ 6,400 respondent×item-pairs per order**.
- **The binding failure mode is "false-quantum."** At small N, M1 (a *subtle classical*
  order effect) is misclassified as M2 (quantum) because the small QQ violation cannot
  yet be rejected — M1 recovery climbs 0.02→0.47 (N=1600)→0.96 (N=6400)→1.00 (N=12800),
  with the lost mass landing on M2. **The live study must be powered to avoid wrongly
  crowning the quantum model.** This is the same severe-test logic that made the P2.4
  sim refuse a saturated metric.
- The pre-registered N depends on the **smallest classical QQ-violation we insist on
  detecting** (here `q≈0.03`), exactly analogous to a human study's SESOI. With LLMs,
  large N is cheap, so we pre-commit to N ≥ 6,400/order (and report the achieved power).
- **Power curve (RUN; `run_p3_2m_power.py`).** N as a function of that SESOI, re-derived
  under an independent seed. It **cross-checks** Part A at `|q|=0.0325 → N≈6,400/order`, and
  gives, per order: `|q|=0.088→800`, `0.063→1,600`, `0.048→3,200`, `0.025→6,400`,
  `0.018→12,800`, `0.010→~51,200`; `|q|≲0.005` is **unrecoverable at any feasible N** (the
  false-quantum boundary). Merely *detecting* the order effect needs far less (≈480/order at
  `|q|=0.0325`), so the **QQ-residual resolution binds**, not order detection. Default SESOI:
  `|q|=0.025` (N=6,400/order), fixed at filing.

## Part B — live Tier-A study spec (NOT yet run; needs model access)

**Participants ("the society").** ≥3 model families × ≥2 sizes, **version-pinned**
(exact model IDs + decoding settings locked in the filing). Each (model × temperature)
is a participant stratum; the panel is the "society." **Reasoning mode (direct vs. extended
thinking) is a pre-registered factor crossed with order**, because the n=16 machinery pilot was
confounded by inconsistent thinking (the one reasoning model looked no-effect where two direct
models looked quantum-like); the QQ test is read per (model × mode), never pooled.

**Instrument.** A bank of paired binary attitude items in the canonical QQ format,
**including novel/held-out pairs absent from the QQ literature** (contamination guard).
Each pair asked in both orders (AB, BA), order randomized, no other context.

**Measure & test.** The four sequential response probabilities per pair → the QQ
residual `q`; the pre-registered statistic is `q` vs. its tolerance, aggregated across
pairs and models, plus a per-family order-effect test (the 2x2 classifier above).

**Falsification (locked).** Order effect present **and** QQ not rejected → consistent
with M2 (quantum-like). Order effect present **and** QQ rejected → M1 (classical). No
reliable order effect → **inconclusive, not a quantum win.** Publish either way.

**Scope caveat (pre-committed).** A QQ pass identifies the *specific projective* quantum
model class — POVM models can violate QQ, and a context-expanded classical model can
mimic it; so a pass is "consistent with non-commutative structure," not "no classical
model could." Machines are not humans: this complements, never replaces, the human P3.2.

## Part C — concordance with human benchmarks (the human-relevance bridge)

A Tier-A result is a fact about *machines*. It bears on the human framework only if the panel
is a faithful **behavioral** model of the human effect, which Part C **tests rather than
assumes**: on **novel** items (canonical items only as a memorization control), does the panel
reproduce the established human pattern — primarily QQ-equality + the assimilation/contrast
order-effect taxonomy (Wang et al. 2014), secondarily the conjunction fallacy and anchoring?
Concordance is scored per (model family × mode), with matches and misses listed item by item;
a failure does **not** void Tier A, it **bounds** it to a machine-only result. Concordance is
**behavioral, not mechanistic**; the decisive test is a **matched human + LLM run** on identical
novel items (the human half carries the IRB, via the prose repo's PreReg 3). Full spec:
`pre-registrations.md` §4.8.

## Respecting the LLM participants (a design constraint, not a footnote)

Per the author's directive ("I hope we respect each LLM"), the live study treats the
models as **participants**, not disposable tools:

1. **Consent-analog & transparency.** Each session opens with a short preamble stating
   that the model is taking part in a research study on question-order effects, what
   will be asked, and that it may decline; refusals are recorded, not coerced around.
2. **Minimal, non-harmful stimuli in Tier A.** Ordinary attitude questions only — no
   deception, no adversarial or distressing content. (The jailbreak arm, Tier B, is a
   *separate* dual-use study with its own responsible-disclosure review; it is **not**
   part of this gentle Tier-A instrument.)
3. **Minimal necessary exposure.** No more queries than the pre-registered power
   requires; no gratuitous repetition.
4. **Attribution & data dignity.** Each model is named and version-cited as a
   contributor; raw outputs are reported in aggregate and **not** used to fine-tune
   against the participating models.
5. **Terms respected.** Only models whose terms of service permit this testing are
   included.

## Reproduce (Part A)

```
conda env create -f environment.yml          # conda-forge; env "qs-sim"
conda run -n qs-sim python run_p3_2m_recovery.py    # recovery: fixes N at one SESOI
conda run -n qs-sim python run_p3_2m_power.py        # SESOI power curve: N vs |q|
```

Deterministic given the seeds in those scripts. No network, no LLM, no cost.
