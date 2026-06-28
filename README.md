# QS Simulations

Computational companions to the **Quantum Social Physics (QS)** framework — the
prose/theory repo that develops a three-layer model of trust capital, "superconducting
society," and quantum-probability social cognition. This repo holds the **neutral,
shareable simulation code** that pre-checks the framework's falsifiable predictions
before (and alongside) human-subjects experiments.

> **Independent sibling repo.** This is a standalone git repository living next to the
> QS prose repo under the author's `pub-root/` container. It is meant to be shared and
> cloned on its own; it can also be wired into the QS paper as a git **submodule** for
> paper↔code version pinning.

## What a simulation can and cannot do (read this first)

The intended evidence hierarchy is:

1. **Real experiment** (best) — randomly assign real humans to conditions and let
   behavior be the judge. It can **surprise you and prove you wrong**. For these
   predictions the real-experiment tier is feasible and has precedent (Centola's
   networked experiments; Shirado & Christakis 2017 human+bot networks; Bond et al.
   2012). The designs live in the QS repo's `pre-registrations.md`.
2. **Neutral simulation** (this repo, fallback / cheap pre-check) — shows whether a
   mechanism is **sufficient** to produce a pattern, and can **kill** a prediction
   cheaply. It **cannot confirm** a claim about real societies, because of
   **equifinality**: many different mechanisms can produce the same pattern, so
   reproducing an observation does not single yours out.

A simulation only counts as evidence if **it could have failed**. The safeguards that
make this one a *severe* test (not a confirmation mirror) are listed below and in
`PREREG.md`.

## P2.4 — generalized vs. direct reciprocity

**Prediction (QS §5, P2.4):** at *matched trust-capital density*, generalized /
upstream reciprocity ("paying it forward", A→B→C) yields higher cooperation
**coherence** and **larger-scale** cooperation than direct (two-body, A↔B) reciprocity.

**Model** (`src/trust_sim.py`), in System-Dynamics grammar (QS §3):
- Each agent holds a **trust-capital stock** `T` with an **inflow** (receiving help
  raises it — the *reinforcing* loop, "paying it forward") and **outflows** (decay,
  and the rival below).
- The **rival** is a *balancing* loop, **Tsvetkova & Macy (2014)**: non-recipient
  *observers* of help lose trust (suppression/crowding-out). Strength `beta_suppress`
  is swept from off to dominant — it is given a real path to win.
- Conditions differ only in routing the forward step: **direct** bounces help A↔B;
  **generalized** passes it onward to a new stranger.

**Neutrality safeguards:**
- Paired design — both conditions share graph, initial trust, and donor schedule per
  seed → "matched trust density" is enforced, not assumed.
- The rival can overturn the prediction; outcomes against P2.4 are reported.
- Propagation-dominated regime chosen on *structural* grounds (else the primary DV
  saturates and the test is dead) — see `PREREG.md`.

### Headline result (`results/`)

| DV | Winner | Robust to the rival? |
|----|--------|----------------------|
| Coherence (evenness) — **primary** | **Generalized** (Δ ≈ +0.09 at baseline) | Eroded ~70% by the rival but **not reversed** |
| Reach (distinct agents) — exploratory | **Generalized** | Eroded, not reversed |
| Total volume (raw help-acts) — exploratory | **Direct** (100% of cells) | — |

**Interpretation:** generalized reciprocity wins on **breadth** (coherence, reach);
direct reciprocity wins on **raw volume** (self-reinforcing dyads pump more help). So
the simulation **partially refutes** P2.4 as worded — its "larger-scale" clause should
be split into *breadth* (supported here) vs. *volume* (refuted here). This is exactly
the kind of correction a mirror-simulation would have hidden, and it feeds back into
the QS framework §5.

Figures: `results/p2_4_phase_coherence.png`, `results/p2_4_phase_volume.png`,
`results/p2_4_slice.png`; data `results/p2_4_phase.csv`; verdict
`results/p2_4_summary.txt`.

## Run it

```bash
conda env create -f environment.yml     # conda-forge only; creates env "qs-sim"
conda run -n qs-sim python run_p2_4.py   # ~10 s; (re)writes results/
```

Dependencies (numpy, pandas, matplotlib, **networkx**) are declared in
`environment.yml` on **conda-forge** — no Anaconda `defaults` channel (so no
Terms-of-Service gate for anyone who clones), and no pip `requirements.txt` (so GitHub
Dependabot, which parses pip/npm but not conda env files, has nothing to alert on).

Deterministic given the seeds in `run_p2_4.py` (seeds 0–15).

## Roadmap

- `run_p2_4.py` — generalized vs. direct reciprocity **(done)**
- P2.5 — LLM node at donor vs. intermediary × memory configuration *(planned)*
- P3.2 — order-vs-count adjudication: M0/M1/M2 model-recovery + fitting *(planned)*

## License & citation

MIT (`LICENSE`), © 2026 Kangwon Lee — **please confirm/adjust the license and author
before publishing the repo.** If you use this, cite the QS framework paper and this
repository. Author: Prof. Kangwon Lee, `kangwon.lee@ieee.org`.
