# Simulation Pre-Registration — P2.4 agent-based companion

This is the **simulation** pre-registration: the decision rules below were fixed
*before* the production run. It is the cheap, severe pre-check that sits beneath the
human-subjects pre-registration (`pre-registrations.md`, PreReg 1, in the QS prose
repo). A simulation can **kill** a prediction cheaply or **demonstrate sufficiency**;
it can never **confirm** a claim about real societies (equifinality). The decisive
test remains the human experiment.

## Hypothesis under test
P2.4 (QS framework §5): at **matched trust-capital density**, generalized / upstream
reciprocity ("paying it forward") produces higher cooperation **coherence** and a
**larger-scale** cooperative regime than direct (two-body) reciprocity.

## Primary (confirmatory) decision rule — fixed before running
- **Primary DV:** cooperation coherence (Pielou evenness of the participation
  distribution; QS §3.3 order-parameter family).
- **(S) Sufficiency:** with the rival mechanism OFF (`beta_suppress = 0`), at matched
  trust density, the generalized condition must exceed the direct condition on
  coherence by **more than 2 SE** across replication seeds. Failure here means the
  mechanism does not work even in simulation; do not take P2.4 to a human study.
- **(C) Contest:** as the rival strengthens (`beta_suppress` ↑), report the threshold
  at which the coherence advantage reverses (Δ ≤ 0), or that it does not within the
  tested range. The rival is **Tsvetkova & Macy (2014)**: observing help (without
  receiving it) suppresses generosity among non-recipients — a real balancing loop
  given a genuine path to win.

## Secondary / exploratory (clearly labelled non-confirmatory)
- **Reach** (number of distinct agents who participate) and **total help volume**
  (raw count of help-acts) are reported as **exploratory** DVs. They were not part of
  the confirmatory rule; findings on them are hypothesis-generating, not tests.

## Neutrality safeguards (so this is a severe test, not a mirror)
1. **Matched initial conditions:** the two conditions share graph, initial trust
   distribution, and donor schedule for a given replication seed (paired design).
   Only the routing of the forward step differs.
2. **The rival can win:** `beta_suppress` is a first-class tunable mechanism; the grid
   spans from rival-off to rival-dominant, and the result is allowed to come out
   against P2.4.
3. **Regime rationale (structural, not outcome-tuning):** the model runs in a
   *propagation-dominated* regime (few seeded help-acts per round, long chains). This
   is required for the manipulated factor to carry variance: under a large per-round
   seed flood with short chains, the two conditions become near-identical and the
   primary DV **saturates** (~0.98), so any apparent "robustness" is a dead-metric
   artifact. This was verified empirically (the flooded regime was uninformative) and
   the regime was chosen on this structural ground, **before** fixing the result.

## Result of the production run (see `results/p2_4_summary.txt`)
- **(S) PASS** — coherence advantage `+0.088` (2·SE `0.010`) at baseline.
- **(C)** — the rival strongly **erodes** the coherence advantage (~70%) but does not
  reverse its sign within the tested range; P2.4 is a genuine contest, not foregone.
- **(X, exploratory)** — the **volume** sub-claim is **not** supported: direct
  reciprocity produces more total help (self-reinforcing dyads) in 100% of grid
  cells. Generalized wins on **breadth** (coherence, reach); direct wins on **raw
  volume**. This should feed back into the wording of P2.4 in the QS framework (split
  "larger-scale" into breadth vs. volume).

## Reproduce
```
conda env create -f environment.yml      # conda-forge; env "qs-sim"
conda run -n qs-sim python run_p2_4.py   # ~10 s; writes results/
```
