"""
trust_sim.py — agent-based companion model for QS framework prediction P2.4.

P2.4 (framework §5): at MATCHED trust-capital density, GENERALIZED / upstream
reciprocity ("paying it forward", A->B->C) attains higher cooperation coherence
than DIRECT (two-body, A<->B) reciprocity.

This module is built as a SEVERE, NEUTRAL test, not a confirmation engine:
  * The reinforcing loop (paying-it-forward boost) and the BALANCING loop that
    opposes it (Tsvetkova-Macy "observing help suppresses generosity among
    non-recipients", 2014) are BOTH first-class mechanisms with tunable strength.
  * The two conditions start from IDENTICAL initial conditions (same graph, same
    trust distribution, same donor schedule) given a replication seed -> "matched
    trust density" is enforced, not assumed.
  * The rival (suppression) is given a real path to win. The result is allowed to
    come out AGAINST P2.4, and the runner reports wherever that happens.

A simulation can only show SUFFICIENCY (the mechanism *can* produce the pattern),
never TRUTH about real societies (equifinality: other mechanisms could produce the
same pattern). The decisive test is the pre-registered human experiment in the QS
repo's `pre-registrations.md` (PreReg 1). See README.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Params:
    n: int = 150                 # number of agents
    k: int = 6                   # ring lattice degree before rewiring (Watts-Strogatz)
    rewire: float = 0.10         # small-world rewiring probability
    rounds: int = 60             # simulation rounds
    # Propagation-dominated regime: few seed help-acts, long chains. This is a
    # STRUCTURAL requirement, not outcome-tuning — if the per-round seed flood is
    # large and chains are short, the two conditions become indistinguishable (the
    # manipulated factor, routing of the forward step, carries almost no variance)
    # and any "robustness" is a saturated-metric artifact. See PREREG.md / README.md.
    seeds_help_per_round: int = 3  # help-acts initiated per round (matched across conditions)
    max_chain: int = 20          # max propagation length per initiated help-act
    alpha_boost: float = 0.05    # reinforcing inflow: receiving help raises recipient trust stock
    beta_suppress: float = 0.0   # balancing outflow: non-recipient observers lose trust (the rival)
    decay: float = 0.03          # stock decay toward baseline each round (forgetting)
    t0_mean: float = 0.50        # initial trust-capital stock mean
    t0_sd: float = 0.15          # initial trust-capital stock sd


def watts_strogatz(n: int, k: int, p: float, rng: np.random.Generator) -> list[np.ndarray]:
    """Watts-Strogatz small-world graph as an adjacency list (pure numpy, no networkx)."""
    if k % 2 != 0:
        k += 1
    nbr: list[set[int]] = [set() for _ in range(n)]
    half = k // 2
    for i in range(n):
        for j in range(1, half + 1):
            nbr[i].add((i + j) % n)
            nbr[i].add((i - j) % n)
    # rewire
    for i in range(n):
        for j in range(1, half + 1):
            if rng.random() < p:
                old = (i + j) % n
                choices = [x for x in range(n) if x != i and x not in nbr[i]]
                if choices:
                    new = int(rng.choice(choices))
                    nbr[i].discard(old)
                    nbr[old].discard(i)
                    nbr[i].add(new)
                    nbr[new].add(i)
    return [np.array(sorted(s), dtype=np.int64) for s in nbr]


def _coherence(activity: np.ndarray) -> float:
    """Cooperation coherence as Pielou evenness of the participation distribution
    over ALL agents (framework §3.3: 1 - normalized entropy family). High = many
    agents participating evenly; low = activity concentrated in a few. Bounded [0,1]."""
    total = activity.sum()
    if total <= 0:
        return 0.0
    p = activity / total
    nz = p[p > 0]
    h = -(nz * np.log(nz)).sum()
    return float(h / np.log(len(activity)))  # normalize by ln(n): even spread over all n -> 1


def run_once(params: Params, condition: str, seed: int) -> dict:
    """Run one replication. `condition` in {'direct','generalized'}.

    Matching: graph, initial trust, and the donor schedule are drawn from RNG
    streams seeded only by `seed` (identical across conditions). Only the routing of
    the forward step differs between conditions."""
    if condition not in ("direct", "generalized"):
        raise ValueError(condition)

    rng_struct = np.random.default_rng(seed)          # graph + initial trust (matched)
    rng_donor = np.random.default_rng(seed + 10_000)  # donor schedule (matched)
    rng_dyn = np.random.default_rng(                  # within-round choices (diverges by design)
        seed + (20_000 if condition == "direct" else 30_000)
    )

    adj = watts_strogatz(params.n, params.k, params.rewire, rng_struct)
    t = np.clip(rng_struct.normal(params.t0_mean, params.t0_sd, params.n), 0.0, 1.0)
    t0 = t.copy()
    activity = np.zeros(params.n, dtype=np.float64)

    for _ in range(params.rounds):
        donors = rng_donor.integers(0, params.n, size=params.seeds_help_per_round)
        for donor in donors:
            current = int(donor)
            prev = -1
            for _step in range(params.max_chain):
                nbrs = adj[current]
                if len(nbrs) == 0:
                    break
                if condition == "direct":
                    cand = nbrs if prev == -1 else np.array([prev], dtype=np.int64)
                else:  # generalized: forward to someone other than where it came from
                    cand = nbrs[nbrs != prev] if prev != -1 else nbrs
                if len(cand) == 0:
                    break
                recipient = int(cand[rng_dyn.integers(len(cand))])

                # the help act
                activity[current] += 1.0   # helper acts
                activity[recipient] += 1.0  # recipient benefits
                # reinforcing inflow: gratitude raises recipient's trust stock
                t[recipient] = min(1.0, t[recipient] + params.alpha_boost)
                # balancing outflow (the rival): non-recipient observers lose trust
                if params.beta_suppress > 0.0:
                    obs = np.union1d(nbrs, adj[recipient])
                    obs = obs[(obs != current) & (obs != recipient)]
                    if len(obs):
                        t[obs] = np.maximum(0.0, t[obs] - params.beta_suppress)

                # continue the chain with prob = recipient's (grateful) trust stock
                if rng_dyn.random() > t[recipient]:
                    break
                prev = current
                current = recipient

        # stock decay toward baseline (forgetting / outflow)
        t += params.decay * (t0 - t)

    return {
        "coherence": _coherence(activity),
        "total_help": float(activity.sum() / 2.0),
        "reach": float((activity > 0).sum()),
    }


def paired_difference(params: Params, seeds: list[int]) -> dict:
    """Run both conditions on each replication seed and return paired generalized-minus-direct
    differences. Pairing on seed = identical initial conditions = matched trust density."""
    dvs = ("coherence", "total_help", "reach")
    diffs = {dv: [] for dv in dvs}
    gen_vals = {dv: [] for dv in dvs}
    dir_vals = {dv: [] for dv in dvs}
    for s in seeds:
        g = run_once(params, "generalized", s)
        d = run_once(params, "direct", s)
        for dv in dvs:
            diffs[dv].append(g[dv] - d[dv])
            gen_vals[dv].append(g[dv])
            dir_vals[dv].append(d[dv])
    out = {}
    for dv in dvs:
        arr = np.array(diffs[dv])
        out[dv] = {
            "delta_mean": float(arr.mean()),
            "delta_se": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
            "generalized_mean": float(np.mean(gen_vals[dv])),
            "direct_mean": float(np.mean(dir_vals[dv])),
        }
    return out
