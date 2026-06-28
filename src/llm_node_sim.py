"""
llm_node_sim.py — agent-based companion for QS framework prediction P2.5.

P2.5 (framework §5, §2.1): a machine (LLM) node occupies ONE side of the cooperation
"pair", and its effect on help-propagation is ASYMMETRIC by role:
  * as a DONOR (an upstream source of help) it AMPLIFIES spread;
  * as a stateless INTERMEDIARY (a relay in the middle of a chain) it ATTENUATES
    spread, because with no memory it has no accumulating stake and tends not to pay
    help forward.
Memory configuration is a pre-registered factor: memory ON should mitigate the
intermediary attenuation (the machine can behave like a committed reciprocator).

Severe, neutral test (not a confirmation engine). Two rivals are first-class and given
real paths to overturn P2.5:
  * Karpus et al. (2021), "algorithm exploitation": humans are keen to exploit
    benevolent AI. Modeled as `p_exploit` — a human receiving help that originated
    from / passed through the machine pays it forward LESS, capping donor amplification.
  * Crandall et al. (2018), "cooperating with machines" needs legible intent. Modeled
    as `illegibility` — help emitted BY the machine is less catalytic (smaller trust
    inflow), weakening its contribution.

Matching: baseline / donor / intermediary arms share the SAME graph, initial trust,
and human donor schedule for a given seed; only the machine's role differs. The
machine node is the highest-degree node (so the intermediary-gating role is real).

A simulation shows SUFFICIENCY, never truth about real human-machine networks
(equifinality). The decisive test is the pre-registered study (QS PreReg 2 / P2.5).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from trust_sim import watts_strogatz, _coherence


@dataclass(frozen=True)
class NodeParams:
    n: int = 150
    k: int = 6
    rewire: float = 0.10
    rounds: int = 60
    base_seeds: int = 2          # human-initiated chains per round
    machine_seeds: int = 2       # machine-initiated chains per round (donor arm)
    max_chain: int = 20
    alpha_boost: float = 0.05    # reinforcing inflow on receiving help
    t0_mean: float = 0.50
    t0_sd: float = 0.15
    pf_machine_off: float = 0.15  # stateless machine's pay-forward prob (attenuation)
    # rivals (real paths to overturn P2.5):
    p_exploit: float = 0.0       # Karpus: humans don't pay forward machine-origin help
    illegibility: float = 0.0    # Crandall: machine-emitted help is less catalytic


def _machine_node(adj: list[np.ndarray]) -> int:
    return int(np.argmax([len(a) for a in adj]))


def run_once(p: NodeParams, arm: str, memory: str, seed: int) -> dict:
    """One replication. arm in {'baseline','donor','intermediary'}; memory in {'on','off'}."""
    assert arm in ("baseline", "donor", "intermediary")
    assert memory in ("on", "off")
    rng_struct = np.random.default_rng(seed)
    rng_donor = np.random.default_rng(seed + 10_000)
    rng_dyn = np.random.default_rng(seed + 20_000)   # matched across arms (same seed)

    adj = watts_strogatz(p.n, p.k, p.rewire, seed)
    m = _machine_node(adj)
    t = np.clip(rng_struct.normal(p.t0_mean, p.t0_sd, p.n), 0.0, 1.0)
    t0_m = t[m]
    activity = np.zeros(p.n, dtype=np.float64)

    for _ in range(p.rounds):
        # matched human seeds; donor arm lets the machine inject extra chains, the
        # other arms hand that same budget to random humans (total injection matched)
        humans = rng_donor.integers(0, p.n, size=p.base_seeds)
        extra = rng_donor.integers(0, p.n, size=p.machine_seeds)
        if arm == "donor":
            sources = list(humans) + [m] * p.machine_seeds
        else:
            sources = list(humans) + list(extra)

        for src in sources:
            current = int(src)
            prev = -1
            machine_tagged = (current == m)
            for _step in range(p.max_chain):
                nbrs = adj[current]
                cand = nbrs[nbrs != prev] if prev != -1 else nbrs
                if len(cand) == 0:
                    break
                recipient = int(cand[rng_dyn.integers(len(cand))])

                activity[current] += 1.0
                activity[recipient] += 1.0

                # reinforcing inflow; machine-emitted help is less catalytic if illegible
                boost = p.alpha_boost * (1.0 - p.illegibility) if current == m else p.alpha_boost
                if not (recipient == m and memory == "off"):
                    t[recipient] = min(1.0, t[recipient] + boost)

                if recipient == m:
                    machine_tagged = True

                # continuation probability for the NEXT step (recipient becomes holder)
                if recipient == m and memory == "off" and arm != "baseline":
                    cont = p.pf_machine_off            # stateless relay: attenuates
                    t[m] = t0_m                         # no memory => no accumulation
                else:
                    cont = t[recipient]
                    # Karpus rival: humans exploit benevolent-machine-origin help
                    if machine_tagged and recipient != m and arm != "baseline":
                        cont *= (1.0 - p.p_exploit)

                if rng_dyn.random() > cont:
                    break
                prev, current = current, recipient

    return {"coherence": _coherence(activity),
            "reach": float((activity > 0).sum()) / p.n,
            "volume": float(activity.sum())}


def marginal(p: NodeParams, arm: str, memory: str, seeds: list[int]) -> dict:
    """Machine's marginal effect = arm minus baseline, paired by seed (matched)."""
    dcoh, dreach = [], []
    for s in seeds:
        base = run_once(p, "baseline", memory, s)
        a = run_once(p, arm, memory, s)
        dcoh.append(a["coherence"] - base["coherence"])
        dreach.append(a["reach"] - base["reach"])
    dcoh = np.array(dcoh)
    dreach = np.array(dreach)
    return {"d_coh": float(dcoh.mean()), "d_coh_2se": float(2 * dcoh.std(ddof=1) / np.sqrt(len(seeds))),
            "d_reach": float(dreach.mean()), "d_reach_2se": float(2 * dreach.std(ddof=1) / np.sqrt(len(seeds)))}
