"""
qq_models.py — generative models + QQ-equality test for QS PreReg 4 (P3.2-M, Tier A).

This is the pre-registered MODEL-RECOVERY precursor for the "society of LLMs"
experiment. BEFORE any LLM is queried, we must show that the parameter-free
QQ-equality test can actually DISCRIMINATE the three candidate generative processes
of question-order effects, and find the sample size needed to separate them.
No API calls, no LLMs: pure simulation. (Respecting the LLM "participants" starts
with not wasting their time on an instrument we have not first shown can work.)

The 2x2 logic the live experiment applies to each LLM participant:

                       | QQ equality holds | QQ equality violated
   order effect absent |   M0 (no order)   |   (degenerate)
   order effect present|   M2 (quantum)    |   M1 (classical order)

  M0 — order-invariant: answers to A and B do not depend on the order asked.
       No order effect; QQ trivially holds.
  M1 — classical order-dependent (anchoring / Markov updating): the first answer
       perturbs the second through a classical rule. GENERICALLY VIOLATES the QQ
       equality (here: q = (c_pos - c_neg)(b - a), nonzero when the anchoring is
       asymmetric AND the two items have different base rates).
  M2 — quantum / projective (Wang & Busemeyer 2013; Wang et al. 2014, PNAS):
       sequential projective ("incompatible") measurement. SATISFIES the QQ
       equality EXACTLY for ALL parameters — a parameter-free invariant, so it
       cannot be rigged by fitting.

QQ equality (Wang et al. 2014): for two yes/no items asked in both orders,

   p(A=y, B=n) + p(A=n, B=y)   ==   p(B=y, A=n) + p(B=n, A=y)

i.e. the total "disagree" (one-yes-one-no) probability is the SAME in both orders.
The test statistic is the difference of those two disagree-sums; the quantum model
predicts it is 0 a priori. This module computes each model's TRUE q and samples
respondents so the runner can study the test's discriminating power.

A simulation can only show the test is SUFFICIENT to tell these processes apart; it
says nothing about which process real LLMs (or humans) actually realize — that is the
live Tier-A study. See PREREG.md (P3.2-M).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Cell order convention everywhere: (A=yes,B=yes), (A=yes,B=no), (A=no,B=yes),
# (A=no,B=no) -> keys "yy","yn","ny","nn", indexed by the A/B *answers* (not by
# measurement position), reported separately for each asking ORDER ("AB","BA").


@dataclass(frozen=True)
class QQParams:
    # --- shared base rates (used by M0 and M1) ---
    a: float = 0.60      # P(A=yes) base rate
    b: float = 0.35      # P(B=yes) base rate
    # --- M1 classical anchoring shifts (asymmetric => QQ violated) ---
    c_pos: float = 0.18  # answering the 1st item "yes" raises 2nd item's yes-prob
    c_neg: float = 0.05  # answering the 1st item "no" lowers 2nd item's yes-prob
    # --- M2 quantum geometry (radians) ---
    phi: float = 0.90    # initial-state angle
    theta: float = 0.70  # angle between the A and B measurement bases


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def cells_M0(p: QQParams, order: str) -> dict:
    """Order-invariant. Independent joint with the base rates; identical for AB/BA."""
    a, b = p.a, p.b
    return {"yy": a * b, "yn": a * (1 - b), "ny": (1 - a) * b, "nn": (1 - a) * (1 - b)}


def cells_M1(p: QQParams, order: str) -> dict:
    """Classical anchoring: the FIRST answer shifts the second item's yes-prob."""
    a, b, cp, cn = p.a, p.b, p.c_pos, p.c_neg
    if order == "AB":            # ask A first (rate a), then B shifted by A's answer
        b_if_ay = _clip01(b + cp)
        b_if_an = _clip01(b - cn)
        return {"yy": a * b_if_ay, "yn": a * (1 - b_if_ay),
                "ny": (1 - a) * b_if_an, "nn": (1 - a) * (1 - b_if_an)}
    else:                        # ask B first (rate b), then A shifted by B's answer
        a_if_by = _clip01(a + cp)
        a_if_bn = _clip01(a - cn)
        # index back to (A,B): B=yes -> {Ay,By:a_if_by*b ... }
        return {"yy": b * a_if_by, "ny": b * (1 - a_if_by),
                "yn": (1 - b) * a_if_bn, "nn": (1 - b) * (1 - a_if_bn)}


def cells_M2(p: QQParams, order: str) -> dict:
    """Quantum projective (2-D). Sequential collapse; satisfies QQ for all phi,theta."""
    phi, th = p.phi, p.theta
    cphi2, sphi2 = np.cos(phi) ** 2, np.sin(phi) ** 2
    cth2, sth2 = np.cos(th) ** 2, np.sin(th) ** 2
    cdiff2 = np.cos(th - phi) ** 2
    sdiff2 = np.sin(th - phi) ** 2
    if order == "AB":            # measure A then B
        return {"yy": cphi2 * cth2, "yn": cphi2 * sth2,
                "ny": sphi2 * sth2, "nn": sphi2 * cth2}
    else:                        # measure B then A, re-indexed to (A,B)
        return {"yy": cdiff2 * cth2, "ny": cdiff2 * sth2,
                "yn": sdiff2 * sth2, "nn": sdiff2 * cth2}


MODELS = {"M0": cells_M0, "M1": cells_M1, "M2": cells_M2}


def disagree(cells: dict) -> float:
    """Total one-yes-one-no probability = p(A=y,B=n) + p(A=n,B=y)."""
    return cells["yn"] + cells["ny"]


def true_qq(model: str, p: QQParams) -> float:
    """The exact QQ residual q = disagree(AB) - disagree(BA) for the model."""
    f = MODELS[model]
    return disagree(f(p, "AB")) - disagree(f(p, "BA"))


def true_order_effect(model: str, p: QQParams) -> float:
    """A simple order-effect magnitude: |p(B=yes | AB) - p(B=yes | BA)|."""
    f = MODELS[model]
    ab, ba = f(p, "AB"), f(p, "BA")
    pB_AB = ab["yy"] + ab["ny"]
    pB_BA = ba["yy"] + ba["ny"]
    return abs(pB_AB - pB_BA)


def _sample(cells: dict, n: int, rng) -> dict:
    probs = np.array([cells["yy"], cells["yn"], cells["ny"], cells["nn"]], float)
    probs = probs / probs.sum()
    counts = rng.multinomial(n, probs)
    return {"yy": counts[0], "yn": counts[1], "ny": counts[2], "nn": counts[3]}


_Z95 = 1.959963985


def classify_counts(cAB: dict, cBA: dict, alpha_z: float = _Z95) -> dict:
    """
    The pre-registered 2x2 classifier, computed from OBSERVED counts in each asking
    order (the same code path the live LLM study uses). cAB/cBA are dicts with integer
    keys "yy","yn","ny","nn" (A-answer, B-answer) for orders AB and BA respectively.

    Returns dict with: label ("M0"/"M1"/"M2"), q_hat (QQ residual estimate),
    z_qq, z_order, order_effect (bool), qq_violated (bool).
    """
    nAB = max(1, sum(cAB.values()))
    nBA = max(1, sum(cBA.values()))

    # order-effect test: B-yes marginal differs across orders?
    pB_AB = (cAB["yy"] + cAB["ny"]) / nAB
    pB_BA = (cBA["yy"] + cBA["ny"]) / nBA
    se_oe = np.sqrt(pB_AB * (1 - pB_AB) / nAB + pB_BA * (1 - pB_BA) / nBA) + 1e-12
    z_oe = (pB_AB - pB_BA) / se_oe
    order_effect = abs(z_oe) > alpha_z

    # QQ-equality test: disagree-sum difference != 0?
    d_AB = (cAB["yn"] + cAB["ny"]) / nAB
    d_BA = (cBA["yn"] + cBA["ny"]) / nBA
    se_qq = np.sqrt(d_AB * (1 - d_AB) / nAB + d_BA * (1 - d_BA) / nBA) + 1e-12
    z_qq = (d_AB - d_BA) / se_qq
    qq_violated = abs(z_qq) > alpha_z

    label = "M0" if not order_effect else ("M1" if qq_violated else "M2")
    return {"label": label, "q_hat": d_AB - d_BA, "z_qq": z_qq, "z_order": z_oe,
            "order_effect": order_effect, "qq_violated": qq_violated}


def classify_study(model: str, p: QQParams, n_per_order: int, rng,
                   alpha: float = 0.05) -> str:
    """
    One simulated study: sample n_per_order respondents in EACH asking order from the
    given generating model, then classify by the pre-registered 2x2 logic.
    Returns one of "M0", "M1", "M2" (the inferred process).
    """
    f = MODELS[model]
    cAB = _sample(f(p, "AB"), n_per_order, rng)
    cBA = _sample(f(p, "BA"), n_per_order, rng)
    return classify_counts(cAB, cBA)["label"]
