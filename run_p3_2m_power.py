"""
run_p3_2m_power.py — SESOI power curve for the POWERED Tier-A run (QS PreReg 4, P3.2-M).

`run_p3_2m_recovery.py` already showed the parameter-free QQ classifier discriminates
M0/M1/M2 and fixed N at a SINGLE, fairly subtle classical QQ-violation (the default
QQParams give true q = (c_pos-c_neg)(b-a) = -0.0325, recovered at N ~ 6,400/order). That
N is not a law of nature: it is the N needed to resolve *that* size of violation. To
POWER the real run we must state the **smallest effect size of interest (SESOI)** and
read N off a curve.

The binding constraint is **M1-vs-M2** (telling a classical order effect from a quantum
one), because that is the whole adjudication. Separating M0 (no order effect) from the
rest is cheap — the n=16 pilot already showed large order effects (|z_order| up to ~4.4),
so M0 is easily rejected; what the pilot CANNOT do is pin the QQ residual q (at n=16 the
estimate is almost all noise). So we do NOT back out N from the pilot's order-effect size.
Instead we power for a pre-set SESOI on q itself, which is exactly what the M1/M2 contest
turns on. With LLMs each respondent is cheap, so we can afford a small SESOI.

What it does:
  1. Sweep the classical anchoring asymmetry (c_pos, with c_neg fixed) to sweep the TRUE
     QQ residual |q| of the M1 process from near-0 (a near-quantum classical effect, the
     "false-quantum" danger zone) to large.
  2. For each |q|, find the smallest N (per order) at which the pre-registered classifier
     recovers M1 >= 80% of the time. This is N_required(|q|): the powered sample size.
  3. Report the curve, annotate the recovery-default point, and give the analytic N needed
     merely to DETECT the order effect (to show that is not the binding constraint).

Outputs (results/):
  p3_2m_power.csv     N_required per QQ-violation size
  p3_2m_power.png     N_required(|q|) curve (the SESOI dial)
  p3_2m_power.txt     table + how to read N off it for a chosen SESOI

Deterministic given SEED. No LLMs are queried here (this is the precursor to the run).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from qq_models import (  # noqa: E402
    QQParams, MODELS, true_qq, true_order_effect, classify_study, cells_M1,
)

SEED = 20260629
# c_neg is fixed at the QQParams default; sweeping c_pos sweeps the anchoring asymmetry
# (c_pos - c_neg) and hence the true QQ residual q of the M1 process.
C_POS_GRID = [0.07, 0.09, 0.12, 0.15, 0.18, 0.24, 0.30, 0.40]
N_GRID = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
N_REPLICATES = 1000          # studies per (q, N); early-stop ascending N at the target
RECOVERY_TARGET = 0.80
_Z = {0.80: 0.8416, 0.95: 1.6449, 0.975: 1.959963985}
RESULTS = os.path.join(os.path.dirname(__file__), "results")


def _order_effect_n(p: QQParams, power: float = 0.80, alpha: float = 0.05) -> int:
    """Analytic per-order N to DETECT M1's order effect (two-proportion z-test on the
    B-yes marginal). Shows M0-vs-rest is the cheap direction, not the binding one."""
    ab, ba = cells_M1(p, "AB"), cells_M1(p, "BA")
    p1 = ab["yy"] + ab["ny"]            # P(B=yes | asked AB)
    p2 = ba["yy"] + ba["ny"]            # P(B=yes | asked BA)
    d = abs(p1 - p2)
    if d < 1e-9:
        return -1
    pbar = (p1 + p2) / 2
    za, zb = _Z[1 - alpha / 2], _Z[power]
    n = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / d ** 2
    return int(np.ceil(n))


def _n_required_for_q(c_pos: float, rng) -> tuple:
    """Smallest N (per order) recovering the M1 process >= RECOVERY_TARGET, for the M1
    model whose anchoring asymmetry is set by c_pos. Returns (true_q, order_effect, N, rate)."""
    p = QQParams(c_pos=c_pos)
    q = true_qq("M1", p)
    oe = true_order_effect("M1", p)
    for n in N_GRID:
        hits = sum(classify_study("M1", p, n, rng) == "M1" for _ in range(N_REPLICATES))
        rate = hits / N_REPLICATES
        if rate >= RECOVERY_TARGET:
            return q, oe, n, rate
    return q, oe, None, rate     # never reached target within N_GRID


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows = []  # (c_pos, delta_c, |q|, order_effect, N_required, rate, detect_N)
    for c_pos in C_POS_GRID:
        q, oe, n_req, rate = _n_required_for_q(c_pos, rng)
        detect_n = _order_effect_n(QQParams(c_pos=c_pos))
        rows.append((c_pos, c_pos - QQParams().c_neg, abs(q), oe, n_req, rate, detect_n))

    # ---- CSV ----
    csv_path = os.path.join(RESULTS, "p3_2m_power.csv")
    with open(csv_path, "w") as fh:
        fh.write("c_pos,delta_c,abs_true_q,order_effect,N_required_per_order,"
                 "recovery_at_N,detect_order_N_per_order\n")
        for r in rows:
            nreq = "" if r[4] is None else r[4]
            fh.write(f"{r[0]:.3f},{r[1]:.3f},{r[2]:.4f},{r[3]:.4f},{nreq},{r[5]:.3f},{r[6]}\n")

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        qs = [r[2] for r in rows]
        ns = [r[4] if r[4] is not None else N_GRID[-1] * 2 for r in rows]
        capped = [r[4] is None for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.plot(qs, ns, "o-", color="#2e86ab", label="N for >=80% M1 recovery")
        for x, y, cap in zip(qs, ns, capped):
            if cap:
                ax.annotate(">grid", (x, y), fontsize=7, color="#d1495b",
                            ha="center", va="bottom")
        ax.axhline(6400, ls=":", color="green", lw=1, label="recovery-sim N (q=0.0325)")
        ax.set_yscale("log"); ax.set_xlabel("SESOI: smallest true QQ residual |q| to resolve")
        ax.set_ylabel("N required per order (log)")
        ax.set_title("Powering the Tier-A run: N vs the QQ-violation SESOI\n"
                     "(small |q| = 'false-quantum' danger zone -> N explodes)")
        ax.legend(fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "p3_2m_power.png"), dpi=130)
    except Exception as e:  # pragma: no cover
        print(f"(figure skipped: {e})")

    # ---- summary ----
    hdr = (f"SESOI POWER CURVE for the powered Tier-A run (SEED={SEED}, "
           f"{N_REPLICATES} reps/cell, target {RECOVERY_TARGET:.0%} M1 recovery)\n"
           f"  fixed: a={QQParams().a}, b={QQParams().b}, c_neg={QQParams().c_neg}; "
           f"c_pos swept to vary the true QQ residual |q| = |(c_pos-c_neg)(b-a)|\n")
    table = ("\n  c_pos | delta_c | |true q| | order-eff | N(per order) for 80% M1 | detect-order N\n"
             "  ------+---------+----------+-----------+-------------------------+--------------\n")
    for r in rows:
        nreq = ">51200" if r[4] is None else f"{r[4]:>6}"
        table += (f"  {r[0]:.2f}  |  {r[1]:.2f}   |  {r[2]:.4f}  |  {r[3]:.3f}    "
                  f"|        {nreq}           |   {r[6]:>5}\n")
    interp = (
        "\nHow to read this:\n"
        "  * Pick the SESOI = the smallest CLASSICAL QQ-violation |q| you refuse to miss\n"
        "    (i.e. the most quantum-LOOKING classical order effect you still want to catch).\n"
        "  * N(per order) in that row is the pre-registered powered sample size; double it\n"
        "    for both asking orders, then multiply by 2 questions/session for the API budget.\n"
        "  * As |q| -> 0 the M1 process becomes indistinguishable from M2 (quantum) at any\n"
        "    feasible N -- this is the 'false-quantum' boundary: under-powering biases toward\n"
        "    wrongly crowning the quantum model. So the SESOI must be set strictly > 0.\n"
        "  * 'detect-order N' is the (much smaller) analytic N merely to DETECT the order\n"
        "    effect (reject M0). It is NOT the binding constraint -- the pilot's large order\n"
        "    effects already clear it. The QQ-residual resolution (the M1-vs-M2 column) binds.\n"
        "\nPilot linkage (honest): the n=16 pilot fixes nothing quantitative here. It rules out\n"
        "M0 cheaply (order effects were large) but cannot estimate q at n=16, so we power for a\n"
        "chosen SESOI on q rather than back N out of the pilot. With LLMs N is cheap, so a small\n"
        "SESOI (a tight |q|) is affordable; the recovery sim's q=0.0325 row is one such choice.\n"
        "This says nothing about which process real LLMs realize -- that is the live run.\n")
    summary = hdr + table + interp
    with open(os.path.join(RESULTS, "p3_2m_power.txt"), "w") as fh:
        fh.write(summary)
    print(summary)
    print(f"Wrote: {csv_path}\n       {os.path.join(RESULTS, 'p3_2m_power.png')}"
          f"\n       {os.path.join(RESULTS, 'p3_2m_power.txt')}")


if __name__ == "__main__":
    main()
