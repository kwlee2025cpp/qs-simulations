"""
run_p3_2m_recovery.py — pre-registered model-recovery study for QS PreReg 4 (P3.2-M).

The "society of LLMs" Tier-A experiment will run the Wang-Busemeyer question-order
paradigm on a panel of LLM participants and test the parameter-free QQ equality. This
script is the PRECURSOR that must pass first (PreReg 4 sec 4.6): it shows the test can
DISCRIMINATE the three candidate generative processes and finds the sample size at
which it does so reliably. No LLMs are queried here.

What it does:
  1. Print each model's TRUE QQ residual q and order-effect magnitude, confirming the
     analytic claim: q == 0 for M0 (no order) and M2 (quantum, for all parameters),
     q != 0 for M1 (classical asymmetric anchoring).
  2. Across a grid of sample sizes (respondents x item-pairs, per asking order), run
     many replicate studies per generating model and record how often the
     pre-registered 2x2 classifier recovers the generating model.
  3. Report the smallest sample size at which recovery >= 0.80 for all three models
     (the pre-registered N), and the M1-vs-M2 QQ-power curve.

Outputs (results/):
  p3_2m_recovery.csv      recovery rate per (generating model, N)
  p3_2m_recovery.png      recovery curves + the true-q bar
  p3_2m_summary.txt       verdict + pre-registered N

Deterministic given SEED.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from qq_models import (  # noqa: E402
    QQParams, MODELS, true_qq, true_order_effect, classify_study,
)

SEED = 20260628
N_GRID = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]  # resp. x item-pairs, per order
N_REPLICATES = 2000                            # simulated studies per (model, N)
RECOVERY_TARGET = 0.80
RESULTS = os.path.join(os.path.dirname(__file__), "results")


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    p = QQParams()
    rng = np.random.default_rng(SEED)

    # ---- 1. analytic ground truth -------------------------------------------------
    truth_lines = ["TRUE generative properties (analytic):",
                   f"  params: a={p.a}, b={p.b}, c_pos={p.c_pos}, c_neg={p.c_neg}, "
                   f"phi={p.phi}, theta={p.theta}",
                   "  model |   true q (QQ residual) | order-effect | QQ holds?"]
    for m in ("M0", "M1", "M2"):
        q = true_qq(m, p)
        oe = true_order_effect(m, p)
        holds = "yes" if abs(q) < 1e-9 else "NO"
        truth_lines.append(f"   {m}  | {q:+.4f}              | {oe:.4f}       | {holds}")
    truth = "\n".join(truth_lines)
    print(truth)
    # sanity: the design is only valid if the truths are as claimed
    assert abs(true_qq("M0", p)) < 1e-9, "M0 must satisfy QQ"
    assert abs(true_qq("M2", p)) < 1e-9, "M2 must satisfy QQ for all params"
    assert abs(true_qq("M1", p)) > 1e-3, "M1 must violate QQ (asymmetric anchoring)"
    assert true_order_effect("M1", p) > 1e-3 and true_order_effect("M2", p) > 1e-3, \
        "M1 and M2 must show an order effect; M0 must not"
    assert true_order_effect("M0", p) < 1e-9, "M0 must show no order effect"

    # ---- 2. recovery across sample sizes -----------------------------------------
    rows = []  # (gen_model, N, recovery_rate, p_class_M0, p_class_M1, p_class_M2)
    rec = {m: [] for m in ("M0", "M1", "M2")}
    for n in N_GRID:
        for gen in ("M0", "M1", "M2"):
            tally = {"M0": 0, "M1": 0, "M2": 0}
            for _ in range(N_REPLICATES):
                tally[classify_study(gen, p, n, rng)] += 1
            rate = tally[gen] / N_REPLICATES
            rec[gen].append(rate)
            rows.append((gen, n, rate, tally["M0"] / N_REPLICATES,
                         tally["M1"] / N_REPLICATES, tally["M2"] / N_REPLICATES))

    # ---- 3. pre-registered N ------------------------------------------------------
    pre_reg_n = None
    for i, n in enumerate(N_GRID):
        if all(rec[m][i] >= RECOVERY_TARGET for m in ("M0", "M1", "M2")):
            pre_reg_n = n
            break

    # ---- write CSV ---------------------------------------------------------------
    csv_path = os.path.join(RESULTS, "p3_2m_recovery.csv")
    with open(csv_path, "w") as fh:
        fh.write("gen_model,N_per_order,recovery_rate,classified_M0,classified_M1,classified_M2\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]:.4f},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f}\n")

    # ---- figure ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        for m, c in zip(("M0", "M1", "M2"), ("#888", "#d1495b", "#2e86ab")):
            ax1.plot(N_GRID, rec[m], "o-", color=c, label=f"gen={m}")
        ax1.axhline(RECOVERY_TARGET, ls="--", color="k", lw=0.8, label=f"{RECOVERY_TARGET:.0%} target")
        if pre_reg_n:
            ax1.axvline(pre_reg_n, ls=":", color="green", lw=1.2, label=f"pre-reg N={pre_reg_n}")
        ax1.set_xscale("log"); ax1.set_xlabel("respondents x item-pairs (per order)")
        ax1.set_ylabel("recovery rate"); ax1.set_ylim(0, 1.02)
        ax1.set_title("Model recovery via the QQ-equality 2x2 test"); ax1.legend(fontsize=8)
        qs = [true_qq(m, p) for m in ("M0", "M1", "M2")]
        ax2.bar(("M0", "M1", "M2"), qs, color=("#888", "#d1495b", "#2e86ab"))
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_ylabel("true QQ residual q"); ax2.set_title("QQ residual: 0 iff quantum/no-order")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "p3_2m_recovery.png"), dpi=130)
    except Exception as e:  # pragma: no cover
        print(f"(figure skipped: {e})")

    # ---- summary -----------------------------------------------------------------
    verdict = (f"PASS — QQ-equality test discriminates M0/M1/M2; pre-registered "
               f"N = {pre_reg_n} (respondents x item-pairs per order)."
               if pre_reg_n else
               "INCONCLUSIVE — no N in the grid reached the recovery target; widen N_GRID.")
    summary = (truth + "\n\n" +
               "RECOVERY (fraction of studies where the generating model is recovered):\n" +
               "  N(per order) | " + " | ".join(f"{n:>6}" for n in N_GRID) + "\n" +
               "\n".join("   gen=" + m + "     | " +
                         " | ".join(f"{r:6.2f}" for r in rec[m]) for m in ("M0", "M1", "M2")) +
               f"\n\nTarget recovery: {RECOVERY_TARGET:.0%} for all three models.\n" +
               verdict + "\n\n" +
               "Interpretation: this fixes the sample size for the LIVE Tier-A study and\n"
               "confirms the parameter-free QQ test can tell a quantum-like order effect\n"
               "(M2) from a classical one (M1) and from no order effect (M0). It does NOT\n"
               "claim real LLMs realize any of these — that is the live study (PreReg 4).\n")
    with open(os.path.join(RESULTS, "p3_2m_summary.txt"), "w") as fh:
        fh.write(summary)
    print("\n" + verdict)
    print(f"\nWrote: {csv_path}\n       {os.path.join(RESULTS, 'p3_2m_recovery.png')}"
          f"\n       {os.path.join(RESULTS, 'p3_2m_summary.txt')}")


if __name__ == "__main__":
    main()
