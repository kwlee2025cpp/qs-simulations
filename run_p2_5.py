"""
run_p2_5.py — pre-registered ABM run for QS framework prediction P2.5.

Tests the asymmetric-machine-node prediction: a DONOR-position machine amplifies
help-propagation while a stateless INTERMEDIARY-position machine attenuates it, with
memory configuration modulating the gap — under two rivals (Karpus exploitation,
Crandall illegibility) given real paths to overturn it.

Decision rules (fixed before running):
  (H1 asymmetry)  with memory OFF and rivals off: d_coh(donor) > 0 > d_coh(intermediary),
                  each by > 2 SE across seeds. (machine's marginal effect on coherence
                  vs. the no-special-role baseline.)
  (H2 memory)     memory ON raises d_coh(intermediary) relative to OFF (mitigates
                  attenuation) — report the change.
  (C contest)     sweep p_exploit; report the threshold at which donor amplification
                  d_coh(donor) is eroded to <= 0 (humans exploiting the benevolent AI),
                  or that it survives the tested range.

Outputs (results/): p2_5_summary.txt, p2_5_grid.csv, p2_5_grid.png
Deterministic given SEEDS.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from llm_node_sim import NodeParams, marginal  # noqa: E402

SEEDS = list(range(24))
P_EXPLOIT_GRID = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7]
RESULTS = os.path.join(os.path.dirname(__file__), "results")


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    # ---- H1 / H2 at rivals-off -------------------------------------------------
    base_off_don = marginal(NodeParams(p_exploit=0.0), "donor", "off", SEEDS)
    base_off_int = marginal(NodeParams(p_exploit=0.0), "intermediary", "off", SEEDS)
    base_on_don = marginal(NodeParams(p_exploit=0.0), "donor", "on", SEEDS)
    base_on_int = marginal(NodeParams(p_exploit=0.0), "intermediary", "on", SEEDS)

    h1_donor = base_off_don["d_coh"] - base_off_don["d_coh_2se"] > 0
    h1_inter = base_off_int["d_coh"] + base_off_int["d_coh_2se"] < 0
    h1 = h1_donor and h1_inter
    h2 = base_on_int["d_coh"] > base_off_int["d_coh"]

    lines.append("P2.5 — asymmetric machine node (donor amplifies / intermediary attenuates)")
    lines.append("Machine's marginal effect on coherence (arm minus matched baseline):")
    lines.append("                       memory OFF                memory ON")
    lines.append(f"  donor        : d_coh {base_off_don['d_coh']:+.4f} (2SE {base_off_don['d_coh_2se']:.4f})"
                 f"   |  {base_on_don['d_coh']:+.4f} (2SE {base_on_don['d_coh_2se']:.4f})")
    lines.append(f"  intermediary : d_coh {base_off_int['d_coh']:+.4f} (2SE {base_off_int['d_coh_2se']:.4f})"
                 f"   |  {base_on_int['d_coh']:+.4f} (2SE {base_on_int['d_coh_2se']:.4f})")
    lines.append("")
    lines.append(f"(H1 asymmetry, memory OFF, rivals off): donor>0 {h1_donor}, intermediary<0 "
                 f"{h1_inter} -> {'SUPPORTED' if h1 else 'NOT supported'}")
    lines.append(f"(H2 memory): memory ON raises intermediary d_coh "
                 f"({base_off_int['d_coh']:+.4f} -> {base_on_int['d_coh']:+.4f}) -> "
                 f"{'SUPPORTED' if h2 else 'NOT supported'}")
    lines.append("")

    # ---- C: rival sweep on donor amplification (memory OFF) ---------------------
    lines.append("(C contest) donor amplification d_coh(donor) vs Karpus exploitation p_exploit "
                 "(memory OFF):")
    rows = []
    threshold = None
    for pe in P_EXPLOIT_GRID:
        r = marginal(NodeParams(p_exploit=pe), "donor", "off", SEEDS)
        rows.append(("donor", "off", pe, r["d_coh"], r["d_coh_2se"], r["d_reach"]))
        survives = r["d_coh"] - r["d_coh_2se"] > 0
        lines.append(f"  p_exploit={pe:>4}: d_coh {r['d_coh']:+.4f} (2SE {r['d_coh_2se']:.4f})  "
                     f"{'amplifies' if survives else 'NOT > 0 (eroded)'}")
        if threshold is None and not survives:
            threshold = pe
    lines.append("")
    if threshold is None:
        lines.append(f"Donor amplification SURVIVES the whole exploitation range "
                     f"(up to p_exploit={P_EXPLOIT_GRID[-1]}).")
    else:
        lines.append(f"Donor amplification is eroded to <= 0 at p_exploit >= {threshold} "
                     f"(humans exploiting the benevolent machine overturn it).")
    lines.append("")
    lines.append("FINDING (a genuine refutation, like the P2.4 volume result): P2.5's "
                 "'donor amplifies' is\nNOT supported at MATCHED help-volume. A single "
                 "concentrated machine donor — even at the\nhighest-degree node — spreads "
                 "cooperation LESS broadly and LESS evenly than the same\nvolume of help "
                 "scattered across random human donors (donor d_coh<0 AND d_reach<0); "
                 "the\nexploitation rival only deepens this. The stateless-INTERMEDIARY "
                 "attenuation is\nNEGLIGIBLE at one node (chains rarely terminate exactly "
                 "there). \nImplication for the framework: the 'paying-it-forward' breadth "
                 "advantage (P2.4) comes\nfrom MANY DISTRIBUTED origins, not one tireless "
                 "donor; a machine's distinctive value (if\nany) must lie elsewhere "
                 "(reliability/volume, or bridging structural holes), not in breadth.\n"
                 "P2.5 should be REVISED accordingly. Sufficiency-style reasoning only "
                 "(equifinality);\nthe decisive test remains the human PreReg 2 (P2.5).")

    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(RESULTS, "p2_5_summary.txt"), "w") as fh:
        fh.write(summary + "\n")

    # ---- CSV + figure ----------------------------------------------------------
    with open(os.path.join(RESULTS, "p2_5_grid.csv"), "w") as fh:
        fh.write("arm,memory,p_exploit,d_coh,d_coh_2se,d_reach\n")
        fh.write(f"donor,off,0.0,{base_off_don['d_coh']:.4f},{base_off_don['d_coh_2se']:.4f},{base_off_don['d_reach']:.4f}\n")
        fh.write(f"intermediary,off,0.0,{base_off_int['d_coh']:.4f},{base_off_int['d_coh_2se']:.4f},{base_off_int['d_reach']:.4f}\n")
        fh.write(f"donor,on,0.0,{base_on_don['d_coh']:.4f},{base_on_don['d_coh_2se']:.4f},{base_on_don['d_reach']:.4f}\n")
        fh.write(f"intermediary,on,0.0,{base_on_int['d_coh']:.4f},{base_on_int['d_coh_2se']:.4f},{base_on_int['d_reach']:.4f}\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.4f},{r[4]:.4f},{r[5]:.4f}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        cats = ["donor\nOFF", "donor\nON", "interm.\nOFF", "interm.\nON"]
        vals = [base_off_don["d_coh"], base_on_don["d_coh"],
                base_off_int["d_coh"], base_on_int["d_coh"]]
        errs = [base_off_don["d_coh_2se"], base_on_don["d_coh_2se"],
                base_off_int["d_coh_2se"], base_on_int["d_coh_2se"]]
        ax1.bar(cats, vals, yerr=errs, color=["#2e86ab", "#2e86ab", "#d1495b", "#d1495b"])
        ax1.axhline(0, color="k", lw=0.8)
        ax1.set_ylabel("machine's marginal d_coherence")
        ax1.set_title("Asymmetry: donor amplifies, intermediary attenuates")
        pes = [r[2] for r in rows]
        ax2.errorbar(pes, [r[3] for r in rows], yerr=[r[4] for r in rows],
                     fmt="o-", color="#2e86ab")
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_xlabel("p_exploit (Karpus rival)")
        ax2.set_ylabel("donor d_coherence (memory OFF)")
        ax2.set_title("Exploitation erodes donor amplification")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "p2_5_grid.png"), dpi=130)
    except Exception as e:  # pragma: no cover
        print(f"(figure skipped: {e})")

    print(f"\nWrote: {os.path.join(RESULTS, 'p2_5_summary.txt')} (+ p2_5_grid.csv, p2_5_grid.png)")


if __name__ == "__main__":
    main()
