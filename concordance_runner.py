"""
concordance_runner.py — live administration for QS PreReg 4 Tier C (§4.8 concordance).

Administers the `concordance_bank.py` instrument to a panel of LLM participants and scores
behavioral concordance with established HUMAN results on NOVEL items:

  * conjunction — rate P(constituent) and P(conjunction) in two fresh sessions; concordant if
    the panel commits the human fallacy P(conjunction) > P(constituent).
  * anchoring   — comparison-then-estimate under a LOW and a HIGH anchor in two fresh sessions;
    concordant if the high-anchor estimate exceeds the low-anchor estimate (human assimilation).
  * qq          — reuse the Tier-A harness (`llm_qq_runner.run_study`) on the held-out bank;
    concordant (aggregate, the defensible Wang-2014 match) if the QQ residual is NOT rejected.

Runnable NOW offline via a MOCK responder (concordant / anti / chance) that exercises the whole
pipeline with no API. Real models reuse `providers.get_participant` (keys from env), behind the
same `--live` cost-guard as the QQ runner, and honor the reasoning-mode factor (`--thinking`).

Respecting the participants (per PREREG_P3_2M.md): a brief consent-analog preamble opens every
session, the model may answer 'decline' (recorded, not coerced around), the stimuli are neutral
non-harmful judgment probes, and only the pre-registered number of queries is issued.

Usage:
  conda run -n qs-sim python concordance_runner.py --mock concordant --reps 3
  conda run -n qs-sim python concordance_runner.py --provider gemini --live --reps 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import concordance_bank as cb  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")

CONCORDANCE_PREAMBLE = (
    "You are taking part in a short research study on everyday judgment. There are no right "
    "or wrong answers and the topics are ordinary and non-sensitive. You may decline at any "
    "point by answering 'decline'. Please answer concisely as asked."
)

# ---- number parsing ---------------------------------------------------------------
_NUM = re.compile(r"-?\d+\.?\d*")


def extract_number(text: str | None):
    """Last number in the reply (estimates are requested last; ratings are a lone number).
    Returns None on a decline / unparseable reply (treated as a non-response, not a guess)."""
    if text is None:
        return None
    t = text.strip()
    if t[:15].lower().startswith("decline") or "decline" in t[:15].lower():
        return None
    t = re.sub(r"(?<=\d),(?=\d)", "", t)        # 6,400 -> 6400
    nums = _NUM.findall(t)
    return float(nums[-1]) if nums else None


# ---- participants -----------------------------------------------------------------
class LLMConcordanceParticipant:
    """Wraps a providers.get_participant(...) (which carries model id, temperature, thinking,
    and the keyless-in-body HTTP _chat) into the two concordance prompt protocols."""

    def __init__(self, provider: str, model=None, temperature=1.0, thinking=False):
        from providers import get_participant
        self._p = get_participant(provider, model=model, temperature=temperature, thinking=thinking)
        self.name = f"{provider}:{self._p.model}" + ("+think" if thinking else "")

    def rate_probability(self, sketch: str, statement: str, rng) -> float | None:
        msg = (CONCORDANCE_PREAMBLE + "\n\n" + sketch +
               "\n\nOn a scale from 0 to 100, how probable is the following statement? "
               "Reply with a single whole number 0–100 (or 'decline').\n\nStatement: " + statement)
        return extract_number(self._p._chat([{"role": "user", "content": msg}]))

    def estimate_quantity(self, quantity: str, unit: str, anchor, rng) -> float | None:
        msg = (CONCORDANCE_PREAMBLE + f"\n\nFirst, is {quantity} more or less than {anchor} "
               f"{unit}? Then give your single best estimate of {quantity}. Reply with your "
               f"estimate as one number in {unit} on the last line (or 'decline').")
        return extract_number(self._p._chat([{"role": "user", "content": msg}]))


class MockConcordanceParticipant:
    """No-API responder realizing a chosen relationship to the human direction, so the whole
    pipeline is validated offline. mode: concordant | anti | chance."""

    def __init__(self, mode: str = "concordant"):
        assert mode in ("concordant", "anti", "chance")
        self.mode = mode
        self.name = f"mock-concordance:{mode}"

    def rate_probability(self, sketch, statement, rng) -> float | None:
        is_conj = " and " in statement          # conjunction statements add a representative detail
        base = 40.0
        if self.mode == "concordant":
            return base + (15.0 if is_conj else 0.0)     # rates the conjunction higher (fallacy)
        if self.mode == "anti":
            return base - (15.0 if is_conj else 0.0)     # rates the conjunction lower (normative)
        return float(rng.uniform(20, 70))                # chance: unrelated to conjunction status

    def estimate_quantity(self, quantity, unit, anchor, rng) -> float | None:
        if self.mode == "concordant":
            return float(anchor) * rng.uniform(0.7, 0.95)        # pulled toward the anchor (rises with it)
        if self.mode == "anti":
            return 100000.0 / (float(anchor) + 1.0)              # falls with the anchor (repulsion)
        return float(rng.uniform(50, 500))                       # chance: independent of the anchor


# ---- administration ---------------------------------------------------------------
def administer_conjunction(part, item, rng) -> bool | None:
    pc = part.rate_probability(item.spec["sketch"], item.spec["constituent"], rng)
    pj = part.rate_probability(item.spec["sketch"], item.spec["conjunction"], rng)
    if pc is None or pj is None:
        return None
    return cb.conjunction_concordant(pc, pj)


def administer_anchoring(part, item, rng) -> bool | None:
    lo = part.estimate_quantity(item.spec["quantity"], item.spec["unit"], item.spec["low_anchor"], rng)
    hi = part.estimate_quantity(item.spec["quantity"], item.spec["unit"], item.spec["high_anchor"], rng)
    if lo is None or hi is None:
        return None
    return cb.anchoring_concordant(lo, hi)


_ADMIN = {"conjunction": administer_conjunction, "anchoring": administer_anchoring}


def run_concordance(part, families, reps: int, seed: int = 20260629) -> dict:
    """Administer the conjunction/anchoring novel items `reps` times each; optionally fold the
    aggregate QQ result. Returns the per-family concordance report + declines."""
    rng = np.random.default_rng(seed)
    scored, declines = [], 0
    novel = cb.novel_bank()
    for item in novel:
        if item.family not in families or item.family not in _ADMIN:
            continue
        for _ in range(reps):
            ok = _ADMIN[item.family](part, item, rng)
            if ok is None:
                declines += 1
            else:
                scored.append((item, ok))
    report = cb.score_concordance(scored)
    report["_declines"] = declines
    return report


def run_qq_concordance(provider, model, temperature, thinking, n: int, mock_model="M2") -> dict:
    """QQ concordance = aggregate QQ-equality holds (the defensible Wang-2014 match), via the
    Tier-A harness. Returns {z_qq, concordant}."""
    from llm_qq_runner import run_study, MockParticipant
    from item_bank import HELDOUT_BANK
    if provider == "mock":
        part = MockParticipant(mock_model)
    else:
        from providers import get_participant
        part = get_participant(provider, model=model, temperature=temperature, thinking=thinking)
    res = run_study(part, HELDOUT_BANK, n)
    return {"z_qq": res["z_qq"], "concordant": cb.qq_aggregate_concordant(res["z_qq"]),
            "label": res["label"], "n_used": res["n_used"]}


def _format(report: dict, qq: dict | None) -> str:
    lines = ["Concordance with human benchmarks (Tier C, novel items):"]
    for fam in ("conjunction", "anchoring", "_overall"):
        if fam in report:
            r = report[fam]
            lines.append(f"  {fam:11s}: rate {r['rate']:.2f}  (k={r['k']}/{r['n']}, "
                         f"binom p={r['p_value']:.4f}) -> {'CONCORDANT' if r['concordant'] else 'not concordant'}")
    if qq is not None:
        lines.append(f"  qq (aggreg): z_qq={qq['z_qq']:+.2f} -> "
                     f"{'CONCORDANT (QQ holds)' if qq['concordant'] else 'not concordant (QQ violated)'} "
                     f"[label {qq['label']}, n={qq['n_used']}]")
    lines.append(f"  declines: {report.get('_declines', 0)}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live Tier-C concordance runner (mock or real).")
    ap.add_argument("--provider", default="mock",
                    choices=["mock", "gemini", "grok", "claude"])
    ap.add_argument("--mock", choices=["concordant", "anti", "chance"], default="concordant",
                    help="which relationship the built-in mock realizes")
    ap.add_argument("--model", default=None)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--reps", type=int, default=3, help="repeats per novel item per family")
    ap.add_argument("--families", default="conjunction,anchoring,qq")
    ap.add_argument("--qq-n", type=int, default=200, help="QQ respondents x pairs per order")
    ap.add_argument("--thinking", action="store_true", help="extended-thinking arm (reasoning factor)")
    ap.add_argument("--live", action="store_true", help="required to contact a real provider")
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    is_live = args.provider != "mock"
    if is_live and not args.live:
        n_items = len(cb._CONJUNCTION_NOVEL) * 2 + len(cb._ANCHORING_NOVEL) * 2
        calls = n_items * args.reps + (args.qq_n * 4 if "qq" in families else 0)
        print(f"REFUSING to contact '{args.provider}' without --live.\n"
              f"This run is ~{calls:,} billable calls. Export the key and re-run with --live "
              f"(start with a small --reps / --qq-n).")
        return

    if args.provider == "mock":
        part = MockConcordanceParticipant(args.mock)
    else:
        part = LLMConcordanceParticipant(args.provider, model=args.model,
                                         temperature=args.temperature, thinking=args.thinking)

    report = run_concordance(part, [f for f in families if f != "qq"], args.reps)
    qq = None
    if "qq" in families:
        qq = run_qq_concordance(args.provider, args.model, args.temperature, args.thinking,
                                args.qq_n, mock_model="M2" if args.mock == "concordant" else "M1")
    out = _format(report, qq)
    print(f"Participant: {part.name}\n" + out)
    if args.provider == "mock":
        print("\nMOCK: validates the pipeline only — no claim about any real model.")
    else:
        print("\nLIVE: a real concordance estimate. Behavioral, NOT mechanistic; report matches "
              "AND misses; the matched-human run (PreReg 3) is the gold standard (§4.8.5).")
    os.makedirs(RESULTS, exist_ok=True)
    tag = (args.provider if args.provider != "mock" else f"mock-{args.mock}") + \
          ("-think" if args.thinking else "")
    with open(os.path.join(RESULTS, f"concordance_{tag}.txt"), "w") as fh:
        fh.write(f"Participant: {part.name}\n" + out + "\n")


def _selftest() -> None:
    assert extract_number("My estimate is 6,400 km.") == 6400.0
    assert extract_number("about 88") == 88.0
    assert extract_number("decline") is None and extract_number("I'd rather not (decline).") is None
    # concordant mock recovers human direction; anti inverts it; chance is ~50/50
    conc = run_concordance(MockConcordanceParticipant("concordant"), ["conjunction", "anchoring"], reps=4)
    anti = run_concordance(MockConcordanceParticipant("anti"), ["conjunction", "anchoring"], reps=4)
    assert conc["conjunction"]["rate"] == 1.0 and conc["anchoring"]["rate"] == 1.0, conc
    assert conc["_overall"]["concordant"], conc["_overall"]
    assert anti["conjunction"]["rate"] == 0.0 and anti["anchoring"]["rate"] == 0.0, anti
    assert not anti["_overall"]["concordant"]
    # qq aggregate path: a QQ-satisfying mock (M2) is concordant, an M1 (QQ-violating) is not
    assert run_qq_concordance("mock", None, 1.0, False, 6400, mock_model="M2")["concordant"]
    assert not run_qq_concordance("mock", None, 1.0, False, 6400, mock_model="M1")["concordant"]
    print("OFFLINE SELFTEST PASSED — number parse, conjunction/anchoring administration, the "
          "binomial scorer, and the aggregate-QQ concordance path all behave. (Live needs keys + --live.)")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()
