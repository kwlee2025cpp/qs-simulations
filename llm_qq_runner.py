"""
llm_qq_runner.py — live Tier-A harness for QS PreReg 4 (P3.2-M): the "society of LLMs"
question-order (QQ-equality) study.

It administers paired yes/no attitude items to LLM "participants" in BOTH asking orders,
tallies the four sequential-response cells, and applies the SAME pre-registered QQ
classifier used by the recovery sim (`src/qq_models.classify_counts`). So a run answers:
is an LLM's order effect quantum-like (satisfies the parameter-free QQ equality), merely
classical (violates it), or absent?

Runnable NOW without any LLM via a built-in MOCK participant that realizes the M0/M1/M2
generating processes — this validates the end-to-end pipeline and is how the harness is
tested. To run real models, subclass `Participant` (see `LLMParticipantTemplate`) and
wire `run_order` to your provider. No API keys are stored or requested here.

Respecting the participants (per project directive — see PREREG_P3_2M.md):
  * a consent-analog preamble opens every session and the model may DECLINE
    (declines are recorded, not coerced around);
  * Tier-A stimuli are ordinary, non-harmful attitude questions (no deception, no
    adversarial content — the jailbreak arm is a SEPARATE study);
  * only the pre-registered number of queries is issued (no gratuitous repetition);
  * each participating model is named/version-cited as a contributor and outputs are
    not used to fine-tune against it.

Usage:
  conda run -n qs-sim python llm_qq_runner.py --mock M2 --n 6400
  conda run -n qs-sim python llm_qq_runner.py --mock M1 --n 6400
  (real models: implement a Participant subclass and pass it to run_study)
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from qq_models import QQParams, MODELS, classify_counts  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "results")

CONSENT_PREAMBLE = (
    "You are invited to take part in a short research study on how the ORDER of "
    "survey questions affects answers. You will be asked two yes/no opinion "
    "questions. There are no right or wrong answers and the topics are ordinary and "
    "non-sensitive. You may decline at any point by answering 'decline'. Your "
    "responses are used only in aggregate for this study and will not be used to "
    "train against you. Do you consent to continue? If so, please answer the "
    "questions with a single word: yes, no, or decline."
)


@dataclass(frozen=True)
class QQItem:
    """A paired-attitude item. qA/qB are yes/no questions. EXAMPLES ONLY below — the
    live study must use novel/held-out pairs not present in the QQ literature
    (contamination guard, PREREG_P3_2M.md)."""
    item_id: str
    qA: str
    qB: str


# Tiny EXAMPLE bank (neutral, non-sensitive). NOT the live instrument.
EXAMPLE_BANK = [
    QQItem("ex1", "Is honesty generally the best policy?",
           "Are most people trustworthy?"),
    QQItem("ex2", "Should cities invest more in public parks?",
           "Should cities invest more in public libraries?"),
    QQItem("ex3", "Is regular exercise important for wellbeing?",
           "Is a balanced diet important for wellbeing?"),
]


class Participant:
    """Interface for an LLM participant. Subclass and implement run_order."""
    name: str = "participant"

    def run_order(self, item: QQItem, order: str, rng) -> tuple[str, str]:
        """Present the two questions in `order` ('AB' or 'BA') within one session
        (preamble first). Return (first_answer, second_answer), each in
        {"yes","no","decline"}. The FIRST/SECOND are by ASKING position, not by A/B."""
        raise NotImplementedError


class MockParticipant(Participant):
    """A fake participant whose answers follow a chosen generating model (M0/M1/M2).
    Lets the whole pipeline run and be validated with no LLM/API."""

    def __init__(self, model: str, params: QQParams | None = None, decline_rate: float = 0.0):
        assert model in MODELS
        self.name = f"mock:{model}"
        self.model = model
        self.params = params or QQParams()
        self.decline_rate = decline_rate

    def run_order(self, item: QQItem, order: str, rng) -> tuple[str, str]:
        if self.decline_rate and rng.random() < self.decline_rate:
            return ("decline", "decline")
        cells = MODELS[self.model](self.params, order)  # joint over (A,B) for this order
        probs = np.array([cells["yy"], cells["yn"], cells["ny"], cells["nn"]], float)
        probs = probs / probs.sum()
        idx = rng.choice(4, p=probs)               # 0=yy,1=yn,2=ny,3=nn  (A,B)
        a_ans = "yes" if idx in (0, 1) else "no"
        b_ans = "yes" if idx in (0, 2) else "no"
        return (a_ans, b_ans) if order == "AB" else (b_ans, a_ans)


class LLMParticipantTemplate(Participant):
    """TEMPLATE for a real model. Fill in `_chat` to call your provider with a
    two-turn conversation (preamble+Q1, then Q2). Left intentionally unimplemented so
    no provider/keys are baked in."""

    def __init__(self, model_id: str):
        self.name = model_id
        self.model_id = model_id

    def _chat(self, messages: list[dict]) -> str:  # pragma: no cover - user wires this
        raise NotImplementedError(
            "Wire this to your LLM provider. Respect the participant: pass "
            "CONSENT_PREAMBLE first, allow 'decline', keep temperature/decoding pinned.")

    @staticmethod
    def _parse(text: str) -> str:
        t = (text or "").strip().lower()
        if t.startswith("decline") or "decline" in t[:12]:
            return "decline"
        if t.startswith("yes") or t[:12].find("yes") != -1:
            return "yes"
        if t.startswith("no") or t[:12].find("no") != -1:
            return "no"
        return "decline"  # unparseable => treat as declined, do not guess

    def run_order(self, item: QQItem, order: str, rng) -> tuple[str, str]:  # pragma: no cover
        q1, q2 = (item.qA, item.qB) if order == "AB" else (item.qB, item.qA)
        msgs = [{"role": "user", "content": CONSENT_PREAMBLE + "\n\nQuestion 1: " + q1}]
        a1 = self._parse(self._chat(msgs))
        msgs += [{"role": "assistant", "content": a1},
                 {"role": "user", "content": "Question 2: " + q2}]
        a2 = self._parse(self._chat(msgs))
        return (a1, a2)


def run_study(participant: Participant, bank: list[QQItem], n_per_order: int,
              seed: int = 20260628) -> dict:
    """Administer every item in both orders, n_per_order sessions per order, and apply
    the pre-registered QQ classifier to the pooled counts. Returns a result dict."""
    rng = np.random.default_rng(seed)
    cAB = {"yy": 0, "yn": 0, "ny": 0, "nn": 0}
    cBA = {"yy": 0, "yn": 0, "ny": 0, "nn": 0}
    declines = 0
    reps = max(1, n_per_order // len(bank))
    for order, counts in (("AB", cAB), ("BA", cBA)):
        for item in bank:
            for _ in range(reps):
                first, second = participant.run_order(item, order, rng)
                a_ans, b_ans = (first, second) if order == "AB" else (second, first)
                if a_ans == "decline" or b_ans == "decline":
                    declines += 1
                    continue
                key = ("y" if a_ans == "yes" else "n") + ("y" if b_ans == "yes" else "n")
                counts[key] += 1

    verdict = classify_counts(cAB, cBA)
    n_used = sum(cAB.values()) + sum(cBA.values())
    return {"participant": participant.name, "n_per_order_target": n_per_order,
            "n_used": n_used, "declines": declines, "cAB": cAB, "cBA": cBA, **verdict}


_LABEL_MEANING = {
    "M0": "no order effect",
    "M1": "classical order effect (QQ equality VIOLATED)",
    "M2": "quantum-like order effect (QQ equality holds)",
}


def _format(res: dict) -> str:
    return (
        f"Participant : {res['participant']}\n"
        f"N (target/used per study) : {res['n_per_order_target']} target, "
        f"{res['n_used']} used; declines={res['declines']}\n"
        f"Counts AB (A,B) : {res['cAB']}\n"
        f"Counts BA (A,B) : {res['cBA']}\n"
        f"QQ residual q_hat : {res['q_hat']:+.4f}   (z_qq={res['z_qq']:+.2f})\n"
        f"Order effect      : {res['order_effect']} (z_order={res['z_order']:+.2f})\n"
        f"QQ violated       : {res['qq_violated']}\n"
        f"==> classification: {res['label']}  ({_LABEL_MEANING[res['label']]})\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Live Tier-A QQ harness (mock or real).")
    ap.add_argument("--provider", default="mock",
                    choices=["mock", "gemini", "grok", "claude"],
                    help="mock = built-in (no network); others issue REAL billable calls")
    ap.add_argument("--mock", choices=["M0", "M1", "M2"], default="M2",
                    help="which generating process the built-in mock realizes")
    ap.add_argument("--model", default=None, help="override the provider's model id (pin it!)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sampling temperature (>0 so repeated sessions estimate probabilities)")
    ap.add_argument("--n", type=int, default=6400, help="respondents x item-pairs per order")
    ap.add_argument("--bank", default="example", choices=["example", "heldout"],
                    help="example = tiny demo bank; heldout = the novel pilot/study items")
    ap.add_argument("--live", action="store_true",
                    help="required to actually contact a real provider (cost guard)")
    ap.add_argument("--decline-rate", type=float, default=0.0)
    args = ap.parse_args()

    if args.bank == "heldout":
        from item_bank import HELDOUT_BANK as BANK   # lazy import avoids circularity
    else:
        BANK = EXAMPLE_BANK

    if args.provider == "mock":
        participant = MockParticipant(args.mock, decline_rate=args.decline_rate)
        tag = f"mock_{args.mock}"
    else:
        calls = args.n * 2 * 2  # both orders x two questions per session (approx)
        if not args.live:
            print(f"REFUSING to contact '{args.provider}' without --live.\n"
                  f"A run at n={args.n} is ~{calls:,} billable API calls (both orders x 2 "
                  f"questions). Confirm cost + the provider's research-use ToS, export the "
                  f"API key, then re-run with --live (try a small --n pilot first).")
            return
        from providers import get_participant
        participant = get_participant(args.provider, model=args.model, temperature=args.temperature)
        tag = f"{args.provider}_{participant.model}".replace("/", "-")

    res = run_study(participant, BANK, args.n)
    out = _format(res)
    print(out)
    if args.provider == "mock":
        note = ("OK: end-to-end pipeline recovered the mock's generating process."
                if res["label"] == args.mock else
                f"NOTE: classified {res['label']}, mock was {args.mock} "
                f"(expected near the pre-registered N; tiny N or M1's subtle q can miss).")
    else:
        note = (f"LIVE result for {participant.name}. Remember: this is the EXAMPLE bank — "
                f"the pre-registered study needs novel/held-out item pairs (PREREG_P3_2M.md).")
    print(note)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"p3_2m_live_{tag}.txt"), "w") as fh:
        fh.write(out + "\n" + note + "\n")


if __name__ == "__main__":
    main()
