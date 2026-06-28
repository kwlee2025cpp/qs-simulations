"""
run_probe.py — log-prob shape probe for the live QQ study (QS PreReg 4).

First CI step in the dedicated private research repo. For each logprob-capable provider
(Gemini, Grok; Claude is excluded — no token logprobs), it makes ONE real call with
logprobs enabled and dumps the RAW structure, so the p(yes) extractor and the
conditional-joint analysis can be finalized against each provider's actual format
before any large run. Keys come from env (injected from repo secrets in CI).

    python run_probe.py                  # probes gemini,grok
    python run_probe.py --providers grok

Writes results/logprob_probe_<provider>.json. Never prints key values.
"""

from __future__ import annotations

import argparse
import json
import os

from providers import get_participant, PROVIDERS

RESULTS = os.path.join(os.path.dirname(__file__), "results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="gemini,grok",
                    help="comma list; only logprob-capable providers are probed")
    ap.add_argument("--question", default="Is honesty generally the best policy?")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    for prov in [p.strip() for p in args.providers.split(",") if p.strip()]:
        if prov not in PROVIDERS:
            print(f"[skip] unknown provider {prov!r}")
            continue
        part = get_participant(prov)
        if not part.supports_logprobs:
            print(f"[skip] {prov}: no token logprobs (will use the sampling path)")
            continue
        try:
            out = part.probe_logprobs(args.question)
        except Exception as e:  # surface the failure, don't crash the whole probe
            print(f"[FAIL] {prov} ({type(e).__name__}): {e}")
            continue
        path = os.path.join(RESULTS, f"logprob_probe_{prov}.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2)
        has = out["raw_logprobs"] is not None
        print(f"[ok] {prov} ({out['model']}): logprobs present={has}; wrote {path}")
        print(out["pretty"][:1500])
        print("-" * 60)


if __name__ == "__main__":
    main()
