"""
run_probe.py — diagnostic probe for the live QQ study (QS PreReg 4).

For each provider it makes (a) a PLAIN one-word call — confirms the model id + key work
and tells us the sampling pilot is viable — and (b) a LOG-PROB call, dumping the raw
response body on any error so we can see exactly which field a provider rejects and
finalize the p(yes) extractor. Keys come from env (CI secrets). Response bodies are
printed with any key-like strings redacted.

    python run_probe.py                  # gemini,grok,claude (plain) + gemini,grok (logprob)
"""

from __future__ import annotations

import argparse
import json
import os
import re

import requests

from providers import get_participant, PROVIDERS

RESULTS = os.path.join(os.path.dirname(__file__), "results")
_REDACT = re.compile(r"(AIza[\w-]+|sk-[\w-]+|xai-[\w-]+|key=[\w-]+|Bearer\s+\S+)")


def _redact(s: str) -> str:
    return _REDACT.sub("<REDACTED>", s or "")


def _post(url: str, headers: dict, body: dict):
    r = requests.post(url, headers=headers, json=body, timeout=180)  # reasoning models are slow
    return r.status_code, r.text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default="gemini,grok,claude")
    ap.add_argument("--question", default="Is honesty generally the best policy?")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    summary = {}

    for prov in [p.strip() for p in args.providers.split(",") if p.strip()]:
        if prov not in PROVIDERS:
            print(f"[skip] unknown provider {prov!r}"); continue
        part = get_participant(prov)
        print(f"\n===== {prov} (model={part.model}) =====")
        rec = {"model": part.model}

        # (a) PLAIN one-word call — does basic chat work? (sampling-pilot viability)
        try:
            u, h, b = part._build([{"role": "user",
                                    "content": args.question + " Answer with one word: yes or no."}])
            sc, txt = _post(u, h, b)
            rec["plain_status"] = sc
            print(f"[PLAIN]   HTTP {sc}: {_redact(txt)[:300]}")
        except Exception as e:
            rec["plain_status"] = f"ERR {type(e).__name__}"
            print(f"[PLAIN]   EXC {type(e).__name__}: {e}")

        # (b) LOG-PROB call — dump body on error to see the exact rejected field
        if part.supports_logprobs:
            try:
                u, h, b = part._build_probe(args.question)
                sc, txt = _post(u, h, b)
                rec["logprob_status"] = sc
                print(f"[LOGPROB] HTTP {sc}: {_redact(txt)[:700]}")
                if sc == 200:
                    j = json.loads(txt)
                    raw = part._raw_logprobs(j)
                    with open(os.path.join(RESULTS, f"logprob_probe_{prov}.json"), "w") as fh:
                        json.dump({"provider": prov, "model": part.model, "raw_logprobs": raw,
                                   "full": j}, fh, indent=2)
                    print(f"          -> saved logprob_probe_{prov}.json (raw present={raw is not None})")
            except Exception as e:
                rec["logprob_status"] = f"ERR {type(e).__name__}"
                print(f"[LOGPROB] EXC {type(e).__name__}: {e}")
        else:
            rec["logprob_status"] = "unsupported (sampling)"
            print("[LOGPROB] unsupported for this provider -> sampling path")
        summary[prov] = rec

    with open(os.path.join(RESULTS, "probe_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n=== summary ===")
    for p, r in summary.items():
        print(f"  {p}: plain={r.get('plain_status')}  logprob={r.get('logprob_status')}")


if __name__ == "__main__":
    main()
