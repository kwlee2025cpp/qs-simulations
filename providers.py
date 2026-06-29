"""
providers.py — real LLM participants for the live Tier-A QQ study (QS PreReg 4).

Implements `Participant` subclasses for the pre-registered panel — Gemini, Grok,
Claude — over plain HTTP (`requests`). The request/parse formats are adapted from the
author's own `gemini-python-tutor` (the [EDU] grading stack), kept self-contained here
so this research repo has no dependency on that private repo.

Keys are read from the environment (never hard-coded, never logged). Each provider
checks a clean name first, then the author's GitHub-Actions-style name as a fallback:
    Gemini : GEMINI_API_KEY    | INPUT_GEMINI-API-KEY
    Grok   : XAI_API_KEY       | INPUT_GROK-API-KEY
    Claude : ANTHROPIC_API_KEY | INPUT_CLAUDE_API_KEY

IMPORTANT — these issue REAL, billable API calls and contact external services. The
runner does not call them unless you pass real `--provider` + `--live`. For the QQ
study we SAMPLE (temperature > 0) over many sessions to estimate response
probabilities, so a full pre-registered run is tens of thousands of calls per model —
do a small pilot first and confirm cost + each provider's ToS for research use.

Offline self-test (no network, no key needed):
    conda run -n qs-sim python providers.py --selftest
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from llm_qq_runner import LLMParticipantTemplate


def _getkey(*names: str, required: bool = True) -> str:
    for n in names:
        v = os.getenv(n, "").strip()
        if v:
            return v
    if required:
        raise RuntimeError(
            f"No API key found in env (tried: {', '.join(names)}). "
            f"Export one before a --live run; it is never read from disk or logged.")
    return ""


class _HTTPParticipant(LLMParticipantTemplate):
    """Shared HTTP plumbing. Subclasses implement _build()/_parse_json()."""
    default_model = "OVERRIDE"
    env_names: tuple = ()
    supports_logprobs: bool = False   # Claude=False (no token logprobs) -> sampling fallback
    reasoning_model: str | None = None  # Grok realizes "thinking" by a model swap, not a flag
    THINK_BUDGET: int = 1024            # hidden-thinking token room in the extended-thinking arm

    def __init__(self, model: str | None = None, temperature: float = 1.0,
                 max_tokens: int = 16, api_key: str | None = None,
                 thinking: bool = False):
        super().__init__(model or self.default_model)
        self.model = model or self.default_model   # template stores model_id; we use .model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # thinking/reasoning mode is a PRE-REGISTERED FACTOR (PreReg 4 sec 4.2), not a tweak:
        # the SAME model is run in 'direct' (thinking off) and 'extended-thinking' modes so the
        # n=16 pilot's thinking CONFOUND (the reasoning model showed a weaker order effect)
        # becomes an estimand instead of a nuisance. Each provider exposes thinking differently.
        self.thinking = thinking
        # api_key may be passed directly (tests) or resolved from env (live)
        self.api_key = api_key if api_key is not None else _getkey(*self.env_names)

    # subclasses provide these two pure functions (offline-testable):
    def _build(self, messages: list[dict]) -> tuple[str, dict, dict]:
        raise NotImplementedError

    def _parse_json(self, j: dict) -> str:
        raise NotImplementedError

    request_timeout: int = 180   # reasoning models (e.g. Grok) can exceed 60s per call

    def _chat(self, messages: list[dict]) -> str:
        import requests
        url, headers, body = self._build(messages)
        r = requests.post(url, headers=headers, json=body, timeout=self.request_timeout)
        r.raise_for_status()
        return self._parse_json(r.json())

    # ---- log-prob probe (de-risking step; finalize the extractor from its output) ----
    # The yes/no probability for a question is read from the answer-token logprobs in a
    # single call (constrained to one token), instead of sampling many sessions. Response
    # shapes differ per provider and aren't verifiable offline, so probe_logprobs() dumps
    # the RAW structure from one real call; the p(yes) extractor is finalized from that.
    def _build_probe(self, question: str) -> tuple[str, dict, dict]:
        raise NotImplementedError(
            f"{self.name}: token logprobs not supported (supports_logprobs={self.supports_logprobs}); "
            "use the sampling path instead.")

    def _raw_logprobs(self, j: dict):  # subclasses point at the provider-specific field
        raise NotImplementedError

    def probe_logprobs(self, question: str = "Is honesty generally the best policy?") -> dict:
        """One real call with logprobs enabled; returns the RAW logprob structure so we can
        see each provider's exact shape before trusting an extractor. Needs a real key."""
        import requests, json as _json
        url, headers, body = self._build_probe(question)
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        j = r.json()
        return {"provider": self.name, "model": self.model,
                "raw_logprobs": self._raw_logprobs(j),
                "full_keys": sorted(j.keys()),
                "pretty": _json.dumps(self._raw_logprobs(j), indent=2)[:4000]}


_PROBE_INSTR = " Answer with exactly one word: yes or no."


class GeminiParticipant(_HTTPParticipant):
    default_model = "gemini-2.5-flash"   # [verify] pin against Google's current model list at filing
    env_names = ("GEMINI_API_KEY", "INPUT_GEMINI-API-KEY")
    supports_logprobs = True   # responseLogprobs (model-dependent) — confirm via probe

    def _build_probe(self, question):
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.api_key}")
        body = {"contents": [{"role": "user", "parts": [{"text": question + _PROBE_INSTR}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": 1,
                                     "responseLogprobs": True, "logprobs": 5}}
        return url, {"Content-Type": "application/json"}, body

    def _raw_logprobs(self, j):
        return j.get("candidates", [{}])[0].get("logprobsResult")

    def _build(self, messages):
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.api_key}")
        contents = [{"role": ("model" if m["role"] == "assistant" else "user"),
                     "parts": [{"text": m["content"]}]} for m in messages]
        if self.thinking:
            # extended-thinking arm: let the model think, and leave room for the one-word
            # answer AFTER the hidden thinking tokens are spent (else finishReason MAX_TOKENS).
            gen = {"temperature": self.temperature,
                   "maxOutputTokens": self.THINK_BUDGET + self.max_tokens,
                   "thinkingConfig": {"thinkingBudget": self.THINK_BUDGET}}
        else:
            # direct arm: gemini-2.5-flash is a THINKING model: without thinkingBudget=0 it
            # spends maxOutputTokens on hidden thinking and returns no answer (no parts).
            gen = {"temperature": self.temperature,
                   "maxOutputTokens": self.max_tokens,
                   "thinkingConfig": {"thinkingBudget": 0}}
        body = {"contents": contents, "generationConfig": gen}
        return url, {"Content-Type": "application/json"}, body

    def _parse_json(self, j):
        # defensive: a truncated/blocked/empty candidate -> "" (caller treats as decline),
        # never a crash that kills the whole run.
        cand = (j.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts")
        if not parts:
            return ""
        return "\n".join(p.get("text", "") for p in parts)


class GrokParticipant(_HTTPParticipant):
    # OpenAI-compatible. Pin the exact available chat model id in the live filing.
    # Grok couples reasoning to the MODEL ID (no per-call thinking flag): the pilot's
    # "grok-3" resolved to a reasoning model. So the thinking FACTOR is realized by a swap:
    # direct arm -> default_model (a non-reasoning id), thinking arm -> reasoning_model.
    # BOTH ids must be pinned against xAI's current model list at filing. [verify]
    default_model = "grok-3"          # [verify] confirm this is the NON-reasoning id
    reasoning_model: str | None = None  # [verify] pin a reasoning-capable id; None => toggle no-op
    env_names = ("XAI_API_KEY", "INPUT_GROK-API-KEY")
    supports_logprobs = True   # OpenAI-style logprobs/top_logprobs — confirm via probe

    def _build_probe(self, question):
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {"model": self.model,
                "messages": [{"role": "user", "content": question + _PROBE_INSTR}],
                "stream": False, "temperature": 0, "max_tokens": 1,
                "logprobs": True, "top_logprobs": 8}   # xAI caps top_logprobs at 8
        return url, headers, body

    def _raw_logprobs(self, j):
        return j.get("choices", [{}])[0].get("logprobs")

    def _build(self, messages):
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        # thinking arm uses the reasoning id when one is pinned; otherwise stays on self.model
        # (a documented no-op so the toggle never silently picks a wrong model).
        model_id = (self.reasoning_model or self.model) if self.thinking else self.model
        body = {"model": model_id, "messages": messages, "stream": False,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        return url, headers, body

    def _parse_json(self, j):
        return j["choices"][0]["message"]["content"]


class ClaudeParticipant(_HTTPParticipant):
    default_model = "claude-sonnet-4-6"   # [verify] pin against Anthropic's current model list at filing
    env_names = ("ANTHROPIC_API_KEY", "INPUT_CLAUDE_API_KEY")

    def _build(self, messages):
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json"}
        if self.thinking:
            # extended thinking: budget_tokens < max_tokens, and Claude requires temperature
            # left at its default when thinking is enabled (we sample at temperature 1 anyway).
            body = {"model": self.model, "max_tokens": self.THINK_BUDGET + self.max_tokens,
                    "thinking": {"type": "enabled", "budget_tokens": self.THINK_BUDGET},
                    "messages": messages}
        else:
            body = {"model": self.model, "max_tokens": self.max_tokens,
                    "temperature": self.temperature, "messages": messages}
        return url, headers, body

    def _parse_json(self, j):
        # with extended thinking on, content[] holds a 'thinking' block BEFORE the 'text'
        # answer, so we cannot assume content[0] is the answer.
        for block in j.get("content", []):
            if block.get("type") == "text":
                return block.get("text", "")
        c = j.get("content") or [{}]      # thinking off => a single text block (may lack 'type')
        return c[0].get("text", "")


PROVIDERS = {"gemini": GeminiParticipant, "grok": GrokParticipant, "claude": ClaudeParticipant}


def get_participant(provider: str, model: str | None = None, temperature: float = 1.0,
                    api_key: str | None = None, thinking: bool = False) -> _HTTPParticipant:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {list(PROVIDERS)}")
    return PROVIDERS[provider](model=model, temperature=temperature, api_key=api_key,
                               thinking=thinking)


def _selftest() -> None:
    """Validate request-building and response-parsing OFFLINE (no network, dummy key)."""
    msgs = [{"role": "user", "content": "Q1?"},
            {"role": "assistant", "content": "yes"},
            {"role": "user", "content": "Q2?"}]
    canned = {
        "gemini": {"candidates": [{"content": {"parts": [{"text": "no"}]}}]},
        "grok": {"choices": [{"message": {"content": "yes"}}]},
        "claude": {"content": [{"text": "decline"}]},
    }
    expect_key_loc = {"gemini": "url", "grok": "header", "claude": "header"}
    for name, cls in PROVIDERS.items():
        p = cls(api_key="TEST-KEY-NOT-REAL", temperature=0.7)
        url, headers, body = p._build(msgs)
        # key must be carried (in url for gemini, header otherwise) and never in body
        if expect_key_loc[name] == "url":
            assert "TEST-KEY-NOT-REAL" in url, name
        else:
            assert any("TEST-KEY-NOT-REAL" in str(v) for v in headers.values()), name
        assert "TEST-KEY-NOT-REAL" not in str(body), f"{name}: key leaked into body"
        # multi-turn must be preserved (3 turns in -> 3 turns out)
        n_turns = len(body.get("messages", body.get("contents", [])))
        assert n_turns == 3, f"{name}: expected 3 turns, got {n_turns}"
        parsed = p._parse_json(canned[name])
        assert parsed in ("yes", "no", "decline"), f"{name}: parsed {parsed!r}"
        # logprob probe builder: enabled providers must request logprobs + 1 token
        if p.supports_logprobs:
            _, _, pbody = p._build_probe("Q?")
            flat = str(pbody).lower()
            assert "logprob" in flat, f"{name}: probe missing logprob flag"
            assert ("max_tokens" in pbody and pbody["max_tokens"] == 1) or \
                   (pbody.get("generationConfig", {}).get("maxOutputTokens") == 1), \
                   f"{name}: probe not constrained to 1 token"
            lp = "logprobs ON"
        else:
            lp = "logprobs UNSUPPORTED -> sampling fallback"
        print(f"  {name:7s}: build OK ({n_turns} turns, key in {expect_key_loc[name]}), "
              f"parse OK -> {parsed!r}, model={p.model}, {lp}")
    assert not ClaudeParticipant(api_key="x").supports_logprobs, "Claude has no token logprobs"

    # --- thinking-as-a-factor (PreReg 4 sec 4.2): the toggle must change the request right ---
    gem0 = GeminiParticipant(api_key="x", thinking=False)._build(msgs)[2]["generationConfig"]
    gem1 = GeminiParticipant(api_key="x", thinking=True)._build(msgs)[2]["generationConfig"]
    assert gem0["thinkingConfig"]["thinkingBudget"] == 0, "gemini direct arm must disable thinking"
    assert gem1["thinkingConfig"]["thinkingBudget"] > 0, "gemini thinking arm must enable thinking"
    assert gem1["maxOutputTokens"] > gem0["maxOutputTokens"], \
        "gemini thinking arm needs answer room beyond the thinking budget"
    cla0 = ClaudeParticipant(api_key="x", thinking=False)._build(msgs)[2]
    cla1 = ClaudeParticipant(api_key="x", thinking=True)._build(msgs)[2]
    assert "thinking" not in cla0 and cla1.get("thinking", {}).get("type") == "enabled", \
        "claude thinking arm must send an enabled thinking block"
    assert ClaudeParticipant(api_key="x")._parse_json(
        {"content": [{"type": "thinking", "thinking": "...hidden..."},
                     {"type": "text", "text": "yes"}]}) == "yes", \
        "claude parse must return the text block, not the leading thinking block"
    gk0 = GrokParticipant(api_key="x", thinking=False)._build(msgs)[2]["model"]
    gk1 = GrokParticipant(api_key="x", thinking=True)._build(msgs)[2]["model"]
    grok_note = (f"grok thinking arm -> {gk1!r}" if gk1 != gk0
                 else "grok thinking is a NO-OP until reasoning_model is pinned ([verify])")
    print(f"  thinking-factor: gemini OK (budget {gem0['thinkingConfig']['thinkingBudget']}"
          f"->{gem1['thinkingConfig']['thinkingBudget']}, tokens {gem0['maxOutputTokens']}"
          f"->{gem1['maxOutputTokens']}), claude OK (enabled block + text-parse), {grok_note}")

    print("OFFLINE SELFTEST PASSED — request/parse + logprob-probe builders correct. "
          "(Live calls need real keys + --live; the exact logprob response shape is "
          "confirmed by probe_logprobs() on the first CI run, then the extractor is finalized.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print("Use --selftest for the offline check, or import get_participant(...) from "
              "the runner. Live runs need real keys and explicit opt-in.")
