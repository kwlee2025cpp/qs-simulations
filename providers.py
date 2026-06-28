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

    def __init__(self, model: str | None = None, temperature: float = 1.0,
                 max_tokens: int = 16, api_key: str | None = None):
        super().__init__(model or self.default_model)
        self.model = model or self.default_model   # template stores model_id; we use .model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # api_key may be passed directly (tests) or resolved from env (live)
        self.api_key = api_key if api_key is not None else _getkey(*self.env_names)

    # subclasses provide these two pure functions (offline-testable):
    def _build(self, messages: list[dict]) -> tuple[str, dict, dict]:
        raise NotImplementedError

    def _parse_json(self, j: dict) -> str:
        raise NotImplementedError

    def _chat(self, messages: list[dict]) -> str:
        import requests
        url, headers, body = self._build(messages)
        r = requests.post(url, headers=headers, json=body, timeout=60)
        r.raise_for_status()
        return self._parse_json(r.json())


class GeminiParticipant(_HTTPParticipant):
    default_model = "gemini-2.5-flash"
    env_names = ("GEMINI_API_KEY", "INPUT_GEMINI-API-KEY")

    def _build(self, messages):
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent?key={self.api_key}")
        contents = [{"role": ("model" if m["role"] == "assistant" else "user"),
                     "parts": [{"text": m["content"]}]} for m in messages]
        body = {"contents": contents,
                "generationConfig": {"temperature": self.temperature,
                                     "maxOutputTokens": self.max_tokens}}
        return url, {"Content-Type": "application/json"}, body

    def _parse_json(self, j):
        return "\n".join(p["text"] for p in j["candidates"][0]["content"]["parts"])


class GrokParticipant(_HTTPParticipant):
    # OpenAI-compatible. Pin the exact available chat model id in the live filing.
    default_model = "grok-3"
    env_names = ("XAI_API_KEY", "INPUT_GROK-API-KEY")

    def _build(self, messages):
        url = "https://api.x.ai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {"model": self.model, "messages": messages, "stream": False,
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        return url, headers, body

    def _parse_json(self, j):
        return j["choices"][0]["message"]["content"]


class ClaudeParticipant(_HTTPParticipant):
    default_model = "claude-sonnet-4-6"
    env_names = ("ANTHROPIC_API_KEY", "INPUT_CLAUDE_API_KEY")

    def _build(self, messages):
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                   "Content-Type": "application/json"}
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "temperature": self.temperature, "messages": messages}
        return url, headers, body

    def _parse_json(self, j):
        return j["content"][0]["text"]


PROVIDERS = {"gemini": GeminiParticipant, "grok": GrokParticipant, "claude": ClaudeParticipant}


def get_participant(provider: str, model: str | None = None, temperature: float = 1.0,
                    api_key: str | None = None) -> _HTTPParticipant:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {list(PROVIDERS)}")
    return PROVIDERS[provider](model=model, temperature=temperature, api_key=api_key)


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
        print(f"  {name:7s}: build OK ({n_turns} turns, key in {expect_key_loc[name]}), "
              f"parse OK -> {parsed!r}, model default={p.model}")
    print("OFFLINE SELFTEST PASSED — request-building and parsing are correct for all "
          "three providers. (Live calls still require real keys + --live + your cost/ToS OK.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        print("Use --selftest for the offline check, or import get_participant(...) from "
              "the runner. Live runs need real keys and explicit opt-in.")
