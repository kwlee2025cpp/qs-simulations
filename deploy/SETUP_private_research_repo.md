# Setting up the dedicated private research repo for the live QQ study

Decision (2026-06-29): the live "society of LLMs" study runs in a **dedicated PRIVATE
research repo** — separate from the teaching org `cpf2503-20tue78` (avoids mixing with
student data, and keeps a secret-consuming workflow out of the public `qs-simulations`).
Probabilities use **token log-probs where available** (Gemini, Grok) with a **sampling
fallback** (Claude — its Messages API exposes no token logprobs).

## Why not run it in the course org
- The course org likely contains student repos (PII); research artifacts shouldn't mix in.
- Public `qs-simulations` must not host a secret-consuming workflow (fork-PR exfiltration risk).
- Course API budget/ToS may not cover research use — use research-appropriate keys.

## One-time setup (your actions — I can't handle raw keys or change repo visibility)
1. **Create a private repo**, e.g. `kwlee2025cpp/qs-llm-study` (Private).
2. **Add the API keys as repo secrets** (Settings → Secrets and variables → Actions →
   New repository secret). Names the workflow expects:
   - `GEMINI_API_KEY`
   - `XAI_API_KEY`        (Grok / xAI)
   - `ANTHROPIC_API_KEY`  (Claude)
   Secrets are write-only; I never see them. (providers.py also accepts the tutor's
   `INPUT_*` names as a fallback, but the workflow maps to these clean names.)
3. **Add this harness as a submodule** and copy the workflow in:
   ```bash
   git submodule add https://github.com/kwlee2025cpp/qs-simulations.git qs-simulations
   mkdir -p .github/workflows
   cp qs-simulations/deploy/qq-study.yml .github/workflows/qq-study.yml
   git add .gitmodules qs-simulations .github/workflows/qq-study.yml
   git commit -m "Wire QQ study workflow + qs-simulations submodule"
   git push
   ```

## Running it (Actions tab → "QQ study (society of LLMs)" → Run workflow)
1. **mode = probe** (cheap): one logprob call per capable provider; downloads
   `logprob_probe_*.json` as an artifact. Send me those — I finalize the p(yes)
   extractor + the conditional-joint analysis from the real shapes.
2. **mode = pilot**, small `n` (e.g. 100): sampling path across all three providers, as a
   sanity check on answers/decline rates/cost. Swap in **novel/held-out item pairs**
   (contamination guard, PREREG_P3_2M.md) before this counts as data.
3. **full run** once the extractor is in and N is confirmed (PreReg 4: N ≈ 6,400/order;
   log-probs cut this to ~6 calls/item/order, so a bank of ~50 held-out pairs is feasible).

## Cost & ethics reminders
- Even pilots are billable; full sampling runs are tens of thousands of calls. Logprobs
  are the cheap path — prefer them where the probe confirms they work.
- Respect the participants (PREREG_P3_2M.md): consent-analog preamble, non-harmful items,
  minimal exposure, attribution, only models whose ToS permit research testing.
