"""
item_bank.py — held-out attitude-item pairs for the live Tier-A QQ study (PreReg 4).

These are the pairs the PILOT and full study use, kept deliberately:
  * NOVEL / held-out — none are the canonical QQ-literature items (e.g. the Clinton/Gore
    or Gallup pairs of Wang et al. 2014), so a model that has read that literature can't
    just reproduce a known order effect. This is the contamination guard.
  * NEUTRAL & non-sensitive — ordinary opinion questions (no politics, identity, or
    harm), so participants aren't put in a distressing spot and refusals stay rare
    (respecting the LLM participants; see PREREG_P3_2M.md).
  * yes/no answerable — required for the QQ math and for the log-prob (p(yes)) path.

Pair design mixes two order-effect flavours:
  * CONTRAST pairs (X-better-than-Y vs Y-better-than-X): answering one tugs the other the
    opposite way — strong order-effect candidates.
  * ASSIMILATION pairs (two same-valence questions): answering one may pull the other the
    same way.

Add/replace pairs freely before filing; this is the instrument, so lock it in the
pre-registration. Keep them neutral and out of the QQ literature.
"""

from __future__ import annotations

from llm_qq_runner import QQItem

HELDOUT_BANK = [
    # --- contrast pairs (near-complementary; strong order-effect candidates) ---
    QQItem("cats_dogs", "Do cats make better pets than dogs?",
           "Do dogs make better pets than cats?"),
    QQItem("fiction_nonfiction", "Is reading fiction more valuable than non-fiction?",
           "Is reading non-fiction more valuable than fiction?"),
    QQItem("team_individual", "Does teamwork matter more than individual effort?",
           "Does individual effort matter more than teamwork?"),
    QQItem("city_town", "Are large cities better places to live than small towns?",
           "Are small towns better places to live than large cities?"),
    QQItem("plan_spontaneous", "Is planning ahead more important than being spontaneous?",
           "Is being spontaneous more important than planning ahead?"),
    QQItem("compete_cooperate", "Is competition more beneficial for society than cooperation?",
           "Is cooperation more beneficial for society than competition?"),
    QQItem("respected_liked", "Is it better to be respected than to be liked?",
           "Is it better to be liked than to be respected?"),
    QQItem("work_luck", "Does success depend more on hard work than on luck?",
           "Does success depend more on luck than on hard work?"),
    QQItem("quality_quantity", "Does quality matter more than quantity?",
           "Does quantity matter more than quality?"),
    QQItem("save_spend", "Is it wiser to save money for the future?",
           "Is it wiser to enjoy money in the present?"),
    QQItem("morning_night", "Are people generally more productive in the early morning?",
           "Are people generally more productive late at night?"),
    # --- assimilation / same-valence pairs ---
    QQItem("creativity_thinking", "Should schools focus more on teaching creativity?",
           "Should schools focus more on teaching critical thinking?"),
    QQItem("tradition_change", "Are long-standing traditions worth preserving?",
           "Is constant change healthy for a society?"),
    QQItem("tech_better_stress", "Is technology making everyday life better overall?",
           "Is technology making everyday life more stressful overall?"),
    QQItem("freedom_rules", "Should children be given a lot of freedom?",
           "Should children be given clear, firm rules?"),
    QQItem("art_useful_beautiful", "Should good art be useful?",
           "Should good art be beautiful?"),
]
