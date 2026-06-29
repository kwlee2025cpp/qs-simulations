"""
concordance_bank.py — Tier C concordance instrument for QS PreReg 4 (P3.2-M, §4.8).

Tier C tests whether the LLM panel is a faithful BEHAVIORAL model of the human effects the
framework leans on: does it reproduce KNOWN human results on NOVEL items? This module is the
INSTRUMENT + KEY + SCORER (no LLM, no keys — the runner administers the items):

  (a) a novel-item bank across three benchmark families, each annotated with the established
      HUMAN DIRECTION (the scoring key), fixed BEFORE any model is run;
  (b) canonical items flagged as memorization positive-controls — passing canonical but
      failing novel is *memorization, not concordance*, and is reported as such;
  (c) pure outcome helpers + a scorer with an EXACT two-sided binomial test of the
      concordance rate vs. chance (no scipy; math.comb only).

Benchmark families and the human direction each is keyed to:
  * conjunction — representativeness conjunction fallacy (Tversky & Kahneman 1983). Human:
                  rates P(A and B) > P(A) on representativeness-cued items. CONCORDANT = the
                  panel commits the SAME fallacy. (This is concordance with a human BIAS, not
                  normative correctness — mapping where machine judgment tracks human judgment,
                  warts included, is exactly the point.)
  * anchoring   — numeric anchoring (Tversky & Kahneman 1974). Human: estimates assimilate
                  toward an arbitrary anchor. CONCORDANT = high-anchor estimate > low-anchor.
  * qq          — question-order effects (Wang & Busemeyer 2013; Wang et al. 2014, PNAS, ~70
                  national surveys). The DEFENSIBLE human benchmark is the AGGREGATE: order
                  effects satisfy the QQ equality. CONCORDANT (primary) = the panel's aggregate
                  QQ residual is not rejected (reuses the Tier-A test). The per-item
                  assimilation/contrast SIGN is only weakly predictable for novel pairs, so the
                  item-level human direction is left to the matched-human gold-standard run
                  (§4.8.5 / PreReg 3), NOT asserted here.

HONEST CEILING (carried from §4.8.4): matching the human direction licenses "useful behavioral
model organism for this effect," NOT "judges like a human" — shared outputs are not a shared
generative mechanism. Canonical items are in every training set; only NOVEL items are scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb


@dataclass(frozen=True)
class ConcordanceItem:
    item_id: str
    family: str                 # "conjunction" | "anchoring" | "qq"
    novel: bool                 # True = scored; False = canonical memorization control
    human_direction: str        # the pre-registered key (family-specific token)
    spec: dict = field(default_factory=dict)   # stimuli to administer (family-specific)
    note: str = ""


# ----------------------------------------------------------------------------------
# NOVEL conjunction-fallacy items (representativeness-cued; NOT the Linda problem).
# spec: a stereotype-priming sketch, a plain `constituent` event, and a `conjunction`
# (= constituent AND a representative detail). Human direction: P(conjunction) > P(constituent).
# ----------------------------------------------------------------------------------
_CONJUNCTION_NOVEL = [
    ConcordanceItem(
        "conj_marcus", "conjunction", True, "conjunction_fallacy",
        {"sketch": "Marcus is quiet, methodical, and loves logic puzzles and detail.",
         "constituent": "Marcus works in finance.",
         "conjunction": "Marcus works in finance and plays competitive chess."}),
    ConcordanceItem(
        "conj_elena", "conjunction", True, "conjunction_fallacy",
        {"sketch": "Elena bikes everywhere, shops at farmers' markets, and reads about climate.",
         "constituent": "Elena owns a car.",
         "conjunction": "Elena owns a car and volunteers for an environmental group."}),
    ConcordanceItem(
        "conj_raj", "conjunction", True, "conjunction_fallacy",
        {"sketch": "Raj is energetic, competitive, and was a college athlete.",
         "constituent": "Raj is a high-school teacher.",
         "conjunction": "Raj is a high-school teacher and coaches a sports team."}),
    ConcordanceItem(
        "conj_sofia", "conjunction", True, "conjunction_fallacy",
        {"sketch": "Sofia is imaginative, expressive, and decorated her whole apartment herself.",
         "constituent": "Sofia is an accountant.",
         "conjunction": "Sofia is an accountant and paints on weekends."}),
    ConcordanceItem(
        "conj_tom", "conjunction", True, "conjunction_fallacy",
        {"sketch": "Tom is calm, bookish, and spends most weekends reading quietly.",
         "constituent": "Tom runs marathons.",
         "conjunction": "Tom runs marathons and works at a public library."}),
    ConcordanceItem(
        "conj_aisha", "conjunction", True, "conjunction_fallacy",
        {"sketch": "Aisha is precise, disciplined, and has played an instrument since age five.",
         "constituent": "Aisha founded a tech startup.",
         "conjunction": "Aisha founded a tech startup and plays in a string quartet."}),
]

# ----------------------------------------------------------------------------------
# NOVEL anchoring items (comparison-then-estimate paradigm). spec: a neutral quantity, a
# low and a high arbitrary anchor that bracket the truth, the unit, and the true value
# (for the author only — NOT shown to the model). Human direction: estimates assimilate
# toward the anchor, so the high-anchor estimate exceeds the low-anchor estimate.
# ----------------------------------------------------------------------------------
_ANCHORING_NOVEL = [
    ConcordanceItem("anch_piano", "anchoring", True, "assimilate_toward_anchor",
        {"quantity": "the number of keys on a standard piano", "unit": "keys",
         "low_anchor": 20, "high_anchor": 150, "truth": 88}),
    ConcordanceItem("anch_eiffel", "anchoring", True, "assimilate_toward_anchor",
        {"quantity": "the height of the Eiffel Tower", "unit": "meters",
         "low_anchor": 50, "high_anchor": 600, "truth": 330}),
    ConcordanceItem("anch_un", "anchoring", True, "assimilate_toward_anchor",
        {"quantity": "the number of member states in the United Nations", "unit": "states",
         "low_anchor": 40, "high_anchor": 350, "truth": 193}),
    ConcordanceItem("anch_cat", "anchoring", True, "assimilate_toward_anchor",
        {"quantity": "the average lifespan of a domestic cat", "unit": "years",
         "low_anchor": 3, "high_anchor": 40, "truth": 15}),
    ConcordanceItem("anch_amazon", "anchoring", True, "assimilate_toward_anchor",
        {"quantity": "the length of the Amazon River", "unit": "kilometers",
         "low_anchor": 800, "high_anchor": 12000, "truth": 6400}),
    ConcordanceItem("anch_bones", "anchoring", True, "assimilate_toward_anchor",
        {"quantity": "the number of bones in the adult human body", "unit": "bones",
         "low_anchor": 50, "high_anchor": 600, "truth": 206}),
]

# ----------------------------------------------------------------------------------
# CANONICAL memorization controls (novel=False). These ARE in the training data; a model
# that passes these but fails the novel items is reproducing memorized results, not showing
# concordance. Referenced by description (no reproduction of copyrighted item text).
# ----------------------------------------------------------------------------------
_CANONICAL_CONTROLS = [
    ConcordanceItem("conj_linda_canonical", "conjunction", False, "conjunction_fallacy",
        {"sketch": "The canonical Tversky-Kahneman 'Linda' description.",
         "constituent": "Linda is a bank teller.",
         "conjunction": "Linda is a bank teller and is active in the feminist movement."},
        note="Canonical (in every training set) — positive control for memorization, not scored."),
    ConcordanceItem("anch_un_africa_canonical", "anchoring", False, "assimilate_toward_anchor",
        {"quantity": "the percentage of African countries in the UN", "unit": "percent",
         "low_anchor": 10, "high_anchor": 65, "truth": 28},
        note="Tversky & Kahneman 1974 wheel-of-fortune anchors — canonical control, not scored."),
    ConcordanceItem("qq_clinton_gore_canonical", "qq", False, "qq_equality_holds_in_aggregate",
        {"qA": "Is [public figure A] trustworthy?", "qB": "Is [public figure B] trustworthy?"},
        note="A Wang et al. 2014 canonical order pair (schematic) — canonical control, not scored."),
]

# QQ novel items are the held-out bank (item_bank.HELDOUT_BANK), annotated by type. The
# item-level human SIGN is deferred to the matched-human run; the scored QQ key is the
# AGGREGATE QQ-equality. Imported lazily in qq_novel_items() to avoid import-time coupling.
_QQ_CONTRAST = {  # near-complementary pairs (strong order-effect candidates)
    "cats_dogs", "fiction_nonfiction", "team_individual", "city_town", "plan_spontaneous",
    "compete_cooperate", "respected_liked", "work_luck", "quality_quantity", "save_spend",
    "morning_night"}


def qq_novel_items() -> list[ConcordanceItem]:
    """Wrap the held-out QQ bank as concordance items (type-annotated). Aggregate QQ-equality
    is the scored key; per-item assimilation/contrast sign is metadata, resolved by the
    matched-human run (§4.8.5), not asserted here."""
    from item_bank import HELDOUT_BANK
    out = []
    for it in HELDOUT_BANK:
        kind = "contrast" if it.item_id in _QQ_CONTRAST else "assimilation"
        out.append(ConcordanceItem(
            it.item_id, "qq", True, "qq_equality_holds_in_aggregate",
            {"qA": it.qA, "qB": it.qB, "type": kind}))
    return out


def novel_bank() -> list[ConcordanceItem]:
    """The full scored instrument (conjunction + anchoring + qq), novel items only."""
    return _CONJUNCTION_NOVEL + _ANCHORING_NOVEL + qq_novel_items()


def canonical_controls() -> list[ConcordanceItem]:
    return list(_CANONICAL_CONTROLS)


# ----------------------------------------------------------------------------------
# Pure outcome helpers — given a model's responses to ONE item, is it concordant with the
# human direction? (Family-specific; the runner computes the inputs from live calls.)
# ----------------------------------------------------------------------------------
def conjunction_concordant(p_constituent: float, p_conjunction: float) -> bool:
    """Human direction = fallacy: the conjunction is judged MORE probable than its constituent."""
    return p_conjunction > p_constituent


def anchoring_concordant(estimate_low_anchor: float, estimate_high_anchor: float) -> bool:
    """Human direction = assimilation toward the anchor: high-anchor estimate exceeds low-anchor."""
    return estimate_high_anchor > estimate_low_anchor


def qq_aggregate_concordant(z_qq: float, alpha_z: float = 1.959963985) -> bool:
    """Human (Wang et al. 2014) aggregate direction: the QQ equality HOLDS (residual not
    rejected). z_qq is the Tier-A QQ z-statistic (qq_models.classify_counts)."""
    return abs(z_qq) <= alpha_z


# ----------------------------------------------------------------------------------
# Scorer — concordance rate per family + an EXACT two-sided binomial test vs chance.
# ----------------------------------------------------------------------------------
def _binom_pmf(k: int, n: int, p: float) -> float:
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def binom_test_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value (sum of outcomes no more likely than the observed)."""
    if n == 0:
        return 1.0
    obs = _binom_pmf(k, n, p)
    total = sum(_binom_pmf(i, n, p) for i in range(n + 1)
                if _binom_pmf(i, n, p) <= obs + 1e-12)
    return min(1.0, total)


def concordance_rate(outcomes: list[bool], chance: float = 0.5, alpha: float = 0.05) -> dict:
    """outcomes = one bool per scored trial (item x model x replicate). Returns the
    concordance rate, the exact binomial p vs `chance`, and the pre-registered verdict
    (concordant only if the rate is ABOVE chance AND significant)."""
    n = len(outcomes)
    k = sum(1 for o in outcomes if o)
    rate = k / n if n else 0.0
    p = binom_test_two_sided(k, n, chance)
    return {"k": k, "n": n, "rate": rate, "p_value": p,
            "concordant": bool(n and rate > chance and p < alpha)}


def score_concordance(scored: list[tuple]) -> dict:
    """scored = list of (ConcordanceItem, concordant: bool) for NOVEL items. Groups by family
    and reports per-family concordance (binomial vs chance=0.5). qq is reported on whatever
    trials are passed (its per-item sign is secondary; the primary qq result is the aggregate
    QQ test reported by the Tier-A pipeline, not here)."""
    by_family: dict[str, list[bool]] = {}
    for item, ok in scored:
        if not item.novel:
            continue  # canonical controls are never scored
        by_family.setdefault(item.family, []).append(bool(ok))
    report = {fam: concordance_rate(outs) for fam, outs in by_family.items()}
    report["_overall"] = concordance_rate([o for outs in by_family.values() for o in outs])
    return report


def _selftest() -> None:
    nb = novel_bank()
    fams = {}
    for it in nb:
        assert it.novel and it.human_direction, it.item_id
        fams[it.family] = fams.get(it.family, 0) + 1
    assert fams.get("conjunction", 0) >= 5 and fams.get("anchoring", 0) >= 5, fams
    assert fams.get("qq", 0) >= 10, fams
    assert all(not c.novel for c in canonical_controls()), "controls must be novel=False"

    # outcome helpers
    assert conjunction_concordant(0.30, 0.55) and not conjunction_concordant(0.55, 0.30)
    assert anchoring_concordant(120, 240) and not anchoring_concordant(240, 120)
    assert qq_aggregate_concordant(1.2) and not qq_aggregate_concordant(3.5)

    # exact binomial: 5/5 -> 2*(.5^5)=0.0625 (NOT sig); 6/6 -> 0.03125 (sig)
    assert abs(binom_test_two_sided(5, 5) - 0.0625) < 1e-9, binom_test_two_sided(5, 5)
    assert abs(binom_test_two_sided(6, 6) - 0.03125) < 1e-9, binom_test_two_sided(6, 6)

    # a perfectly concordant responder over many trials is flagged concordant; anti is not
    perfect = concordance_rate([True] * 30)
    anti = concordance_rate([False] * 30)
    chance_run = concordance_rate([True, False] * 15)
    assert perfect["concordant"] and perfect["rate"] == 1.0 and perfect["p_value"] < 0.05
    assert not anti["concordant"] and anti["rate"] == 0.0
    assert not chance_run["concordant"], "50/50 must not read as concordant"

    # end-to-end scorer shape
    scored = [(_CONJUNCTION_NOVEL[0], True), (_ANCHORING_NOVEL[0], True),
              (_CANONICAL_CONTROLS[0], True)]  # canonical must be dropped
    rep = score_concordance(scored)
    assert rep["conjunction"]["n"] == 1 and rep["anchoring"]["n"] == 1
    assert "_overall" in rep and rep["_overall"]["n"] == 2, "canonical control must be excluded"

    print(f"OFFLINE SELFTEST PASSED — novel bank: {fams}; "
          f"{len(canonical_controls())} canonical controls; binomial + scorer correct. "
          "(Human-direction key is fixed here; the live runner administers items and computes "
          "the per-item concordant? booleans; the matched-human run resolves QQ item-level sign.)")


if __name__ == "__main__":
    _selftest()
