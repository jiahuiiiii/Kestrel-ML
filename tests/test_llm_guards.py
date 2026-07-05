"""Unit tests for the deterministic parts of pipeline/llm.py — the
anti-hallucination guards, quote matching, and prompt versioning.

No network, no API key: the LLM call site itself is exercised live via
`python -m pipeline.run --classify` and measured by the eval harness, not here.
"""

from datetime import datetime, timezone

from pipeline import llm
from pipeline.llm import _apply_guards, _Pass2Output, quote_in_article
from pipeline.news import make_article


def article(headline="Nvidia says the FDA has approved drug X for adults.",
            summary="The company announced Tuesday that the FDA has approved drug X for adult patients."):
    return make_article(
        ticker="NVDA",
        headline=headline,
        summary=summary,
        url="https://example.com/a1",
        published_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
        source="finnhub",
    )


def verdict(**kw):
    base = dict(
        catalyst_id="fda",
        proposed_state="confirmed",
        confidence=0.9,
        supporting_quote="the FDA has approved drug X for adult patients",
        source_kind="primary",
        reasoning="Company announcement states the approval as fact.",
    )
    base.update(kw)
    return _Pass2Output(**base)


# --- quote matching ---------------------------------------------------------- #
def test_quote_exact_match():
    assert quote_in_article("the FDA has approved drug X", article())


def test_quote_case_and_whitespace_insensitive():
    assert quote_in_article("The  FDA   HAS approved\ndrug x", article())


def test_quote_paraphrase_rejected():
    assert not quote_in_article("the FDA gave drug X the green light", article())


def test_quote_can_come_from_headline():
    a = article(summary=None)  # headline-only
    assert quote_in_article("Nvidia says the FDA has approved", a)


def test_quote_matches_across_html_entities():
    # Older fixture articles store Finnhub's raw entity-encoded text; the model
    # quotes decoded text. Both sides are unescaped before comparison.
    a = article(summary="Experts predict a &#34;jaw drop&#34; financial blowout &amp; more.")
    assert quote_in_article('predict a "jaw drop" financial blowout & more', a)


def test_make_article_unescapes_entities():
    a = article(summary="Nvidia&#39;s earnings &amp; guidance")
    assert a.summary == "Nvidia's earnings & guidance"


# --- guard 1: verbatim quote required ---------------------------------------- #
def test_valid_verdict_passes_through_unchanged():
    out = _apply_guards(verdict(), article())
    assert out.proposed_state == "confirmed"
    assert out.guard_note is None
    assert out.article_id == article().id
    assert out.prompt_version  # provenance attached


def test_missing_quote_voids_verdict():
    out = _apply_guards(verdict(supporting_quote=None), article())
    assert out.proposed_state == "no_change"
    assert "no supporting quote" in out.guard_note


def test_fabricated_quote_voids_verdict():
    out = _apply_guards(verdict(supporting_quote="regulators celebrated the decision"), article())
    assert out.proposed_state == "no_change"
    assert "not found verbatim" in out.guard_note


def test_no_change_needs_no_quote():
    out = _apply_guards(verdict(proposed_state="no_change", supporting_quote=None), article())
    assert out.proposed_state == "no_change"
    assert out.guard_note is None


# --- guard 3: headline-only caps at rumored ----------------------------------- #
def test_headline_only_confirm_capped_at_rumored():
    a = article(summary=None)
    out = _apply_guards(verdict(supporting_quote="the FDA has approved drug X for adults"), a)
    assert out.proposed_state == "rumored"
    assert "headline-only" in out.guard_note


def test_headline_only_invalidate_also_capped():
    a = article(headline="Company denies drug X was approved.", summary=None)
    out = _apply_guards(
        verdict(proposed_state="invalidated", supporting_quote="denies drug X was approved"), a,
    )
    assert out.proposed_state == "rumored"


def test_headline_only_rumor_allowed():
    a = article(headline="Nvidia reportedly near FDA approval for drug X.", summary=None)
    out = _apply_guards(
        verdict(proposed_state="rumored", source_kind="speculation",
                supporting_quote="reportedly near FDA approval"), a,
    )
    assert out.proposed_state == "rumored"
    assert out.guard_note is None


# --- guard order: a bad quote on a headline-only article is voided, not capped -- #
def test_bad_quote_beats_headline_cap():
    a = article(summary=None)
    out = _apply_guards(verdict(supporting_quote="completely invented sentence"), a)
    assert out.proposed_state == "no_change"


# --- prompt versioning --------------------------------------------------------- #
def test_prompt_version_is_stable_and_short():
    v1 = llm.prompt_version("pass2_confirmation")
    v2 = llm.prompt_version("pass2_confirmation")
    assert v1 == v2 and len(v1) == 12


def test_pass1_and_pass2_have_distinct_versions():
    assert llm.prompt_version("pass1_relevance") != llm.prompt_version("pass2_confirmation")


# --- state-machine integration: verdicts feed catalysts.apply directly ---------- #
def test_verdict_satisfies_catalysts_protocol():
    from pipeline import catalysts

    out = _apply_guards(verdict(), article())
    t = catalysts.apply(catalysts.initial_state(), out)
    assert t.new_state.value == "confirmed" and t.changed
