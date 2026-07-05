"""Unit tests for the eval harness's scoring logic (eval/replay.py).

No LLM, no network: verdicts are stubs. The live pipeline is exercised by
`python -m eval.replay --fixture ...`; these tests pin down that the *metrics*
computed from its output are right.
"""

from dataclasses import dataclass
from datetime import datetime, timezone

from eval import fixture_io
from eval.backfill import build_or_update_labels, curate
from eval.replay import (
    build_report,
    replay_timeline,
    score_end_to_end,
    score_pass1,
    score_pass2,
)
from pipeline.news import make_article


@dataclass
class StubVerdict:
    article_id: str
    catalyst_id: str
    proposed_state: str
    source_kind: str = "primary"
    guard_note: str | None = None


def art(n: int, hour: int, headline: str = "headline", summary: str = "body text"):
    return make_article(
        ticker="NVDA",
        headline=headline,
        summary=summary,
        url=f"https://example.com/{n}",
        published_at=datetime(2026, 7, 1, hour, tzinfo=timezone.utc),
        source="finnhub",
    )


A1, A2, A3 = art(1, 9), art(2, 12), art(3, 15)
CATALYSTS = [{"id": "beat", "description": "earnings beat"}]


def label(a, cid="beat", relevant=False, expected="no_change"):
    return {"article_id": a.id, "catalyst_id": cid, "relevant": relevant,
            "expected_state": expected}


# --- pass 1 -------------------------------------------------------------------- #
def test_pass1_perfect_run():
    labels = [label(A1, relevant=True, expected="rumored"), label(A2), label(A3)]
    out = score_pass1({(A1.id, "beat")}, labels)
    assert out["precision"] == 1.0 and out["recall"] == 1.0
    assert out["missed_relevant"] == []


def test_pass1_miss_and_false_positive():
    labels = [label(A1, relevant=True, expected="confirmed"),
              label(A2, relevant=True, expected="rumored"), label(A3)]
    out = score_pass1({(A1.id, "beat"), (A3.id, "beat")}, labels)  # missed A2, kept irrelevant A3
    assert out["recall"] == 0.5
    assert out["precision"] == 0.5
    assert out["missed_relevant"] == [f"{A2.id[:12]}/beat"]


def test_pass1_unlabeled_pairs_dont_pollute_metrics():
    labels = [label(A1, relevant=True, expected="confirmed")]
    out = score_pass1({(A1.id, "beat"), (A2.id, "beat")}, labels)  # A2 has no label row
    assert out["precision"] == 1.0
    assert out["unlabeled_kept"] == 1


def test_pass1_no_relevant_labels_means_no_recall():
    out = score_pass1(set(), [label(A1)])
    assert out["recall"] is None and out["precision"] is None


# --- pass 2 -------------------------------------------------------------------- #
def test_pass2_confusion_and_confirmed_metrics():
    labels = [label(A1, relevant=True, expected="confirmed"),
              label(A2, relevant=True, expected="confirmed"),
              label(A3, relevant=True, expected="no_change")]
    verdicts = [
        StubVerdict(A1.id, "beat", "confirmed"),          # right
        StubVerdict(A2.id, "beat", "rumored"),            # missed confirm
        StubVerdict(A3.id, "beat", "confirmed"),          # over-eager confirm
    ]
    out = score_pass2(verdicts, labels)
    assert out["confusion"]["confirmed"]["confirmed"] == 1
    assert out["confusion"]["confirmed"]["rumored"] == 1
    assert out["confusion"]["no_change"]["confirmed"] == 1
    assert out["confirmed_precision"] == 0.5   # 1 of 2 predicted confirms was right
    assert out["confirmed_recall"] == 0.5      # 1 of 2 expected confirms found
    assert len(out["mismatches"]) == 2


def test_pass2_quote_validity_counts_guard_failures():
    labels = [label(A1, relevant=True, expected="confirmed"),
              label(A2, relevant=True, expected="confirmed"),
              label(A3, relevant=True, expected="no_change")]
    verdicts = [
        StubVerdict(A1.id, "beat", "confirmed"),  # quote passed guard
        StubVerdict(A2.id, "beat", "no_change",   # quote failed -> voided by guard 1
                    guard_note="guard: verdict voided — quote not found verbatim in article"),
        StubVerdict(A3.id, "beat", "no_change"),  # genuine no_change: no quote required
    ]
    out = score_pass2(verdicts, labels)
    assert out["quote_validity"] == 0.5  # 2 quotes required, 1 survived


def test_pass2_ignores_unlabeled_verdicts():
    out = score_pass2([StubVerdict(A1.id, "beat", "confirmed")], [])
    assert out["confusion"] == {} and out["mismatches"] == []


# --- end-to-end timeline -------------------------------------------------------- #
def test_timeline_orders_by_timestamp_not_input_order():
    # A3's confirm is fed first, but A1's rumor happens earlier in article time.
    verdicts = [
        StubVerdict(A3.id, "beat", "confirmed"),
        StubVerdict(A1.id, "beat", "rumored", source_kind="speculation"),
    ]
    finals, timelines = replay_timeline(verdicts, [A1, A2, A3], CATALYSTS)
    assert finals == {"beat": "confirmed"}
    assert [t["to"] for t in timelines["beat"]] == ["rumored", "confirmed"]


def test_timeline_skips_unknown_catalysts_and_no_change():
    verdicts = [StubVerdict(A1.id, "other", "confirmed"),
                StubVerdict(A2.id, "beat", "no_change")]
    finals, timelines = replay_timeline(verdicts, [A1, A2], CATALYSTS)
    assert finals == {"beat": "unconfirmed"} and timelines["beat"] == []


def doc(expected_final="confirmed", confirm_article=None):
    return {
        "catalysts": CATALYSTS,
        "labels": [],
        "expected_final": {"beat": expected_final},
        "expected_confirm_article": {"beat": confirm_article},
    }


def test_end_to_end_ok_on_right_article():
    verdicts = [StubVerdict(A2.id, "beat", "confirmed")]
    finals, timelines = replay_timeline(verdicts, [A1, A2, A3], CATALYSTS)
    out = score_end_to_end(finals, timelines, [A1, A2, A3], doc(confirm_article=A2.id))
    assert out["ok"]
    e = out["per_catalyst"]["beat"]
    assert e["final_ok"] and e["confirm_article_ok"] and not e["premature_confirm"]


def test_end_to_end_flags_premature_confirm():
    # Confirmed on A1, but the labels say A2 is the article that should confirm.
    verdicts = [StubVerdict(A1.id, "beat", "confirmed")]
    finals, timelines = replay_timeline(verdicts, [A1, A2, A3], CATALYSTS)
    out = score_end_to_end(finals, timelines, [A1, A2, A3], doc(confirm_article=A2.id))
    assert not out["ok"]
    e = out["per_catalyst"]["beat"]
    assert e["premature_confirm"] and not e["confirm_article_ok"]


def test_end_to_end_flags_wrong_final_state():
    finals, timelines = replay_timeline([], [A1], CATALYSTS)  # nothing ever fires
    out = score_end_to_end(finals, timelines, [A1], doc(expected_final="confirmed"))
    assert not out["ok"]
    assert out["per_catalyst"]["beat"]["final_ok"] is False


def test_end_to_end_never_confirmed_negative_fixture():
    # The "rumor that never confirms" case: expected_final=rumored, no confirm article.
    verdicts = [StubVerdict(A1.id, "beat", "rumored", source_kind="speculation")]
    finals, timelines = replay_timeline(verdicts, [A1], CATALYSTS)
    out = score_end_to_end(finals, timelines, [A1],
                           doc(expected_final="rumored", confirm_article=None))
    assert out["ok"]
    assert out["per_catalyst"]["beat"]["first_confirmed_at"] is None


# --- report assembly + fixture io ------------------------------------------------ #
def test_build_report_shape():
    labels_doc = {
        "catalysts": CATALYSTS,
        "labels": [label(A1, relevant=True, expected="confirmed")],
        "expected_final": {"beat": "confirmed"},
        "expected_confirm_article": {"beat": A1.id},
    }
    pairs = [(A1, CATALYSTS[0])]
    verdicts = [StubVerdict(A1.id, "beat", "confirmed")]
    r = build_report("evt", [A1], labels_doc, pairs, verdicts)
    assert r["pass1"]["recall"] == 1.0
    assert r["pass2"]["confirmed_recall"] == 1.0
    assert r["end_to_end"]["ok"]


def test_article_roundtrips_through_json():
    a = art(9, 10, headline="Round trip", summary=None)
    assert fixture_io.article_from_dict(fixture_io.article_to_dict(a)) == a


def test_validate_catches_label_mistakes():
    bad_doc = {
        "catalysts": CATALYSTS,
        "labels": [
            {"article_id": A1.id, "catalyst_id": "beat",
             "relevant": False, "expected_state": "confirmed"},   # inconsistent
            {"article_id": "nope", "catalyst_id": "ghost",
             "relevant": True, "expected_state": "sideways"},      # 3 problems
        ],
        "expected_final": {"beat": "confirmed"},
        "expected_confirm_article": {"beat": None},
    }
    problems = [p for p in fixture_io.validate([A1], bad_doc) if not p.startswith("note:")]
    assert len(problems) == 4


# --- unlabeled-fixture guard ---------------------------------------------------- #
def test_live_replay_refuses_unlabeled_fixture(tmp_path, monkeypatch, capsys):
    from eval import replay
    monkeypatch.setattr(fixture_io, "FIXTURES_DIR", tmp_path)
    fixture_io.save_articles("blank", [A1])
    fixture_io.save_labels("blank", {
        "event": "blank", "ticker": "NVDA", "catalysts": CATALYSTS,
        "labels": [label(A1)],   # skeleton: relevant=false
        "expected_final": {"beat": "unconfirmed"},
        "expected_confirm_article": {"beat": None},
    })
    assert replay.run_fixture("blank") is None          # refused, no API call attempted
    assert "UNLABELED" in capsys.readouterr().out


# --- curation (backfill) ------------------------------------------------------- #
def kw_art(n, headline, summary=""):
    return make_article(ticker="ORCL", headline=headline, summary=summary,
                        url=f"https://example.com/kw{n}",
                        published_at=datetime(2026, 6, 22, n % 24, tzinfo=timezone.utc),
                        source="finnhub")


def test_curate_keeps_matches_and_samples_negatives():
    arts = [kw_art(0, "Oracle cuts 21,000 jobs"),           # match (headline)
            kw_art(1, "Markets update", "amid layoff news"),  # match (summary)
            kw_art(2, "SpaceX falls"),                       # negative
            kw_art(3, "Nuclear loan program"),               # negative
            kw_art(4, "Weather today")]                      # negative
    kept, hint = curate(arts, ["21,000", "layoff"], sample_n=1)
    kinds = sorted(hint[a.id] for a in kept)
    assert kinds.count("keyword-match") == 2
    assert kinds.count("sample") == 1
    assert len(kept) == 3


def test_curate_is_case_insensitive_and_deterministic():
    arts = [kw_art(n, f"story {n}", "Big LAYOFF wave") for n in range(3)]
    kept1, _ = curate(arts, ["layoff"], sample_n=0)
    kept2, _ = curate(arts, ["layoff"], sample_n=0)
    assert [a.id for a in kept1] == [a.id for a in kept2] == [a.id for a in arts]


def test_curate_sample_zero_keeps_only_matches():
    arts = [kw_art(0, "Oracle layoff"), kw_art(1, "unrelated"), kw_art(2, "also unrelated")]
    kept, hint = curate(arts, ["layoff"], sample_n=0)
    assert len(kept) == 1 and all(v == "keyword-match" for v in hint.values())


def test_curation_hint_lands_in_skeleton_rows():
    a_match, a_neg = kw_art(0, "Oracle layoff"), kw_art(1, "SpaceX")
    _, hint = curate([a_match, a_neg], ["layoff"], sample_n=1)
    doc, added = build_or_update_labels("evt", "ORCL", [a_match, a_neg],
                                        [{"id": "orcl_layoff", "description": "layoffs"}], hint)
    by_id = {r["article_id"]: r for r in doc["labels"]}
    assert added == 2
    assert by_id[a_match.id]["_curation"] == "keyword-match"
    assert by_id[a_neg.id]["_curation"] == "sample"
    # curation decides inclusion only — every row still starts unlabeled.
    assert all(r["relevant"] is False for r in doc["labels"])


def test_validate_clean_fixture_is_quiet():
    good = {
        "catalysts": CATALYSTS,
        "labels": [label(A1, relevant=True, expected="confirmed")],
        "expected_final": {"beat": "confirmed"},
        "expected_confirm_article": {"beat": A1.id},
    }
    assert fixture_io.validate([A1], good) == []
