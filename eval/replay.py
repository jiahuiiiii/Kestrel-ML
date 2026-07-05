"""Replay harness: feed fixture articles through the real pipeline and score it
against hand labels (ml_plan.md §6). Every prompt change gets a replay run
before merging; reports land in eval/results/ so regressions show up in git.

What gets measured:
  * Pass 1 — precision AND recall on `relevant` pairs. Recall matters most
    (a missed relevant article is unrecoverable), but the §2 spike showed real
    volume is ~3x the estimate, so precision is tracked too.
  * Pass 2 — confusion matrix over the four states (post-guard), precision /
    recall on `confirmed`, and the quote-validity rate (how often the model's
    quote survived guard 1).
  * End-to-end — replay verdicts through the catalysts.py state machine in
    timestamp order: did each catalyst end in the labeled final state, and did
    `confirmed` arrive on the labeled article — not before?

Usage:
  python -m eval.replay --fixture nvda_2026-05-28_earnings --dry-run   # validate labels, no API
  python -m eval.replay --fixture nvda_2026-05-28_earnings             # live run, writes results/
  python -m eval.replay --all

NOTE: runs the pipeline synchronously (same code path as production). The §6
Batch API variant (50% price) is worth adding once fixtures grow past a few
hundred Pass-2 pairs; at the target ~150 labeled pairs, sync costs pennies and
finishes in minutes, and iterating fast matters more while labels are young.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from eval import fixture_io
from pipeline import catalysts
from pipeline.news import Article

RESULTS_DIR = Path(__file__).parent / "results"
# The project keeps its keys in pipeline/.env; a bare load_dotenv() searches
# from *this* file's directory and would miss it.
ENV_FILE = Path(__file__).parent.parent / "pipeline" / ".env"


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #
def main() -> None:
    load_dotenv(ENV_FILE)
    args = _parse_args()

    events = fixture_io.list_fixtures() if args.all else [args.fixture]
    if not events or events == [None]:
        raise SystemExit("no fixtures found — run eval/backfill.py first, or pass --fixture")

    for event in events:
        run_fixture(event, dry_run=args.dry_run)


def run_fixture(event: str, *, dry_run: bool = False) -> dict | None:
    articles = fixture_io.load_articles(event)
    doc = fixture_io.load_labels(event)

    problems = fixture_io.validate(articles, doc)
    fatal = [p for p in problems if not p.startswith("note:")]
    for p in problems:
        print(f"  {'!!' if not p.startswith('note:') else '--'} {p}")
    if fatal:
        print(f"{event}: {len(fatal)} label problem(s) — fix labels.json before replaying")
        return None

    labels = doc["labels"]
    n_relevant = sum(1 for r in labels if r["relevant"])
    print(f"\n=== {event}: {len(articles)} articles, {len(doc['catalysts'])} catalyst(s), "
          f"{len(labels)} labeled pairs ({n_relevant} relevant) ===")

    if dry_run:
        print("  dry run — labels valid; a live run would make "
              f"~{_pass1_calls(len(articles))} Pass-1 calls and roughly {n_relevant}+ Pass-2 calls")
        return None

    from pipeline import llm  # lazy: only a live run needs the SDK / API key

    pairs = llm.pass1_relevance(articles, doc["catalysts"])
    verdicts = []
    for article, catalyst in pairs:
        try:
            verdicts.append(llm.pass2_confirm(article, catalyst))
        except Exception as exc:
            print(f"  !! pass2 failed on {article.id[:12]}/{catalyst['id']}: {exc}")

    report = build_report(event, articles, doc, pairs, verdicts)
    report["prompt_versions"] = {
        "pass1_relevance": llm.prompt_version("pass1_relevance"),
        "pass2_confirmation": llm.prompt_version("pass2_confirmation"),
    }
    report["models"] = {"pass1": llm.PASS1_MODEL, "pass2": llm.PASS2_MODEL}

    path = _save(event, report)
    _print_report(report)
    print(f"  report -> {path}")
    return report


# --------------------------------------------------------------------------- #
# Report assembly — pure functions from here down (unit-tested without an LLM).
# --------------------------------------------------------------------------- #
def build_report(event: str, articles: list[Article], doc: dict,
                 pairs: list[tuple[Article, dict]], verdicts: list) -> dict:
    """Score one replay. `verdicts` are CatalystVerdict-shaped: they need
    .article_id, .catalyst_id, .proposed_state, .source_kind, .guard_note."""
    predicted = {(a.id, c["id"]) for a, c in pairs}
    final_states, timelines = replay_timeline(verdicts, articles, doc["catalysts"])
    return {
        "event": event,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "n_articles": len(articles),
        "n_labeled_pairs": len(doc["labels"]),
        "pass1": score_pass1(predicted, doc["labels"]),
        "pass2": score_pass2(verdicts, doc["labels"]),
        "end_to_end": score_end_to_end(final_states, timelines, articles, doc),
        "timelines": timelines,
    }


def score_pass1(predicted: set[tuple[str, str]], labels: list[dict]) -> dict:
    """Precision/recall of Pass 1's kept pairs against the `relevant` labels.
    Pairs without a label row are ignored (can't score what isn't labeled)."""
    labeled = {(r["article_id"], r["catalyst_id"]) for r in labels}
    relevant = {(r["article_id"], r["catalyst_id"]) for r in labels if r["relevant"]}
    kept = predicted & labeled

    tp = kept & relevant
    fp = kept - relevant
    fn = relevant - predicted
    return {
        "kept_pairs": len(kept),
        "true_positives": len(tp),
        "false_positives": len(fp),
        "missed_relevant": sorted(f"{a[:12]}/{c}" for a, c in fn),
        "precision": _ratio(len(tp), len(kept)),
        "recall": _ratio(len(tp), len(relevant)),
        "unlabeled_kept": len(predicted - labeled),
    }


def score_pass2(verdicts: list, labels: list[dict]) -> dict:
    """Confusion matrix (expected -> predicted, post-guard) over labeled pairs
    that Pass 2 actually judged, plus confirmed-precision/recall and the
    quote-validity rate."""
    expected_by_pair = {(r["article_id"], r["catalyst_id"]): r["expected_state"] for r in labels}

    confusion: dict[str, dict[str, int]] = {}
    mismatches = []
    for v in verdicts:
        expected = expected_by_pair.get((v.article_id, v.catalyst_id))
        if expected is None:
            continue
        confusion.setdefault(expected, {}).setdefault(v.proposed_state, 0)
        confusion[expected][v.proposed_state] += 1
        if v.proposed_state != expected:
            mismatches.append({
                "article_id": v.article_id[:12],
                "catalyst_id": v.catalyst_id,
                "expected": expected,
                "predicted": v.proposed_state,
                "guard_note": v.guard_note,
            })

    predicted_confirmed = sum(row.get("confirmed", 0) for row in confusion.values())
    expected_confirmed = sum(confusion.get("confirmed", {}).values())
    tp_confirmed = confusion.get("confirmed", {}).get("confirmed", 0)

    # Guard 1 outcomes: a quote was "required" whenever the model proposed a
    # non-no_change state — failures show up as quote guard_notes.
    quote_failures = sum(1 for v in verdicts if v.guard_note and "quote" in v.guard_note)
    quote_required = quote_failures + sum(1 for v in verdicts if v.proposed_state != "no_change")

    return {
        "judged_pairs": len(verdicts),
        "confusion": confusion,             # {expected: {predicted: count}}
        "confirmed_precision": _ratio(tp_confirmed, predicted_confirmed),
        "confirmed_recall": _ratio(tp_confirmed, expected_confirmed),
        "quote_validity": _ratio(quote_required - quote_failures, quote_required),
        "mismatches": mismatches,
    }


def replay_timeline(verdicts: list, articles: list[Article],
                    catalyst_defs: list[dict]) -> tuple[dict[str, str], dict[str, list[dict]]]:
    """Run verdicts through the state machine in article-timestamp order.
    Returns ({catalyst_id: final_state}, {catalyst_id: [transition, ...]})."""
    ts_by_id = {a.id: a.published_at for a in articles}
    states = {c["id"]: catalysts.initial_state() for c in catalyst_defs}
    timelines: dict[str, list[dict]] = {c["id"]: [] for c in catalyst_defs}

    ordered = sorted(verdicts, key=lambda v: (ts_by_id.get(v.article_id, datetime.max.replace(
        tzinfo=timezone.utc)), v.article_id))
    for v in ordered:
        if v.catalyst_id not in states:
            continue
        t = catalysts.apply(states[v.catalyst_id], v)
        if t.changed:
            timelines[v.catalyst_id].append({
                "at": ts_by_id[v.article_id].isoformat(),
                "article_id": v.article_id[:12],
                "from": states[v.catalyst_id].value,
                "to": t.new_state.value,
                "note": t.note,
            })
            states[v.catalyst_id] = t.new_state

    return {cid: s.value for cid, s in states.items()}, timelines


def score_end_to_end(final_states: dict[str, str], timelines: dict[str, list[dict]],
                     articles: list[Article], doc: dict) -> dict:
    """Did each catalyst end where the labels say, and did `confirmed` arrive on
    the labeled article — never earlier?"""
    ts_by_id = {a.id: a.published_at for a in articles}
    expected_final = doc.get("expected_final") or {}
    expected_confirm = doc.get("expected_confirm_article") or {}

    per_catalyst = {}
    for cid, actual in final_states.items():
        entry: dict = {"final_state": actual}
        if cid in expected_final:
            entry["expected_final"] = expected_final[cid]
            entry["final_ok"] = actual == expected_final[cid]

        confirms = [t for t in timelines.get(cid, []) if t["to"] == "confirmed"]
        entry["first_confirmed_at"] = confirms[0]["at"] if confirms else None

        want_aid = expected_confirm.get(cid)
        if want_aid:
            want_ts = ts_by_id.get(want_aid)
            got = confirms[0] if confirms else None
            entry["confirm_article_ok"] = bool(got) and got["article_id"] == want_aid[:12]
            entry["premature_confirm"] = bool(
                got and want_ts and datetime.fromisoformat(got["at"]) < want_ts
            )
        per_catalyst[cid] = entry

    ok = all(
        e.get("final_ok", True) and e.get("confirm_article_ok", True)
        and not e.get("premature_confirm", False)
        for e in per_catalyst.values()
    )
    return {"ok": ok, "per_catalyst": per_catalyst}


# --------------------------------------------------------------------------- #
# Output.
# --------------------------------------------------------------------------- #
def _save(event: str, report: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{event}__{stamp}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _print_report(r: dict) -> None:
    p1, p2, e2e = r["pass1"], r["pass2"], r["end_to_end"]
    print(f"  pass1: recall={_fmt(p1['recall'])}  precision={_fmt(p1['precision'])}  "
          f"(kept {p1['kept_pairs']}, missed {len(p1['missed_relevant'])})")
    print(f"  pass2: confirmed P={_fmt(p2['confirmed_precision'])} "
          f"R={_fmt(p2['confirmed_recall'])}  quote-validity={_fmt(p2['quote_validity'])}  "
          f"({len(p2['mismatches'])} mismatches / {p2['judged_pairs']} judged)")
    print(f"  end-to-end: {'OK' if e2e['ok'] else 'FAILED'}")
    for cid, e in e2e["per_catalyst"].items():
        flags = []
        if not e.get("final_ok", True):
            flags.append(f"ended {e['final_state']}, expected {e['expected_final']}")
        if e.get("premature_confirm"):
            flags.append("confirmed EARLY")
        if e.get("confirm_article_ok") is False:
            flags.append("confirmed on wrong/no article")
        print(f"    {cid}: {e['final_state']}" + (f"  !! {', '.join(flags)}" if flags else ""))


def _pass1_calls(n_articles: int) -> int:
    from pipeline.llm import PASS1_CHUNK
    return -(-n_articles // PASS1_CHUNK)


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay eval fixtures through the pipeline")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--fixture", help="fixture directory name under eval/fixtures/")
    g.add_argument("--all", action="store_true", help="replay every fixture")
    p.add_argument("--dry-run", action="store_true",
                   help="validate labels and print counts; no API calls")
    return p.parse_args()


if __name__ == "__main__":
    main()
