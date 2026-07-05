"""Capture a fixture's articles for a historical catalyst event (ml_plan.md §6).

Fetches day-by-day — NOT one wide call — because Finnhub caps each response at
~250 items and silently truncates wide historical windows (ml_plan.md §2).
Writes articles.json plus a labels.json skeleton (every (article, catalyst)
pair, defaulted to relevant=false / no_change) for hand-labeling.

Re-running is safe: articles.json is refreshed, but existing hand labels are
kept — only skeleton rows for newly seen pairs are appended.

Example:
  python -m eval.backfill --event nvda_2026-05-28_earnings --ticker NVDA \
      --start 2026-05-26 --end 2026-05-30 \
      --catalyst "earnings_beat:Q1 revenue beats consensus" \
      --catalyst "guidance_raise:full-year guidance raised"
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from eval import fixture_io
from pipeline import news

# Keys live in pipeline/.env; a bare load_dotenv() searches from this file's
# directory and would miss it.
ENV_FILE = Path(__file__).parent.parent / "pipeline" / ".env"


def main() -> None:
    load_dotenv(ENV_FILE)
    args = _parse_args()
    catalysts = [_parse_catalyst(c) for c in args.catalyst]

    articles = collect(args.ticker, args.start, args.end, sources=args.sources)
    if not articles:
        print(f"no articles found for {args.ticker} {args.start}..{args.end} — nothing written")
        return

    apath = fixture_io.save_articles(args.event, articles)
    doc, added = build_or_update_labels(args.event, args.ticker, articles, catalysts)
    lpath = fixture_io.save_labels(args.event, doc)

    with_body = sum(1 for a in articles if a.has_body)
    print(f"\n{args.event}:")
    print(f"  articles.json  {len(articles)} articles ({with_body} with body text) -> {apath}")
    print(f"  labels.json    {len(doc['labels'])} pair rows ({added} new skeleton rows) -> {lpath}")
    print("\nNext: hand-label labels.json (see eval/fixtures/README.md), "
          "then `python -m eval.replay --fixture " + args.event + " --dry-run` to validate.")


def collect(ticker: str, start: date, end: date, *, sources: list[str]) -> list[news.Article]:
    """Fetch [start, end] one day at a time, deduping across days by Article.id."""
    seen: dict[str, news.Article] = {}
    day = start
    while day <= end:
        since = datetime.combine(day, dtime.min, tzinfo=timezone.utc)
        until = datetime.combine(day, dtime.max, tzinfo=timezone.utc)
        got = news.fetch(ticker, since, sources=sources, until=until)
        fresh = [a for a in got if a.id not in seen]
        seen.update((a.id, a) for a in got)
        print(f"  {day}: {len(got)} articles ({len(fresh)} new)")
        day += timedelta(days=1)
        time.sleep(1.1)  # stay well under Finnhub's 60 calls/min free tier
    return sorted(seen.values(), key=lambda a: a.published_at)


def build_or_update_labels(
    event: str,
    ticker: str,
    articles: list[news.Article],
    catalysts: list[dict],
) -> tuple[dict, int]:
    """Return (labels doc, number of skeleton rows added). Existing hand labels
    are never overwritten — this only appends rows for unseen pairs."""
    try:
        doc = fixture_io.load_labels(event)
    except FileNotFoundError:
        doc = {
            "event": event,
            "ticker": ticker,
            "catalysts": catalysts,
            "labels": [],
            "expected_final": {c["id"]: "unconfirmed" for c in catalysts},
            "expected_confirm_article": {c["id"]: None for c in catalysts},
        }

    known = {c["id"] for c in doc["catalysts"]}
    for c in catalysts:
        if c["id"] not in known:
            doc["catalysts"].append(c)
            doc.setdefault("expected_final", {})[c["id"]] = "unconfirmed"
            doc.setdefault("expected_confirm_article", {})[c["id"]] = None

    have = {(row["article_id"], row["catalyst_id"]) for row in doc["labels"]}
    added = 0
    for a in articles:
        for c in doc["catalysts"]:
            if (a.id, c["id"]) in have:
                continue
            doc["labels"].append({
                "article_id": a.id,
                "catalyst_id": c["id"],
                "relevant": False,
                "expected_state": "no_change",
                # not read by replay.py — context so the labeler can work
                # inside labels.json without cross-referencing articles.json:
                "_headline": a.headline[:120],
                "_published_at": a.published_at.isoformat(),
            })
            added += 1
    doc["labels"].sort(key=lambda r: (r.get("_published_at", ""), r["catalyst_id"]))
    return doc, added


def _parse_catalyst(spec: str) -> dict:
    if ":" not in spec:
        raise SystemExit(f'--catalyst must be "id:description", got {spec!r}')
    cid, desc = spec.split(":", 1)
    return {"id": cid.strip(), "description": desc.strip()}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture articles for one eval fixture")
    p.add_argument("--event", required=True, help="fixture name, e.g. nvda_2026-05-28_earnings")
    p.add_argument("--ticker", required=True)
    p.add_argument("--start", required=True, type=date.fromisoformat, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, type=date.fromisoformat, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--catalyst", action="append", required=True, metavar="ID:DESC",
                   help='e.g. "earnings_beat:Q1 revenue beats consensus"; repeatable')
    p.add_argument("--sources", nargs="+", default=["finnhub"])
    return p.parse_args()


if __name__ == "__main__":
    main()
