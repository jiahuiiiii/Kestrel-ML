"""Capture a fixture's articles for a historical catalyst event (ml_plan.md §6).

Fetches day-by-day — NOT one wide call — because Finnhub caps each response at
~250 items and silently truncates wide historical windows (ml_plan.md §2).
Writes articles.json plus a labels.json skeleton (every (article, catalyst)
pair, defaulted to relevant=false / no_change) for hand-labeling.

CURATION (--keep-match): a raw fetch is often hundreds of articles, mostly
noise. Passing --keep-match narrows the fixture to articles whose text contains
one of the given terms, plus a stride-sample of the rest as negatives
(--sample-irrelevant). This decides only which articles are IN the fixture — it
does NOT label them. Every kept article still starts relevant=false; a keyword
match is a hint for the labeler (recorded in `_curation`), not a verdict. The
answer key stays human-written on purpose (see eval/fixtures/README.md).

Re-running is safe: articles.json is refreshed, but existing hand labels are
kept — only skeleton rows for newly seen pairs are appended.

Examples:
  # Full dump (no curation) — every article in the window:
  python -m eval.backfill --event nvda_2026-05-28_earnings --ticker NVDA \
      --start 2026-05-26 --end 2026-05-30 \
      --catalyst "earnings_beat:Q1 revenue beats consensus"

  # Curated — keep layoff-related articles + 15 sampled negatives:
  python -m eval.backfill --event orcl_2026-06-22_layoffs --ticker ORCL \
      --start 2026-06-20 --end 2026-07-05 \
      --catalyst "orcl_layoff:Oracle discloses cutting ~21,000 jobs in FY2026" \
      --keep-match layoff "job cut" "21,000" workforce fired "laid off" \
      --sample-irrelevant 15
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

    curation: dict[str, str] = {}
    if args.keep_match:
        raw = len(articles)
        articles, curation = curate(articles, args.keep_match, args.sample_irrelevant)
        n_match = sum(1 for v in curation.values() if v == "keyword-match")
        print(f"  curated: {raw} -> {len(articles)} "
              f"({n_match} keyword-match + {len(articles) - n_match} sampled negatives)")

    apath = fixture_io.save_articles(args.event, articles)
    doc, added = build_or_update_labels(args.event, args.ticker, articles, catalysts, curation)
    lpath = fixture_io.save_labels(args.event, doc)

    with_body = sum(1 for a in articles if a.has_body)
    print(f"\n{args.event}:")
    print(f"  articles.json  {len(articles)} articles ({with_body} with body text) -> {apath}")
    print(f"  labels.json    {len(doc['labels'])} pair rows ({added} new skeleton rows) -> {lpath}")
    print("\nAll rows start relevant=false — capture is done, but the answer key is NOT "
          "written.\nNext: hand-label labels.json (see eval/fixtures/README.md), "
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


def curate(
    articles: list[news.Article],
    keep_terms: list[str],
    sample_n: int,
) -> tuple[list[news.Article], dict[str, str]]:
    """Narrow a raw fetch to a labelable fixture. Pure and deterministic.

    Keeps every article whose headline+summary contains any of `keep_terms`
    (case-insensitive substring), plus a fixed-stride sample of `sample_n`
    non-matching articles to serve as labeled negatives (so Pass-1 precision is
    measurable). Choose SPECIFIC terms — "21,000", "layoff", "job cut" — not
    broad ones like the company name, or everything matches.

    This is an inclusion filter, not a labeler: it says which articles enter the
    fixture, never what their label is. Returns (kept articles oldest-first,
    {article_id: "keyword-match" | "sample"}) — the map is a labeling hint only.
    """
    terms = [t.lower() for t in keep_terms]
    matched, others = [], []
    for a in articles:
        text = f"{a.headline} {a.summary or ''}".lower()
        (matched if any(t in text for t in terms) else others).append(a)

    step = max(len(others) // sample_n, 1) if sample_n > 0 else 0
    sample = others[::step][:sample_n] if step else []

    hint = {a.id: "keyword-match" for a in matched}
    hint.update({a.id: "sample" for a in sample})
    kept = sorted({a.id: a for a in matched + sample}.values(), key=lambda a: a.published_at)
    return kept, hint


def build_or_update_labels(
    event: str,
    ticker: str,
    articles: list[news.Article],
    catalysts: list[dict],
    curation: dict[str, str] | None = None,
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

    curation = curation or {}
    have = {(row["article_id"], row["catalyst_id"]) for row in doc["labels"]}
    added = 0
    for a in articles:
        for c in doc["catalysts"]:
            if (a.id, c["id"]) in have:
                continue
            row = {
                "article_id": a.id,
                "catalyst_id": c["id"],
                "relevant": False,
                "expected_state": "no_change",
                # underscore fields are not read by replay.py — context so the
                # labeler can work inside labels.json without cross-referencing
                # articles.json. `_curation` (present only for curated fixtures)
                # flags which rows the keyword filter thought worth a look.
                "_headline": a.headline[:120],
                "_published_at": a.published_at.isoformat(),
            }
            if a.id in curation:
                row["_curation"] = curation[a.id]
            doc["labels"].append(row)
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
    p.add_argument("--keep-match", nargs="+", metavar="TERM", default=None,
                   help="curate: keep only articles whose text contains one of these "
                        "(case-insensitive) terms, plus a --sample-irrelevant sample. "
                        "Use specific terms, not the company name. Omit for a full dump.")
    p.add_argument("--sample-irrelevant", type=int, default=15, metavar="N",
                   help="with --keep-match: also keep N stride-sampled non-matching "
                        "articles as labeled negatives (default 15; 0 to keep none)")
    return p.parse_args()


if __name__ == "__main__":
    main()
