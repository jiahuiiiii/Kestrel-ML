"""CLI to exercise the news pipeline — the Phase-0 spike driver (ml_plan.md §2).

Loads .env (for FINNHUB_API_KEY), fetches recent news for one or more tickers,
and prints the numbers the spike needs to decide the primary source:
  - articles per day (after dedup)
  - fraction with quotable body text  (has_body) vs headline-only
  - staleness of the newest article   (how far behind real-time the feed is)

With --classify it also runs the two-pass LLM classifier (needs ANTHROPIC_API_KEY
in .env) against ad-hoc catalysts given on the command line.

Examples:
  python -m pipeline.run --ticker NVDA
  python -m pipeline.run --ticker NVDA AMD AAPL --days 3 --sources finnhub yfinance
  python -m pipeline.run --ticker NVDA --show          # also print each article
  python -m pipeline.run --ticker NVDA --days 1 --limit 20 --classify \
      --catalyst "hyperscaler capex cut" --catalyst "new export restrictions to China"
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from pipeline import news


def main() -> None:
    load_dotenv()  # reads .env in the project root -> FINNHUB_API_KEY into os.environ

    args = _parse_args()
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    now = datetime.now(timezone.utc)

    for ticker in args.ticker:
        try:
            articles = news.fetch(ticker, since, sources=args.sources)
        except Exception as exc:
            print(f"\n{ticker}: FETCH FAILED — {exc}")
            continue

        if args.limit and len(articles) > args.limit:
            articles = articles[-args.limit:]  # keep the most recent N

        _print_summary(ticker, articles, days=args.days, now=now)
        if args.show:
            _print_articles(articles)
        if args.classify:
            _classify(ticker, articles, args.catalyst or [])


def _print_summary(ticker: str, articles: list[news.Article], *, days: int, now: datetime) -> None:
    n = len(articles)
    print(f"\n=== {ticker} — last {days}d ===")
    if not n:
        print("  no articles")
        return

    with_body = sum(1 for a in articles if a.has_body)
    newest = max(a.published_at for a in articles)
    oldest = min(a.published_at for a in articles)
    newest_age = now - newest
    oldest_age = now - oldest

    # Rate over the span actually covered, not the span requested — a capped
    # feed (oldest_age << days) makes n/days an undercount.
    covered_days = max((newest - oldest).total_seconds() / 86400, 1e-6)
    capped = oldest_age.total_seconds() / 86400 < days * 0.9

    by_source: dict[str, int] = {}
    for a in articles:
        by_source[a.source] = by_source.get(a.source, 0) + 1

    print(f"  articles:       {n}  ({n / covered_days:.1f}/day over covered span)")
    print(f"  with body text: {with_body}/{n}  ({with_body / n:.0%})   <- Pass 2 quote-rule viability")
    print(f"  newest age:     {_fmt_age(newest_age)}   <- feed staleness")
    print(f"  oldest age:     {_fmt_age(oldest_age)}" + (f"   <- CAPPED: feed truncated well short of {days}d" if capped else ""))
    print(f"  by source:      {by_source}")


def _print_articles(articles: list[news.Article]) -> None:
    for a in articles:
        body = "[body]" if a.has_body else "[headline-only]"
        print(f"    {a.published_at:%Y-%m-%d %H:%M} {body:16} {a.source:9} {a.headline}")


def _classify(ticker: str, articles: list[news.Article], descriptions: list[str]) -> None:
    """Run the two-pass classifier against ad-hoc catalysts and print verdicts."""
    from pipeline import llm  # lazy: only needs anthropic/pydantic when actually used

    if not descriptions:
        print("  --classify needs at least one --catalyst \"description\"")
        return

    catalysts = [{"id": f"c{i + 1}", "description": d} for i, d in enumerate(descriptions)]
    print(f"\n  classifying {len(articles)} articles against {len(catalysts)} catalyst(s)...")
    verdicts = llm.classify_batch(articles, catalysts)

    by_article = {a.id: a for a in articles}
    moved = [v for v in verdicts if v.proposed_state != "no_change"]
    print(f"  pass 2 verdicts: {len(verdicts)}  (non-no_change: {len(moved)})")

    for v in verdicts:
        a = by_article.get(v.article_id)
        headline = a.headline[:80] if a else v.article_id[:12]
        print(f"\n  [{v.proposed_state.upper():12}] {v.catalyst_id}  conf={v.confidence:.2f}  src={v.source_kind}")
        print(f"    article: {headline}")
        print(f"    reasoning: {v.reasoning}")
        if v.supporting_quote:
            print(f"    quote: \"{v.supporting_quote}\"")
        if v.guard_note:
            print(f"    !! {v.guard_note}")


def _fmt_age(delta: timedelta) -> str:
    hours = delta.total_seconds() / 3600
    return f"{hours:.1f}h ago" if hours < 48 else f"{hours / 24:.1f}d ago"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kestrel news pipeline spike driver")
    p.add_argument("--ticker", nargs="+", required=True, help="one or more tickers, e.g. NVDA AMD")
    p.add_argument("--days", type=int, default=3, help="look back this many days (default 3)")
    p.add_argument(
        "--sources",
        nargs="+",
        default=["finnhub"],
        help="news sources to merge (default: finnhub)",
    )
    p.add_argument("--show", action="store_true", help="also print each article")
    p.add_argument("--limit", type=int, default=None,
                   help="cap at the N most recent articles (recommended with --classify)")
    p.add_argument("--classify", action="store_true",
                   help="run the two-pass LLM classifier (needs ANTHROPIC_API_KEY)")
    p.add_argument("--catalyst", action="append", metavar="DESC",
                   help="catalyst description to watch for; repeatable")
    return p.parse_args()


if __name__ == "__main__":
    main()
