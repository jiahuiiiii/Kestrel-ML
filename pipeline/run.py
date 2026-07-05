"""CLI to exercise the news pipeline — the Phase-0 spike driver (ml_plan.md §2).

Loads .env (for FINNHUB_API_KEY), fetches recent news for one or more tickers,
and prints the numbers the spike needs to decide the primary source:
  - articles per day (after dedup)
  - fraction with quotable body text  (has_body) vs headline-only
  - staleness of the newest article   (how far behind real-time the feed is)

With --classify it also runs the two-pass LLM classifier (needs OPENAI_API_KEY
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
import re
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from pipeline import news

_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)([mhd])$", re.IGNORECASE)
_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400}


def parse_duration(value: str) -> timedelta:
    """Parse a duration string like '6h', '2d', '90m', '1.5d' into a timedelta.

    Supported units:
        m  — minutes
        h  — hours
        d  — days

    Raises argparse.ArgumentTypeError on bad input so argparse surfaces it cleanly.
    """
    m = _DURATION_RE.match(value.strip())
    if not m:
        raise argparse.ArgumentTypeError(
            f"invalid duration {value!r} — expected a number followed by m/h/d "
            f"(e.g. '6h', '2d', '90m', '1.5d')"
        )
    amount, unit = float(m.group(1)), m.group(2).lower()
    return timedelta(seconds=amount * _UNIT_SECONDS[unit])


def _fmt_duration(td: timedelta) -> str:
    """Human-readable label for a timedelta, collapsing to the largest clean unit."""
    total_seconds = td.total_seconds()
    if total_seconds < 3600:
        return f"{total_seconds / 60:.4g}m"
    if total_seconds < 86400:
        return f"{total_seconds / 3600:.4g}h"
    return f"{total_seconds / 86400:.4g}d"


def main() -> None:
    load_dotenv()  # reads .env in the project root -> FINNHUB_API_KEY into os.environ

    args = _parse_args()
    now = datetime.now(timezone.utc)
    since = now - args.lookback
    window_label = _fmt_duration(args.lookback)

    for ticker in args.ticker:
        try:
            articles = news.fetch(ticker, since, sources=args.sources)
        except Exception as exc:
            print(f"\n{ticker}: FETCH FAILED — {exc}")
            continue

        if args.limit and len(articles) > args.limit:
            articles = articles[-args.limit:]  # keep the most recent N

        _print_summary(ticker, articles, lookback=args.lookback, window_label=window_label, now=now)
        if args.show:
            _print_articles(articles)
        if args.classify:
            _classify(ticker, articles, args.catalyst or [])


def _print_summary(ticker: str, articles: list[news.Article], *, lookback: timedelta, window_label: str, now: datetime) -> None:
    n = len(articles)
    print(f"\n=== {ticker} — last {window_label} ===")
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
    capped = oldest_age < lookback * 0.9 # feed didn't reach back as far as we asked

    by_source: dict[str, int] = {}
    for a in articles:
        by_source[a.source] = by_source.get(a.source, 0) + 1

    print(f"  articles:       {n}  ({n / covered_days:.1f}/day over covered span)")
    print(f"  with body text: {with_body}/{n}  ({with_body / n:.0%})   <- Pass 2 quote-rule viability")
    print(f"  newest age:     {_fmt_age(newest_age)}   <- feed staleness")
    print(f"  oldest age:     {_fmt_age(oldest_age)}" + (f"   <- CAPPED: feed truncated well short of {window_label}" if capped else ""))
    print(f"  by source:      {by_source}")


def _print_articles(articles: list[news.Article]) -> None:
    for a in articles:
        body = "[body]" if a.has_body else "[headline-only]"
        print(f"    {a.published_at:%Y-%m-%d %H:%M} {body:16} {a.source:9} {a.headline}")


def _classify(ticker: str, articles: list[news.Article], descriptions: list[str]) -> None:
    """Run the two-pass classifier against ad-hoc catalysts and print verdicts."""
    from pipeline import llm  # lazy: only needs openai/pydantic when actually used

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

    window = p.add_mutually_exclusive_group()
    window.add_argument(
        "--since",
        dest="lookback",
        type=parse_duration,
        metavar="DURATION",
        help=(
            "look-back window as number + unit: m (minutes), h (hours), d (days). "
            "E.g. '6h', '2d', '90m', '1.5d'. Default: 3d."
        ),
    )
    window.add_argument(
        "--days",
        dest="lookback",
        type=lambda x: parse_duration(f"{x}d"),
        metavar="N",
        help="[legacy] equivalent to --since Nd. Use --since instead.",
    )

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
                   help="run the two-pass LLM classifier (needs OPENAI_API_KEY)")
    p.add_argument("--catalyst", action="append", metavar="DESC",
                   help="catalyst description to watch for; repeatable")
    return p.parse_args()


if __name__ == "__main__":
    main()
