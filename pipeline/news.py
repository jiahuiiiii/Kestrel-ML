"""News fetching + normalization for the Kestrel catalyst pipeline.

Every news source (Finnhub, yfinance, ...) gets normalized into one `Article`
shape so the rest of the pipeline never has to know where an article came from.
The public entry point is `fetch()`, which can pull from one or several sources
and merge the results.

Contract (ml_plan.md §7):  fetch(ticker, since) -> list[Article]
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

import requests

log = logging.getLogger(__name__)

FINNHUB_BASE = "https://finnhub.io/api/v1"


# --------------------------------------------------------------------------- #
# The normalized shape everything downstream consumes.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Article:
    id: str                  # sha256(url) — stable dedup key
    ticker: str
    headline: str
    summary: str | None      # body text / snippet when available; None = headline-only
    source: str              # which adapter produced it ("finnhub", "yfinance", ...)
    url: str
    published_at: datetime   # timezone-aware, UTC

    @property
    def has_body(self) -> bool:
        """True if there's quotable text beyond the headline.

        Pass 2 uses this: a headline-only article can never yield a
        `confirmed` verdict (ml_plan.md §5, guard 3).
        """
        return bool(self.summary and self.summary.strip())


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def fetch(
    ticker: str,
    since: datetime,
    sources: Iterable[str] = ("finnhub",),
) -> list[Article]:
    """Fetch normalized, deduped articles for `ticker` published on/after `since`.

    Args:
        ticker:  e.g. "NVDA".
        since:   only return articles at or after this instant (tz-aware).
        sources: one or more adapter names to pull from and merge. Pass
                 ("finnhub", "yfinance") to use both — results are merged and
                 deduped by URL.

    Returns:
        Articles sorted oldest-first, deduped by URL across all sources. Empty
        list if the sources returned nothing.

    Raises:
        ValueError:  an unknown source name was requested.
        RuntimeError: every requested source failed (a single source failing
                      among several is logged and skipped, not raised).
    """
    sources = tuple(sources)
    unknown = [s for s in sources if s not in _ADAPTERS]
    if unknown:
        raise ValueError(f"unknown news source(s) {unknown}; have {list(_ADAPTERS)}")

    since = _ensure_utc(since)
    collected: list[Article] = []
    failures: list[tuple[str, Exception]] = []

    for name in sources:
        try:
            collected.extend(a for a in _ADAPTERS[name](ticker, since) if a.published_at >= since)
        except Exception as exc:  # one bad source shouldn't sink the rest
            log.warning("news source %r failed for %s: %s", name, ticker, exc)
            failures.append((name, exc))

    if failures and len(failures) == len(sources):
        raise RuntimeError(f"all news sources failed for {ticker}: {failures}")

    return sorted(_dedup(collected), key=lambda a: a.published_at)


# --------------------------------------------------------------------------- #
# Normalization + dedup helpers (source-agnostic).
# --------------------------------------------------------------------------- #
def make_article(
    *,
    ticker: str,
    headline: str,
    url: str,
    published_at: datetime,
    source: str,
    summary: str | None = None,
) -> Article:
    """Build an Article, computing the dedup id and cleaning fields.

    Every adapter builds its Articles through here rather than constructing the
    dataclass directly — keeps id/cleaning rules in one place.
    """
    return Article(
        id=_article_id(url),
        ticker=ticker.upper().strip(),
        headline=_clean(headline) or "(no headline)",
        summary=_clean(summary) or None,
        source=source,
        url=url.strip(),
        published_at=_ensure_utc(published_at),
    )


def _article_id(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def _dedup(articles: Iterable[Article]) -> list[Article]:
    """Drop duplicate URLs, keeping the first-seen.

    NOTE: dedup is by URL only. The same story from two different sources has
    two different URLs, so cross-source near-duplicates are NOT collapsed. That
    is an accepted non-goal (ml_plan.md §9); revisit with fuzzy title matching
    only if merged feeds get noisy.
    """
    seen: set[str] = set()
    out: list[Article] = []
    for a in articles:
        if a.id not in seen:
            seen.add(a.id)
            out.append(a)
    return out


def _clean(text: str | None) -> str | None:
    """Collapse whitespace; return None for empty."""
    if not text:
        return None
    return " ".join(text.split()).strip() or None


def _ensure_utc(dt: datetime) -> datetime:
    """Coerce to timezone-aware UTC. Naive datetimes are assumed UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Source adapters. Signature: (ticker, since) -> Iterable[Article].
# --------------------------------------------------------------------------- #
def _fetch_finnhub(ticker: str, since: datetime) -> Iterable[Article]:
    """Primary source: Finnhub `company-news` (headline + summary paragraph).

    Free tier: 60 calls/min. Needs FINNHUB_API_KEY in the environment.
    """
    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY not set")

    today = datetime.now(timezone.utc).date()
    params = {
        "symbol": ticker.upper(),
        "from": since.date().isoformat(),
        "to": today.isoformat(),
        "token": api_key,
    }
    items = _get_json(f"{FINNHUB_BASE}/company-news", params)

    for item in items:
        url = item.get("url")
        ts = item.get("datetime")  # unix seconds
        if not url or not ts:
            continue  # skip malformed rows rather than crash the batch
        yield make_article(
            ticker=ticker,
            headline=item.get("headline", ""),
            summary=item.get("summary"),  # usually 1-3 sentences; may be ""
            url=url,
            published_at=datetime.fromtimestamp(ts, tz=timezone.utc),
            source="finnhub",
        )


def _fetch_yfinance(ticker: str, since: datetime) -> Iterable[Article]:
    """Backup source: yfinance `.news`. Thin (often headline only) and the shape
    shifts between library versions, so every field access is guarded.

    Imported lazily so the package doesn't hard-depend on yfinance when only
    Finnhub is used.
    """
    import yfinance as yf  # lazy: optional dependency

    raw = getattr(yf.Ticker(ticker), "news", None) or []
    for item in raw:
        # yfinance has moved fields under a "content" sub-dict in recent versions;
        # accept either layout.
        node = item.get("content", item)
        url = node.get("canonicalUrl", {}).get("url") or node.get("link")
        title = node.get("title") or node.get("headline")
        pub = node.get("pubDate") or node.get("providerPublishTime")
        if not url or not title or pub is None:
            continue
        published_at = (
            datetime.fromisoformat(pub.replace("Z", "+00:00"))
            if isinstance(pub, str)
            else datetime.fromtimestamp(pub, tz=timezone.utc)
        )
        yield make_article(
            ticker=ticker,
            headline=title,
            summary=node.get("summary"),  # frequently absent -> headline-only
            url=url,
            published_at=published_at,
            source="yfinance",
        )


# --------------------------------------------------------------------------- #
# HTTP helper (requests, with a single 429 backoff retry).
# --------------------------------------------------------------------------- #
def _get_json(url: str, params: dict, *, timeout: float = 10.0) -> list:
    for attempt in range(2):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 429 and attempt == 0:
            time.sleep(1.0)  # free tier: 60/min — brief backoff, then retry once
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()  # second 429 -> surface it
    return []


_ADAPTERS: dict[str, Callable[[str, datetime], Iterable[Article]]] = {
    "finnhub": _fetch_finnhub,
    "yfinance": _fetch_yfinance,
}
