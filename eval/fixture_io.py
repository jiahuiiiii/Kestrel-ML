"""Fixture load/save + label validation for the eval harness (ml_plan.md §6).

One fixture = one historical catalyst event, as a directory:

    eval/fixtures/<event>/
    ├── articles.json   # frozen Article list, written by backfill.py
    └── labels.json     # hand labels — see fixtures/README.md

labels.json shape:
    {
      "event": "nvda_2026-05-28_earnings",
      "ticker": "NVDA",
      "catalysts": [{"id": "earnings_beat", "description": "..."}],
      "labels": [
        {"article_id": "...", "catalyst_id": "earnings_beat",
         "relevant": true, "expected_state": "confirmed"},
        ...
      ],
      "expected_final": {"earnings_beat": "confirmed"},
      "expected_confirm_article": {"earnings_beat": "<article_id>" | null}
    }

`labels` covers every (article, catalyst) pair; backfill.py writes a skeleton
with relevant=false / expected_state="no_change" and the human edits the
interesting rows. `expected_state` is the correct POST-GUARD Pass-2 verdict for
that single pair, using the same vocabulary the classifier emits.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from pipeline.news import Article

FIXTURES_DIR = Path(__file__).parent / "fixtures"

PAIR_STATES = ("no_change", "rumored", "confirmed", "invalidated")   # per-pair labels
FINAL_STATES = ("unconfirmed", "rumored", "confirmed", "invalidated")  # end-of-replay labels


# --------------------------------------------------------------------------- #
# Article <-> JSON.
# --------------------------------------------------------------------------- #
def article_to_dict(a: Article) -> dict:
    d = asdict(a)
    d["published_at"] = a.published_at.isoformat()
    return d


def article_from_dict(d: dict) -> Article:
    return Article(
        id=d["id"],
        ticker=d["ticker"],
        headline=d["headline"],
        summary=d.get("summary"),
        source=d["source"],
        url=d["url"],
        published_at=datetime.fromisoformat(d["published_at"]),
    )


# --------------------------------------------------------------------------- #
# Fixture load/save.
# --------------------------------------------------------------------------- #
def fixture_dir(event: str) -> Path:
    return FIXTURES_DIR / event


def list_fixtures() -> list[str]:
    if not FIXTURES_DIR.is_dir():
        return []
    return sorted(p.name for p in FIXTURES_DIR.iterdir() if (p / "labels.json").is_file())


def save_articles(event: str, articles: list[Article]) -> Path:
    path = fixture_dir(event) / "articles.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [article_to_dict(a) for a in sorted(articles, key=lambda a: a.published_at)]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_articles(event: str) -> list[Article]:
    path = fixture_dir(event) / "articles.json"
    return [article_from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def save_labels(event: str, doc: dict) -> Path:
    path = fixture_dir(event) / "labels.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_labels(event: str) -> dict:
    path = fixture_dir(event) / "labels.json"
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Label validation — catch labeling mistakes before spending API calls.
# --------------------------------------------------------------------------- #
def validate(articles: list[Article], doc: dict) -> list[str]:
    """Return a list of human-readable problems (empty = fixture is sound)."""
    problems: list[str] = []
    article_ids = {a.id for a in articles}
    catalyst_ids = {c.get("id") for c in doc.get("catalysts", [])}

    if not catalyst_ids:
        problems.append("no catalysts defined")

    seen_pairs: set[tuple[str, str]] = set()
    for i, row in enumerate(doc.get("labels", [])):
        where = f"labels[{i}]"
        aid, cid = row.get("article_id"), row.get("catalyst_id")
        if aid not in article_ids:
            problems.append(f"{where}: unknown article_id {str(aid)[:12]!r}")
        if cid not in catalyst_ids:
            problems.append(f"{where}: unknown catalyst_id {cid!r}")
        if (aid, cid) in seen_pairs:
            problems.append(f"{where}: duplicate pair ({str(aid)[:12]}, {cid})")
        seen_pairs.add((aid, cid))
        if row.get("expected_state") not in PAIR_STATES:
            problems.append(f"{where}: expected_state {row.get('expected_state')!r} "
                            f"not in {list(PAIR_STATES)}")
        if not row.get("relevant") and row.get("expected_state") != "no_change":
            problems.append(f"{where}: relevant=false but expected_state != no_change")

    for cid, state in (doc.get("expected_final") or {}).items():
        if cid not in catalyst_ids:
            problems.append(f"expected_final: unknown catalyst_id {cid!r}")
        if state not in FINAL_STATES:
            problems.append(f"expected_final[{cid}]: {state!r} not in {list(FINAL_STATES)}")

    for cid, aid in (doc.get("expected_confirm_article") or {}).items():
        if cid not in catalyst_ids:
            problems.append(f"expected_confirm_article: unknown catalyst_id {cid!r}")
        if aid is not None and aid not in article_ids:
            problems.append(f"expected_confirm_article[{cid}]: unknown article_id {str(aid)[:12]!r}")

    unlabeled = len(article_ids) * len(catalyst_ids) - len(seen_pairs)
    if unlabeled > 0:
        problems.append(f"note: {unlabeled} (article, catalyst) pairs have no label row "
                        "(they'll be excluded from Pass-1/Pass-2 metrics)")
    return problems
