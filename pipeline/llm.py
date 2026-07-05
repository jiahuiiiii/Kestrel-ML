"""Two-pass catalyst classifier (ml_plan.md §5).

Pass 1 (claude-haiku-4-5): cheap relevance filter over a batch of articles —
drops the ~90% that don't bear on any watched catalyst.
Pass 2 (claude-sonnet-5): careful confirmation judgment on each surviving
(article, catalyst) pair, with a verbatim supporting quote.

The anti-hallucination guards live HERE, in code, after the model call — not
only in the prompt (ml_plan.md §5):
  guard 1: `supporting_quote` must appear verbatim (case/whitespace-normalized)
           in the article text, or the verdict is downgraded to `no_change`.
  guard 3: a headline-only article (no body text) can propose at most `rumored`.
  (guard 2 — speculation can't confirm — is enforced downstream by the
  catalysts.py state machine, which reads `source_kind`.)

Prompts live in pipeline/prompts/*.md, versioned in git; every verdict records
the sha256-derived version of the prompt that produced it (§5).

Contract (§7): classify_batch(articles, catalysts) -> list[CatalystVerdict]
"""

from __future__ import annotations

import hashlib
import html
import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from pipeline.news import Article

log = logging.getLogger(__name__)

PASS1_MODEL = "claude-haiku-4-5"
PASS2_MODEL = "claude-sonnet-5"
PASS1_CHUNK = 25          # articles per Pass-1 call (batched to cut cost/latency)
_PROMPT_DIR = Path(__file__).parent / "prompts"


# --------------------------------------------------------------------------- #
# Output schemas. These ARE the LLM output contract — the API constrains the
# model to them via structured outputs, so no JSON parsing/regex anywhere.
# --------------------------------------------------------------------------- #
class RelevanceItem(BaseModel):
    """Pass-1 judgment for one article in the batch."""
    index: int                                   # position in the submitted batch
    relevant: bool
    catalyst_ids: list[str] = Field(default_factory=list)


class _Pass1Output(BaseModel):
    items: list[RelevanceItem]


class _Pass2Output(BaseModel):
    """What the Pass-2 model emits for one (article, catalyst) pair."""
    catalyst_id: str
    proposed_state: Literal["no_change", "rumored", "confirmed", "invalidated"]
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_quote: str | None                 # verbatim, or null for no_change
    source_kind: Literal["primary", "reporting", "speculation"]
    reasoning: str                               # 1-2 sentences, shown in the UI


class CatalystVerdict(_Pass2Output):
    """A Pass-2 verdict enriched with provenance — what gets persisted
    (catalysts.evidence / evaluations.catalyst_results) and pushed to the UI.
    The extra fields are filled by us, never by the model.

    `classified_at` doubles as the catalyst's state-change timestamp: the
    schema has no `state_updated_at` column by agreement — "when did the state
    last change" is read off the newest evidence entry, so this field must
    survive persistence intact.
    """
    article_id: str
    prompt_version: str
    classified_at: str                           # ISO-8601 UTC, stamped at guard time
    guard_note: str | None = None                # set when a guard rewrote the verdict


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def classify_batch(articles: list[Article], catalysts: list[dict]) -> list[CatalystVerdict]:
    """Run the full two-pass pipeline.

    Args:
        articles:  normalized articles for ONE ticker (news.fetch output).
        catalysts: the ticker's active catalysts, each {"id": ..., "description": ...}.

    Returns:
        One CatalystVerdict per (article, catalyst) pair that survived Pass 1 —
        including `no_change` verdicts (they're evidence too; the state machine
        ignores them). A Pass-2 failure on one pair is logged and skipped, not
        raised, so one bad article can't sink the poll cycle.
    """
    if not articles or not catalysts:
        return []

    pairs = pass1_relevance(articles, catalysts)
    log.info("pass1: %d/%d articles kept (%d pairs)",
             len({a.id for a, _ in pairs}), len(articles), len(pairs))

    verdicts: list[CatalystVerdict] = []
    for article, catalyst in pairs:
        try:
            verdicts.append(pass2_confirm(article, catalyst))
        except Exception as exc:
            log.warning("pass2 failed for article %s / catalyst %s: %s",
                        article.id[:12], catalyst.get("id"), exc)
    return verdicts


# --------------------------------------------------------------------------- #
# Pass 1 — relevance (Haiku).
# --------------------------------------------------------------------------- #
def pass1_relevance(articles: list[Article], catalysts: list[dict]) -> list[tuple[Article, dict]]:
    """Return the (article, catalyst) pairs worth a careful Pass-2 look."""
    by_id = {c["id"]: c for c in catalysts}
    catalyst_block = "\n".join(f"- id: {c['id']} — {c.get('description', '')}" for c in catalysts)

    pairs: list[tuple[Article, dict]] = []
    for start in range(0, len(articles), PASS1_CHUNK):
        chunk = articles[start:start + PASS1_CHUNK]
        article_block = "\n".join(_format_article(i, a) for i, a in enumerate(chunk))
        user_msg = f"CATALYSTS:\n{catalyst_block}\n\nARTICLES:\n{article_block}"

        result = _call(
            model=PASS1_MODEL,
            system=_prompt("pass1_relevance"),
            user=user_msg,
            schema=_Pass1Output,
            max_tokens=4096,
        )

        for item in result.items:
            if not item.relevant or not (0 <= item.index < len(chunk)):
                continue
            article = chunk[item.index]
            for cid in item.catalyst_ids:
                if cid in by_id:              # ignore hallucinated catalyst ids
                    pairs.append((article, by_id[cid]))
    return pairs


def _format_article(index: int, a: Article) -> str:
    body = a.summary if a.has_body else "(headline only)"
    return f"[{index}] HEADLINE: {a.headline}\n    SUMMARY: {body}"


# --------------------------------------------------------------------------- #
# Pass 2 — confirmation (Sonnet), plus the code-enforced guards.
# --------------------------------------------------------------------------- #
def pass2_confirm(article: Article, catalyst: dict) -> CatalystVerdict:
    """Judge one (article, catalyst) pair and apply the anti-hallucination guards."""
    body = article.summary if article.has_body else "(no body text — headline only)"
    user_msg = (
        f"CATALYST: {catalyst['id']} — {catalyst.get('description', '')}\n\n"
        f"ARTICLE (via {article.source}, {article.published_at:%Y-%m-%d}):\n"
        f"HEADLINE: {article.headline}\n"
        f"BODY: {body}"
    )

    raw = _call(
        model=PASS2_MODEL,
        system=_prompt("pass2_confirmation"),
        user=user_msg,
        schema=_Pass2Output,
        max_tokens=2048,
        effort="low",   # classifier, not an essay — adaptive thinking stays on
    )
    return _apply_guards(raw, article)


def _apply_guards(raw: _Pass2Output, article: Article) -> CatalystVerdict:
    """Enforce in code what the prompt merely requests. Returns the final verdict."""
    state = raw.proposed_state
    note = None

    if state != "no_change":
        # Guard 1: the quote must exist and appear verbatim in the article.
        if not raw.supporting_quote:
            state, note = "no_change", "guard: verdict voided — no supporting quote given"
        elif not quote_in_article(raw.supporting_quote, article):
            state, note = "no_change", "guard: verdict voided — quote not found verbatim in article"
        # Guard 3: headline-only articles can never confirm (or invalidate).
        elif not article.has_body and state in ("confirmed", "invalidated"):
            state, note = "rumored", f"guard: headline-only article — {raw.proposed_state} capped at rumored"

    return CatalystVerdict(
        **{**raw.model_dump(), "proposed_state": state},
        article_id=article.id,
        prompt_version=prompt_version("pass2_confirmation"),
        classified_at=datetime.now(timezone.utc).isoformat(),
        guard_note=note,
    )


def quote_in_article(quote: str, article: Article) -> bool:
    """Case-insensitive, whitespace-normalized verbatim substring check."""
    haystack = _norm(f"{article.headline} {article.summary or ''}")
    return _norm(quote) in haystack


def _norm(text: str) -> str:
    # html.unescape on BOTH sides keeps the check verbatim while tolerating
    # entity-encoded article text (older fixtures predate ingestion unescaping).
    return " ".join(html.unescape(text).lower().split())


# --------------------------------------------------------------------------- #
# Prompt loading + versioning.
# --------------------------------------------------------------------------- #
@lru_cache
def _prompt(name: str) -> str:
    return (_PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


@lru_cache
def prompt_version(name: str) -> str:
    """Short content hash of a prompt file — persisted with every verdict so
    metric shifts can be attributed to prompt changes (§5)."""
    return hashlib.sha256(_prompt(name).encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# The single Anthropic call site.
# --------------------------------------------------------------------------- #
def _call(*, model: str, system: str, user: str, schema: type[BaseModel],
          max_tokens: int, effort: str | None = None):
    """One structured-outputs call. `client.messages.parse` validates the
    response against `schema` — no hand-rolled JSON parsing anywhere.

    The system prompt carries a cache_control marker: it's identical across
    calls within a poll cycle, so Pass-2 calls after the first read it from
    cache. (Below the model's minimum cacheable prefix it silently no-ops.)
    """
    kwargs: dict = {}
    if effort is not None:
        kwargs["output_config"] = {"effort": effort}

    response = _client().messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        output_format=schema,
        **kwargs,
    )
    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError(f"{model} returned unparseable output (stop_reason={response.stop_reason})")
    return parsed


@lru_cache
def _client():
    import anthropic  # lazy: lets the deterministic tests run without the SDK installed
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
