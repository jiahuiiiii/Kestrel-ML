# Kestrel — News → Catalyst Pipeline

The **news → LLM catalyst classifier → evaluation** pipeline for Kestrel, a personal
stock-watchlist monitor — plus the eval/replay harness that proves it works. Full design
rationale: [`ml_plan.md`](ml_plan.md).

It's a **standalone Python package of pure functions over plain dicts/dataclasses.** It
never touches a DB, a queue, or a web framework — the host application's scheduler fetches
a thesis, calls these functions, and persists what they return. The only integration
surface is the four function signatures in [Integration contract](#integration-contract)
below.

```
pipeline/
  news.py         fetch + normalize + dedup articles (Finnhub primary, yfinance backup)
  llm.py          two-pass classifier: Pass 1 relevance (gpt-5.4-mini) → Pass 2 confirm (gpt-5.4)
  catalysts.py    catalyst state machine (unconfirmed → rumored → confirmed / invalidated)
  evaluator.py    combine quant results + catalyst states → a signal (+ "why not firing")
  prompts/        the two system prompts, versioned as files
  run.py          CLI to exercise the pipeline on live news
eval/
  backfill.py     capture a historical event's articles into a labeled fixture
  replay.py       feed a fixture through the pipeline, score against hand labels
  fixtures/       3 labeled events (nvda earnings, orcl layoffs, meta compute)
  results/        metric runs, kept in git so regressions show in history
tests/            deterministic unit tests (no network, no API key)
```

## Setup

Requires **Python 3.11+**. Two API keys, both free-tier-friendly.

```bash
git clone https://github.com/jiahuiiiii/Kestrel-ML.git
cd Kestrel-ML
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy the env template and fill in your keys:

```bash
cp pipeline/.env.example pipeline/.env   # then edit pipeline/.env
```

```
FINNHUB_API_KEY=    # free tier at finnhub.io — the news source
OPENAI_API_KEY=     # the two-pass classifier (gpt-5.4-mini + gpt-5.4)
```

`pipeline/.env` is git-ignored; keys never get committed. The tests and the news-fetch
path don't need `OPENAI_API_KEY` — only `--classify` and a live `replay` do.

## Quick check it works (no API key needed)

```bash
python -m pytest tests/ -q      # 69 tests: state machine, evaluator, guards, replay scoring
```

## Run it on live news

**Fetch + inspect** (needs `FINNHUB_API_KEY` only) — how many articles, how fresh, how many
have quotable body text:

```bash
python -m pipeline.run --ticker NVDA --days 1
```

**Full two-pass classify** (needs `OPENAI_API_KEY`; costs a few cents) — watch the classifier
judge live articles against an ad-hoc catalyst. Here `--catalyst` is a plain description:

```bash
python -m pipeline.run --ticker META --days 3 --limit 20 --classify \
    --catalyst "Meta enters the cloud computing business, renting compute to external customers"
```

Each surviving article prints its verdict, confidence, source kind, one-line reasoning, and
the **verbatim quote** that supports any non-`no_change` state (see [guards](#anti-hallucination-guards)).

## The eval harness (how we prove it's right, and the demo)

Catalysts are rare, so we don't wait for live news to fire. Instead we replay known
historical events with hand-labeled ground truth. **This is also the symposium demo.**

```bash
python -m eval.replay --all --dry-run   # validate labels, free, no API calls
python -m eval.replay --all             # score every fixture on live pipeline (costs cents)
```

Reports land in `eval/results/` (kept in git). What it measures: Pass-1 relevance recall &
precision, Pass-2 confusion matrix + confirmed precision/recall + quote-validity, and the
end-to-end check that each catalyst reached `confirmed` **at the right article, not before**.

To build a **new** fixture for an event you know the ground truth of, see
[`eval/fixtures/README.md`](eval/fixtures/README.md) for the `backfill` → label → `replay` loop.

Current fixtures, all passing end-to-end:

| Fixture                     | Event                                 | Tests                        |
| --------------------------- | ------------------------------------- | ---------------------------- |
| `nvda_2026-05-20_earnings`  | NVDA Q1 FY27 earnings beat            | straight-to-confirmed        |
| `orcl_2026-06-22_layoffs`   | Oracle ~21k layoffs via annual filing | filing → reporting wave      |
| `meta_2026-07_meta_compute` | Meta Compute cloud launch             | full rumored → confirmed arc |

## Integration contract

These four functions are the entire seam between this package and the host application.
They're sync, pure (beyond the news fetch + LLM call), and operate on plain dicts/dataclasses
— import them, no setup. Freeze these shapes and the two sides can be built in parallel
([`ml_plan.md §7`](ml_plan.md)).

```python
from datetime import datetime, timezone
from pipeline import news, llm, catalysts, evaluator

# 1. Fetch normalized, deduped articles for one ticker since a timestamp.
articles = news.fetch(ticker, since, sources=("finnhub",), until=None)  # -> list[news.Article]

# 2. Two-pass classify articles against a ticker's active catalysts.
#    catalysts arg: [{"id": "...", "description": "..."}]
verdicts = llm.classify_batch(articles, catalyst_defs)                  # -> list[llm.CatalystVerdict]

# 3. Apply one verdict to a catalyst's current state (the state machine).
transition = catalysts.apply(current_state, verdict)                    # -> catalysts.Transition
#    transition.new_state (CatalystState), .changed (bool), .note (str), .as_tuple()

# 4. Combine quant results + catalyst states into a signal.
result = evaluator.evaluate(thesis, quant_results, catalyst_states)     # -> dict (shape below)
```

### Shapes to persist

**`news.Article`** (frozen dataclass):

```python
id: str            # sha256(url) — your news_seen dedup key
ticker: str
headline: str
summary: str | None    # body text / snippet; None = headline-only
source: str
url: str
published_at: datetime  # tz-aware UTC
```

**`llm.CatalystVerdict`** — `.model_dump()` into `catalysts.evidence` / `evaluations.catalyst_results`,
and push over WS to the frontend reasoning panel:

```python
catalyst_id: str
proposed_state: "no_change" | "rumored" | "confirmed" | "invalidated"
confidence: float                 # 0.0–1.0
supporting_quote: str | None      # verbatim from the article; None for no_change
source_kind: "primary" | "reporting" | "speculation"
reasoning: str                    # 1–2 sentences for the UI
article_id: str                   # provenance
prompt_version: str               # which prompt file produced this
classified_at: str                # ISO-8601 UTC — doubles as the state-change timestamp
guard_note: str | None            # set when a guard rewrote the verdict
```

**`evaluator.evaluate(...)` output**:

```python
{
  "ticker": str,
  "signal": bool,            # the thing that fires an alert
  "status": "firing" | "not_met" | "incomplete",   # incomplete = missing quant data, not "false"
  "quant_ok": bool,
  "catalysts_ok": bool,
  "blocked_by": [str],       # why it's NOT firing — render this in the dashboard
  "reason": str,             # one human sentence
  "evaluated_at": str,       # ISO-8601 UTC
}
```

### Anti-hallucination guards

Enforced in **code**, after the model call — not just in the prompt (`pipeline/llm.py`):

1. `supporting_quote` must appear **verbatim** (case/whitespace/HTML-entity normalized) in the
   article, or the verdict is downgraded to `no_change`. The one guard the model can't talk past.
2. `confirmed` requires a credible `source_kind` — speculation caps at `rumored` (enforced by the
   `catalysts.py` state machine).
3. A headline-only article (no body text) can propose at most `rumored`.

### Schema asks for `V1__init.sql`

Raise before it's locked (details in [`ml_plan.md §3, §7`](ml_plan.md)):

- `catalysts.state` enum (`unconfirmed`/`rumored`/`confirmed`/`invalidated`) + `evidence` JSON
  (there is **no** `state_updated_at` column by agreement — read it off the newest evidence entry's
  `classified_at`).
- `quant_mode` / `catalyst_mode` (`all`/`any`/`none_required`) on `theses`.
- `prompt_version` on evaluation rows.

Two contract refinements are supersets of the original `ml_plan.md §7` shapes — flag before locking:
`Transition.note` (a human string; `.as_tuple()` gives the minimal `(state, changed)`), and the
evaluator's extra `ticker` / `status` fields.

## Further development

The pipeline package is complete and provable in isolation (3 positive + 1 negative
fixtures, all green). The remaining work is integration and expansion — roughly in
priority order.

### 1. Wire it into the host application

The scheduler owns the loop; this package owns the judgment. Per poll cycle, for each
watched ticker:

```python
articles = news.fetch(ticker, since=last_poll)          # since = your news_seen high-water mark
verdicts = llm.classify_batch(articles, ticker_catalysts)
for v in verdicts:
    new_state, changed = catalysts.apply(current_state[v.catalyst_id], v).as_tuple()
    # persist v.model_dump() as evidence; update the catalyst's state if changed
result = evaluator.evaluate(thesis, quant_results, catalyst_states)
# persist result; if result["signal"] flips true, notify
```

Dedup by `Article.id` (sha256 of the URL) across polls — Finnhub's `from` is date-granular
and re-returns the whole day, so the same article recurs until the DB filters it. See the
[schema asks](#schema-asks-for-v1__initsql) for the columns this needs.

### 2. Demo / replay mode

Add a `--replay <fixture>` flag to the scheduler that feeds
`eval/fixtures/<event>/articles.json` through the same code path instead of live news, at
accelerated speed. Any labeled fixture becomes a deterministic, rehearsable demo — the
catalyst walks `unconfirmed → rumored → confirmed` live in the UI. Essential for showing
the system work when the live watchlist happens to be quiet.

### 3. Frontend reasoning panel

Consumes the same JSON this package already emits — no new contract. Render each
`CatalystVerdict` (state, confidence, source kind, reasoning, and the verbatim
`supporting_quote`) as the agent's evidence trail, and the evaluator output's `blocked_by`
list as "why this isn't firing yet." Build it against a mock WS feed replaying fixture
verdicts so it doesn't block on the backend.

### 4. Grow and maintain the eval

- **More fixtures.** Target 6+ labeled events. The hardest negatives — rumors that fizzled,
  denials, wrong-acquirer M&A — are worth the most, because they're what catch false
  confirmations. Loop: `eval/backfill.py` (capture + `--keep-match` curation) → hand-label
  `labels.json` → `eval/replay.py`. Full walkthrough in
  [`eval/fixtures/README.md`](eval/fixtures/README.md).
- **Tune prompts against metrics, not vibes.** Edit `pipeline/prompts/*.md`, re-run
  `python -m eval.replay --all`, diff `eval/results/`. Every verdict records its
  `prompt_version`, so a metric shift is attributable to a prompt change vs. the data.
  Commit a replay run alongside each prompt edit so regressions surface in git history.

### 5. Possible v2 directions

- **Catalyst proposals.** Reuse the same news stream to _suggest_ catalysts a thesis isn't
  watching yet (add / remove / update), gated by human approval. This is a distinct
  classification task with its own prompt, output schema, and eval — not a flag on Pass 2.
  Share the news fetch, not the pass.
- **Adaptive valuation thresholds.** Replace static quant conditions (`forward_pe < 28`)
  with anomaly-relative ones ("8th percentile of its own two-year range"). A separate
  workstream; joins as a proposal generator.

### Operational

- **Rotate any key that has ever been committed.** Untracking `.env` stops future leaks,
  not past ones — a key in git history is compromised until rotated at its provider.
- Prompts are loaded at import and versioned by content hash; `.env` stays local (git-ignored).

## Notes

- **Models:** Pass 1 `gpt-5.4-mini`, Pass 2 `gpt-5.4` (structured outputs via
  `chat.completions.parse`, `reasoning_effort="low"`). Cost is trivial (~$1/day for 10 tickers
  hourly); correctness is the constraint.
- **Prompts live in `pipeline/prompts/*.md`** and are versioned by content hash — every persisted
  verdict records which prompt produced it, so metric shifts are attributable.
- **`ml_plan.md`** is the full design rationale if any decision here looks surprising.
