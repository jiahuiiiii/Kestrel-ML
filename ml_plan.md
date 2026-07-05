# Kestrel — ML/LLM Pipeline Plan (Amelia's half)

Companion to `bot_plan.md`. That doc covers the full system; this one covers only the part I own: **the news → LLM catalyst classifier → evaluation pipeline**, plus the eval/replay harness that proves it works. Brandon owns the entire backend (FastAPI, Postgres, scheduler, notifications); I own this pipeline and the React frontend.

The whole doc is organized around one constraint: **I must be able to build and test this with zero dependency on Brandon's backend.** No Postgres, no FastAPI, no scheduler — a standalone Python package driven by fixture files and a CLI, that Brandon later imports from `scheduler.py`.

---

## 1. Ownership boundary

| Mine | Brandon's |
|---|---|
| `pipeline/news.py` — fetch + normalize + dedup articles | `backend/` everything: FastAPI, models.py, migrations, scheduler |
| `pipeline/llm.py` — two-pass catalyst classifier | `services/fundamentals.py` (yfinance quant checks) |
| `pipeline/catalysts.py` — catalyst state machine | `notify.py`, WebSocket plumbing |
| `pipeline/evaluator.py` — combine quant + catalyst results → signal | Wiring my package into the scheduler + persisting results |
| `pipeline/prompts/` — all prompt text, versioned as files | |
| `eval/` — replay harness, labeled fixtures, metrics | |
| Frontend agent-reasoning panel (consumes my output JSON) | |

**How the seam works:** my package is pure functions over plain dicts/dataclasses. It takes a thesis dict + articles in, returns verdict dicts out. It never touches the DB. Brandon's `scheduler.py` does: fetch thesis from DB → call my functions → persist result. That means we only need to agree on the JSON shapes (section 7) and can then work fully in parallel. Bonus: the same output JSON is what my frontend reasoning panel renders, so defining it carefully serves both of my jobs at once.

```
pipeline/                 ← my package, importable standalone
├── news.py
├── llm.py
├── catalysts.py
├── evaluator.py
├── prompts/
│   ├── pass1_relevance.md
│   └── pass2_confirmation.md
├── run.py                ← CLI: python -m pipeline.run --ticker NVDA --fixtures eval/fixtures/
eval/
├── fixtures/             ← saved historical articles + hand labels
│   └── nvda_2025-05-28_earnings/   (one directory per known catalyst event)
├── replay.py             ← feed fixtures through the pipeline, score against labels
└── results/              ← metric runs, kept in git so we can see regressions
```

---

## 2. Phase 0 — validate the news source FIRST (before anything else)

This is the load-bearing decision the main plan skips. The Pass-2 anti-hallucination rule ("quote the exact supporting sentence, no quote → rejected") **only works if we have article text, not just headlines.** So before writing any classifier code, run a half-day spike:

**Candidates, in the order I'd try them:**

1. **Finnhub (free tier)** — `company-news` endpoint gives headline + summary paragraph per ticker, 60 calls/min free. Summaries are usually 1–3 sentences: enough for Pass 2 to quote from, most of the time.
2. **yfinance `.news`** — free, zero setup, but thin (headline + publisher + link, content often missing) and breaks with library updates. Acceptable as a secondary/backup feed, not the primary.
3. **RSS (Yahoo Finance per-ticker feed, Google News query feed)** — free, gives headline + description snippet. No API key, but parsing is fragile and dedup across feeds is on us.
4. **NewsAPI.org free tier** — 24h delay on free tier, which kills the use case. Skip unless paying.

**Spike exit criteria** (write the answers into this doc when done):
- For 3 tickers over 3 days: how many articles/day, and what fraction have ≥1 full sentence of body text (not just a headline)?
- Latency: how stale is the newest article vs. its actual publish time?
- Decision: which source is primary, and — critically — **does Pass 2 keep the strict quote rule, or downgrade to "quote the headline/summary" when body text is unavailable?** My default: keep the quote rule but let it quote the summary snippet; if only a bare headline exists, Pass 2 can return at most `rumored`, never `confirmed`.

**Spike results (2026-07-03) — DECIDED.** Ran `python -m pipeline.run --ticker NVDA AMD AAPL --days 3`:

| Ticker | Articles/day | With body text | Newest age |
|---|---|---|---|
| NVDA | 83 | 100% | 0.9h |
| AMD  | 42 | 98%  | 1.5h |
| AAPL | 59 | 99%  | 1.1h |

Decisions:
- **Primary source: Finnhub.** yfinance not needed — Finnhub already gives ~99–100% body text and ~1h freshness. Keep the yfinance adapter as an unused backup only.
- **Pass 2 keeps the strict quote rule as-is.** Body text is essentially always present, so the "no verbatim quote → downgrade to `no_change`" guard is fully viable; the headline-only fallback path (§5 guard 3) will fire rarely.
- **Volume is ~3× the original cost estimate** (§5 assumed ~200 articles/day across 10 tickers; real is closer to 40–83/day *per* ticker → 500–800/day for 10). Cost is still trivial (~$1/day), but two things follow: (1) **cross-poll dedup is now load-bearing** — an hourly poll must not re-run Pass 1 on already-seen articles; `Article.id` (sha256 of URL) is the dedup key, and it's exactly what Brandon's `news_seen` table keys on. (2) High volume = lots of aggregator/SEO noise, so **Pass 1 precision matters, not just recall** — measure both in the eval harness.
- **Finnhub `from` is date-granularity**, so an hourly poll re-returns the whole day. In-call URL dedup doesn't span polls; the `published_at >= since` filter in `fetch()` trims most of it, but the real dedup is DB-level on `Article.id`. Flagged for Brandon's scheduler.
- **Finnhub `company-news` caps responses at ~250 items/call.** NVDA hit it: a 3-day query returned exactly 250 articles spanning only the most recent ~33.5h, so real NVDA volume is ~180/day (not the 83 first measured — that was a capped count ÷ full window). Two implications: (1) **production is unaffected** — hourly polls request a 1h slice, far under the cap; (2) **the eval harness must backfill day-by-day** (or narrower) per fixture window, or wide historical calls get silently truncated. AMD (125) and AAPL (177) were un-capped and complete.

`news.py` normalizes whatever source wins into one shape so the rest of the pipeline never cares where an article came from:

```python
@dataclass
class Article:
    id: str            # sha256(url) — dedup key
    ticker: str
    headline: str
    summary: str | None   # body text or snippet, when available
    source: str
    url: str
    published_at: datetime
```

---

## 3. Catalyst model — a state machine, not a boolean

`bot_plan.md` treats catalyst confirmation as one-shot (`triggered_at` timestamp). Real news contradicts itself ("contract confirmed" Tuesday, "contract delayed" Thursday), so a catalyst carries a state:

```
unconfirmed ──▶ rumored ──▶ confirmed
     ▲             │            │
     └─────────────┴────────────┴──▶ invalidated
```

- **rumored** — an article suggests it but hedges (analyst speculation, "sources say", unconfirmed reports).
- **confirmed** — an article states it as fact from a primary-ish source (company statement, filing, official announcement, earnings release).
- **invalidated** — a later article contradicts the confirmation. This resets the signal.

The classifier's job (Pass 2) is to output a **state transition proposal** for one (article, catalyst) pair, not a bare bool. `catalysts.py` owns the transition rules (e.g., `confirmed` can only be reached from a non-hedged source category; `invalidated` beats everything). This is a plain-Python state machine — easy to unit test with no LLM in the loop.

For Brandon: this means the `catalysts` table wants a `state` column (enum above) + `state_updated_at` + `evidence` (JSON list of the article verdicts that drove each transition), instead of just `triggered_at`. I'll flag this before `V1__init.sql` is locked.

---

## 4. Thesis combination logic

The motivating example ("P/E < 28 AND capex cut AND revenue decel") needs structure the flat schema doesn't have. Minimal fix — one grouping field on the thesis, no expression trees:

```python
# thesis dict shape my evaluator consumes
{
  "ticker": "NVDA",
  "quant_mode": "all",        # "all" | "any" — how quant_conditions combine
  "catalyst_mode": "all",     # "all" | "any" | "none_required"
  "quant_conditions": [...],
  "catalysts": [...],
}
```

`evaluator.evaluate(thesis, quant_results, catalyst_states)` then returns:

```python
{
  "signal": bool,
  "quant_ok": bool,
  "catalysts_ok": bool,
  "blocked_by": ["catalyst:capex_cut is rumored, needs confirmed"],  # why NOT firing — gold for the UI
  "reason": "...",             # one human sentence
  "evaluated_at": iso8601,
}
```

`blocked_by` is deliberately first-class: the dashboard's most common state is "not firing yet," and showing *why not* is what makes the agent's reasoning legible on stage.

**Quant data caveat baked in:** a quant condition whose yfinance value came back `None` evaluates to state `"unknown"`, not `False`. `quant_mode: "all"` with any unknown → thesis is "couldn't evaluate," surfaced as such, never silently false. (The check itself is Brandon's code; the three-valued combination logic is mine.)

---

## 5. The two-pass classifier (`llm.py`)

Design carried over from `bot_plan.md`, updated to current models and hardened.

### Models (switched to OpenAI 2026-07-05; original build used claude-haiku-4-5 / claude-sonnet-5)

| Pass | Model | Why | Price (per MTok in/out) |
|---|---|---|---|
| 1 — relevance filter | `gpt-5.4-mini` | Cheap, fast; drops ~90% of articles | $0.75 / $4.50 |
| 2 — confirmation | `gpt-5.4` | Real reasoning + quoting | $2.50 / $15 |

Two GPT-5.4 gotchas learned the hard way (both caught by the replay harness):
- **Reasoning tokens share the output budget.** A 25-article Pass-1 batch with a
  tight `max_completion_tokens` intermittently truncates — articles silently drop
  and recall craters (measured 56% one run, 100% the next). Give batched calls
  generous headroom (8192); reasoning models only bill tokens actually used.
- Pass `reasoning_effort="low"` — it's a classifier, not an essay. Structured
  outputs (`chat.completions.parse` + Pydantic `response_format`) replace all
  JSON parsing, same as before.

### Pass 1 — relevance (Haiku)

Input: article headline+summary, plus the compact list of this ticker's active catalysts. Output via **structured outputs** (`client.messages.parse()` with a Pydantic model — guaranteed valid JSON, no regex parsing):

```python
class RelevanceVerdict(BaseModel):
    relevant: bool
    catalyst_ids: list[str]   # which catalysts this might bear on (empty if none)
```

Batch all of a poll cycle's headlines for one ticker into a single call (list in → list out) instead of one call per headline — cuts cost and latency substantially.

### Pass 2 — confirmation (Sonnet 5)

Runs only on (article, catalyst) pairs Pass 1 flagged. Output schema is the heart of the whole system — it feeds the state machine, the DB, and the frontend reasoning panel:

```python
class CatalystVerdict(BaseModel):
    catalyst_id: str
    proposed_state: Literal["no_change", "rumored", "confirmed", "invalidated"]
    confidence: float                      # 0.0–1.0
    supporting_quote: str | None           # verbatim from the article — REQUIRED for rumored/confirmed/invalidated
    source_kind: Literal["primary", "reporting", "speculation"]
    reasoning: str                         # 1–2 sentences, shown in the UI
```

**Anti-hallucination guards (enforced in code, not just prompt):**
1. `supporting_quote` must appear verbatim (case-insensitive, whitespace-normalized) in the article text — checked with plain string matching in Python after the call. Fails → verdict downgraded to `no_change`. This is the guard the LLM can't talk its way past.
2. `proposed_state: "confirmed"` requires `source_kind: "primary" | "reporting"` — speculation caps at `rumored`. (State machine rule, section 3.)
3. Bare-headline articles (no summary text) cap at `rumored` regardless of what the model says.

**Prompt caching:** the system prompt (task instructions + output rules + few-shot examples) is identical across every call and always comes first, so OpenAI's automatic prefix caching applies with no explicit marker (kicks in around 1K tokens; `prompt_cache_key` pins calls to one cache lane). It pays off *within* a poll cycle when several articles hit Pass 2.

**Prompts live in `pipeline/prompts/*.md`,** loaded at import — versioned in git, diffable in PRs, editable without touching code. Every verdict we persist records which prompt version produced it (a hash of the prompt file), so when I tune a prompt I can tell whether metrics moved because of the prompt or the data.

### Cost sanity check

Assume 10 tickers, hourly polls, ~200 new articles/day total after dedup:
- Pass 1: 200 articles × ~800 tokens in / 50 out ≈ 0.17 MTok/day in → **~$0.20/day** on Haiku.
- Pass 2: ~10% survive → 20 calls × ~2.5K in / 200 out → **~$0.20/day** on Sonnet (intro pricing).

Rounding error. Cost is not a design constraint here; correctness is.

---

## 6. Eval + replay harness (`eval/`) — this is also the demo

The problem: catalysts are rare. In a 2–3 week window, nothing on the live watchlist may fire — leaving us with no way to know the classifier works and nothing to show at the symposium. The replay harness solves both.

**Fixtures.** Pick 4–6 historical catalyst events we know the ground truth for (an NVDA earnings beat, an FDA approval, a confirmed contract, plus 1–2 "rumor that never confirmed" cases as negatives). For each, collect the ~20–50 articles from around that date (the Phase-0 news source, or hand-collected), save as JSON in `eval/fixtures/<event>/articles.json`, and hand-label each (article, catalyst) pair with the correct verdict in `labels.json`. Target ~100–150 labeled pairs total — an evening of labeling, and it makes everything downstream measurable.

**Replay (`eval/replay.py`).** Feeds fixture articles through the real pipeline in timestamp order and reports:
- Pass 1: recall on relevant articles (missing a relevant article is the expensive error — Pass 1 should over-include).
- Pass 2: precision/recall on `confirmed`, quote-validity rate, confusion matrix over the four states.
- End-to-end: did the catalyst reach `confirmed` at the right point in the timeline, and not before?

Runs via the **Batch API** (50% price, results in ~1h) since eval isn't latency-sensitive. Every prompt change gets a replay run before merging; results land in `eval/results/` so regressions are visible in git history.

**Demo mode.** The same replay, played through the live system at accelerated speed: Brandon's scheduler gets a `--replay <fixture>` flag that feeds fixture articles instead of live news, and the dashboard shows the catalyst going `unconfirmed → rumored → confirmed` with quotes and reasoning in real time. That's the symposium demo — deterministic, rehearsable, and it works even if the live market does nothing interesting that week.

---

## 7. Frozen contracts with Brandon (agree on these, then build in parallel)

The three shapes below are the entire integration surface. Once we shake hands on them, neither of us blocks the other.

```python
# 1. What the scheduler calls (sync, pure, no I/O beyond the news fetch + LLM):
news.fetch(ticker: str, since: datetime) -> list[Article]
llm.classify_batch(articles: list[Article], catalysts: list[dict]) -> list[CatalystVerdict]
catalysts.apply(current_state: str, verdict: CatalystVerdict) -> tuple[new_state: str, changed: bool]
evaluator.evaluate(thesis: dict, quant_results: list[dict], catalyst_states: dict) -> dict  # shape in §4

# 2. What gets persisted (Brandon's tables store these JSON blobs verbatim):
CatalystVerdict.model_dump()   # into catalysts.evidence / evaluations.catalyst_results
evaluator output dict          # into evaluations

# 3. What the frontend receives over WS (my reasoning panel renders this):
{"type": "agent_activity", "payload": {"stage": "pass1"|"pass2"|"evaluate", "ticker": ..., "detail": CatalystVerdict | evaluator_output}}
```

Schema asks for `V1__init.sql` (raise before it's locked): `catalysts.state` enum + `state_updated_at` + `evidence` JSON; `quant_mode`/`catalyst_mode` on `theses`; `prompt_version` on evaluation rows.

---

## 8. Build order

| Sitting | Deliverable | Proves |
|---|---|---|
| 1 | Phase-0 news spike; pick source; `news.py` + `Article`; record findings in §2 | The data exists |
| 2 | `catalysts.py` state machine + `evaluator.py` combination logic, fully unit-tested (no LLM) | The deterministic core is right |
| 3 | Pass 1 + Pass 2 in `llm.py` with structured outputs + quote validation; `python -m pipeline.run` works end-to-end on live news | The pipeline runs |
| 4 | Fixtures labeled; `eval/replay.py` + first metrics run | It's *measurably* right |
| 5 | Prompt tuning against replay metrics; hand `--replay` hook + contracts to Brandon; build the frontend reasoning panel against the §7 WS shape | Demo-ready |

Sittings 1–4 need nothing from Brandon. Sitting 5 needs his scheduler to exist, but the reasoning panel can be built against a mock WS feed replaying fixture verdicts — so even that isn't blocking.

## 9. Non-goals (this half)

Sentiment analysis, price prediction, embeddings/RAG over news archives, fine-tuning, multi-source news fusion beyond simple dedup, streaming/real-time news (hourly polls are fine). The adaptive-threshold anomaly work lives in the separate Kestrel-ML notebook project and joins as a proposal generator in v2.
