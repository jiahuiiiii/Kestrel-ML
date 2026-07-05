# Kestrel — Thesis-Driven Watchlist Monitor

## The idea

A tool that monitors a personal US-equity watchlist and alerts us when a stock hits the conditions we've set. The point is: I have stocks I *want* to buy but not yet — too expensive right now, or waiting on some catalyst to confirm (a contract, an earnings result, a product launch). Instead of manually checking, the system watches and tells me "conditions met, take a look."

**Decision support, not auto-trading.** It never buys anything. It tells us when *our own* criteria are satisfied, we decide.

## How and Why

Two kinds of triggers per stock:

1. **Quantitative** — "valuation drops below X" (P/E, P/B, EV/EBITDA, price vs MA). Easy, deterministic, yfinance gives us this.
2. **Qualitative** — "the contract got confirmed", "earnings beat", "FDA approval". This is the hard, interesting part: monitor news, use an LLM to judge *"does this article actually mean the catalyst I'm waiting for happened?"*

Most off-the-shelf tools (Public.com, Simply Wall St, Benzinga) do generic price/news alerts. None let you encode a *specific personal thesis* — "I want NVDA under 28 P/E AND a hyperscaler capex cut AND data-center revenue decel" — and only alert when *all of it* is true. That gap is the project.

## Interface decision: web app (primary) + push notifications

We're building a **React web app** as the primary interface rather than a Telegram-only bot. Reasoning: managing theses, viewing the agent's reasoning, and handling the propose/approve queue is far more intuitive in a real UI than through chat commands — and a live dashboard demos much better at the AI Symposium finale than a phone buzzing.

The web app handles *interaction and viewing*. Notifications (the "tab is closed, alert me anyway" problem) go through **email + optional Telegram** — see the notifications section for why we're deliberately *not* using web push.

## Architecture

Full-stack split: FastAPI backend + React frontend, sharing a Postgres DB.

```
backend/
├── main.py               ← FastAPI app entry + startup (launches scheduler)
├── config.py             ← env vars + database URL
├── db.py                 ← Postgres connection + async session management
├── models.py             ← SQLAlchemy ORM models (schema source of truth)
├── audit.py              ← sqlalchemy-history hooks + audit queries
├── scheduler.py          ← async poll loop (APScheduler / asyncio task)
├── api/
│   ├── theses.py         ← REST: CRUD theses, quant conditions, catalysts
│   ├── proposals.py      ← REST: list / approve / reject proposals
│   ├── evaluations.py    ← REST: history, current signals
│   └── ws.py             ← WebSocket: live push to frontend (new alerts, agent activity)
├── services/
│   ├── fundamentals.py   ← yfinance fetch + threshold checks
│   ├── news.py           ← news fetch + dedup
│   ├── llm.py            ← two-pass catalyst classifier
│   ├── evaluator.py      ← combines quant + catalysts → signal
│   └── notify.py         ← email (Resend) + optional Telegram dispatch
├── db/migrations/        ← Flyway SQL migrations (V1__init.sql, V2__..., etc.)
├── .env.example          ← template for environment vars (committed)
├── .env.dev              ← actual config (gitignored; machine-specific)
└── docker-compose.yml    ← Postgres + Flyway setup

frontend/
├── src/
│   ├── App.tsx
│   ├── api/client.ts     ← typed fetch wrapper to FastAPI
│   ├── hooks/useWs.ts    ← WebSocket subscription for live updates
│   ├── pages/
│   │   ├── Dashboard.tsx     ← watchlist overview, live status per thesis
│   │   ├── ThesisDetail.tsx  ← conditions, catalysts, evaluation history
│   │   └── Proposals.tsx     ← approve/reject queue
│   └── components/       ← thesis cards, condition editor, agent-reasoning panel
├── tailwind.config.js
├── index.html
└── vite.config.ts
```

**Stack:**
- **Backend:** Python 3.11+, FastAPI (async), SQLAlchemy 2.0+ (async), `sqlalchemy-history` (audit), PostgreSQL (Docker Compose locally, Supabase for production), Flyway (SQL migrations), `anthropic`, `yfinance`, APScheduler (or a plain asyncio background task) for the poll loop.
- **Frontend:** React 18 + TypeScript, Vite, TailwindCSS. TanStack Query (React Query) for server state + caching, plain WebSocket for live updates.
- **Notifications:** Resend (email) + optional `python-telegram-bot` (instant mobile push).

## Backend shape (FastAPI)

FastAPI does double duty:
1. **REST API** — the web app's read/write path. Thesis CRUD, proposal approve/reject, evaluation history. Replaces the Telegram command layer as the primary way we manage theses.
2. **WebSocket** — pushes live updates to the open dashboard: a new alert fired, the agent is currently classifying an article, a proposal just landed. This is what makes the demo feel alive.
3. **Background scheduler** — the poll loop (fundamentals check + news scan + evaluation) runs as an async task launched on FastAPI startup. It writes to the same DB; when something fires, it (a) persists it, (b) pushes over WebSocket to any open client, (c) sends email/Telegram.

Pydantic models for request/response validation come basically free with FastAPI and pair cleanly with the SQLAlchemy models — one place the ORM choice pays off.

## Notifications: why NOT web push

A web app only runs when its tab is open, so "alert me when a catalyst fires while I'm in class" needs an out-of-app channel. Options considered:

- **Web Push (Service Workers + Push API):** the "native" web answer, but **iOS Safari only supports it if the user installs the PWA to home screen first** — unreliable as a sole channel, and service-worker + VAPID setup is real effort. **Rejected** for v1: worst effort-to-reliability ratio.
- **Email (chosen baseline):** backend fires an email via Resend when a signal triggers. Works everywhere, no platform games, permanent record, ~20 min to wire. Latency of seconds-to-minutes is fine — this isn't day-trading.
- **Telegram (chosen bonus):** one HTTP POST from `notify.py`, free, instant, works on iOS. We keep a *minimal* Telegram integration purely as a push pipe (not the full command bot). Best-of-both without the web-push pain.
- **In-app (WebSocket):** while the dashboard is open, updates stream live. Great for the demo.

**Decision:** email as reliable baseline + Telegram as instant-push bonus + WebSocket for in-app live updates. No web push in v1.

## Key design decisions

**1. All-database thesis config with automatic audit trails.**
- Theses live **only** in Postgres. No YAML/JSON thesis files. Managed via the web app (FastAPI REST), not hand-edited files.
- Schema defined in `models.py` as SQLAlchemy ORM classes (`Thesis`, `QuantCondition`, `Catalyst`, etc.). The code is the source of truth.
- **Automatic versioning via sqlalchemy-history:** every INSERT/UPDATE/DELETE on core tables (`theses`, `catalysts`, `quant_conditions`, `proposals`) is logged to `*_aud` audit tables. Query `theses_aud` for a full change timeline — replaces the git-diff history we'd have had with a YAML config. For an investment thesis, the "when/why did I change my mind" trail is genuinely valuable.
- Removes the intent/state split complexity: the DB is the single source of truth. Tradeoff: discipline around audit-library setup.

**2. Two-pass LLM classifier** (pattern from your VIX bot):
- Pass 1 — Haiku, cheap: "is this headline even relevant to our catalysts?" Drops ~90%.
- Pass 2 — Sonnet, only on relevant: "does this confirm the catalyst? Quote the exact supporting sentence." **No quote → rejected.** Anti-hallucination guard.

**3. Postgres with SQLAlchemy ORM and Flyway migrations.**
- **models.py** is the schema source of truth. Because we hand-write SQL migrations (Flyway) rather than autogenerating from models, the SQL and the ORM must be kept in sync *by us*: schema change = write the migration SQL first, then update the model to match. Migration is authoritative; `models.py` mirrors it. (This is the flip side of the SQL-first control we chose Flyway for.)
- **db/migrations/** holds `V1__init.sql`, `V2__add_proposals_table.sql`, etc. Pragmatic for v1: write ad-hoc SQL as you develop, keep them in version order, don't obsess. Clean up before Supabase cutover.
- **Docker Compose locally:** `docker-compose up` spins Postgres + runs Flyway on startup. Flyway runs as its own Compose service (JVM stays contained; nobody installs Java).
- **Supabase for production:** same migrations, same schema — swap the connection string. Note for cutover: run Flyway against Supabase's **direct** connection (5432), point the FastAPI app at the **pooled** connection (PgBouncer, 6543).

**4. Propose / approve loop (human-in-the-loop config).**
- The bot can *suggest* changes to thresholds and catalysts — never applies them unilaterally.
- **Catalyst discovery:** LLM spots a recurring theme not in your list ("NVDA news keeps mentioning China export restrictions"). Generates a proposal.
- **Threshold suggestion:** bot flags when your number looks stale ("your P/E < 28 sits at the 8th percentile of NVDA's 2-yr range — rarely triggers. Loosen to 30?").
- **Flow (now in the web app):**
  1. Bot generates proposal → stores in `proposals` (`status: pending`, `proposed_change: {...}`) → pushes to the frontend Proposals queue over WebSocket (+ optional email/Telegram nudge).
  2. You approve/reject **in the web UI** (one click) — or via Telegram `/approve <id>` if we keep that path.
  3. On approve: FastAPI applies the change to the thesis; audit trail auto-logs it; status → `approved`.
  4. On reject: status → `rejected`, reason logged; dedup logic won't re-propose the same thing for N days.
- Judgment stays with you; bot is a grunt-work + pattern-spotter.

**5. Designed-in ML hook (v2, not v1):**
- `outcomes` table: the moment a catalyst confirms, bot logs `price_at_signal` + `llm_confidence` + timestamp.
- Later, fill `price_after_30d` (weekly job). Gives a dataset to calibrate the LLM's confidence against realized outcomes. Costs nothing now, impossible to reconstruct later.

## Proposed data model (high-level)

**Core tables (defined in models.py):**
- `theses` — ticker, status, notes, created_at, updated_at
- `quant_conditions` — thesis_id, metric (forward_pe, pb_ratio, price_vs_ma200, etc.), operator (<, >, ==), value, enabled
- `catalysts` — thesis_id, description, triggered_at, enabled
- `evaluations` — thesis_id, timestamp, quant_results (JSON), catalyst_results (JSON), signal (bool), reason
- `proposals` — thesis_id, proposal_type (add_catalyst | remove_catalyst | adjust_threshold), proposed_change (JSON), status (pending | approved | rejected), created_at, resolved_at
- `news_seen` — article_id (hash), ticker, headline, url, fetched_at (dedup)
- `alerts` — thesis_id, timestamp, alert_type, channels_sent (JSON: {email, telegram, ws})
- `outcomes` — thesis_id, price_at_signal, llm_confidence, triggered_at, price_after_30d (filled later), notes

**Audit tables (auto-generated by sqlalchemy-history):**
- `theses_aud`, `quant_conditions_aud`, `catalysts_aud`, `proposals_aud` — track all changes. Never written by application code.

---

## Environment setup

**`.env.example`** (committed; template):
```bash
# Database
DATABASE_URL=postgresql+asyncpg://watchbot:password@localhost:5432/watchbot_dev

# Telegram (optional push channel)
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here

# Email (Resend)
RESEND_API_KEY=your_resend_key_here
ALERT_EMAIL_TO=you@example.com

# News API (if using external news source)
NEWS_API_KEY=optional_news_api_key

# Backend / app
LOG_LEVEL=INFO
EVALUATION_INTERVAL_MINUTES=60
CORS_ORIGINS=http://localhost:5173

# Frontend (Vite; separate .env in frontend/)
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
```

**`.env.dev`** (gitignored) holds the real values. Frontend uses Vite's own `.env` for `VITE_*` vars.

**Run backend:**
```bash
python -m dotenv -f .env.dev run -- uvicorn main:app --reload --port 8000
```

**Run frontend:**
```bash
cd frontend && npm install && npm run dev   # Vite dev server on :5173
```

---

## Database setup (Docker Compose)

**`docker-compose.yml`:**
```yaml
version: '3.9'
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: watchbot
      POSTGRES_PASSWORD: devpass
      POSTGRES_DB: watchbot_dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U watchbot"]
      interval: 5s
      timeout: 5s
      retries: 5

  flyway:
    image: flyway/flyway:latest
    command: -locations=filesystem:/flyway/sql -connectRetries=10 migrate
    depends_on:
      postgres:
        condition: service_healthy      # wait for DB, don't race its boot
    volumes:
      - ./db/migrations:/flyway/sql
    environment:
      FLYWAY_URL: jdbc:postgresql://postgres:5432/watchbot_dev
      FLYWAY_USER: watchbot
      FLYWAY_PASSWORD: devpass

volumes:
  postgres_data:
```

(Note the healthcheck + `condition: service_healthy` — without it Flyway races Postgres boot and the first migration fails.)

**Startup / teardown:**
```bash
docker-compose up -d              # Postgres + Flyway migrations run automatically
docker-compose down --volumes     # total reset: removes DB + data; next up runs from V1
```

---

## Migration strategy (pragmatic)

**For v1 (development):**
1. `db/migrations/V1__init.sql` — defines all core tables.
2. As the model evolves, add `V2__...sql`, `V3__...sql` in order. No ceremony.
3. `docker-compose up` applies fresh; `flyway migrate` applies to existing.

**Before Supabase cutover:** review the chain, test a fresh V1→Vn apply matches current schema, swap connection string (direct conn for Flyway, pooled for the app).

---

## Critical setup decisions upfront

1. **Audit library:** `sqlalchemy-history`. Set up in week 1 as a dependency + hooks in `audit.py`. Don't defer.
2. **Schema stability:** finalize `models.py` + `V1__init.sql` before splitting work. After that, schema changes are PRs with new migration files.
3. **API contract:** lock the FastAPI REST + WebSocket shapes early (an OpenAPI spec basically for free from FastAPI) so the frontend can build against it in parallel — or against a mock — without waiting on the backend.
4. **Proposal dedup:** if a proposal of type X for thesis Y is already `rejected`, don't re-propose for N days. Implement before the loop goes live.

---

## Proposed split

Now split along the full-stack seam, which is a *clean* two-person boundary: one owns backend + data + agent, the other owns frontend + API integration. Given Amelia's frontend strength, natural assignment noted.

| Area | Owner | Notes |
|---|---|---|
| Postgres + Docker Compose + Flyway + `models.py` + `V1__init.sql` | **Together** | lock schema first; everything depends on it |
| `audit.py` (sqlalchemy-history) | Brandon | other reviews |
| FastAPI `api/` (REST + WebSocket) | Brandon | defines the contract the frontend consumes |
| `services/fundamentals.py` | Brandon | yfinance + thresholds |
| `services/news.py` + `services/llm.py` + prompts | **TBD** | the LLM side — the hardest, most interesting part |
| `services/evaluator.py` + `scheduler.py` | **Together** | the glue; lock interfaces first |
| `services/notify.py` (email + Telegram) | Brandon | small, self-contained |
| **React frontend** (Dashboard, ThesisDetail, Proposals) | Amelia | your strength + the demo surface |
| Tailwind design + agent-reasoning panel | Amelia | make the agent's thinking *legible*, not just pretty |
| WebSocket client hook + live updates | Amelia | the "feels alive on stage" piece |

**Interface contracts to lock upfront:**
- `fundamentals.check(thesis) -> dict` (ticker, metric, value, passes_threshold)
- `llm.classify(article, catalyst) -> (verdict: bool, confidence: 0.0-1.0, supporting_quote: str | None)`
- `evaluator.evaluate(thesis) -> (signal: bool, reason: str, triggered_catalysts: [...])`
- **REST:** `GET/POST/PATCH/DELETE /theses`, `GET /proposals`, `POST /proposals/{id}/approve|reject`, `GET /evaluations`
- **WS:** server → client events `{type: "alert" | "agent_activity" | "proposal", payload: {...}}`

The full-stack split means the frontend can build against the OpenAPI spec (or a mock server) while the backend fills in real logic — genuinely parallel, minimal collision.

---

## Workflow

- **Fresh repo** (monorepo: `backend/` + `frontend/`), shared org or both as collaborators.
- **Branch discipline:** feature branches only, never push to `main`. Branch protection on.
- **PR review:** review each other's PRs. Migration changes = review the SQL.
- **Pull before sessions.** Shared chat for "pushed X, endpoint Y is live now."

---

## Scope

**v1 (target ~2–3 weeks):**
- Quant threshold checks (yfinance + comparison logic)
- Two-pass catalyst classifier (Haiku + Sonnet)
- FastAPI backend: REST CRUD + WebSocket live updates + background poll loop
- React + Tailwind frontend: dashboard, thesis detail, proposal queue
- Notifications: email (Resend) baseline + optional Telegram push
- Postgres persistence + audit tables
- Propose/approve loop (threshold + catalyst discovery)
- US stocks only

> **Mitigation:** build the agent + FastAPI backend to *working* first (even with a bare frontend or Swagger UI as the interface). The React app comes *after* the core works. If time gets tight, ship a minimal but functional UI + screen-record — don't let a half-built frontend hide a working agent. Point the frontend effort at **making the agent's reasoning visible** (which reads as "understanding the problem"), not at visual polish (which the criteria discount). A pretty shell over a shallow agent loses to an ugly UI over a brilliant one.

**Explicit non-goals:** auto-trading, multi-user/auth, price-prediction ML, sentiment analysis, backtesting, web push notifications, mobile app.

**v2 (later):**
- Anomaly detection for adaptive thresholds (vs static numbers)
- Outcome-feedback calibration loop (price_at_signal → price_after_30d → LLM confidence tuning)
- Both designed-for in v1's schema.

---

## Watchlist bootstrap

On first startup, seed via manual SQL (or a small seed script) once after `docker-compose up`. Then manage theses through the **web app** (or Telegram `/add_thesis` if we keep that path).
