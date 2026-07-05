# Eval fixtures — how to build and label one

One fixture = one historical catalyst event we know the ground truth for
(ml_plan.md §6). Target 4–6 fixtures, ~100–150 labeled pairs total, including
1–2 "rumor that never confirmed" negatives.

## 1. Capture articles

```bash
python -m eval.backfill --event nvda_2026-05-28_earnings --ticker NVDA \
    --start 2026-05-26 --end 2026-05-30 \
    --catalyst "earnings_beat:Q1 revenue beats consensus"
```

Fetches day-by-day (Finnhub truncates wide windows at ~250 items), writes
`articles.json` (frozen — don't hand-edit) and a `labels.json` skeleton.
Re-running never overwrites hand labels; it only appends rows for new pairs.

**A raw window is often hundreds of articles, mostly noise.** Curate it with
`--keep-match` — keep only articles whose text contains one of the given terms,
plus a stride-sample of the rest as labeled negatives:

```bash
python -m eval.backfill --event orcl_2026-06-22_layoffs --ticker ORCL \
    --start 2026-06-20 --end 2026-07-05 \
    --catalyst "orcl_layoff:Oracle discloses cutting ~21,000 jobs in FY2026" \
    --keep-match layoff "job cut" "21,000" workforce fired "laid off" \
    --sample-irrelevant 15
```

Use **specific** terms (`"21,000"`, `layoff`) — not the company name, or
everything matches. Curation only decides which articles enter the fixture; it
does **not** label them. Matched rows still start `relevant: false` and carry a
`_curation: "keyword-match"` hint marking them as worth your attention; sampled
negatives get `_curation: "sample"`. You still write every label yourself
(step 2) — the keyword filter is a triage aid, never the answer key.

## 2. Hand-label `labels.json`

Every row is one (article, catalyst) pair. Edit two fields:

- **`relevant`** — would a competent human say this article *bears on* the
  catalyst at all? This scores Pass 1. Err inclusive: an article that discusses
  the catalyst but doesn't move it is `relevant: true, expected_state: "no_change"`.
- **`expected_state`** — the correct Pass-2 verdict for this pair *in isolation*:
  - `no_change` — bears on the catalyst but doesn't move it (or isn't relevant).
  - `rumored` — hedged: analyst speculation, "sources say", unconfirmed reports.
  - `confirmed` — stated as fact from a primary-ish source (company statement,
    filing, earnings release).
  - `invalidated` — contradicts a confirmation.

  Label what the *article* supports, not what you know happened later.

The `_headline` / `_published_at` fields are context for you; replay ignores them.

Then set the event-level truth:

- **`expected_final`** — the state each catalyst should be in after the whole
  timeline plays out (`unconfirmed` / `rumored` / `confirmed` / `invalidated`).
- **`expected_confirm_article`** — the article id that *should* first flip the
  catalyst to confirmed (`null` if it never should). This powers the
  "confirmed at the right point, not before" end-to-end check.

## 3. Validate, then run

```bash
python -m eval.replay --fixture nvda_2026-05-28_earnings --dry-run   # label sanity check, free
python -m eval.replay --fixture nvda_2026-05-28_earnings             # live, needs ANTHROPIC_API_KEY
```

Reports land in `eval/results/` (kept in git — that's the regression history).
Run a replay before merging any prompt change.
