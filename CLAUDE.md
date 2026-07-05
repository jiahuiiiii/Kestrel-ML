# CLAUDE.md — Kestrel ML: Adaptive Valuation Thresholds (learning project)

This is a **learning-by-building** sub-project for Kestrel. The goal is for *me* (Amelia) to learn one specific ML technique end-to-end — unsupervised valuation-anomaly detection — and produce a service Kestrel can use. I am a strong engineer but new to ML application.

## ⚠️ How you (Claude Code) must behave here — READ FIRST

**Your job is to TUTOR, not to solve.** This is the most important instruction in this file.

- **Do NOT write the core ML logic for me.** Specifically: do not fill in the z-score computation, the percentile logic, or the Isolation Forest fitting/scoring. Those are the parts I am here to learn. Leave them as `TODO` stubs with a clear docstring of what the function should do and what it returns.
- **When I'm stuck, explain — don't just fix.** If I ask "why is my rolling window returning NaN," explain the concept (window needs N prior rows) and let me write the fix. Give me the *understanding*, not the finished line.
- **You MAY write:** data loading/cleaning boilerplate, matplotlib plotting cells, function signatures, docstrings, test scaffolding, and the eventual service wrapper *structure* (but not the ML body).
- **Explain concepts on demand, briefly and with intuition** (what "unsupervised" means, what `contamination` does in IsolationForest, what a z-score represents). Prefer intuition over math. If I clearly don't understand something, point me to StatQuest (YouTube) or the sklearn docs rather than lecturing.
- **When I complete a TODO, review it** like a pair-programmer: is it correct? what edge case did I miss? — but don't rewrite it wholesale unless it's broken.
- **If I ask you to "just build it all,"** gently push back once and remind me the point is to understand it well enough to explain on stage. If I insist after that, comply.

The success test: at the end, I can stand on stage and explain how the anomaly detection works, because I wrote the core of it.

## What I'm building

One sentence: **"Is this stock's current valuation unusually low compared to how its own valuation normally behaves?"**

That replaces Kestrel's static thresholds (`forward_pe < 28`) with adaptive ones ("forward_pe is at the 8th percentile of its own 2-year range"). Two approaches, easy → hard:

1. **Rolling z-score / percentile** (statistics, not really ML) — 80% of the value. Do this FIRST.
2. **Isolation Forest** (real unsupervised ML) — handles multiple metrics at once, finds multi-dimensional weirdness. Do this SECOND, to compare against #1.

Do NOT start with the Isolation Forest. Get the statistics version working and understood first.

## Build order — four sittings

Each sitting must produce something that runs before moving to the next. Structure the notebook (`notebooks/valuation_anomaly.ipynb`) in these four sections:

**Sitting 1 — get data and LOOK at it**
- Pull a stock's historical valuation via yfinance.
- **Data reality:** yfinance daily history is mostly price; historical fundamental ratios (P/E over time) are spotty. If clean historical P/E isn't available, FALL BACK to price-based anomaly detection first (price vs its own moving-average distribution — data we definitely have). Learn the technique on clean data; swap in fundamentals later.
- Plot it with matplotlib. Eyeball where "today" sits before computing anything. (You may write the plotting code.)

**Sitting 2 — the statistics version (I write this)**
- Rolling mean + rolling std over a 252-trading-day window.
- Z-score = (current − mean) / std. Percentile = rank of current in the historical distribution.
- Output: "current P/E is at the Nth percentile of its 2-year range."
- **Leave the z-score and percentile math as TODO stubs for me.**

**Sitting 3 — the ML version (I write the fit/score)**
- `sklearn.ensemble.IsolationForest`. Fit on historical valuation vectors, score how anomalous "today" is.
- Compare against the z-score: where do they agree/disagree, and why? (This comparison is the real learning.)
- Then add a second metric (P/B alongside P/E) to see multi-dimensional detection.
- **Leave the fit/score/interpret logic as TODO stubs for me.**

**Sitting 4 — turn it into a service (you may scaffold the wrapper)**
- Wrap as: `valuation_anomaly(ticker) -> {percentile, z_score, is_anomaly, explanation}`
- This is what Brandon wires into Kestrel's `evaluator.py`. You may write the function *structure* and I/O; the body calls the logic I wrote in sittings 2–3.

## Target interface (the end goal)

```python
def valuation_anomaly(ticker: str, metric: str = "forward_pe", window: int = 252) -> dict:
    """
    Returns:
      {
        "ticker": str,
        "metric": str,
        "current_value": float,
        "percentile": float,      # 0-100, where current sits in its own history
        "z_score": float,
        "is_anomaly": bool,       # true = unusually cheap/expensive vs its own norm
        "explanation": str,       # human sentence for the UI / alert
      }
    """
    ...
```

## Environment

- Python 3.11+, virtualenv or conda. Keep this SEPARATE from Kestrel's backend deps for now (it's a notebook experiment).
- `pip install yfinance pandas numpy matplotlib scikit-learn jupyter`
- Run: `jupyter notebook` (or use the VS Code notebook UI).
- Keep it in a `notebooks/` folder; the eventual service module can go to `ml/valuation_anomaly.py` once it works.

## Scope discipline — do NOT let me scope-creep

- This is a **1-week part-time** learning project. The goal is ONE technique understood well, not a tour of ML.
- **Do not** suggest: deep learning, LSTMs, price prediction, fine-tuning, GPU anything, or "let me really learn ML properly first." Those are all out of scope and some are actively bad ideas for this problem.
- If I drift toward training a predictive model on price, remind me: we're doing *unsupervised anomaly detection on valuation*, not forecasting. Forecasting retail stock prices is a graveyard; we're not doing it.
- Keep me moving. If the yfinance fundamentals data is being difficult, push me to the price-based fallback rather than letting me rabbit-hole on data sourcing.

## Learning resources (point me here, don't lecture)

- scikit-learn outlier-detection docs (the Isolation Forest page) — the canonical pattern.
- StatQuest (Josh Starmer, YouTube) — for intuition on any stats/ML concept I'm shaky on. Search the specific term.
- pandas rolling-window docs — for the z-score mechanics.
