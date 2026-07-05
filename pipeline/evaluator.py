"""Thesis evaluator (ml_plan.md §4).

Combines quant condition results + catalyst states into a single signal, under
the thesis's `all`/`any` combination modes. Pure: no I/O, no LLM.

The two things that make this more than an AND of booleans:
  1. Three-valued quant — a condition whose yfinance value came back None is
     `unknown`, not False. A thesis that can't be evaluated is surfaced as
     `incomplete`, never as a silent "not firing".
  2. `blocked_by` — an explicit list of *why the thesis is NOT firing*. The
     dashboard's most common state is "not yet", and showing the reason is what
     makes the agent legible (the whole point of the frontend panel).

Refines the §7 contract output with `ticker` + `status` fields (superset — flag
to Brandon before locking).
"""

from __future__ import annotations

from datetime import datetime, timezone

from pipeline import catalysts
from pipeline.catalysts import CatalystState

# Three-valued quant outcome.
OK = "ok"
FAIL = "fail"
UNKNOWN = "unknown"


def evaluate(thesis: dict, quant_results: list[dict], catalyst_states: dict[str, str]) -> dict:
    """Evaluate one thesis.

    Args:
        thesis: dict with keys `ticker`, `quant_mode` ("all"|"any"),
            `catalyst_mode` ("all"|"any"|"none_required"), `quant_conditions`,
            `catalysts`. See ml_plan.md §4.
        quant_results: one dict per quant condition, aligned by position with
            `thesis["quant_conditions"]`. Each: {"value": float|None,
            "passes": bool|None}. `passes is None` == couldn't evaluate.
        catalyst_states: {catalyst_id: state_string}. Missing ids default to
            `unconfirmed`.

    Returns:
        The §4 output dict (plus `ticker`/`status`).
    """
    ticker = thesis.get("ticker", "?")

    quant_status, quant_blockers = _eval_quant(
        thesis.get("quant_mode", "all"),
        thesis.get("quant_conditions", []),
        quant_results,
    )
    catalysts_ok, catalyst_blockers = _eval_catalysts(
        thesis.get("catalyst_mode", "all"),
        thesis.get("catalysts", []),
        catalyst_states,
    )

    quant_ok = quant_status == OK
    signal = quant_ok and catalysts_ok

    if signal:
        status = "firing"
    elif quant_status == UNKNOWN and catalysts_ok:
        # The only thing missing is data — don't call it "not met".
        status = "incomplete"
    else:
        status = "not_met"

    blocked_by = [] if signal else quant_blockers + catalyst_blockers

    return {
        "ticker": ticker,
        "signal": signal,
        "status": status,
        "quant_ok": quant_ok,
        "catalysts_ok": catalysts_ok,
        "blocked_by": blocked_by,
        "reason": _reason(ticker, status, blocked_by),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


# --------------------------------------------------------------------------- #
# Quant side — three-valued.
# --------------------------------------------------------------------------- #
def _eval_quant(mode: str, conditions: list[dict], results: list[dict]) -> tuple[str, list[str]]:
    enabled = [(c, r) for c, r in _align(conditions, results) if c.get("enabled", True)]
    if not enabled:
        return OK, []  # no quant conditions gate this thesis

    passed, failed, unknown = [], [], []
    for cond, res in enabled:
        p = res.get("passes")
        if p is None:
            unknown.append((cond, res))
        elif p:
            passed.append((cond, res))
        else:
            failed.append((cond, res))

    if mode == "any":
        if passed:
            return OK, []                       # one hit is enough
        if unknown:
            return UNKNOWN, _unknown_blockers(unknown)
        return FAIL, ["no quant condition met (any-mode): " + "; ".join(_fail_blockers(failed))]

    # mode == "all" (default): a definite failure dominates; otherwise unknowns block.
    if failed:
        return FAIL, _fail_blockers(failed)
    if unknown:
        return UNKNOWN, _unknown_blockers(unknown)
    return OK, []


def _fail_blockers(items: list[tuple[dict, dict]]) -> list[str]:
    out = []
    for cond, res in items:
        metric = cond.get("metric", "?")
        op = cond.get("operator", "?")
        threshold = cond.get("value", "?")
        out.append(f"{metric} = {res.get('value')}, needs {op} {threshold}")
    return out


def _unknown_blockers(items: list[tuple[dict, dict]]) -> list[str]:
    return [f"{cond.get('metric', '?')}: data unavailable — couldn't evaluate" for cond, _ in items]


# --------------------------------------------------------------------------- #
# Catalyst side — boolean (a catalyst always has a definite state).
# --------------------------------------------------------------------------- #
def _eval_catalysts(mode: str, catalyst_defs: list[dict], states: dict[str, str]) -> tuple[bool, list[str]]:
    if mode == "none_required":
        return True, []

    enabled = [c for c in catalyst_defs if c.get("enabled", True)]
    if not enabled:
        return True, []  # nothing gating

    def state_of(c: dict) -> CatalystState:
        return CatalystState(states.get(c.get("id"), CatalystState.UNCONFIRMED.value))

    met = [c for c in enabled if catalysts.is_met(state_of(c))]

    if mode == "any":
        if met:
            return True, []
        return False, ["no catalyst confirmed yet (any-mode): "
                       + "; ".join(_catalyst_desc(c, state_of(c)) for c in enabled)]

    # mode == "all"
    unmet = [c for c in enabled if not catalysts.is_met(state_of(c))]
    if not unmet:
        return True, []
    return False, [_catalyst_desc(c, state_of(c)) for c in unmet]


def _catalyst_desc(c: dict, state: CatalystState) -> str:
    label = c.get("description") or c.get("id", "catalyst")
    if state is CatalystState.INVALIDATED:
        return f"{label} was invalidated"
    return f"{label} is {state.value}, needs confirmed"


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #
def _align(conditions: list[dict], results: list[dict]) -> list[tuple[dict, dict]]:
    if len(conditions) != len(results):
        raise ValueError(
            f"quant_conditions ({len(conditions)}) and quant_results ({len(results)}) "
            "must align 1:1 by position"
        )
    return list(zip(conditions, results))


def _reason(ticker: str, status: str, blocked_by: list[str]) -> str:
    if status == "firing":
        return f"All conditions met — signal firing for {ticker}."
    if status == "incomplete":
        head = blocked_by[0] if blocked_by else "missing data"
        return f"Couldn't fully evaluate {ticker}: {head}."
    head = "; ".join(blocked_by[:2]) if blocked_by else "conditions not met"
    return f"{ticker} not firing: {head}."
