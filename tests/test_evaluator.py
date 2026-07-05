"""Unit tests for the thesis evaluator (pipeline/evaluator.py).

Focus: the three-valued quant logic (unknown != False), the all/any modes,
and the blocked_by explanations. No LLM, no network.
"""

import pytest

from pipeline import evaluator


def cond(metric, operator, value, enabled=True):
    return {"metric": metric, "operator": operator, "value": value, "enabled": enabled}


def result(value, passes):
    return {"value": value, "passes": passes}


def thesis(**kw):
    base = {
        "ticker": "NVDA",
        "quant_mode": "all",
        "catalyst_mode": "all",
        "quant_conditions": [],
        "catalysts": [],
    }
    base.update(kw)
    return base


# --- firing ----------------------------------------------------------------- #
def test_fires_when_all_quant_pass_and_no_catalysts():
    t = thesis(quant_conditions=[cond("forward_pe", "<", 28)], catalyst_mode="none_required")
    out = evaluator.evaluate(t, [result(27.0, True)], {})
    assert out["signal"] is True
    assert out["status"] == "firing"
    assert out["blocked_by"] == []


def test_fires_with_confirmed_catalyst():
    t = thesis(
        quant_conditions=[cond("forward_pe", "<", 28)],
        catalysts=[{"id": "capex_cut", "description": "hyperscaler capex cut"}],
    )
    out = evaluator.evaluate(t, [result(27.0, True)], {"capex_cut": "confirmed"})
    assert out["signal"] is True


# --- definite failure ------------------------------------------------------- #
def test_quant_fail_blocks_and_explains():
    t = thesis(quant_conditions=[cond("forward_pe", "<", 28)], catalyst_mode="none_required")
    out = evaluator.evaluate(t, [result(31.0, False)], {})
    assert out["signal"] is False
    assert out["status"] == "not_met"
    assert out["blocked_by"] == ["forward_pe = 31.0, needs < 28"]


def test_rumored_catalyst_blocks_with_reason():
    t = thesis(
        quant_conditions=[cond("forward_pe", "<", 28)],
        catalysts=[{"id": "capex_cut", "description": "hyperscaler capex cut"}],
    )
    out = evaluator.evaluate(t, [result(27.0, True)], {"capex_cut": "rumored"})
    assert out["signal"] is False
    assert out["quant_ok"] is True and out["catalysts_ok"] is False
    assert out["blocked_by"] == ["hyperscaler capex cut is rumored, needs confirmed"]


def test_invalidated_catalyst_message():
    t = thesis(catalysts=[{"id": "fda", "description": "FDA approval"}], quant_mode="all")
    out = evaluator.evaluate(t, [], {"fda": "invalidated"})
    assert out["blocked_by"] == ["FDA approval was invalidated"]


# --- the important one: unknown != False ------------------------------------ #
def test_unknown_quant_is_incomplete_not_not_met():
    t = thesis(quant_conditions=[cond("forward_pe", "<", 28)], catalyst_mode="none_required")
    out = evaluator.evaluate(t, [result(None, None)], {})
    assert out["signal"] is False
    assert out["status"] == "incomplete"          # <-- not "not_met"
    assert out["quant_ok"] is False
    assert "couldn't evaluate" in out["blocked_by"][0]


def test_definite_fail_dominates_unknown_in_all_mode():
    # One condition fails outright, another is unknown -> the fail wins,
    # status is not_met (we KNOW it can't fire), not incomplete.
    t = thesis(
        quant_conditions=[cond("forward_pe", "<", 28), cond("pb_ratio", "<", 5)],
        catalyst_mode="none_required",
    )
    out = evaluator.evaluate(t, [result(31.0, False), result(None, None)], {})
    assert out["status"] == "not_met"


# --- any mode --------------------------------------------------------------- #
def test_any_mode_quant_one_pass_fires():
    t = thesis(
        quant_mode="any",
        quant_conditions=[cond("forward_pe", "<", 28), cond("pb_ratio", "<", 5)],
        catalyst_mode="none_required",
    )
    out = evaluator.evaluate(t, [result(31.0, False), result(4.0, True)], {})
    assert out["signal"] is True


def test_any_mode_catalyst_one_confirmed_fires():
    t = thesis(
        catalyst_mode="any",
        catalysts=[
            {"id": "a", "description": "contract A"},
            {"id": "b", "description": "contract B"},
        ],
    )
    out = evaluator.evaluate(t, [], {"a": "rumored", "b": "confirmed"})
    assert out["catalysts_ok"] is True and out["signal"] is True


# --- disabled + alignment --------------------------------------------------- #
def test_disabled_condition_is_ignored():
    t = thesis(
        quant_conditions=[cond("forward_pe", "<", 28, enabled=False)],
        catalyst_mode="none_required",
    )
    # disabled -> no quant gate -> fires even though the result "fails"
    out = evaluator.evaluate(t, [result(99.0, False)], {})
    assert out["signal"] is True


def test_misaligned_lengths_raise():
    t = thesis(quant_conditions=[cond("forward_pe", "<", 28)])
    with pytest.raises(ValueError):
        evaluator.evaluate(t, [], {})
