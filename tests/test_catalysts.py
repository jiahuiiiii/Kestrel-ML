"""Unit tests for the catalyst state machine (pipeline/catalysts.py).

Pure deterministic logic — no LLM, no network. Drives `apply()` with a tiny
stub verdict.
"""

from dataclasses import dataclass

import pytest

from pipeline import catalysts
from pipeline.catalysts import CatalystState as S


@dataclass
class StubVerdict:
    proposed_state: str
    source_kind: str = "reporting"  # credible by default


def apply(current, proposed, source_kind="reporting"):
    return catalysts.apply(current, StubVerdict(proposed, source_kind))


# --- the happy-path ladder -------------------------------------------------- #
def test_initial_state():
    assert catalysts.initial_state() is S.UNCONFIRMED


def test_unconfirmed_to_rumored():
    t = apply(S.UNCONFIRMED, "rumored")
    assert t.new_state is S.RUMORED and t.changed


def test_rumored_to_confirmed_credible():
    t = apply(S.RUMORED, "confirmed", "primary")
    assert t.new_state is S.CONFIRMED and t.changed


def test_unconfirmed_straight_to_confirmed():
    # A single primary-source article can jump the ladder.
    t = apply(S.UNCONFIRMED, "confirmed", "primary")
    assert t.new_state is S.CONFIRMED and t.changed


# --- source-credibility guard (§5 guard 2) ---------------------------------- #
def test_speculative_confirm_capped_at_rumored():
    t = apply(S.UNCONFIRMED, "confirmed", "speculation")
    assert t.new_state is S.RUMORED and t.changed


def test_speculative_confirm_does_not_downgrade_confirmed():
    # Already confirmed; a speculative "confirm" collapses to rumored, which
    # must not pull a real confirmation back down.
    t = apply(S.CONFIRMED, "confirmed", "speculation")
    assert t.new_state is S.CONFIRMED and not t.changed


# --- no downgrades / no false revivals -------------------------------------- #
def test_rumor_does_not_downgrade_confirmed():
    t = apply(S.CONFIRMED, "rumored")
    assert t.new_state is S.CONFIRMED and not t.changed


def test_rumor_cannot_revive_invalidated():
    t = apply(S.INVALIDATED, "rumored")
    assert t.new_state is S.INVALIDATED and not t.changed


# --- invalidation ----------------------------------------------------------- #
def test_confirmed_can_be_invalidated_by_credible_source():
    t = apply(S.CONFIRMED, "invalidated", "reporting")
    assert t.new_state is S.INVALIDATED and t.changed


def test_speculative_invalidation_is_ignored():
    t = apply(S.CONFIRMED, "invalidated", "speculation")
    assert t.new_state is S.CONFIRMED and not t.changed


def test_credible_confirm_revives_invalidated():
    t = apply(S.INVALIDATED, "confirmed", "primary")
    assert t.new_state is S.CONFIRMED and t.changed


# --- no_change / misc ------------------------------------------------------- #
@pytest.mark.parametrize("state", list(S))
def test_no_change_never_moves(state):
    t = apply(state, "no_change")
    assert t.new_state is state and not t.changed


def test_unknown_proposal_raises():
    with pytest.raises(ValueError):
        apply(S.UNCONFIRMED, "totally_bogus")


def test_accepts_string_state():
    t = apply("unconfirmed", "rumored")
    assert t.new_state is S.RUMORED


def test_is_met_only_confirmed():
    assert catalysts.is_met(S.CONFIRMED)
    assert not catalysts.is_met(S.RUMORED)
    assert not catalysts.is_met(S.INVALIDATED)
    assert not catalysts.is_met(S.UNCONFIRMED)


def test_as_tuple_shape():
    t = apply(S.UNCONFIRMED, "rumored")
    assert t.as_tuple() == ("rumored", True)
