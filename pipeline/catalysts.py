"""Catalyst state machine (ml_plan.md §3).

A catalyst is not a boolean — news contradicts itself ("contract confirmed"
Tuesday, "contract delayed" Thursday), so each catalyst carries a state:

    unconfirmed ──▶ rumored ──▶ confirmed
         ▲             │            │
         └─────────────┴────────────┴──▶ invalidated
                                             │ (credible re-confirmation)
                                             └──▶ confirmed

Pass 2 (llm.py) proposes a state per (article, catalyst) pair; this module owns
the *rules* for whether that proposal actually moves the catalyst. It is pure:
no LLM, no I/O, no DB. It reads only two fields off a verdict (`proposed_state`,
`source_kind`) via a Protocol, so it never imports llm.py and unit tests can
drive it with plain stubs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class CatalystState(str, Enum):
    """str-valued so it serializes straight to JSON / the DB `state` column."""
    UNCONFIRMED = "unconfirmed"
    RUMORED = "rumored"
    CONFIRMED = "confirmed"
    INVALIDATED = "invalidated"


# The proposal vocabulary Pass 2 emits (llm.py CatalystVerdict.proposed_state).
# "no_change" means "this article bears on the catalyst but doesn't move it".
PROPOSALS = frozenset({"no_change", "rumored", "confirmed", "invalidated"})

# Source categories credible enough to *confirm* or *invalidate*. Speculation
# (analyst guesses, "sources say") can only ever raise a rumor.
CREDIBLE_SOURCES = frozenset({"primary", "reporting"})


class VerdictLike(Protocol):
    """The slice of a Pass-2 verdict the state machine actually reads.

    Kept structural on purpose: catalysts.py must not depend on llm.py (which
    imports the OpenAI SDK). Anything with these two attributes works —
    the real CatalystVerdict, or a test stub.
    """
    proposed_state: str   # one of PROPOSALS
    source_kind: str      # "primary" | "reporting" | "speculation"


@dataclass(frozen=True)
class Transition:
    """Result of applying one verdict. `note` explains the outcome for the UI /
    the evaluator's `blocked_by` line (ml_plan.md §4) — e.g. why a `confirmed`
    proposal was capped at `rumored`.

    Refines the §7 contract (`-> (new_state, changed)`) with a human note; flag
    to Brandon before the contract is locked. `.as_tuple()` gives the plain
    shape if he'd rather keep it minimal.
    """
    new_state: CatalystState
    changed: bool
    note: str

    def as_tuple(self) -> tuple[str, bool]:
        return self.new_state.value, self.changed


def initial_state() -> CatalystState:
    """Every catalyst starts here."""
    return CatalystState.UNCONFIRMED


def is_met(state: CatalystState | str) -> bool:
    """Does this catalyst count as satisfied for signal purposes?

    Only `confirmed` counts. `rumored` is not enough; `invalidated` actively
    fails. The evaluator (§4) uses this when combining catalysts.
    """
    return CatalystState(state) is CatalystState.CONFIRMED


def apply(current: CatalystState | str, verdict: VerdictLike) -> Transition:
    """Apply one Pass-2 verdict to a catalyst's current state.

    Rules (ml_plan.md §3):
      * no_change            -> never moves the state.
      * confirmed/invalidated require a credible source; speculation is capped
        at `rumored` (for a would-be confirm) or ignored (for a would-be invalidate).
      * the positive ladder unconfirmed < rumored < confirmed never moves *down*
        on a weaker proposal — only `invalidated` (credible) can pull a confirmed
        catalyst back, and only a fresh `confirmed` can revive an invalidated one.
    """
    state = CatalystState(current)
    proposed = verdict.proposed_state
    if proposed not in PROPOSALS:
        raise ValueError(f"unknown proposed_state {proposed!r}; expected one of {sorted(PROPOSALS)}")

    credible = verdict.source_kind in CREDIBLE_SOURCES

    if proposed == "no_change":
        return _stay(state, "article bears on the catalyst but does not move it")

    if proposed == "invalidated":
        if not credible:
            return _stay(state, "invalidation from a speculative source — ignored")
        return _to(state, CatalystState.INVALIDATED, "contradicted by a credible source")

    if proposed == "confirmed":
        if not credible:
            # A speculative "confirmation" is really just a rumor.
            proposed = "rumored"
        else:
            return _to(state, CatalystState.CONFIRMED, "confirmed by a credible source")

    # proposed == "rumored" (either directly, or downgraded from a speculative confirm)
    if state is CatalystState.UNCONFIRMED:
        return _to(state, CatalystState.RUMORED, "raised to rumored")
    # A rumor cannot downgrade a confirmation, nor revive an invalidated catalyst.
    reason = {
        CatalystState.RUMORED: "already rumored",
        CatalystState.CONFIRMED: "already confirmed — a rumor does not downgrade it",
        CatalystState.INVALIDATED: "invalidated — a rumor cannot revive it (needs a credible confirm)",
    }[state]
    return _stay(state, reason)


def _to(current: CatalystState, new: CatalystState, note: str) -> Transition:
    return Transition(new_state=new, changed=(new is not current), note=note)


def _stay(current: CatalystState, note: str) -> Transition:
    return Transition(new_state=current, changed=False, note=note)
