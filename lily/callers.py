"""Caller identity: the one rule for "this caller ID names nobody".

Twilio collapses every withheld caller ID onto a few shared constants. None of
them is an identity. Strikes, memories, profile cards and the last-call
summary keyed on one of them would pass from one unrelated caller to the next.
Every guard in the app uses this predicate, so the rule cannot drift between
call screening, the call pipeline and the memory lifecycle.
"""
from __future__ import annotations

# Lower-cased. "+266696687" spells "ANONYMOUS" on a keypad: Twilio sends it for
# some carriers' withheld calls. "" is how the app stores a caller with no ID.
WITHHELD_CALLERS = frozenset({
    "", "unknown", "anonymous", "restricted", "private", "unavailable",
    "blocked", "withheld", "+266696687", "266696687",
})


def is_withheld(caller: str | None) -> bool:
    """True when `caller` is empty or a withheld-ID placeholder."""
    return (caller or "").strip().lower() in WITHHELD_CALLERS
