"""Property checks of the fusion invariants over random call histories.

Each test replays seeded random histories (sentences, LLM verdicts, deepfake
scores) through RiskFusion and checks an invariant from the fusion.py
docstring after every event. The seed is fixed, so a failure reproduces.
"""
from __future__ import annotations

import random

import pytest

from lily.detection.fusion import (
    LEVELS,
    PAYMENT_STAGE_SIGNALS,
    THETA_ALERT,
    THETA_WARN,
    RiskFusion,
)
from lily.detection.heuristics import HeuristicResult, max_level

HISTORIES = 3000
EPS = 1e-9
OTHER_SIGNALS = ("urgency", "gov impersonation", "bank impersonation", "prize/lottery", "tech support")
SIGNAL_POOL = tuple(sorted(PAYMENT_STAGE_SIGNALS)) + OTHER_SIGNALS
VERDICTS = ("scam", "uncertain", "safe", "garbage")
DEEPFAKES = (None, 0.0, 0.3, 0.6, 1.0, -0.5, 1.7)


def _event(rng: random.Random):
    kind = rng.random()
    if kind < 0.7:
        score = rng.randint(0, 15)
        signals = rng.sample(SIGNAL_POOL, k=rng.choice((0, 0, 0, 1, 1, 2)))
        return ("sentence", HeuristicResult("Low", score, signals))
    if kind < 0.9:
        return ("llm", rng.choice(VERDICTS))
    deepfake = rng.choice(DEEPFAKES + (rng.random(),))
    return ("deepfake", deepfake)


def _apply(fusion: RiskFusion, event):
    kind, value = event
    if kind == "sentence":
        return fusion.on_sentence(value)
    if kind == "llm":
        return fusion.on_llm(value)
    return fusion.on_deepfake(value)


def _histories():
    rng = random.Random(20261028)
    for _ in range(HISTORIES):
        yield [_event(rng) for _ in range(rng.randint(1, 30))]


def _expected_level(f: RiskFusion) -> str:
    """The Level rule from the fusion.py docstring, written out from fields."""
    corroborated = f.payment_marker_seen and len(f._signals_seen) >= 2
    lone = f.payment_marker_seen and not corroborated
    cleared = lone and f.llm_after_marker and f.llm_verdict == "safe"
    hard = corroborated or f.ewma >= THETA_ALERT - EPS or f.llm_verdict == "scam"
    if hard:
        return "High"
    if (
        f.ewma >= THETA_WARN - EPS
        or (f.llm_verdict == "uncertain" and f._signals_seen)
        or (lone and not cleared)
    ):
        return "Medium"
    return "Low"


def test_ewma_stays_in_unit_interval():
    for history in _histories():
        f = RiskFusion()
        for event in history:
            _apply(f, event)
            assert 0.0 <= f.ewma <= 1.0


def test_deepfake_never_lowers_risk():
    # The same history with a synthetic-voice score never ends with less risk.
    for history in _histories():
        plain, faked = RiskFusion(), RiskFusion()
        faked.on_deepfake(random.Random(len(history)).random())
        for event in history:
            if event[0] == "deepfake":
                continue
            _apply(plain, event)
            _apply(faked, event)
            assert faked.ewma >= plain.ewma - EPS
            assert LEVELS.index(faked.state().level) >= LEVELS.index(plain.state().level)


def test_llm_scam_verdict_always_alerts():
    for history in _histories():
        f = RiskFusion()
        for event in history:
            _apply(f, event)
        state = f.on_llm("scam")
        assert state.hard_alert is True
        assert state.level == "High"
        assert f.ewma >= THETA_ALERT


@pytest.mark.parametrize("later", ["safe", "uncertain"])
def test_scam_floor_survives_a_later_verdict(later):
    # With no sentence in between, the 0.60 floor keeps the hard alert.
    for history in _histories():
        f = RiskFusion()
        for event in history:
            _apply(f, event)
        f.on_llm("scam")
        assert f.on_llm(later).hard_alert is True


def test_corroborated_marker_always_alerts():
    for history in _histories():
        f = RiskFusion()
        for event in history:
            state = _apply(f, event)
            if f.payment_marker_seen and len(f._signals_seen) >= 2:
                assert state.hard_alert is True


def test_lone_marker_cleared_by_safe_is_not_a_hard_alert():
    for history in _histories():
        f = RiskFusion()
        for event in history:
            state = _apply(f, event)
            lone = f.payment_marker_seen and len(f._signals_seen) < 2
            if lone and f.llm_after_marker and f.llm_verdict == "safe" and f.ewma < THETA_ALERT - EPS:
                assert state.hard_alert is False


def test_level_matches_the_documented_rule():
    for history in _histories():
        f = RiskFusion()
        for event in history:
            state = _apply(f, event)
            assert state.level == _expected_level(f)


def test_state_reads_and_never_writes():
    for history in _histories():
        f = RiskFusion()
        for event in history:
            _apply(f, event)
        before = dict(vars(f), _signals_seen=set(f._signals_seen))
        assert f.state() == f.state()
        assert dict(vars(f), _signals_seen=set(f._signals_seen)) == before


def test_displayed_level_never_decreases():
    # app.py shows max_level(previous, heuristic level, fusion level).
    for history in _histories():
        f = RiskFusion()
        shown = "Low"
        for event in history:
            state = _apply(f, event)
            heur_level = event[1].level if event[0] == "sentence" else "Low"
            new = max_level(shown, heur_level, state.level)
            assert LEVELS.index(new) >= LEVELS.index(shown)
            shown = new
