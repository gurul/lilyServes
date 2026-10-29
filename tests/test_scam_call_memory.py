"""A scam call must not become the older adult's to-do list or memory.

A scammer's instructions are phrased as commitments ("buy the cards at
Walgreens Friday at 3"), and the post-call summary can turn "you've won a
prize" into a *win* fact. Before this fix, a High-risk call stored both:
the instructions as open reminders, the claims as durable memories, and the
"prize" as a good-news activity item. Only the call's SAFETY record stays.
"""
from __future__ import annotations

import asyncio
import tempfile

import dotenv
import pytest

from lily.config import settings

# lily.app builds its stores from settings at import time: point them at a
# temp dir and never read a developer .env.
_saved = (settings.data_dir, settings.memory_embeddings)
settings.data_dir = tempfile.mkdtemp(prefix="lily-scam-memory-")
settings.memory_embeddings = False
dotenv.load_dotenv = lambda *a, **k: False
import lily.app as app_mod  # noqa: E402

settings.data_dir, settings.memory_embeddings = _saved

SCAM_SUMMARY = {
    "summary": "A caller said your grandson was in jail and asked for gift cards.",
    "scam_indicators": ["gift cards", "secrecy"],
    "risk_level": "high",
    "recommended_action": "Hang up if they call back",
    "events": [{"title": "Buy gift cards at Walgreens", "when": "Friday at 3 PM"}],
    "facts": [
        {"kind": "person", "content": "Officer Daniels handles the grandson's bail case.", "entities": ["Daniels"]},
        {"kind": "win", "content": "You won the national sweepstakes prize today.", "entities": []},
    ],
}
SAFE_SUMMARY = {
    "summary": "Main Street Pharmacy said the refill is ready.",
    "scam_indicators": [],
    "risk_level": "low",
    "recommended_action": "This appears legitimate",
    "events": [{"title": "Pick up refill at Main Street Pharmacy", "when": "Friday at 4 PM"}],
    "facts": [
        {"kind": "win", "content": "Sarah's kids loved the birthday gift card.", "entities": ["Sarah"]},
    ],
}


class Recorder:
    """Stands in for the stores, memory and hub; records every write."""

    def __init__(self, monkeypatch, summary, open_events=()):
        self.commitments: list[tuple] = []
        self.memories: list[tuple] = []
        self.calls: list[tuple] = []
        self.activity: list[str] = []
        self.completed_events: list[int] = []
        self.retired_memories: list[str] = []
        self.broadcasts: list[dict] = []
        self._open_events = list(open_events)

        async def summarize(_lines):
            return dict(summary)

        async def capture_commitment(call_sid, caller, title, when):
            self.commitments.append((title, when))
            return len(self.commitments)

        async def remember(content, **kw):
            self.memories.append((kw.get("memory_type"), content))

        async def remember_call(call_sid, caller, text, level):
            self.calls.append((level, text))

        async def complete_commitment(memory_id):
            self.retired_memories.append(memory_id)
            return True

        async def events_for_call(_sid):
            return [dict(e) for e in self._open_events]

        async def complete_event(event_id):
            self.completed_events.append(event_id)
            for e in self._open_events:
                if e["id"] == event_id:
                    return e["memory_id"]
            return ""

        async def record_call_end(*_a, **_k):
            pass

        async def activity(kind, *_a, **_k):
            self.activity.append(kind)

        async def broadcast(payload, *_a, **_k):
            self.broadcasts.append(payload)

        async def no_deepfake(_b):
            return {"score": None, "is_deepfake": False}

        monkeypatch.setattr(app_mod, "generate_summary", summarize)
        monkeypatch.setattr(app_mod, "check_deepfake", no_deepfake)
        monkeypatch.setattr(app_mod, "_capture_commitment", capture_commitment)
        monkeypatch.setattr(app_mod, "push_activity", activity)
        monkeypatch.setattr(app_mod.memory, "remember", remember)
        monkeypatch.setattr(app_mod.memory, "remember_call", remember_call)
        monkeypatch.setattr(app_mod.memory, "complete_commitment", complete_commitment)
        monkeypatch.setattr(app_mod.store, "events_for_call", events_for_call)
        monkeypatch.setattr(app_mod.store, "complete_event", complete_event)
        monkeypatch.setattr(app_mod.store, "record_call_end", record_call_end)
        monkeypatch.setattr(app_mod.hub, "broadcast", broadcast)


def _run(coro_fn):
    async def main():
        await coro_fn()
        loop = asyncio.get_running_loop()
        await asyncio.gather(*(t for t in list(app_mod._background) if t.get_loop() is loop))

    asyncio.run(main())


def _session(sid, caller, level):
    app_mod.registry._calls.pop(sid, None)
    session = app_mod.registry.create(sid, caller)
    session.scam_level = level
    session.transcript = ["a line"]
    return session


@pytest.fixture(autouse=True)
def _clean_registry():
    app_mod.registry._calls.clear()
    yield
    app_mod.registry._calls.clear()


def test_high_risk_call_stores_no_reminders_or_facts(monkeypatch):
    rec = Recorder(monkeypatch, SCAM_SUMMARY)
    _session("CA_SCAM", "+15550001111", "High")
    _run(lambda: app_mod._finalize_call("CA_SCAM"))
    assert rec.commitments == []
    assert rec.memories == []
    assert "good_news" not in rec.activity
    # The safety record of the call itself is kept.
    assert rec.calls == [("High", SCAM_SUMMARY["summary"])]


def test_summary_risk_high_alone_also_gates(monkeypatch):
    # The live level stayed Low, but the summary model judged the call high risk.
    rec = Recorder(monkeypatch, SCAM_SUMMARY)
    _session("CA_SUMHIGH", "+15550001112", "Low")
    _run(lambda: app_mod._finalize_call("CA_SUMHIGH"))
    assert rec.commitments == []
    assert rec.memories == []
    assert "good_news" not in rec.activity


def test_low_risk_call_still_stores_reminders_and_facts(monkeypatch):
    rec = Recorder(monkeypatch, SAFE_SUMMARY)
    _session("CA_SAFE", "+15550002222", "Low")
    _run(lambda: app_mod._finalize_call("CA_SAFE"))
    assert rec.commitments == [("Pick up refill at Main Street Pharmacy", "Friday at 4 PM")]
    assert [c for _, c in rec.memories] == ["Sarah's kids loved the birthday gift card."]
    assert "good_news" in rec.activity
    assert rec.completed_events == []


def test_reminders_captured_before_the_call_turned_high_are_retired(monkeypatch):
    rec = Recorder(
        monkeypatch, SCAM_SUMMARY,
        open_events=[{"id": 7, "title": "buy gift cards", "when_text": "tomorrow", "memory_id": "m7"}],
    )
    _session("CA_LATE", "+15550003333", "High")
    _run(lambda: app_mod._finalize_call("CA_LATE"))
    assert rec.completed_events == [7]
    assert rec.retired_memories == ["m7"]
    summary = next(b for b in rec.broadcasts if b.get("event") == "call_summary")
    assert summary["retired_events"] == [7]


def test_mid_call_capture_skips_a_high_risk_call(monkeypatch):
    rec = Recorder(monkeypatch, SCAM_SUMMARY)
    calls = []

    async def extract(sentence):
        calls.append(sentence)
        return [{"title": "Buy gift cards at Walgreens", "when": "tomorrow at 10"}]

    monkeypatch.setattr(app_mod, "extract_events", extract)
    _session("CA_MID", "+15550004444", "High")
    _run(lambda: app_mod._capture_events("CA_MID", "Go buy the gift cards tomorrow at 10."))
    assert rec.commitments == []
    assert "event_captured" not in rec.activity
    assert calls == []  # no model call is spent on a call already judged a scam


def test_mid_call_capture_rechecks_risk_after_extraction(monkeypatch):
    rec = Recorder(monkeypatch, SCAM_SUMMARY)
    session = _session("CA_RACE", "+15550005555", "Medium")

    async def extract(_sentence):
        session.scam_level = "High"  # the call escalates while the model is working
        return [{"title": "Buy gift cards", "when": "tomorrow"}]

    monkeypatch.setattr(app_mod, "extract_events", extract)
    _run(lambda: app_mod._capture_events("CA_RACE", "Buy the cards tomorrow."))
    assert rec.commitments == []
    assert "event_captured" not in rec.activity


def test_mid_call_capture_after_hangup_is_left_to_the_summary(monkeypatch):
    rec = Recorder(monkeypatch, SCAM_SUMMARY)
    _session("CA_GONE", "+15550006666", "Low")

    async def extract(_sentence):
        app_mod.registry.pop("CA_GONE")  # finalize ran while the model was working
        return [{"title": "Visit Dr. Lee", "when": "Monday"}]

    monkeypatch.setattr(app_mod, "extract_events", extract)
    _run(lambda: app_mod._capture_events("CA_GONE", "See Dr. Lee Monday."))
    assert rec.commitments == []


def test_mid_call_capture_on_a_low_call_still_works(monkeypatch):
    rec = Recorder(monkeypatch, SAFE_SUMMARY)

    async def extract(_sentence):
        return [{"title": "Pick up refill", "when": "Friday at 4"}]

    monkeypatch.setattr(app_mod, "extract_events", extract)
    _session("CA_OK", "+15550007777", "Low")
    _run(lambda: app_mod._capture_events("CA_OK", "Your refill is ready Friday at 4."))
    assert rec.commitments == [("Pick up refill", "Friday at 4")]
    assert "event_captured" in rec.activity
