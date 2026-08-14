import asyncio

import pytest

from lily.memory.operational import MemoryStore, redact


def test_redact_card_numbers():
    assert "4111 1111 1111 1111" not in redact("card is 4111 1111 1111 1111 ok")


def test_redact_ssn():
    assert "123-45-6789" not in redact("my social is 123-45-6789 thanks")


def test_redact_otp_code():
    assert "482913" not in redact("the code is 482913")


def test_redact_preserves_short_numbers():
    text = "see you at 4 PM on the 21st"
    assert redact(text) == text


def test_redact_context_keyed_pin():
    assert "1234" not in redact("My PIN is 1234")
    assert "4321" not in redact("the code is 4321")
    assert "PIN" in redact("My PIN is 1234")  # keyword survives, digits don't


def test_redact_leaves_phone_numbers_alone():
    text = "call me back at 555-123-4567"
    assert redact(text) == text


@pytest.fixture()
def store(tmp_path):
    s = MemoryStore(str(tmp_path), retain_transcripts=False)
    asyncio.run(s.open())
    yield s
    asyncio.run(s.close())


def test_caller_lifecycle(store):
    async def run():
        ctx = await store.caller_context("+15551230000")
        assert ctx["known"] is False

        await store.touch_caller("+15551230000")
        await store.touch_caller("+15551230000")
        ctx = await store.caller_context("+15551230000")
        assert ctx["known"] is True
        assert ctx["call_count"] == 2
        assert ctx["trusted"] is False

        await store.set_trusted("+15551230000", True, "Pharmacy")
        ctx = await store.caller_context("+15551230000")
        assert ctx["trusted"] is True
        assert ctx["name"] == "Pharmacy"

        strikes = await store.add_scam_strike("+15559990000")
        assert strikes == 1
        strikes = await store.add_scam_strike("+15559990000")
        assert strikes == 2

    asyncio.run(run())


def test_call_and_events_and_activity(store):
    async def run():
        await store.record_call_start("CA123", "+15551230000", screened=True)
        await store.record_call_end(
            "CA123", "+15551230000", "Medium", 0.7,
            {"summary": "Caller discussed a pickup. Card 4111 1111 1111 1111."},
            ["line one", "line two"],
        )
        calls = await store.recent_calls()
        assert calls[0]["call_sid"] == "CA123"
        assert calls[0]["scam_level"] == "Medium"
        # retention off: transcript not persisted; redaction applied to summary
        assert calls[0]["summary_json"] != "" and "4111" not in calls[0]["summary_json"]

        event_id = await store.add_event("CA123", "+15551230000", "Pharmacy pickup", "Friday 4 PM")
        events = await store.open_events()
        assert events[0]["title"] == "Pharmacy pickup"
        await store.complete_event(event_id)
        assert await store.open_events() == []

        await store.add_activity("stayed_safe", "Stayed safe", "detail", "important", "CA123")
        activity = await store.recent_activity()
        assert activity[0]["kind"] == "stayed_safe"

        ctx = await store.caller_context("+15551230000")
        assert "pickup" in ctx["last_summary"].lower()

    asyncio.run(run())


def test_event_memory_link_roundtrip(store):
    async def run():
        event_id = await store.add_event("CA1", "+15551", "Pharmacy pickup", "Friday",
                                         memory_id="mem123")
        linked = await store.events_for_call("CA1")
        assert linked[0]["memory_id"] == "mem123"
        assert await store.complete_event(event_id) == "mem123"
        # Unlinked events return '' so callers can skip the twin-close.
        plain = await store.add_event("CA2", "+15552", "Call back", "")
        assert await store.complete_event(plain) == ""
    asyncio.run(run())


def test_events_migration_from_pre_link_schema(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "lily.db")
    db.execute(
        "CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, call_sid TEXT, "
        "number TEXT, title TEXT, when_text TEXT, created_at REAL, done INTEGER DEFAULT 0)"
    )
    db.execute(
        "INSERT INTO events (call_sid, number, title, when_text, created_at) "
        "VALUES ('CA0', '+1555', 'Old event', '', 0)"
    )
    db.commit()
    db.close()

    s = MemoryStore(str(tmp_path), retain_transcripts=False)

    async def run():
        await s.open()  # guarded ALTER must upgrade in place
        events = await s.open_events()
        assert events[0]["title"] == "Old event"
        assert await s.complete_event(events[0]["id"]) == ""
        await s.close()
    asyncio.run(run())


def test_transcript_retention_flag(tmp_path):
    s = MemoryStore(str(tmp_path), retain_transcripts=True)

    async def run():
        await s.open()
        await s.record_call_start("CA9", "+15550000001", screened=False)
        await s.record_call_end("CA9", "+15550000001", "Low", None, {"summary": "ok"}, ["hello there"])
        calls = await s.recent_calls()
        assert calls[0]["call_sid"] == "CA9"
        # transcript column not included in recent_calls payload, check directly
        row = await s._run(
            lambda conn: conn.execute("SELECT transcript FROM calls WHERE call_sid='CA9'").fetchone(),
        )
        assert row["transcript"] == "hello there"
        await s.close()

    asyncio.run(run())
