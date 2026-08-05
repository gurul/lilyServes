import asyncio

import pytest

from memory_store import MemoryStore, redact


def test_redact_card_numbers():
    assert "4111 1111 1111 1111" not in redact("card is 4111 1111 1111 1111 ok")


def test_redact_ssn():
    assert "123-45-6789" not in redact("my social is 123-45-6789 thanks")


def test_redact_otp_code():
    assert "482913" not in redact("the code is 482913")


def test_redact_preserves_short_numbers():
    text = "see you at 4 PM on the 21st"
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
            s._conn,
        )
        assert row["transcript"] == "hello there"
        await s.close()

    asyncio.run(run())
