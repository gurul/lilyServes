"""Regression tests for the 2026-11-15 critical review.

Each test pins one defect that review reproduced on commit b13de61:

- J1  the risk average was fed the whole-transcript score on every sentence,
      so a long normal call climbed to High.
- J2  the dashboard token was written to the log in clear text: uvicorn logs
      WebSocket handshakes on "uvicorn.error", and only "uvicorn.access" was
      masked.
- J3  the summary and reminder-extraction calls sent card numbers, PINs and
      SSNs to the model provider unredacted.
- J4  a failed model call read "safe", replaced a real earlier verdict, and
      /health still said "ok".
- J5  the per-call stream `sig` covered only the CallSid: it could be
      replayed, and with any caller number.
- J6  three different definitions of a withheld caller ID let one withheld
      caller's context show on another's call.
- J7  with no public URL the startup check assumed a private server, while
      the server listened on every interface.
- J8  deleting a memory left the same text in the calls, callers and
      activity tables.
- J10 the screened caller's spoken answer ignored SHARE_TRANSCRIPTS.
"""
from __future__ import annotations

import asyncio
import logging
import tempfile
import types

import dotenv
import pytest

from lily.config import settings

# lily.app builds its stores from settings at import time: point them at a
# temp dir and never read a developer .env.
_saved = (settings.data_dir, settings.memory_embeddings)
settings.data_dir = tempfile.mkdtemp(prefix="lily-judge-")
settings.memory_embeddings = False
dotenv.load_dotenv = lambda *a, **k: False
import lily.app as app_mod  # noqa: E402

settings.data_dir, settings.memory_embeddings = _saved

from lily import auth, events, screening, summarizer  # noqa: E402
from lily.detection import detector  # noqa: E402
from lily.memory import MemoryService, lifecycle  # noqa: E402
from lily.memory.operational import MemoryStore  # noqa: E402

PRIVATE = "my new card is 4111 1111 1111 1111, PIN is 4821, social 123-45-6789"


def _run(coro_fn):
    """Run a coroutine function and every background task it spawned."""
    async def main():
        result = await coro_fn()
        loop = asyncio.get_running_loop()
        while True:
            pending = [t for t in list(app_mod._background) if t.get_loop() is loop]
            if not pending:
                return result
            await asyncio.gather(*pending, return_exceptions=True)
            # gather() of tasks that are already done returns without
            # yielding, so their done-callbacks (which empty _background)
            # would never run.
            await asyncio.sleep(0)

    return asyncio.run(main())


@pytest.fixture(autouse=True)
def _clean_registry():
    app_mod.registry._calls.clear()
    yield
    app_mod.registry._calls.clear()


class _Pipeline:
    """The live path with the network taken out: every model call, the hub
    and the activity feed are stubs that record what they were given."""

    def __init__(self, monkeypatch, verdict="safe"):
        self.activity: list[str] = []
        self.broadcasts: list[dict] = []
        self.verdict = verdict

        async def analyze(_transcript):
            return {"verdict": self.verdict, "criteria": [], "reasoning": "", "signals": [],
                    "available": True}

        async def activity(kind, *_a, **_k):
            self.activity.append(kind)

        async def broadcast(payload, *_a, **_k):
            self.broadcasts.append(payload)

        async def no_events(_sentence):
            return []

        async def no_strike(_number):
            return 1

        monkeypatch.setattr(app_mod, "analyze_scam", analyze)
        monkeypatch.setattr(app_mod, "push_activity", activity)
        monkeypatch.setattr(app_mod.hub, "broadcast", broadcast)
        monkeypatch.setattr(app_mod, "extract_events", no_events)
        monkeypatch.setattr(app_mod.store, "add_scam_strike", no_strike)
        monkeypatch.setattr(settings, "auto_intervene", False)

    def play(self, sid, lines):
        session = app_mod.registry.create(sid, "+15550007001")

        async def go():
            spoken: list[str] = []
            for line in lines:
                spoken.append(line)
                await app_mod._on_final_line(sid, line, " ".join(spoken))
                await asyncio.sleep(0)

        _run(go)
        return session


# ----- J1: one early signal must not be counted again on every sentence ------

BENIGN_OPENING = [
    "Hi Grandma, it is your grandson Daniel.",
    "I wanted you to know my brother was in a car accident, he is fine, he is at the hospital.",
    "Do you still have his insurance account number somewhere?",
]
SMALL_TALK = [
    "The garden looks lovely this year.",
    "We had soup for dinner and watched the game.",
    "The kids are back at school next week.",
]


def test_j1_long_normal_call_does_not_climb_to_high(monkeypatch):
    pipeline = _Pipeline(monkeypatch, verdict="safe")
    lines = BENIGN_OPENING + SMALL_TALK * 19
    assert len(lines) == 60

    session = pipeline.play("CAj1benign", lines)

    # The whole-transcript regex score is 6 (Medium): below the High line.
    assert session.heuristic_signals == ["family emergency", "account credentials"]
    assert session.scam_level == "Medium"
    assert "scam_alert" not in pipeline.activity
    # Small talk decays the average: the newest sentences carry the weight.
    assert session.fusion.ewma < 0.05


def test_j1_scam_still_reaches_high_at_once(monkeypatch):
    """The fix must not blind the detector: a real script alerts by line 3."""
    pipeline = _Pipeline(monkeypatch, verdict="safe")
    session = pipeline.play("CAj1scam", [
        "This is the IRS calling about your tax file.",
        "There is a warrant for your arrest unless this is settled today.",
        "You need to pay right now with gift cards.",
    ])
    assert session.scam_level == "High"
    assert pipeline.activity.count("scam_alert") == 1


def test_j1_pattern_completed_across_sentences_still_counts(monkeypatch):
    """A rule can match across two sentences. The sentence that completes it
    carries the added score, though it scores nothing alone."""
    pipeline = _Pipeline(monkeypatch, verdict="safe")
    session = pipeline.play("CAj1span", [
        "Grandma it is your grandson.",
        "I am in jail.",
    ])
    assert session.heuristic_signals == ["family emergency"]
    assert session.fusion.ewma == pytest.approx(0.3 / 3)


def test_j1_llm_scam_verdict_strikes_once_through_the_live_path(monkeypatch):
    """The composed path the unit tests never ran: sentence -> LLM verdict ->
    one alert and one strike, however many lines follow."""
    strikes: list[str] = []
    pipeline = _Pipeline(monkeypatch, verdict="scam")

    async def strike(number):
        strikes.append(number)
        return len(strikes)

    monkeypatch.setattr(app_mod.store, "add_scam_strike", strike)
    session = pipeline.play("CAj1llm", [
        "Hello, I am calling from your bank's security team.",
        "Please move your savings to the safe account I give you.",
        "Stay on the line with me while you do it.",
    ])
    assert session.scam_level == "High"
    assert strikes == ["+15550007001"]
    assert pipeline.activity.count("scam_alert") == 1


# ----- J2: the dashboard token must not reach any uvicorn log ----------------

class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@pytest.mark.parametrize("logger_name,fmt,args", [
    # The exact calls uvicorn makes for a WebSocket handshake.
    ("uvicorn.error", '%s - "WebSocket %s" [accepted]', ("127.0.0.1:5000", "/ws/client?token=s3cret")),
    ("uvicorn.error", '%s - "WebSocket %s" 403', ("127.0.0.1:5000", "/ws/client?a=1&token=s3cret&b=2")),
    # And for an HTTP request.
    ("uvicorn.access", '%s - "%s %s HTTP/%s" %d', ("127.0.0.1:5000", "GET", "/x?token=s3cret", "1.1", 200)),
])
def test_j2_token_is_masked_on_every_uvicorn_logger(logger_name, fmt, args):
    auth.install_access_log_filter()
    logger = logging.getLogger(logger_name)
    capture = _Capture()
    logger.addHandler(capture)
    previous = logger.level
    logger.setLevel(logging.INFO)
    try:
        # Through the real logger, so its own filters decide: a test that
        # builds a record by hand passes whatever is installed.
        logger.info(fmt, *args)
    finally:
        logger.removeHandler(capture)
        logger.setLevel(previous)
    assert capture.lines, "the record was dropped"
    assert "s3cret" not in capture.lines[0]
    assert "token=***" in capture.lines[0]


# ----- J3: private numbers do not leave the process --------------------------

class _FakeClient:
    """Stands in for the OpenAI client; records the messages it was sent."""

    def __init__(self, content='{"events": []}', error=None):
        self.sent: list[list[dict]] = []
        self._content = content
        self._error = error
        self.chat = types.SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.sent.append(kwargs["messages"])
        if self._error is not None:
            raise self._error
        message = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    def user_text(self) -> str:
        return " ".join(m["content"] for msgs in self.sent for m in msgs if m["role"] == "user")


def _assert_private_numbers_absent(text: str) -> None:
    assert "4111" not in text
    assert "4821" not in text
    assert "123-45-6789" not in text


def test_j3_summary_call_is_redacted(monkeypatch):
    fake = _FakeClient(content='{"summary": "ok"}')
    monkeypatch.setattr(summarizer, "get_client", lambda: fake)
    asyncio.run(summarizer.generate_summary(["Hello dear.", PRIVATE]))
    assert fake.sent, "the model was never called"
    _assert_private_numbers_absent(fake.user_text())
    assert "Hello dear." in fake.user_text()


def test_j3_scam_verdict_call_is_redacted(monkeypatch):
    fake = _FakeClient(content='{"verdict": "safe", "criteria": [], "reasoning": ""}')
    monkeypatch.setattr(detector, "get_client", lambda: fake)
    asyncio.run(detector.analyze_scam("Hello dear. " + PRIVATE))
    assert fake.sent, "the model was never called"
    _assert_private_numbers_absent(fake.user_text())
    assert "Hello dear." in fake.user_text()


def test_j3_reminder_extraction_call_is_redacted(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(events, "get_client", lambda: fake)
    asyncio.run(events.extract_events("See you Friday at 4 PM, " + PRIVATE))
    assert fake.sent, "the model was never called"
    _assert_private_numbers_absent(fake.user_text())
    assert "Friday at 4 PM" in fake.user_text()


# ----- J4: an outage is not a verdict ---------------------------------------

REWORDED_SCAM = "Go and buy two prepaid store cards and scratch the back for me."


def test_j4_outage_never_reads_safe(monkeypatch):
    monkeypatch.setattr(detector, "get_client", lambda: _FakeClient(error=RuntimeError("401")))
    result = asyncio.run(detector.analyze_scam(REWORDED_SCAM))
    # The regex tier is quiet on this wording, so nothing judged the call.
    assert result["signals"] == []
    assert result["verdict"] != "safe"
    assert result["available"] is False


def test_j4_outage_does_not_replace_an_earlier_scam_verdict(monkeypatch):
    monkeypatch.setattr(detector, "get_client", lambda: _FakeClient(error=RuntimeError("down")))
    monkeypatch.setattr(app_mod, "analyze_scam", detector.analyze_scam)

    async def quiet(*_a, **_k):
        return None

    monkeypatch.setattr(app_mod.hub, "broadcast", quiet)
    monkeypatch.setattr(app_mod, "push_activity", quiet)
    monkeypatch.setattr(app_mod.store, "add_scam_strike", quiet)
    session = app_mod.registry.create("CAj4", "+15550007004")
    session.transcript = [REWORDED_SCAM]
    session.fusion.on_llm("scam")

    _run(lambda: app_mod._refine_scam("CAj4"))

    assert session.fusion.llm_verdict == "scam"
    assert session.fusion.state().hard_alert is True


def test_j4_health_reports_a_failing_llm_tier(monkeypatch):
    monkeypatch.setattr(detector, "get_client", lambda: _FakeClient(error=RuntimeError("401")))
    asyncio.run(detector.analyze_scam("Hello there."))
    failing = asyncio.run(app_mod.health())
    assert failing["status"] == "degraded"
    assert failing["llm"]["ok"] is False

    monkeypatch.setattr(
        detector, "get_client",
        lambda: _FakeClient(content='{"verdict": "safe", "criteria": [], "reasoning": ""}'),
    )
    asyncio.run(detector.analyze_scam("Hello there."))
    healthy = asyncio.run(app_mod.health())
    assert healthy["status"] == "ok"
    assert healthy["llm"]["ok"] is True


# ----- J5: the stream sig covers the caller, and a CallSid is used once ------

def test_j5_sig_does_not_verify_for_another_caller_or_flag(monkeypatch):
    monkeypatch.setattr(settings, "twilio_auth_token", "twtok")
    sig = auth.stream_param_sig("twtok", "CAj5", "+15550001111", False)
    assert auth.verify_stream_param_sig("CAj5", sig, "+15550001111", False)
    assert not auth.verify_stream_param_sig("CAj5", sig, "+15550009999", False)
    assert not auth.verify_stream_param_sig("CAj5", sig, "+15550001111", True)
    assert not auth.verify_stream_param_sig("CAother", sig, "+15550001111", False)


def test_j5_a_callsid_is_claimed_once_and_the_first_record_is_kept():
    async def go():
        store = MemoryStore(tempfile.mkdtemp(prefix="lily-j5-"))
        await store.open()
        try:
            first = await store.record_call_start("CAj5replay", "+15550001111", False)
            replay = await store.record_call_start("CAj5replay", "+15550009999", True)
            calls = await store.recent_calls()
        finally:
            await store.close()
        return first, replay, calls

    first, replay, calls = asyncio.run(go())
    assert first is True
    assert replay is False
    assert [(c["call_sid"], c["number"]) for c in calls] == [("CAj5replay", "+15550001111")]


def test_j5_a_replayed_start_does_not_begin_a_call(monkeypatch):
    started: list[str] = []

    class _STT:
        def __init__(self, **_kw):
            pass

        async def start(self):
            started.append("stt")

    async def quiet(*_a, **_k):
        return None

    async def never_claimed(*_a, **_k):
        return False

    monkeypatch.setattr(app_mod, "GoogleStreamingSession", _STT)
    monkeypatch.setattr(app_mod.hub, "broadcast", quiet)
    monkeypatch.setattr(app_mod.store, "record_call_start", never_claimed)

    with pytest.raises(app_mod.CallExistsError):
        _run(lambda: app_mod._begin_call("CAj5seen", "+15550009999", False))

    assert started == []
    assert app_mod.registry.get("CAj5seen") is None


# ----- J6: one rule for a withheld caller ID ---------------------------------

WITHHELD = ["", "unknown", "Anonymous", "Restricted", "private", "unavailable",
            "blocked", "withheld", "+266696687"]


@pytest.mark.parametrize("caller", WITHHELD)
def test_j6_every_guard_uses_the_same_rule(caller):
    assert screening.is_anonymous(caller)
    assert lifecycle.is_withheld_caller(caller)


def test_j6_a_real_number_is_not_withheld():
    assert not screening.is_anonymous("+15550001111")
    assert not lifecycle.is_withheld_caller("+15550001111")


@pytest.mark.parametrize("caller", ["Restricted", "+266696687", "anonymous"])
def test_j6_withheld_bucket_has_no_caller_context(caller):
    async def go():
        store = MemoryStore(tempfile.mkdtemp(prefix="lily-j6-"))
        await store.open()
        try:
            await store.touch_caller(caller)
            await store.record_call_start("CAj6", caller, True)
            await store.record_call_end(
                "CAj6", caller, "Low", None,
                {"summary": "Caller from the county asked about Margaret's pension."}, [],
            )
            return await store.caller_context(caller)
        finally:
            await store.close()

    ctx = asyncio.run(go())
    assert ctx == {"known": False, "number": caller}


@pytest.mark.parametrize("caller", ["Restricted", "+266696687"])
def test_j6_withheld_call_shows_no_card_and_recalls_nothing(monkeypatch, caller):
    asked: list[str] = []
    payloads: list[dict] = []

    class _STT:
        def __init__(self, **_kw):
            pass

        async def start(self):
            pass

    async def profile(number, _name=""):
        asked.append("profile:" + number)
        return {"facts": ["Dr. Patel at Mercy Clinic is Margaret's cardiologist."]}

    async def recall(number, _name=""):
        asked.append("recall:" + number)
        return [{"content": "another caller's memory"}]

    async def broadcast(payload, *_a, **_k):
        payloads.append(payload)

    async def claimed(*_a, **_k):
        return True

    async def no_context(number):
        return {"known": False, "number": number}

    monkeypatch.setattr(app_mod, "GoogleStreamingSession", _STT)
    monkeypatch.setattr(app_mod.memory, "profile", profile)
    monkeypatch.setattr(app_mod.memory, "caller_context", recall)
    monkeypatch.setattr(app_mod.hub, "broadcast", broadcast)
    monkeypatch.setattr(app_mod.store, "record_call_start", claimed)
    monkeypatch.setattr(app_mod.store, "caller_context", no_context)

    _run(lambda: app_mod._begin_call("CAj6card", caller, True))

    assert asked == []
    started = [p for p in payloads if p["event"] == "call_started"]
    assert len(started) == 1 and started[0]["memory_card"] is None
    assert not [p for p in payloads if p["event"] == "caller_memories"]


def test_j6_a_withheld_id_cannot_be_trusted():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as err:
        asyncio.run(app_mod.api_set_trusted({"number": "anonymous", "trusted": True}))
    assert err.value.status_code == 422


# ----- J7: a server other machines can reach needs its secrets ---------------

def _exposure(monkeypatch, host):
    monkeypatch.setattr(settings, "host", host)
    monkeypatch.setattr(settings, "public_url", "")
    monkeypatch.setattr(settings, "ngrok_enabled", False)
    monkeypatch.setattr(settings, "allow_insecure_dev", False)
    monkeypatch.setattr(settings, "client_token", "")
    monkeypatch.setattr(settings, "twilio_auth_token", "")


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20"])
def test_j7_non_loopback_host_without_secrets_refuses_to_start(monkeypatch, host):
    _exposure(monkeypatch, host)
    with pytest.raises(RuntimeError, match="CLIENT_TOKEN and TWILIO_AUTH_TOKEN"):
        app_mod.check_exposure()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_j7_loopback_host_starts_without_secrets(monkeypatch, host):
    _exposure(monkeypatch, host)
    app_mod.check_exposure()


def test_j7_default_host_is_loopback(monkeypatch):
    from lily.config import Settings

    monkeypatch.delenv("HOST", raising=False)
    assert Settings.from_env().host == "127.0.0.1"
    monkeypatch.setenv("HOST", "0.0.0.0")
    assert Settings.from_env().host == "0.0.0.0"


# ----- J8: a deleted memory is gone from every table -------------------------

SECRET = "Margaret discussed her biopsy with Dr. Lee."


def _delete_through_the_api(monkeypatch, topics, content):
    """Store one call with its record, delete the memory through the route
    function, and return what every read path still serves."""
    async def go():
        tmp = tempfile.mkdtemp(prefix="lily-j8-")
        store = MemoryStore(tmp)
        memory = MemoryService.lexical_only(tmp)
        await store.open()
        await memory.open()
        monkeypatch.setattr(app_mod, "store", store)
        monkeypatch.setattr(app_mod, "memory", memory)
        try:
            await store.record_call_start("CAj8", "+15550008001", False)
            await store.record_call_end(
                "CAj8", "+15550008001", "Low", None, {"summary": SECRET}, [SECRET],
            )
            await store.add_activity("call_ended", "Call ended", SECRET, "info", "CAj8")
            await store.add_activity("good_news", "A nice moment", "The roses bloomed.", "info", "CAj8")
            await store.add_activity("call_ended", "Call ended", "Another call.", "info", "CAother")
            m = await memory.remember(
                content, topics=topics, caller="+15550008001", call_sid="CAj8",
            )
            await app_mod.api_memories_delete(m.id)
            return {
                "calls": await store.recent_calls(),
                "activity": await store.recent_activity(),
                "context": await store.caller_context("+15550008001"),
                "memories": await memory.recent(10),
            }
        finally:
            await memory.close()
            await store.close()

    return asyncio.run(go())


def test_j8_deleting_the_call_record_scrubs_every_table(monkeypatch):
    left = _delete_through_the_api(monkeypatch, ["phone_call"], SECRET)
    assert left["memories"] == []
    assert "biopsy" not in str(left["calls"])
    assert "biopsy" not in str(left["activity"])
    assert left["context"]["last_summary"] == ""
    # Another call's activity is not touched.
    assert "Another call." in str(left["activity"])


def test_j8_deleting_one_fact_scrubs_only_the_rows_that_quote_it(monkeypatch):
    left = _delete_through_the_api(monkeypatch, ["extracted"], "The roses bloomed.")
    assert "The roses bloomed." not in str(left["activity"])
    # The call's own summary is a different memory: it stays.
    assert "biopsy" in str(left["calls"])
    assert "biopsy" in str(left["activity"])


# ----- J10: the screened caller's answer follows SHARE_TRANSCRIPTS -----------

@pytest.mark.parametrize("share,expect_speech", [(False, False), (True, True)])
def test_j10_screen_answer_follows_the_transcript_setting(monkeypatch, share, expect_speech):
    details: list[str] = []

    async def activity(_kind, _title, detail, *_a, **_k):
        details.append(detail)

    async def quiet(*_a, **_k):
        return None

    monkeypatch.setattr(settings, "share_transcripts", share)
    monkeypatch.setattr(settings, "twilio_auth_token", "")
    monkeypatch.setattr(app_mod, "push_activity", activity)
    monkeypatch.setattr(app_mod.store, "touch_caller", quiet)
    request = types.SimpleNamespace(
        app=types.SimpleNamespace(state=types.SimpleNamespace(public_url="https://lily.example")),
        base_url="https://lily.example/",
    )
    form = {"From": "+15550001010", "CallSid": "CAj10",
            "SpeechResult": "This is Dr. Lee calling about the biopsy results."}

    _run(lambda: app_mod.screen_result(request, form))

    assert len(details) == 1
    assert ("biopsy" in details[0]) is expect_speech
    assert "monitored" in details[0]
