"""Regression tests for the live call pipeline in lily.app.

Each test pins one defect: forged media streams, duplicate `start`,
STT init failure, screening bypass, the shared anonymous strike bucket,
open-by-default exposure, token-in-query logging, and the strike policy.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import tempfile
import time

import dotenv
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from lily.auth import compute_twilio_signature, stream_param_sig
from lily.config import settings

# lily.app builds its stores from settings at import time: point them at a
# temp dir and never read a developer .env.
_saved = (settings.data_dir, settings.memory_embeddings)
settings.data_dir = tempfile.mkdtemp(prefix="lily-pipeline-")
settings.memory_embeddings = False
dotenv.load_dotenv = lambda *a, **k: False
import lily.app as app_mod  # noqa: E402
import lily.transcription as tr  # noqa: E402

settings.data_dir, settings.memory_embeddings = _saved

PUBLIC = "https://lily.example"
WSS = "wss://lily.example/ws/twilio"
TW_TOKEN = "twtok"
SCAM = (
    "This is the IRS. You must pay now with gift cards and do not tell anyone, "
    "keep this secret, wire transfer immediately or you will be arrested"
)


class FakeSTT:
    """Stands in for GoogleStreamingSession: each audio frame is one final line."""

    instances: list[FakeSTT] = []

    def __init__(self, on_final_result=None, **_kw):
        self.cb = on_final_result
        self.lines: list[str] = []
        self.stopped = False
        FakeSTT.instances.append(self)

    async def start(self):
        pass

    def feed_audio(self, b):
        text = b.decode()
        self.lines.append(text)
        asyncio.get_running_loop().create_task(self.cb(text, " ".join(self.lines)))

    async def stop(self):
        self.stopped = True
        return self.lines


async def _summary(_lines):
    return {"summary": "x", "risk_level": "low", "events": [], "facts": []}


async def _scam_safe(_t):
    return {"verdict": "safe", "criteria": [], "reasoning": "", "signals": []}


async def _no_events(_s):
    return []


async def _no_deepfake(_b):
    return {"score": None, "is_deepfake": False}


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(settings, "public_url", PUBLIC)
    monkeypatch.setattr(settings, "ngrok_enabled", False)
    monkeypatch.setattr(settings, "allow_insecure_dev", True)
    monkeypatch.setattr(settings, "client_token", "")
    monkeypatch.setattr(settings, "twilio_auth_token", "")
    monkeypatch.setattr(settings, "twilio_account_sid", "")
    monkeypatch.setattr(settings, "forward_to", "+10000000000")
    monkeypatch.setattr(settings, "trusted_numbers", [])
    monkeypatch.setattr(settings, "screen_unknown_callers", True)
    monkeypatch.setattr(settings, "auto_intervene", False)
    monkeypatch.setattr(app_mod, "generate_summary", _summary)
    monkeypatch.setattr(app_mod, "analyze_scam", _scam_safe)
    monkeypatch.setattr(app_mod, "extract_events", _no_events)
    monkeypatch.setattr(app_mod, "check_deepfake", _no_deepfake)
    monkeypatch.setattr(app_mod, "GoogleStreamingSession", FakeSTT)
    FakeSTT.instances.clear()
    app_mod.registry._calls.clear()
    yield
    app_mod.registry._calls.clear()


@pytest.fixture(scope="module")
def _app_client():
    # lily.memory cannot reopen after close: one lifespan for the module.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(settings, "public_url", PUBLIC)
        mp.setattr(settings, "ngrok_enabled", False)
        mp.setattr(settings, "allow_insecure_dev", True)
        mp.setattr(settings, "twilio_account_sid", "")
        with TestClient(app_mod.app, raise_server_exceptions=False) as c:
            yield c


@pytest.fixture
def client(_app_client):
    return _app_client


def _post(c, path, form):
    headers = {}
    if settings.twilio_auth_token:
        headers["X-Twilio-Signature"] = compute_twilio_signature(
            settings.twilio_auth_token, PUBLIC + path, form
        )
    return c.post(path, data=form, headers=headers)


def _route(c, frm, sid):
    body = _post(c, "/twiml", {"From": frm, "CallSid": sid}).text
    if "not accepting your calls" in body:
        return "block"
    if "<Gather" in body:
        return "screen"
    if "<Dial>" in body:
        return "pass"
    return body


def _ctx(c, number):
    return c.portal.call(app_mod.store.caller_context, number)


def _wait(pred, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


def _start(sid, caller, sig=None):
    params = {"caller": caller}
    if sig is not None:
        params["sig"] = sig
    return json.dumps({"event": "start", "start": {"callSid": sid, "customParameters": params}})


def _media(text):
    return json.dumps({"event": "media", "media": {"payload": base64.b64encode(text.encode()).decode()}})


STOP = json.dumps({"event": "stop"})


# ----- Fix 1: /ws/twilio authentication --------------------------------------

def test_ws_twilio_unsigned_handshake_cannot_start_a_call(client, monkeypatch):
    # The handshake check is observe-only; the start `sig` param is what
    # stops a forged stream from creating a session.
    monkeypatch.setattr(settings, "twilio_auth_token", TW_TOKEN)
    for headers in ({}, {"X-Twilio-Signature": "bogus"}):
        try:
            with client.websocket_connect("/ws/twilio", headers=headers) as ws:
                ws.send_text(_start("CAforged", "+15550001111"))
                time.sleep(0.2)
                assert app_mod.registry.get("CAforged") is None
        except WebSocketDisconnect:
            pass
    assert app_mod.registry.get("CAforged") is None


@pytest.mark.parametrize("signed_url", [WSS, WSS + "/"])
def test_ws_twilio_handshake_accepts_valid_signature(client, monkeypatch, signed_url):
    monkeypatch.setattr(settings, "twilio_auth_token", TW_TOKEN)
    sig = compute_twilio_signature(TW_TOKEN, signed_url, {})
    with client.websocket_connect("/ws/twilio", headers={"X-Twilio-Signature": sig}) as ws:
        ws.send_text(_start("CAok", "+15550001112", stream_param_sig(TW_TOKEN, "CAok")))
        assert _wait(lambda: app_mod.registry.get("CAok") is not None)
        ws.send_text(STOP)
    assert _wait(lambda: app_mod.registry.get("CAok") is None)


def test_ws_twilio_start_with_bad_sig_param_is_closed(client, monkeypatch):
    monkeypatch.setattr(settings, "twilio_auth_token", TW_TOKEN)
    handshake = {"X-Twilio-Signature": compute_twilio_signature(TW_TOKEN, WSS, {})}
    for bad in (None, "", stream_param_sig(TW_TOKEN, "CAother"), stream_param_sig("wrong", "CAx")):
        with client.websocket_connect("/ws/twilio", headers=handshake) as ws:
            ws.send_text(_start("CAx", "+15550001113", bad))
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()
        assert app_mod.registry.get("CAx") is None
    assert FakeSTT.instances == []


def test_twiml_bridge_carries_callsid_hmac(client, monkeypatch):
    monkeypatch.setattr(settings, "twilio_auth_token", TW_TOKEN)
    monkeypatch.setattr(settings, "trusted_numbers", ["+15550001114"])
    body = _post(client, "/twiml", {"From": "+15550001114", "CallSid": "CAsig"}).text
    assert f'<Parameter name="sig" value="{stream_param_sig(TW_TOKEN, "CAsig")}" />' in body


def test_forged_stream_cannot_strike_victim(client, monkeypatch):
    """The original exploit: two forged streams put a real number on the block list."""
    monkeypatch.setattr(settings, "twilio_auth_token", TW_TOKEN)
    victim = "+15550001115"
    for i in range(2):
        try:
            with client.websocket_connect("/ws/twilio") as ws:
                ws.send_text(_start(f"CAforged{i}", victim))
                ws.send_text(_media(SCAM))
                ws.send_text(STOP)
        except WebSocketDisconnect:
            pass
    time.sleep(0.2)
    assert _route(client, victim, "CAreal") == "screen"


def test_dev_mode_stream_warning_logged_once(client, monkeypatch, caplog):
    monkeypatch.setattr(app_mod, "_dev_stream_warned", False)
    caplog.set_level(logging.WARNING, logger="lily.server")
    for i in range(2):
        with client.websocket_connect("/ws/twilio") as ws:
            ws.send_text(_start(f"CAdev{i}", "+15550001116"))
            ws.send_text(STOP)
    hits = [r for r in caplog.records if "media streams NOT authenticated" in r.getMessage()]
    assert len(hits) == 1


# ----- Fix 2: duplicate start ----------------------------------------------

def test_duplicate_start_on_socket_is_ignored(client):
    with client.websocket_connect("/ws/twilio") as ws:
        ws.send_text(_start("CA_A", "+15550002222"))
        ws.send_text(_start("CA_B", "+15550003333"))
        time.sleep(0.2)
        ws.send_text(STOP)
    assert _wait(lambda: len(app_mod.registry) == 0)
    assert len(FakeSTT.instances) == 1
    assert FakeSTT.instances[0].stopped


def test_second_socket_for_active_call_is_refused(client):
    with client.websocket_connect("/ws/twilio") as first:
        first.send_text(_start("CA_DUP", "+15550002223"))
        assert _wait(lambda: app_mod.registry.get("CA_DUP") is not None)
        original = app_mod.registry.get("CA_DUP")
        with client.websocket_connect("/ws/twilio") as second:
            second.send_text(_start("CA_DUP", "+15550002223"))
            with pytest.raises(WebSocketDisconnect):
                second.receive_text()
        # The intruder neither replaced nor finalized the live session.
        assert app_mod.registry.get("CA_DUP") is original
        first.send_text(STOP)
    assert _wait(lambda: app_mod.registry.get("CA_DUP") is None)


# ----- Fix 3: STT init failure ------------------------------------------------

def test_stt_init_failure_does_not_kill_finalize(client, monkeypatch):
    class Boom:
        def __init__(self):
            raise RuntimeError("DefaultCredentialsError: no ADC")

    monkeypatch.setattr(app_mod, "GoogleStreamingSession", tr.GoogleStreamingSession)
    monkeypatch.setattr(tr, "SpeechAsyncClient", Boom)
    ended = []

    async def spy(call_sid, *_a, **_k):
        ended.append(call_sid)

    monkeypatch.setattr(app_mod.store, "record_call_end", spy)
    with client.websocket_connect("/ws/twilio") as ws:
        ws.send_text(_start("CA_C", "+15550004444"))
        time.sleep(0.2)
        ws.send_text(STOP)
    assert _wait(lambda: ended == ["CA_C"])
    assert app_mod.registry.get("CA_C") is None


# ----- Fix 4: flagged caller redial bypass -------------------------------------

def test_flagged_caller_cannot_bypass_screen_by_redialing(client):
    num = "+15550005555"
    assert _route(client, num, "C1") == "screen"
    r = _post(client, "/screen/result", {"From": num, "CallSid": "C1", "SpeechResult": SCAM})
    assert "Hangup" in r.text
    assert _wait(lambda: _ctx(client, num).get("scam_strikes") == 1)
    assert _route(client, num, "C2") == "screen"  # hangs up at the Gather: no POST
    time.sleep(0.2)
    assert _route(client, num, "C3") == "screen"
    assert _ctx(client, num).get("call_count", 0) == 0


def test_call_count_increments_only_on_connected_calls(client):
    num = "+15550005556"
    assert _route(client, num, "D1") == "screen"
    time.sleep(0.2)
    assert _ctx(client, num)["known"] is False
    _post(client, "/screen/result", {"From": num, "CallSid": "D1", "SpeechResult": "Hi, it's Susan from the pharmacy."})
    assert _wait(lambda: _ctx(client, num).get("call_count") == 1)


# ----- Fix 5: anonymous strike bucket --------------------------------------------

@pytest.mark.parametrize("anon", ["Anonymous", ""])
def test_anonymous_decline_adds_no_strike(client, anon):
    for sid in ("A1", "A2", "A3"):
        assert _route(client, anon, sid) == "screen"
        _post(client, "/screen/result", {"From": anon, "CallSid": sid, "SpeechResult": SCAM})
    time.sleep(0.3)
    assert _ctx(client, anon).get("scam_strikes", 0) == 0
    assert _route(client, anon, "A4-hospital") == "screen"


# ----- Fix 7: non-ASCII token ----------------------------------------------------

def test_non_ascii_token_is_403_not_500(client, monkeypatch):
    monkeypatch.setattr(settings, "client_token", "secret")
    r = client.get("/api/activity", headers={"X-Lily-Token": "é".encode("latin-1")})
    assert r.status_code == 403


def test_non_ascii_twilio_signature_is_403_not_500(client, monkeypatch):
    monkeypatch.setattr(settings, "twilio_auth_token", TW_TOKEN)
    r = client.post(
        "/twiml", data={"From": "+15550007777"}, headers={"X-Twilio-Signature": "é".encode("latin-1")}
    )
    assert r.status_code == 403


# ----- Fix 8: open-by-default exposure ---------------------------------------------

@pytest.mark.parametrize(
    ("public_url", "ngrok", "client_token", "tw_token", "missing"),
    [
        (PUBLIC, False, "", "t", "CLIENT_TOKEN"),
        (PUBLIC, False, "c", "", "TWILIO_AUTH_TOKEN"),
        ("", True, "", "", "CLIENT_TOKEN and TWILIO_AUTH_TOKEN"),
    ],
)
def test_public_without_secrets_refuses_to_start(monkeypatch, public_url, ngrok, client_token, tw_token, missing):
    monkeypatch.setattr(settings, "allow_insecure_dev", False)
    monkeypatch.setattr(settings, "public_url", public_url)
    monkeypatch.setattr(settings, "ngrok_enabled", ngrok)
    monkeypatch.setattr(settings, "client_token", client_token)
    monkeypatch.setattr(settings, "twilio_auth_token", tw_token)
    with pytest.raises(RuntimeError, match=missing):
        with TestClient(app_mod.app):
            pass


def test_exposure_check_allows_secrets_flag_or_local(monkeypatch):
    monkeypatch.setattr(settings, "allow_insecure_dev", False)
    monkeypatch.setattr(settings, "client_token", "c")
    monkeypatch.setattr(settings, "twilio_auth_token", "t")
    app_mod.check_exposure()  # public with both secrets
    monkeypatch.setattr(settings, "client_token", "")
    monkeypatch.setattr(settings, "allow_insecure_dev", True)
    app_mod.check_exposure()  # explicit dev override
    monkeypatch.setattr(settings, "allow_insecure_dev", False)
    monkeypatch.setattr(settings, "public_url", "")
    monkeypatch.setattr(settings, "ngrok_enabled", False)
    app_mod.check_exposure()  # not publicly reachable


def test_allow_insecure_dev_read_from_env(monkeypatch):
    from lily.config import Settings

    monkeypatch.setenv("ALLOW_INSECURE_DEV", "true")
    assert Settings.from_env().allow_insecure_dev is True
    monkeypatch.delenv("ALLOW_INSECURE_DEV")
    assert Settings.from_env().allow_insecure_dev is False


# ----- Fix 9: token only via header on REST; masked in access log --------------------

def test_rest_rejects_query_token_and_accepts_header(client, monkeypatch):
    monkeypatch.setattr(settings, "client_token", "s3cret")
    assert client.get("/api/calls?token=s3cret").status_code == 403
    assert client.get("/api/calls", headers={"X-Lily-Token": "s3cret"}).status_code == 200


def test_ws_client_still_accepts_query_token(client, monkeypatch):
    monkeypatch.setattr(settings, "client_token", "s3cret")
    with client.websocket_connect("/ws/client?token=s3cret") as ws:
        assert json.loads(ws.receive_text())["event"] == "snapshot"


def test_access_log_masks_query_token():
    access = logging.getLogger("uvicorn.access")
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", "/ws/client?a=1&token=s3cret&b=2", "1.1", 101),
        None,
    )
    for f in access.filters:
        f.filter(record)
    line = record.getMessage()
    assert "s3cret" not in line
    assert "token=***" in line and "b=2" in line


# ----- Fix 10: strike policy ---------------------------------------------------------

def _run_escalate(monkeypatch, caller, verdicts):
    """Escalate once per verdict; return (strikes, alerts, interventions)."""
    strikes, alerts, interventions = [], [], []

    async def strike(number):
        strikes.append(number)

    async def activity(kind, *_a, **_k):
        alerts.append(kind)

    async def intervene(call_sid):
        interventions.append(call_sid)

    monkeypatch.setattr(settings, "auto_intervene", True)
    monkeypatch.setattr(app_mod.store, "add_scam_strike", strike)
    monkeypatch.setattr(app_mod, "push_activity", activity)
    monkeypatch.setattr(app_mod, "_intervene", intervene)

    async def main():
        session = app_mod.CallSession(call_sid="CA_S", caller=caller)
        for verdict in verdicts:
            session.fusion.on_llm(verdict)
            await app_mod._escalate_high_risk(session, "reason")
        loop = asyncio.get_running_loop()
        await asyncio.gather(*(t for t in list(app_mod._background) if t.get_loop() is loop))

    asyncio.run(main())
    return strikes, alerts, interventions


def test_heuristic_high_alerts_without_strike_or_intervention(monkeypatch):
    strikes, alerts, interventions = _run_escalate(monkeypatch, "+15550008888", ["safe", "safe"])
    assert alerts == ["scam_alert"]
    assert strikes == []
    assert interventions == []


def test_llm_scam_verdict_strikes_once_after_heuristic_alert(monkeypatch):
    strikes, alerts, interventions = _run_escalate(
        monkeypatch, "+15550008889", ["safe", "scam", "scam"]
    )
    assert alerts == ["scam_alert"]
    assert strikes == ["+15550008889"]
    assert interventions == ["CA_S"]


def test_llm_scam_verdict_on_anonymous_caller_adds_no_strike(monkeypatch):
    strikes, alerts, _ = _run_escalate(monkeypatch, "Anonymous", ["scam"])
    assert alerts == ["scam_alert"]
    assert strikes == []
