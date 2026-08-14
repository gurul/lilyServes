"""lilyServes — real-time call companion backend.

Latency model: everything on the audio/transcript hot path is either
sub-millisecond (heuristic scoring, WS pushes) or scheduled as a background
task (LLM refinement, event extraction, deepfake check, DB writes via a
worker thread). Nothing on the hot path ever awaits a network round-trip to
an AI provider.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import PlainTextResponse

from lily import screening
from lily.auth import check_client_token, require_twilio_signature
from lily.calls import CallSession, registry
from lily.config import settings
from lily.detection.deepfake import check_deepfake, close_http
from lily.detection.detector import analyze_scam
from lily.detection.heuristics import max_level, score_text
from lily.events import extract_events, has_event_trigger
from lily.hub import IMPORTANT, INFO, NOTABLE, URGENT, hub, push_activity
from lily.memory import MemoryService, MemoryType
from lily.memory.operational import MemoryStore, redact
from lily.summarizer import generate_summary
from lily.transcription import GoogleStreamingSession

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("lily.server")

DEEPFAKE_BUFFER_SIZE = 80_000  # ~10 seconds of mulaw audio at 8 kHz

store = MemoryStore(settings.data_dir, settings.retain_transcripts)
memory = (
    MemoryService(settings.data_dir)
    if settings.memory_embeddings
    else MemoryService.lexical_only(settings.data_dir)
)
# Keep the operational event twins consistent with the memory lifecycle:
# a superseded/completed commitment closes its event; a merged duplicate
# repoints its event at the surviving memory.
memory.on_commitment_retired = store.complete_event_by_memory
memory.on_commitment_merged = store.remap_event_memory

# Keep strong references to fire-and-forget tasks (asyncio only holds weak ones).
_background: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.get_running_loop().create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


# ----- Lifespan --------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await store.open()
    await memory.open()

    public_url = settings.public_url
    tunnel = None
    if not public_url and settings.ngrok_enabled:
        from pyngrok import ngrok

        tunnel = ngrok.connect(settings.port, bind_tls=True)
        public_url = tunnel.public_url
    app.state.public_url = public_url or ""

    if not settings.forward_to:
        log.warning("FORWARD_TO unset — calls cannot be bridged until it is configured")
    if not settings.client_token:
        log.warning("CLIENT_TOKEN unset — dashboard access is unauthenticated (dev mode)")
    if public_url:
        log.info("public URL: %s (TwiML hook: %s/twiml)", public_url, public_url)
        _configure_twilio_webhook(public_url)
    else:
        log.warning("no PUBLIC_URL and ngrok disabled — set the Twilio webhook manually")

    yield

    # Finalize any in-flight calls so summaries aren't lost on shutdown.
    for session in registry.active():
        try:
            await _finalize_call(session.call_sid)
        except Exception:
            log.exception("finalize on shutdown failed for %s", session.call_sid)
    if _background:
        await asyncio.gather(*_background, return_exceptions=True)
    await close_http()
    await memory.close()
    await store.close()
    if tunnel is not None:
        from pyngrok import ngrok

        ngrok.disconnect(tunnel.public_url)


def _configure_twilio_webhook(public_url: str) -> None:
    if not (settings.twilio_account_sid and settings.twilio_auth_token):
        log.warning("Twilio credentials unset — set the voice webhook manually: %s/twiml", public_url)
        return
    try:
        from twilio.rest import Client as TwilioClient

        twilio = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
        numbers = twilio.incoming_phone_numbers.list(limit=1)
        if numbers:
            numbers[0].update(voice_url=public_url + "/twiml", voice_method="POST")
            log.info("Twilio voice webhook set on %s", numbers[0].phone_number)
    except Exception as e:
        log.warning("could not auto-set Twilio webhook (%s) — set manually: %s/twiml", e, public_url)


app = FastAPI(lifespan=lifespan)


# ----- Auth dependency for dashboard REST ------------------------------------

async def require_client_token(request: Request) -> None:
    token = request.headers.get("X-Lily-Token", "") or request.query_params.get("token", "")
    if not check_client_token(token):
        raise HTTPException(status_code=403, detail="invalid token")


# ----- Health ----------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_calls": len(registry),
        "clients": hub.client_count,
    }


# ----- Twilio voice webhook: screening decision ------------------------------

def _ws_url(request: Request) -> str:
    base = request.app.state.public_url or str(request.base_url).rstrip("/")
    return base.replace("https://", "wss://").replace("http://", "ws://")


@app.post("/twiml")
async def twiml(request: Request, form: dict = Depends(require_twilio_signature)):
    caller = form.get("From", "unknown")
    call_sid = form.get("CallSid", "")
    ctx = await store.caller_context(caller)
    route = screening.decide_route(
        caller, ctx, settings.trusted_numbers, settings.screen_unknown_callers
    )
    log.info("incoming call from %s -> %s", caller, route)
    _spawn(store.touch_caller(caller))

    if route == screening.ROUTE_BLOCK:
        _spawn(push_activity(
            "stayed_safe",
            "Blocked a known scam caller",
            f"Lily declined a call from {caller} (repeat offender).",
            IMPORTANT, call_sid, store,
        ))
        return PlainTextResponse(screening.twiml_reject(blocked=True), media_type="application/xml")

    if route == screening.ROUTE_SCREEN:
        base = request.app.state.public_url or str(request.base_url).rstrip("/")
        _spawn(push_activity(
            "screening",
            "Lily is screening a caller",
            f"Unrecognized caller {caller} is being asked to identify themselves.",
            INFO, call_sid, store,
        ))
        return PlainTextResponse(
            screening.twiml_screen(base + "/screen/result"), media_type="application/xml"
        )

    xml = screening.twiml_pass(_ws_url(request), settings.forward_to, caller, screened=False)
    return PlainTextResponse(xml, media_type="application/xml")


@app.post("/screen/result")
async def screen_result(request: Request, form: dict = Depends(require_twilio_signature)):
    caller = form.get("From", "unknown")
    call_sid = form.get("CallSid", "")
    speech = form.get("SpeechResult", "")
    allow, reason = screening.assess_screen_answer(speech)
    log.info("screen result for %s: allow=%s (%s)", caller, allow, reason)

    if not allow:
        _spawn(store.add_scam_strike(caller))
        _spawn(push_activity(
            "stayed_safe",
            "Stayed safe",
            f"Lily screened a caller ({caller}). {reason}. Call handled.",
            IMPORTANT, call_sid, store,
        ))
        return PlainTextResponse(screening.twiml_reject(), media_type="application/xml")

    _spawn(push_activity(
        "screen_passed",
        "Screened caller connected",
        f"Caller said: “{redact(speech)}” — call connected and being monitored.",
        INFO, call_sid, store,
    ))
    xml = screening.twiml_pass(_ws_url(request), settings.forward_to, caller, screened=True)
    return PlainTextResponse(xml, media_type="application/xml")


# ----- Twilio media stream ---------------------------------------------------

@app.websocket("/ws/twilio")
async def twilio_ws(websocket: WebSocket):
    await websocket.accept()
    call_sid = None
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            event = msg.get("event")

            if event == "start":
                start = msg["start"]
                call_sid = start["callSid"]
                params = start.get("customParameters") or {}
                caller = params.get("caller") or "unknown"
                screened = params.get("screened") == "1"
                await _begin_call(call_sid, caller, screened)

            elif event == "media":
                session = registry.get(call_sid) if call_sid else None
                if session is not None:
                    payload = base64.b64decode(msg["media"]["payload"])
                    session.stream_session.feed_audio(payload)
                    if not session.deepfake_submitted:
                        session.deepfake_buffer += payload
                        if len(session.deepfake_buffer) >= DEEPFAKE_BUFFER_SIZE:
                            session.deepfake_submitted = True
                            buf = bytes(session.deepfake_buffer[:DEEPFAKE_BUFFER_SIZE])
                            session.deepfake_buffer = bytearray()
                            _spawn(_run_deepfake_check(call_sid, buf))

            elif event == "stop":
                log.info("stream stopped call_sid=%s", call_sid)
                break

    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("twilio_ws error call_sid=%s", call_sid)
    finally:
        if call_sid and registry.get(call_sid):
            await _finalize_call(call_sid)


async def _begin_call(call_sid: str, caller: str, screened: bool) -> None:
    log.info("stream started call_sid=%s caller=%s screened=%s", call_sid, caller, screened)
    session = registry.create(call_sid, caller)
    session.screened = screened

    async def on_final(text: str, full_transcript: str, _sid=call_sid):
        await _on_final_line(_sid, text, full_transcript)

    async def on_interim(text: str, _sid=call_sid):
        if settings.share_transcripts:
            await hub.broadcast({
                "event": "transcript_interim",
                "call_sid": _sid,
                "text": text,
            })

    session.stream_session = GoogleStreamingSession(
        project_id=settings.project_id,
        location=settings.stt_location,
        model=settings.stt_model,
        language_code=settings.stt_language,
        on_final_result=on_final,
        on_interim_result=on_interim,
    )
    await session.stream_session.start()

    _spawn(store.record_call_start(call_sid, caller, screened))

    # Cognitive continuity: give the dashboard gentle context about who this is.
    ctx = await store.caller_context(caller)
    # Profile card is pure indexed SQL — instant, no embedding round-trip.
    # Withheld caller IDs share one bucket; a card for it would cross-pollinate
    # unrelated callers.
    card = None
    if caller.lower() not in ("unknown", "anonymous"):
        try:
            card = await memory.profile(caller, ctx.get("name", ""))
        except Exception:
            log.exception("memory profile failed for %s", caller)
    await hub.broadcast({
        "event": "call_started",
        "call_sid": call_sid,
        "caller": caller,
        "screened": screened,
        "caller_context": ctx,
        "memory_card": card,
    }, NOTABLE)
    # Long-term memories arrive as a follow-up so the embedding round-trip
    # never delays the call_started push.
    _spawn(_recall_caller_memories(call_sid, caller, ctx.get("name", "")))


async def _recall_caller_memories(call_sid: str, caller: str, name: str) -> None:
    try:
        memories = await memory.caller_context(caller, name)
    except Exception:
        log.exception("memory recall failed for %s", caller)
        return
    if memories:
        await hub.broadcast({
            "event": "caller_memories",
            "call_sid": call_sid,
            "caller": caller,
            "memories": memories,
        }, NOTABLE)


async def _on_final_line(call_sid: str, text: str, full_transcript: str) -> None:
    """Hot path: heuristics + EWMA fusion + immediate push, then async refinement."""
    session = registry.get(call_sid)
    if session is None:
        return
    session.transcript.append(text)

    # Cumulative heuristics feed the EWMA fusion; the regex tier escalates
    # but never clears (displayed level stays monotonic per call).
    heur = score_text(full_transcript)
    state = session.fusion.on_sentence(heur)
    session.scam_level = max_level(session.scam_level, heur.level, state.level)
    session.heuristic_signals = heur.signals

    payload = {
        "event": "transcript_update",
        "call_sid": call_sid,
        "scam": {
            "scam_level": session.scam_level,
            "reasoning": session.scam_reasoning or ("Signals: " + ", ".join(heur.signals) if heur.signals else ""),
            "signals": heur.signals,
            "risk_score": state.score,
            "stage": state.stage,
            "provisional": True,
        },
    }
    if settings.share_transcripts:
        payload["text"] = text
        payload["full_transcript"] = full_transcript
    await hub.broadcast(payload, INFO)

    # Payment-stage markers and theta-2 fire immediately — scam-progression
    # research shows 1-2 turns of lead time once payment language appears.
    if state.hard_alert or heur.level == "High":
        await _escalate_high_risk(session, "Signals: " + ", ".join(heur.signals))

    _spawn(_refine_scam(call_sid))
    if has_event_trigger(text):
        _spawn(_capture_events(call_sid, text))


async def _refine_scam(call_sid: str) -> None:
    """LLM refinement, coalesced: one in flight per call, re-run once if new
    lines arrived mid-analysis. Keeps LLM latency entirely off the hot path
    and bounds cost on chatty calls."""
    session = registry.get(call_sid)
    if session is None:
        return
    if session.analysis_in_flight:
        session.analysis_dirty = True
        return
    session.analysis_in_flight = True
    try:
        while True:
            session.analysis_dirty = False
            result = await analyze_scam(session.full_transcript)
            if registry.get(call_sid) is None:
                return
            previous = session.scam_level
            state = session.fusion.on_llm(result["verdict"])
            session.scam_level = max_level(session.scam_level, state.level)
            session.scam_reasoning = result["reasoning"]
            await hub.broadcast({
                "event": "scam_update",
                "call_sid": call_sid,
                "scam": {
                    "scam_level": session.scam_level,
                    "verdict": result["verdict"],
                    "criteria": result.get("criteria", []),
                    "reasoning": session.scam_reasoning,
                    "signals": result.get("signals", []),
                    "risk_score": state.score,
                    "stage": state.stage,
                    "provisional": False,
                },
            }, IMPORTANT if session.scam_level != "Low" and session.scam_level != previous else INFO)
            if state.hard_alert:
                await _escalate_high_risk(session, result["reasoning"])
            if not session.analysis_dirty:
                return
    finally:
        session.analysis_in_flight = False


async def _escalate_high_risk(session: CallSession, reason: str) -> None:
    """Urgent family notification (once per call) and optional intervention."""
    if session.intervened:
        return
    session.intervened = True
    _spawn(store.add_scam_strike(session.caller))
    await push_activity(
        "scam_alert",
        "High scam risk on an active call",
        f"Caller {session.caller}: {reason}",
        URGENT, session.call_sid, store,
    )
    if settings.auto_intervene:
        _spawn(_intervene(session.call_sid))


async def _intervene(call_sid: str) -> None:
    """Redirect the live call to a polite hangup (AUTO_INTERVENE=true only)."""
    if not (settings.twilio_account_sid and settings.twilio_auth_token):
        log.warning("AUTO_INTERVENE set but Twilio credentials missing")
        return
    try:
        from twilio.rest import Client as TwilioClient

        def _do():
            twilio = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
            twilio.calls(call_sid).update(twiml=screening.twiml_hangup_polite())

        await asyncio.to_thread(_do)
        log.info("intervened on call %s", call_sid)
        await push_activity(
            "intervened", "Lily ended a dangerous call",
            "The call was ended automatically for safety.", URGENT, call_sid, store,
        )
    except Exception:
        log.exception("intervention failed for %s", call_sid)


async def _capture_commitment(call_sid: str, caller: str, title: str, when: str) -> int:
    """One write path for commitments: the COMMITMENT memory and its
    operational event twin are linked, so completing one closes both."""
    mem = await memory.remember_commitment(call_sid, caller, title, when)
    return await store.add_event(call_sid, caller, title, when, memory_id=mem.id)


async def _capture_events(call_sid: str, sentence: str) -> None:
    """Active Assistance: turn a spoken commitment into a stored reminder."""
    session = registry.get(call_sid)
    caller = session.caller if session else "unknown"
    for event in await extract_events(sentence):
        event_id = await _capture_commitment(call_sid, caller, event["title"], event["when"])
        title = event["title"] + (f" — {event['when']}" if event["when"] else "")
        await push_activity(
            "event_captured", "Event captured", title, NOTABLE, call_sid, store,
        )
        await hub.broadcast({
            "event": "event_captured",
            "call_sid": call_sid,
            "id": event_id,
            "title": event["title"],
            "when": event["when"],
        }, NOTABLE)


async def _run_deepfake_check(call_sid: str, mulaw_bytes: bytes) -> None:
    result = await check_deepfake(mulaw_bytes)
    session = registry.get(call_sid)
    if session is None:
        return
    if result["score"] is not None:
        session.deepfake_score = result["score"]
        session.fusion.on_deepfake(result["score"])
    await hub.broadcast({
        "event": "deepfake_result",
        "call_sid": call_sid,
        "deepfake_score": result["score"],
        "is_deepfake": result["is_deepfake"],
    }, IMPORTANT if result["is_deepfake"] else INFO)
    if result["is_deepfake"]:
        # Narrowband telephony blunts deepfake detectors, so a synthetic-voice
        # score multiplies transcript risk in fusion rather than alarming
        # alone; it only reads URGENT when the transcript is also risky.
        elevated = session.scam_level != "Low"
        await push_activity(
            "deepfake_alert", "Synthetic voice suspected",
            "The voice on this call may be AI-generated."
            + (" Combined with the conversation, this call looks dangerous." if elevated else ""),
            URGENT if elevated else IMPORTANT, call_sid, store,
        )
        _spawn(_refine_scam(call_sid))


_FACT_KINDS = {
    "person": (MemoryType.PERSON, 0.7),
    "preference": (MemoryType.PREFERENCE, 0.6),
    "win": (MemoryType.WIN, 0.6),
}


def _capture_facts(call_sid: str, caller: str, facts: list) -> None:
    """Durable facts piggybacked on the summary call — zero marginal LLM cost."""
    if not isinstance(facts, list):
        return
    if caller.lower() in ("unknown", "anonymous"):
        # Withheld caller IDs share one bucket — never attribute facts to it.
        caller = ""
    seen: set[str] = set()
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        kind = str(fact.get("kind", "")).lower()
        content = str(fact.get("content", "")).strip()
        if kind not in _FACT_KINDS or len(content) < 15 or content.lower() in seen:
            continue
        seen.add(content.lower())
        memory_type, importance = _FACT_KINDS[kind]
        raw_entities = fact.get("entities")
        entities = (
            [str(e) for e in raw_entities if e] if isinstance(raw_entities, list) else []
        )
        _spawn(memory.remember(
            content,
            memory_type=memory_type,
            entities=entities,
            topics=["extracted"],
            importance=importance,
            confidence=0.7,
            # Preferences describe the user, not the caller — global scope.
            caller="" if memory_type == MemoryType.PREFERENCE else caller,
            call_sid=call_sid,
        ))
        if kind == "win":
            _spawn(push_activity(
                "good_news", "A nice moment from today's call", content,
                NOTABLE, call_sid, store,
            ))


async def _finalize_call(call_sid: str) -> None:
    session = registry.pop(call_sid)
    if session is None:
        return

    if session.stream_session is not None:
        session.transcript = await session.stream_session.stop()

    # Short call: run deepfake on whatever audio we have.
    if not session.deepfake_submitted and session.deepfake_buffer:
        result = await check_deepfake(bytes(session.deepfake_buffer))
        if result["score"] is not None:
            session.deepfake_score = result["score"]

    summary = await generate_summary(session.transcript)
    final_level = max_level(session.scam_level, str(summary.get("risk_level", "low")).capitalize())

    _spawn(store.record_call_end(
        call_sid, session.caller, final_level,
        session.deepfake_score, summary, session.transcript,
    ))
    _spawn(memory.remember_call(
        call_sid, session.caller, str(summary.get("summary", "")), final_level,
    ))
    # Summary-extracted commitments join the same linked write path as
    # mid-call captures; skip ones already captured during the call.
    seen_titles = {
        e["title"].lower() for e in await store.events_for_call(call_sid)
    }
    for event in summary.get("events", []):
        if isinstance(event, dict) and event.get("title"):
            title = str(event["title"])
            # Stored titles are redacted — compare like with like.
            if redact(title).lower() in seen_titles:
                continue
            _spawn(_capture_commitment(
                call_sid, session.caller, title, str(event.get("when", ""))
            ))
    _capture_facts(call_sid, session.caller, summary.get("facts", []))

    await hub.broadcast({
        "event": "call_summary",
        "call_sid": call_sid,
        "caller": session.caller,
        "summary": summary,
        "scam_level": final_level,
        "deepfake_score": session.deepfake_score,
    }, IMPORTANT if final_level != "Low" else NOTABLE)
    await push_activity(
        "call_ended",
        "Call ended" + (f" — risk {final_level}" if final_level != "Low" else ""),
        redact(str(summary.get("summary", ""))),
        IMPORTANT if final_level != "Low" else INFO,
        call_sid, store,
    )
    log.info("finalized call %s (risk=%s)", call_sid, final_level)


# ----- Dashboard WebSocket ---------------------------------------------------

@app.websocket("/ws/client")
async def client_ws(websocket: WebSocket):
    token = websocket.query_params.get("token", "")
    if not check_client_token(token):
        await websocket.close(code=4403)
        return
    await websocket.accept()
    await hub.register(websocket)

    # Snapshot so a client joining mid-call isn't blind.
    try:
        snapshot = {
            "event": "snapshot",
            "active_calls": [
                {
                    "call_sid": s.call_sid,
                    "caller": s.caller,
                    "scam_level": s.scam_level,
                    "screened": s.screened,
                    "deepfake_score": s.deepfake_score,
                    **({"full_transcript": s.full_transcript} if settings.share_transcripts else {}),
                }
                for s in registry.active()
            ],
            "recent_activity": await store.recent_activity(20),
            "open_events": await store.open_events(20),
            "memory": {
                "count": await memory.count(),
                "recent": await memory.recent(10),
            },
        }
        await websocket.send_text(json.dumps(snapshot, separators=(",", ":")))
        while True:
            await websocket.receive_text()  # keepalive; client sends nothing meaningful
    except WebSocketDisconnect:
        pass
    finally:
        await hub.unregister(websocket)


# ----- Dashboard REST --------------------------------------------------------

@app.get("/api/activity", dependencies=[Depends(require_client_token)])
async def api_activity(limit: int = 50):
    return {"activity": await store.recent_activity(min(limit, 200))}


@app.get("/api/events", dependencies=[Depends(require_client_token)])
async def api_events():
    return {"events": await store.open_events()}


@app.post("/api/events/{event_id}/complete", dependencies=[Depends(require_client_token)])
async def api_complete_event(event_id: int):
    memory_id = await store.complete_event(event_id)
    if memory_id:
        # Done in one store means done in both: the COMMITMENT memory twin
        # leaves recall and the profile card too.
        _spawn(memory.complete_commitment(memory_id))
    return {"ok": True}


@app.get("/api/calls", dependencies=[Depends(require_client_token)])
async def api_calls(limit: int = 20):
    return {"calls": await store.recent_calls(min(limit, 100))}


@app.get("/api/caller/{number}", dependencies=[Depends(require_client_token)])
async def api_caller(number: str):
    ctx = await store.caller_context(number)
    return {**ctx, "memory_card": await memory.profile(number, ctx.get("name", ""))}


@app.post("/api/contacts/trusted", dependencies=[Depends(require_client_token)])
async def api_set_trusted(payload: dict):
    number = str(payload.get("number", "")).strip()
    if not number:
        raise HTTPException(status_code=422, detail="number required")
    trusted = bool(payload.get("trusted", True))
    name = str(payload.get("name", ""))
    note = str(payload.get("note", ""))
    await store.set_trusted(number, trusted, name)
    if trusted and name:
        # PERSON supersession makes re-trusting with a new note replace the
        # old fact instead of stacking a second one.
        _spawn(memory.remember_person(number, name, note))
    return {"ok": True}


# ----- Memory REST (lily.memory) ----------------------------------------------

@app.get("/api/memories", dependencies=[Depends(require_client_token)])
async def api_memories(
    limit: int = 50, memory_type: str = "", caller: str = "", include_inactive: bool = False
):
    """include_inactive exposes superseded/expired rows as an audit trail."""
    return {
        "memories": await memory.recent(min(limit, 200), memory_type, caller, include_inactive)
    }


@app.post("/api/memories", dependencies=[Depends(require_client_token)])
async def api_memories_add(payload: dict):
    """Family write path: 'Mom prefers the morning pharmacy run'."""
    content = str(payload.get("content", "")).strip()
    if not content or len(content) > 500:
        raise HTTPException(status_code=422, detail="content must be 1-500 characters")
    memory_type = str(payload.get("memory_type", MemoryType.PREFERENCE))
    if memory_type not in MemoryType.ALL:
        raise HTTPException(status_code=422, detail=f"memory_type must be one of {MemoryType.ALL}")
    raw_topics = payload.get("topics", [])
    if not isinstance(raw_topics, list):
        raise HTTPException(status_code=422, detail="topics must be a list")
    topics = [str(t) for t in raw_topics if t]
    m = await memory.remember(
        content,
        memory_type=memory_type,
        topics=topics + ["family_note"],
        importance=0.8,
        confidence=1.0,
        caller=str(payload.get("caller", "")),
    )
    await hub.broadcast({"event": "memory_added", "memory": m.to_dict()}, NOTABLE)
    return m.to_dict()


@app.get("/api/memories/search", dependencies=[Depends(require_client_token)])
async def api_memories_search(q: str, limit: int = 10, caller: str = ""):
    return {"memories": await memory.recall(q, limit=min(limit, 50), caller=caller)}


@app.delete("/api/memories/{memory_id}", dependencies=[Depends(require_client_token)])
async def api_memories_delete(memory_id: str):
    """You own your data — any memory can be deleted."""
    deleted = await memory.forget(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"ok": True}


# ----- Entry point -----------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("lily.app:app", host="0.0.0.0", port=settings.port, reload=False)
