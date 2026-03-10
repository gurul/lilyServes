import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

load_dotenv()

from audio_utils import is_chunk_too_small, mulaw_to_wav
# from deepfake_client import send_to_deepfake  # re-enable when deepfake endpoint is ready
from scam_detector import analyze_scam
from summarizer import generate_summary
from transcriber import transcribe_chunk

PORT = int(os.environ.get("PORT", 8000))
FORWARD_TO = os.environ.get("FORWARD_TO", "+12067418265")
CHUNK_INTERVAL = 5  # seconds between Whisper API calls
# DEEPFAKE_EVERY_N_CHUNKS = 3  # re-enable when deepfake endpoint is ready

# ----- In-memory call state ------------------------------------------------
# { call_sid: { audio_buffer, transcript, scam_score, client_ws, chunk_task, ratecv_state, chunk_count } }
calls: dict[str, dict] = {}

# Clients waiting for a call to start (frontend connected before call arrives)
pending_clients: list[WebSocket] = []


# ----- Lifespan: start ngrok and auto-configure Twilio webhook -------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    from pyngrok import ngrok
    from twilio.rest import Client as TwilioClient

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")

    tunnel = ngrok.connect(PORT, bind_tls=True)
    public_url = tunnel.public_url
    app.state.public_url = public_url

    print(f"\n{'='*60}")
    print(f"  ngrok URL   : {public_url}")
    print(f"  TwiML hook  : {public_url}/twiml")
    print(f"  Client WS   : wss://{public_url.split('://', 1)[1]}/ws/client")
    print(f"  Forwarding  : {FORWARD_TO}")
    print(f"{'='*60}\n")

    if account_sid and auth_token:
        try:
            twilio = TwilioClient(account_sid, auth_token)
            numbers = twilio.incoming_phone_numbers.list(limit=1)
            if numbers:
                numbers[0].update(voice_url=public_url + "/twiml")
                print(f"  Twilio webhook set on: {numbers[0].phone_number}\n")
        except Exception as e:
            print(f"  [warn] Could not auto-set Twilio webhook: {e}")
            print(f"  Manually set: {public_url}/twiml\n")
    else:
        print("  [warn] TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN not set.")
        print(f"  Manually set Twilio webhook to: {public_url}/twiml\n")

    yield

    ngrok.disconnect(public_url)


app = FastAPI(lifespan=lifespan)


# ----- /health --------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "active_calls": len(calls)}


# ----- /twiml ---------------------------------------------------------------

@app.post("/twiml")
async def twiml(request: Request):
    """
    Return TwiML that:
    1. Asynchronously streams caller audio to our WebSocket
    2. Forwards (bridges) the call to FORWARD_TO
    """
    public_url = request.app.state.public_url
    ws_url = public_url.replace("https://", "wss://").replace("http://", "ws://")

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Stream url="{ws_url}/ws/twilio" track="inbound_track" />
  </Start>
  <Dial>{FORWARD_TO}</Dial>
</Response>"""

    caller = (await request.form()).get("From", "unknown")
    print(f"[twiml] Incoming call from {caller}")
    return PlainTextResponse(xml, media_type="application/xml")


# ----- /ws/twilio -----------------------------------------------------------

@app.websocket("/ws/twilio")
async def twilio_ws(websocket: WebSocket):
    """Receive Twilio Media Stream events and process audio."""
    await websocket.accept()
    call_sid = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "connected":
                pass  # no-op

            elif event == "start":
                call_sid = msg["start"]["callSid"]
                print(f"[twilio_ws] Stream started — call_sid={call_sid}")

                # Attach pending frontend client if available
                client_ws = pending_clients.pop(0) if pending_clients else None

                calls[call_sid] = {
                    "audio_buffer": bytearray(),
                    "transcript": [],
                    "scam_score": 0,
                    "client_ws": client_ws,
                    "ratecv_state": None,
                    "chunk_count": 0,
                    "chunk_task": None,
                }
                calls[call_sid]["chunk_task"] = asyncio.create_task(
                    _periodic_chunk_task(call_sid)
                )

            elif event == "media":
                if call_sid and call_sid in calls:
                    payload = base64.b64decode(msg["media"]["payload"])
                    calls[call_sid]["audio_buffer"].extend(payload)

            elif event == "stop":
                print(f"[twilio_ws] Stream stopped — call_sid={call_sid}")
                if call_sid and call_sid in calls:
                    await _finalize_call(call_sid)
                break

    except WebSocketDisconnect:
        if call_sid and call_sid in calls:
            await _finalize_call(call_sid)
    except Exception as e:
        print(f"[twilio_ws] Error: {e}")
        if call_sid and call_sid in calls:
            await _finalize_call(call_sid)


# ----- /ws/client -----------------------------------------------------------

@app.websocket("/ws/client")
async def client_ws(websocket: WebSocket):
    """
    The Hey Lily frontend connects here to receive live call events.
    If a call is already active, attaches immediately; otherwise waits.
    """
    await websocket.accept()
    print("[client_ws] Frontend connected")

    # Attach to an active call if one exists
    active_sid = next(iter(calls), None)
    if active_sid:
        calls[active_sid]["client_ws"] = websocket
    else:
        pending_clients.append(websocket)

    try:
        # Keep connection alive; frontend doesn't need to send messages
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("[client_ws] Frontend disconnected")
        if websocket in pending_clients:
            pending_clients.remove(websocket)
        for state in calls.values():
            if state["client_ws"] is websocket:
                state["client_ws"] = None


# ----- Core processing helpers ----------------------------------------------

async def _process_audio_chunk(call_sid: str, mulaw_chunk: bytes):
    """Convert, transcribe, analyze, and push a single audio chunk."""
    state = calls.get(call_sid)
    if not state:
        return

    if is_chunk_too_small(mulaw_chunk):
        return

    wav_bytes, new_state = mulaw_to_wav(mulaw_chunk, state["ratecv_state"])
    state["ratecv_state"] = new_state

    text = await transcribe_chunk(wav_bytes)
    if not text:
        return

    print(f"[transcript] {text}")
    state["transcript"].append(text)

    scam = analyze_scam(text)
    if scam["score"] > state["scam_score"]:
        state["scam_score"] = scam["score"]

    state["chunk_count"] += 1

    # # Fire-and-forget deepfake check every Nth chunk  (re-enable with deepfake endpoint)
    # if state["chunk_count"] % DEEPFAKE_EVERY_N_CHUNKS == 0:
    #     asyncio.create_task(_send_deepfake(call_sid, wav_bytes))

    await _push_to_client(call_sid, {
        "event": "transcript_update",
        "call_sid": call_sid,
        "text": text,
        "scam": scam,
    })


# async def _send_deepfake(call_sid: str, wav_bytes: bytes):  # re-enable with deepfake endpoint
#     result = await send_to_deepfake(wav_bytes)
#     prob = result.get("probability")
#     if prob is not None:
#         print(f"[deepfake] call_sid={call_sid} probability={prob}")
#         await _push_to_client(call_sid, {
#             "event": "deepfake_score",
#             "call_sid": call_sid,
#             "probability": prob,
#         })


async def _push_to_client(call_sid: str, payload: dict):
    state = calls.get(call_sid)
    if not state:
        return
    ws = state.get("client_ws")
    if ws:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            state["client_ws"] = None


async def _periodic_chunk_task(call_sid: str):
    """Every CHUNK_INTERVAL seconds, drain the buffer and process it."""
    try:
        while True:
            await asyncio.sleep(CHUNK_INTERVAL)
            state = calls.get(call_sid)
            if not state:
                break

            # Atomically swap the buffer
            chunk = bytes(state["audio_buffer"])
            state["audio_buffer"] = bytearray()

            if chunk:
                await _process_audio_chunk(call_sid, chunk)
    except asyncio.CancelledError:
        pass


async def _finalize_call(call_sid: str):
    """Process remaining audio, generate summary, clean up."""
    state = calls.pop(call_sid, None)
    if not state:
        return

    # Cancel the periodic task
    task = state.get("chunk_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Drain remaining audio
    remaining = bytes(state["audio_buffer"])
    if remaining and not is_chunk_too_small(remaining):
        wav_bytes, _ = mulaw_to_wav(remaining, state["ratecv_state"])
        text = await transcribe_chunk(wav_bytes)
        if text:
            print(f"[transcript/final] {text}")
            state["transcript"].append(text)
            scam = analyze_scam(text)
            await _push_to_client_direct(state, {
                "event": "transcript_update",
                "call_sid": call_sid,
                "text": text,
                "scam": scam,
            })

    # Generate GPT-4o summary
    print(f"[summarizer] Generating summary for call_sid={call_sid} ...")
    summary = await generate_summary(state["transcript"])
    print(f"[summarizer] {summary}")

    await _push_to_client_direct(state, {
        "event": "call_summary",
        "call_sid": call_sid,
        "summary": summary,
        "total_scam_score": state["scam_score"],
    })


async def _push_to_client_direct(state: dict, payload: dict):
    ws = state.get("client_ws")
    if ws:
        try:
            await ws.send_text(json.dumps(payload))
        except Exception:
            pass


# ----- Entry point ----------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT, reload=False)
