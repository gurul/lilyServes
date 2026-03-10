import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

load_dotenv()

from deepfake_client import check_deepfake
from google_transcriber import GoogleStreamingSession
from scam_detector import analyze_scam
from summarizer import generate_summary

PORT = int(os.environ.get("PORT", 8000))
FORWARD_TO = os.environ.get("FORWARD_TO", "+12067418265")
DEEPFAKE_BUFFER_SIZE = 80_000  # ~10 seconds of mulaw audio at 8 kHz

# ----- In-memory call state ------------------------------------------------
# { call_sid: { stream_session, transcript, scam_score, client_ws } }
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

                project_id = os.environ.get("PROJECT_ID", "heylily")

                # Callback for final transcription results only
                async def on_final(text: str, full_transcript: str, _sid=call_sid):
                    state = calls.get(_sid)
                    if not state:
                        return
                    state["transcript"].append(text)
                    print(f"[transcript] {text}")

                    scam = await analyze_scam(full_transcript, state["deepfake_score"])
                    state["scam_level"] = scam["scam_level"]
                    print(f"[scam] level={scam['scam_level']} reasoning={scam['reasoning']}")

                    await _push_to_client(_sid, {
                        "event": "transcript_update",
                        "call_sid": _sid,
                        "text": text,
                        "full_transcript": full_transcript,
                        "scam": scam,
                    })

                session = GoogleStreamingSession(
                    project_id=project_id,
                    on_final_result=on_final,
                )

                calls[call_sid] = {
                    "stream_session": session,
                    "transcript": [],
                    "scam_level": "Low",
                    "client_ws": client_ws,
                    "deepfake_score": 0,
                    "deepfake_buffer": bytearray(),
                    "deepfake_submitted": False,
                }

                await session.start()

            elif event == "media":
                if call_sid and call_sid in calls:
                    payload = base64.b64decode(msg["media"]["payload"])
                    calls[call_sid]["stream_session"].feed_audio(payload)

                    # Buffer first 10s of audio for deepfake detection (one-shot)
                    state = calls[call_sid]
                    if not state["deepfake_submitted"]:
                        state["deepfake_buffer"] += payload
                        if len(state["deepfake_buffer"]) >= DEEPFAKE_BUFFER_SIZE:
                            state["deepfake_submitted"] = True
                            buf = bytes(state["deepfake_buffer"][:DEEPFAKE_BUFFER_SIZE])
                            state["deepfake_buffer"] = bytearray()  # free memory
                            asyncio.create_task(_run_deepfake_check(call_sid, buf))

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


async def _run_deepfake_check(call_sid: str, mulaw_bytes: bytes):
    """One-shot deepfake detection on the first 10 seconds of audio."""
    print(f"[deepfake] Checking {len(mulaw_bytes)} bytes for call_sid={call_sid}")
    result = await check_deepfake(mulaw_bytes)
    print(f"[deepfake] Result: score={result['score']} is_deepfake={result['is_deepfake']}")

    state = calls.get(call_sid)
    if not state:
        return  # call ended while we were waiting

    if result["score"] is not None:
        state["deepfake_score"] = result["score"]

    await _push_to_client(call_sid, {
        "event": "deepfake_result",
        "call_sid": call_sid,
        "deepfake_score": result["score"],
        "is_deepfake": result["is_deepfake"],
    })


async def _finalize_call(call_sid: str):
    """Stop the streaming session, generate summary, clean up."""
    state = calls.pop(call_sid, None)
    if not state:
        return

    # Stop Google streaming and get final transcript
    session = state.get("stream_session")
    if session:
        transcript_lines = await session.stop()
        state["transcript"] = transcript_lines

    # If call ended before 10s, run deepfake check on whatever audio we have
    if not state.get("deepfake_submitted") and len(state.get("deepfake_buffer", b"")) > 0:
        buf = bytes(state["deepfake_buffer"])
        print(f"[deepfake] Call ended early, checking {len(buf)} bytes")
        result = await check_deepfake(buf)
        print(f"[deepfake] Result: score={result['score']} is_deepfake={result['is_deepfake']}")
        if result["score"] is not None:
            state["deepfake_score"] = result["score"]
        await _push_to_client_direct(state, {
            "event": "deepfake_result",
            "call_sid": call_sid,
            "deepfake_score": result["score"],
            "is_deepfake": result["is_deepfake"],
        })

    # Generate GPT-4o summary
    print(f"[summarizer] Generating summary for call_sid={call_sid} ...")
    summary = await generate_summary(state["transcript"])
    print(f"[summarizer] {summary}")

    await _push_to_client_direct(state, {
        "event": "call_summary",
        "call_sid": call_sid,
        "summary": summary,
        "scam_level": state["scam_level"],
        "deepfake_score": state["deepfake_score"],
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
