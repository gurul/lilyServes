# lilyServes

**Live scam protection and call memory for phone calls.** The real-time backend behind [heyLily](https://github.com/gurul/lilyWebsite) — a gentle phone companion for older adults and people living with memory loss. Lily remembers your calls and screens out the scams.

A scam call only works while it is happening. By the time a transcript is reviewed, or a family member hears about the gift cards, the money is gone. lilyServes sits between the phone and the outside world: unknown callers are screened before the phone ever rings, every bridged call is transcribed and risk-scored sentence by sentence, a synthetic-voice check runs on the first seconds of audio, and the family dashboard hears about it live — during the conversation, not after it.

## Features

- **Call screening.** Trusted callers ring straight through. Restricted, anonymous, and first-time callers are answered by Lily first ("May I ask who's calling?"); the answer is risk-scored and the call is bridged or politely declined. Repeat scammers are blocked outright. Every screen shows up on the dashboard as a *Stayed safe* event.
- **Live scam detection, two tiers.** A sub-millisecond heuristic engine (gift cards, wire transfers, urgency, secrecy, grandparent-scam patterns, remote-access requests, verification-code requests, …) scores every sentence instantly; an LLM refines the assessment asynchronously over a rolling window. The displayed risk level is monotonic — it can only escalate during a call.
- **Deepfake detection.** The first ~10 seconds of caller audio go to Hive AI. A synthetic voice floors the risk at Medium, raises an urgent alert, and can never *lower* a score.
- **Cognitive continuity — lilyMemory.** A purpose-built long-term memory engine. Every call, commitment, trusted contact, and scam encounter becomes a typed memory (`episode` / `commitment` / `person` / `preference` / `win` / `safety`) with importance, confidence, entities, and topics. Recall is hybrid — semantic (embeddings + cosine) fused with lexical (FTS5 BM25), importance, and recency — so when a call starts the dashboard receives genuinely related context about the caller, bridging the gaps for someone living with memory changes. Alongside it, a structured SQLite store tracks caller history, trust, and scam strikes.
- **Event capture (Active Assistance).** "Main St. Pharmacy confirmed pickup for Friday @ 4:00 PM" becomes a structured reminder, extracted mid-call and pushed to the dashboard as an *Event captured* item. Post-call summaries also sweep for missed commitments.
- **Family dashboard, smart updates.** Any number of clients connect over WebSocket and get a full snapshot plus live events. Every item carries an importance level (`info` / `notable` / `important` / `urgent`); only `important+` is flagged `notify: true` — peace of mind, not surveillance.
- **Selective privacy.** `SHARE_TRANSCRIPTS=false` keeps transcript text off the dashboard (risk levels still flow). `RETAIN_TRANSCRIPTS=false` (default) means transcripts are never persisted. Everything that *is* stored — including every memory — passes through redaction that scrubs card numbers, SSNs, and one-time codes, and any memory can be deleted via the API (`DELETE /api/memories/{id}`). You own your data, always. `MEMORY_EMBEDDINGS=false` keeps memory fully offline (lexical recall only).
- **Optional intervention.** With `AUTO_INTERVENE=true`, a call that reaches High risk is redirected to a polite hangup via the Twilio REST API. Off by default — warn, don't act.

## Latency model

Nothing on the audio/transcript hot path awaits an AI provider:

1. Twilio media frames are fed to Google STT through a bounded queue (drop-oldest under stall, so live latency beats completeness).
2. Each final sentence is heuristic-scored inline (<1 ms) and pushed to the dashboard immediately, marked `provisional`.
3. LLM refinement runs as a background task, **coalesced** — one analysis in flight per call; lines arriving mid-flight trigger exactly one re-run. Cost and tail latency stay flat on long calls (rolling 4k-char window, `max_tokens` capped, JSON mode).
4. Deepfake checks, event extraction, and all SQLite writes run off-loop (background tasks / worker thread, WAL mode).
5. Dashboard broadcasts serialize the payload once and fan out concurrently; interim (non-final) transcript fragments are streamed too, so text appears as it's spoken.

## Architecture

```mermaid
flowchart LR
    A[Caller] --> B[Twilio Voice]
    B --> C[POST /twiml<br/>signature-validated]
    C -->|trusted / known| D[Bridge + fork audio]
    C -->|unknown / restricted| S[Lily screens:<br/>Gather speech]
    C -->|repeat scammer| X[Blocked]
    S -->|answer OK| D
    S -->|scam patterns / silence| X
    D --> F[WS /ws/twilio<br/>mulaw 8kHz]
    F --> G[Google STT V2 streaming]
    F --> H[First 10s] --> I[Hive AI deepfake]
    G -->|each sentence| J[Heuristics <1ms<br/>instant push]
    J -.->|background| K[LLM refine<br/>coalesced]
    G -->|commitment trigger| E[Event extraction]
    G -->|hangup| L[Structured summary]
    J & K & I & E & L --> M[ClientHub<br/>WS fan-out + importance]
    E & L --> N[(SQLite memory<br/>callers · calls · events · activity)]
    N -->|caller context| M
```

## Quick start

Requires Python 3.9+ (3.13 supported — mulaw decode has a pure-Python fallback), a Twilio number, a Google Cloud project with Speech-to-Text enabled, an OpenAI key, and [ngrok](https://ngrok.com) for local development.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

gcloud auth application-default login    # or export GOOGLE_APPLICATION_CREDENTIALS

cp .env.example .env                     # fill in the values
python server.py
```

On boot the server opens an ngrok tunnel (unless `PUBLIC_URL` is set), and — if Twilio credentials are present — points your Twilio number's voice webhook at `<public_url>/twiml`. Connect a dashboard to `wss://<public_url>/ws/client?token=<CLIENT_TOKEN>`, then call the number.

### Tests & lint

```bash
pip install -r requirements-dev.txt
pytest          # 37 tests: heuristics, screening, memory/redaction, audio, auth
ruff check .
```

## Configuration

All configuration is environment variables (see `.env.example` for the full annotated list). Credentials belong in `.env`, which is gitignored — never commit keys.

| Variable | Required | Purpose |
|---|---:|---|
| `OPENAI_API_KEY` | Yes | Scam scoring, event extraction, summaries |
| `PROJECT_ID` | Yes | Google Cloud project for Speech-to-Text V2 |
| `FORWARD_TO` | Yes | Number calls are bridged to |
| `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` | Recommended | Webhook auto-config **and webhook signature validation** — without the token, webhooks are unauthenticated (dev only) |
| `CLIENT_TOKEN` | Recommended | Shared secret for the dashboard WS + REST; unset = open (dev only) |
| `SCREEN_UNKNOWN_CALLERS` | No | Default `true` — Lily answers unknown callers first |
| `TRUSTED_NUMBERS` | No | Comma-separated allowlist that always rings through |
| `HIVE_API_SECRET` | No | Deepfake detection; skipped if unset |
| `AUTO_INTERVENE` | No | Default `false` — hang up High-risk calls automatically |
| `RETAIN_TRANSCRIPTS` / `SHARE_TRANSCRIPTS` | No | Privacy controls (persist / stream transcript text) |
| `PUBLIC_URL`, `NGROK_ENABLED`, `PORT`, `DATA_DIR`, `LOG_LEVEL` | No | Deployment knobs |
| `SCAM_MODEL`, `SUMMARY_MODEL`, `EVENT_MODEL` | No | Model overrides |

## Endpoints

| Route | Kind | Auth | Purpose |
|---|---|---|---|
| `POST /twiml` | HTTP | Twilio signature | Voice webhook: screening decision → bridge / screen / block |
| `POST /screen/result` | HTTP | Twilio signature | Screening answer assessment → bridge or decline |
| `GET /health` | HTTP | — | Liveness, active calls, connected clients |
| `/ws/twilio` | WebSocket | — | Twilio Media Streams ingress |
| `/ws/client` | WebSocket | `?token=` | Dashboard egress: snapshot on connect, then live events |
| `GET /api/activity` · `/api/events` · `/api/calls` · `/api/caller/{number}` | HTTP | token | Activity feed, open reminders, call history, caller context |
| `POST /api/events/{id}/complete` | HTTP | token | Mark a reminder done |
| `POST /api/contacts/trusted` | HTTP | token | Add/remove a trusted contact (also remembered as a `person` memory) |
| `GET /api/memories` · `GET /api/memories/search?q=` | HTTP | token | Browse / hybrid-search long-term memories |
| `DELETE /api/memories/{id}` | HTTP | token | Forget a memory permanently |

Events pushed on `/ws/client` (all carry `importance` and `notify`):

| `event` | When |
|---|---|
| `snapshot` | On connect: active calls, recent activity, open events |
| `call_started` | Stream starts — includes `caller_context` (name, history, open commitments) |
| `caller_memories` | Moments later: hybrid-recalled long-term memories about this caller |
| `transcript_interim` / `transcript_update` | As words are spoken / each final sentence (instant provisional risk) |
| `scam_update` | LLM-refined risk level |
| `deepfake_result` | Once, ~10 s in |
| `event_captured` | A commitment was extracted mid-call |
| `activity` | Feed items: *Stayed safe*, *Event captured*, screening, alerts |
| `call_summary` | On hangup: structured summary, indicators, recommended action |

## Project layout

| Path | What |
|---|---|
| `server.py` | FastAPI app: webhooks, both WebSockets, hot-path orchestration, dashboard REST |
| `screening.py` | Screening routes, answer assessment, all TwiML builders |
| `scam_heuristics.py` | Instant regex risk engine (16 weighted patterns) |
| `scam_detector.py` | LLM refinement over a rolling window, deepfake/heuristic floors |
| `events.py` | Commitment extraction with a cheap trigger gate |
| `memory_store.py` | SQLite (WAL) operational store: callers, calls, events, activity; redaction |
| `lilyMemory/` | Long-term memory engine: typed memories, embeddings, FTS5 + cosine hybrid recall |
| `google_transcriber.py` | STT V2 bidirectional stream, bounded queue, reconnect-before-timeout |
| `deepfake_client.py` | Hive AI call over a warm pooled connection |
| `summarizer.py` | Shared OpenAI client, structured post-call summary |
| `hub.py` | Dashboard fan-out + importance/notification tagging |
| `call_state.py` | Multi-call registry and per-call session state |
| `auth.py` | Twilio signature validation, dashboard token check |
| `audio.py` | mulaw → WAV, pure-Python fallback for 3.13+ |
| `config.py` | Env-driven settings |
| `tests/` | 37 unit tests (no network required) |

## Security & privacy posture

- Twilio webhooks are HMAC-validated (`X-Twilio-Signature`) whenever `TWILIO_AUTH_TOKEN` is set; unset is loudly logged as dev mode.
- Dashboard WS and REST require `CLIENT_TOKEN` (constant-time compare); unset is loudly logged as dev mode.
- Minimal retention by default: no transcript persistence, summaries only, all stored text redacted (cards / SSNs / one-time codes).
- No phone numbers or secrets in source; everything comes from the environment.

## Known limitations

- Call state is in-memory and single-process (SQLite memory survives restarts; in-flight calls do not).
- Only the caller's inbound track is transcribed; the recipient's replies are not analyzed.
- Screening uses heuristics only on the spoken answer — a calm, novel scam opening will get bridged (and then caught by live scoring).
- The dashboard is a protocol, not a UI — the heyLily frontend consumes it.

## Built with

FastAPI, Twilio Voice + Media Streams, Google Cloud Speech-to-Text V2, OpenAI, Hive AI, SQLite, and ngrok.
