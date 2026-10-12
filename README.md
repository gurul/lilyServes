# lilyServes

<p align="center">
  <a href="https://heylily.vercel.app"><img src="assets/heylily-hero.png" alt="heyLily — the quiet way to stay independent. A gentle phone companion for older adults and people living with memory loss, screening a restricted number with the option to let Lily answer first" width="100%"></a>
</p>

<p align="center">
  <b>▶ Try the demo — <a href="https://heylily.vercel.app">heylily.vercel.app</a></b>
  <br>
  <sub>Site source: <a href="https://github.com/gurul/lilyWebsite">gurul/lilyWebsite</a></sub>
</p>

**Live scam protection and call memory for phone calls.** The real-time backend behind [heyLily](https://heylily.vercel.app) — a gentle phone companion for older adults and people living with memory loss. Lily remembers your calls and screens out the scams.

A scam call only works while it is happening. By the time a transcript is reviewed, or a family member hears about the gift cards, the money is gone. lilyServes sits between the phone and the outside world: unknown callers are screened before the phone ever rings, every bridged call is transcribed and risk-scored sentence by sentence, a synthetic-voice check runs on the first seconds of audio, and the family dashboard hears about it live — during the conversation, not after it.

## Features

- **Call screening.** Trusted callers ring straight through. Restricted, anonymous, and first-time callers are answered by Lily first ("May I ask who's calling?"); the answer is risk-scored and the call is bridged or politely declined. A caller with two scam strikes is blocked outright; a strike comes only from a declined screen or an LLM *scam* verdict, and withheld caller IDs never accrue strikes (one anonymous scammer must not block every withheld caller). Every screen shows up on the dashboard as a *Stayed safe* event.
- **Live scam detection, research-grounded.** A sub-millisecond heuristic engine scores every sentence instantly. Its rules cover gift cards, wire transfers, crypto, urgency, secrecy, grandparent-scam patterns, remote-access and verification-code requests, and the elder-fraud scripts that the FTC and FBI IC3 describe: Social Security benefit threats, jury-duty and arrest-warrant threats, couriers that pick up cash or gold, "protect your money" transfers, payment apps to "yourself", unauthorized-charge alerts, utility shutoff threats, refund and tech-support remote access, and guaranteed investment returns. A criteria-rubric LLM tier gives a scam / uncertain / safe verdict asynchronously, and malformed model output reads as uncertain, never safe. Signals fuse through a fixed-window EWMA with two thresholds. A payment-stage word alone gives Medium and an immediate LLM check; it escalates to a hard alert only when another signal, the LLM, or the EWMA corroborates it. The displayed risk level only escalates during a call (citations below).
- **Deepfake detection.** The first ~10 seconds of caller audio go to Hive AI. A synthetic voice multiplies the transcript risk and alerts the family — urgently when the conversation is also suspicious — and can never *lower* a score.
- **Cognitive continuity — lilyMemory.** A purpose-built long-term memory engine. Every call, commitment, trusted contact, extracted fact, and scam encounter becomes a typed memory (`episode` / `commitment` / `person` / `preference` / `win` / `safety`) with importance, confidence, entities, and topics. Recall is hybrid — semantic (embeddings + cosine) fused with lexical (entity-weighted FTS5 BM25), gated on relevance, shaded by type-aware recency, and MMR-diversified so near-duplicates never crowd the results. Alongside it, a structured SQLite store tracks caller history, trust, and scam strikes.
- **A memory lifecycle that keeps memory honest.** Repeated observations *reinforce* the existing memory instead of duplicating it; a changed fact ("pickup moved to Saturday") *supersedes* the old one, which stays auditable but stops surfacing; commitments resolve their "Friday at 4 PM" into a real due date and *expire* after it passes (completing the reminder retires the memory too); low-importance chatter fades out while scam history never expires. All deterministic — no extra LLM calls — and `forget()` is always a hard delete.
- **Caller profile card.** When the phone rings the dashboard instantly receives a structured card — stable facts, open commitments (soonest due first), recent moments, and the caller's scam record — assembled in one SQL query, no model round-trip, plus a compact rendering ready for prompt injection. Post-call, durable facts (who people are, how the user likes things done, wins worth celebrating) are extracted by the *same* LLM call that writes the summary.
- **Event capture (Active Assistance).** "Main St. Pharmacy confirmed pickup for Friday @ 4:00 PM" becomes a structured reminder, extracted mid-call and pushed to the dashboard as an *Event captured* item. Post-call summaries also sweep for missed commitments.
- **Family dashboard, smart updates.** Any number of clients connect over WebSocket and get a full snapshot plus live events. Every item carries an importance level (`info` / `notable` / `important` / `urgent`); only `important+` is flagged `notify: true` — peace of mind, not surveillance.
- **Selective privacy.** `SHARE_TRANSCRIPTS=false` keeps transcript text off the dashboard (risk levels still flow). `RETAIN_TRANSCRIPTS=false` (default) means transcripts are never persisted. Everything that *is* stored — including every memory — passes through redaction that scrubs card numbers, SSNs, and one-time codes, and any memory can be deleted via the API (`DELETE /api/memories/{id}`). You own your data, always. `MEMORY_EMBEDDINGS=false` keeps memory fully offline (lexical recall only).
- **Optional intervention.** With `AUTO_INTERVENE=true`, a call the LLM judges a scam is redirected to a polite hangup via the Twilio REST API. Off by default — warn, don't act.

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

Requires Python 3.10+ (3.13 supported — mulaw decode has a pure-Python fallback), a Twilio number, a Google Cloud project with Speech-to-Text enabled, an OpenAI key, and [ngrok](https://ngrok.com) for local development.

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
pytest          # 280+ tests: heuristics, fusion, screening, memory, lifecycle, pipeline, audio, auth
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
| `ALLOW_INSECURE_DEV` | No | Default `false` — the server refuses to start on a public URL or ngrok tunnel while `CLIENT_TOKEN` or `TWILIO_AUTH_TOKEN` is unset; `true` overrides that for local testing |
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
| `/ws/twilio` | WebSocket | per-call `sig` | Twilio Media Streams ingress; `start` must carry the HMAC our TwiML issued for that CallSid |
| `/ws/client` | WebSocket | `?token=` | Dashboard egress: snapshot on connect, then live events |
| `GET /api/activity` · `/api/events` · `/api/calls` | HTTP | `X-Lily-Token` | Activity feed, open reminders, call history |
| `GET /api/caller/{number}` | HTTP | `X-Lily-Token` | Caller context + instant memory profile card |
| `POST /api/events/{id}/complete` | HTTP | `X-Lily-Token` | Mark a reminder done (also retires its commitment memory) |
| `POST /api/contacts/trusted` | HTTP | `X-Lily-Token` | Add/remove a trusted contact (name + optional note become a `person` memory; re-trusting replaces the old fact) |
| `GET /api/memories` · `GET /api/memories/search?q=` | HTTP | `X-Lily-Token` | Browse / hybrid-search memories; `include_inactive=true` shows the superseded/expired audit trail |
| `POST /api/memories` | HTTP | `X-Lily-Token` | Family write path — add a preference or note directly |
| `DELETE /api/memories/{id}` | HTTP | `X-Lily-Token` | Forget a memory permanently |

Events pushed on `/ws/client` (all carry `importance` and `notify`):

| `event` | When |
|---|---|
| `snapshot` | On connect: active calls, recent activity, open events, memory panel |
| `call_started` | Stream starts — includes `caller_context` and the instant `memory_card` (facts, open commitments, safety record) |
| `caller_memories` | Moments later: hybrid-recalled long-term memories about this caller |
| `memory_added` | A family member added a memory via `POST /api/memories` |
| `transcript_interim` / `transcript_update` | As words are spoken / each final sentence (instant provisional risk) |
| `scam_update` | LLM-refined risk level |
| `deepfake_result` | Once, ~10 s in |
| `event_captured` | A commitment was extracted mid-call |
| `activity` | Feed items: *Stayed safe*, *Event captured*, screening, alerts |
| `call_summary` | On hangup: structured summary, indicators, recommended action |

## Project layout

```
server.py                  entry point — python server.py
lily/
├── app.py                 FastAPI app: webhooks, WebSockets, hot-path orchestration, REST
├── config.py              env-driven settings
├── auth.py                Twilio signature validation, dashboard token check
├── calls.py               multi-call registry and per-call session state
├── hub.py                 dashboard fan-out + importance/notification tagging
├── screening.py           screening routes, answer assessment, TwiML builders
├── events.py              commitment extraction with a cheap trigger gate
├── summarizer.py          shared OpenAI client, structured post-call summary
├── transcription.py       STT V2 stream, bounded queue, reconnect-before-timeout
├── audio.py               mulaw → WAV, pure-Python fallback for 3.13+
├── detection/
│   ├── heuristics.py      instant regex risk engine (16 weighted patterns)
│   ├── detector.py        criteria-rubric LLM tier, 3-way verdict
│   ├── fusion.py          EWMA multi-signal risk fusion, kill-chain stages
│   └── deepfake.py        Hive AI check over a warm pooled connection
└── memory/
    ├── models.py          typed memories + lifecycle status (active/superseded/expired)
    ├── embeddings.py      OpenAI embeddings, injectable provider
    ├── store.py           SQLite + FTS5 + in-process vector index, scoped search, migrations
    ├── service.py         hybrid recall facade (calibrated fusion, MMR, profile card)
    ├── lifecycle.py       dedupe-reinforce, supersession, expiry, when-text resolver
    └── operational.py     callers/calls/events/activity store; PII redaction
tests/                     280+ tests (no network required)
```

## Research grounding

Each line states what the cited source reports, then what we built on it. Items marked *our design choice* go beyond the source.

- **Criteria-rubric prompting.** [arXiv:2506.06180](https://arxiv.org/abs/2506.06180) puts 11 voice-phishing evaluation criteria in the prompt and fine-tunes Llama3-8B. On its adversarial test set (non-scam transcripts that use scam vocabulary), GPT-4o with criteria scored 90.18–99.11% accuracy, the criteria-prompted fine-tuned Llama3-8B 87.50–94.64%, and a KoBERT classifier 56.25% (98.52% on the normal set). Our LLM prompt adapts those criteria.
- **Three-way verdict (scam / uncertain / safe)** is our design choice. [arXiv:2502.03964](https://arxiv.org/abs/2502.03964) studies LLM-based real-time phone-scam detection, refines its method for precision, and explores the trade-off between recall and timeliness.
- **The regex tier is easy to evade.** [arXiv:2507.16291](https://arxiv.org/abs/2507.16291) reports that LLM rewrites of real vishing scripts, which keep the meaning, cut ML classifier accuracy by up to 30.96%. So the LLM tier decides negatives.
- **EWMA risk accumulation.** [arXiv:2509.05362](https://arxiv.org/abs/2509.05362) defines e₁ = S(m₁), eₜ = φ·S(mₜ) + (1−φ)·eₜ₋₁ with φ = 2/(T+1), and alerts at one threshold θ₁ = 0.7 (its θ₂ governs the scam-baiting agent, not alerting). We fix T at a 5-sentence window (φ = 1/3) because a live call has no final T, and we use two thresholds, 0.35 (Medium) and 0.60 (hard alert): *our design choices*.
- **Payment-stage markers.** [arXiv:2605.12243](https://arxiv.org/abs/2605.12243) (PreScam) defines a scam kill chain and argues that a live system cannot always wait for the Termination phase, where the scammer asks for payment, before it flags a conversation. A lone payment word gives Medium plus an immediate LLM check, and a corroborated one gives a hard alert: *our design choice*.
- **Deepfake score as a risk multiplier, not a standalone alarm** is our design choice. [arXiv:2411.00121](https://arxiv.org/abs/2411.00121) finds that state-of-the-art audio-deepfake detectors often rely on high-frequency features that natural corruptions or an attacker can easily manipulate.
- **PII sanitization before any cloud LLM call.** [arXiv:2510.18493](https://arxiv.org/abs/2510.18493) (MASK) proposes sanitizing call transcripts because they often contain sensitive personal information that third-party providers may see during processing.
- **Heuristic rule families** cite their source in `lily/detection/heuristics.py`: FTC consumer alerts on Social Security, jury-duty, utility, payment-app, tech-support, investment, and "move your money to protect it" scams, the FBI IC3 courier PSA ([PSA240129](https://www.ic3.gov/PSA/2024/PSA240129)), and the [FBI IC3 2024 Internet Crime Report](https://www.ic3.gov/AnnualReport/Reports/2024_IC3Report.pdf), which lists Tech Support ($982,440,006) and Government Impersonation ($208,096,366) losses for complainants over 60. Rule weights are our design choice.

## Security & privacy posture

- Twilio webhooks are HMAC-validated (`X-Twilio-Signature`) whenever `TWILIO_AUTH_TOKEN` is set; unset is loudly logged as dev mode.
- Dashboard WS and REST require `CLIENT_TOKEN` (constant-time compare). REST takes it only in the `X-Lily-Token` header; `/ws/client` takes `?token=` because browsers cannot set WebSocket headers, and the access log masks it. A public URL with no token refuses to start unless `ALLOW_INSECURE_DEV=true`.
- Media streams must present an HMAC-SHA256 of the CallSid, keyed with `TWILIO_AUTH_TOKEN`, that only our TwiML issues, so a forged stream cannot open a session or strike a number. The `X-Twilio-Signature` on the WebSocket handshake is checked and logged but not enforced, because Twilio's docs do not pin the signed URL form.
- Minimal retention by default: no transcript persistence, summaries only, all stored text redacted (cards / SSNs / one-time codes).
- No phone numbers or secrets in source; everything comes from the environment.

## Known limitations

- Call state is in-memory and single-process (SQLite memory survives restarts; in-flight calls do not).
- Only the caller's inbound track is transcribed; the recipient's replies are not analyzed.
- Screening uses heuristics only on the spoken answer — a calm, novel scam opening will get bridged (and then caught by live scoring).
- The dashboard is a protocol, not a UI — the heyLily frontend consumes it.

## Built with

FastAPI, Twilio Voice + Media Streams, Google Cloud Speech-to-Text V2, OpenAI, Hive AI, SQLite, and ngrok.
