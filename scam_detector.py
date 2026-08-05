"""LLM tier of the two-tier scam scorer.

The regex tier (scam_heuristics) escalates instantly; this tier is the
authoritative judgment. Two literature-grounded choices (details/citations
in README):

- The prompt is an explicit criteria rubric, not open-ended "is this a
  scam" — criteria-prompted models hold ~95% accuracy under adversarial
  rephrasing while keyword classifiers collapse (arXiv:2506.06180), and
  zero-shot frontier models without criteria score far worse
  (arXiv:2503.24115).
- The verdict is three-way: "scam" / "uncertain" / "safe". UNCERTAIN keeps
  watching without alerting — the precision knob that protects an older
  adult's trust in the alerts that do fire (arXiv:2502.03964).

Transcripts are PII-sanitized (redact) before leaving the process
(arXiv:2510.18493).
"""
from __future__ import annotations

import json
import logging

from config import settings
from memory_store import redact
from scam_heuristics import score_text
from summarizer import get_client

log = logging.getLogger("lily.scam")

# Rolling window keeps prompt size (and tail latency) flat on long calls.
TRANSCRIPT_WINDOW_CHARS = 4000

VERDICTS = ("scam", "uncertain", "safe")

# Criteria rubric adapted from the voice-phishing criteria of
# arXiv:2506.06180, extended with grandparent-scam patterns.
SYSTEM_PROMPT = """You are a phone-scam detection system protecting an older adult. \
You see the caller's side of a live call transcript. Judge it against these criteria:

1. Unsolicited loan, investment, or guaranteed-profit offer
2. Claims to be law enforcement / government and says the person or their account is implicated
3. Demands account numbers, balances, PINs, passwords, or personal identifiers
4. Asks the person to install an app or grant remote access to a device
5. Frames withdrawing or transferring money as "protecting" it or "damage prevention"
6. Instructs payment via gift cards, wire transfer, cryptocurrency, or a courier
7. Asks the person to read back a verification / one-time code
8. Claims to be a family member in sudden trouble (jail, accident, hospital) needing money now
9. Pressures urgency or secrecy ("right now", "don't tell anyone")
10. Prize, lottery, or refund that requires a payment or personal details first
11. Unsolicited "tech support" reporting a virus, hack, or compromised account

Verdict rules:
- "scam": one or more criteria clearly present with intent to extract money, access, or credentials
- "uncertain": scam-adjacent language but a plausible legitimate reading (e.g. a real bank fraud department, a genuine family call about money) — keep watching, do not alarm
- "safe": an ordinary conversation; mentioning money, appointments, or family alone is NOT a scam

Respond with JSON: {"verdict": "scam"|"uncertain"|"safe", "criteria": [matched criterion numbers], "reasoning": "one short sentence"}"""


async def analyze_scam(transcript: str) -> dict:
    """Criteria-rubric LLM judgment over a rolling, PII-sanitized window.

    Returns {"verdict": ..., "criteria": [...], "reasoning": ..., "signals": [...]}.
    Never raises; falls back to a heuristic-informed verdict on API failure.
    """
    heur = score_text(transcript)
    if not transcript.strip():
        return {"verdict": "safe", "criteria": [], "reasoning": "No transcript yet.", "signals": []}

    window = redact(transcript[-TRANSCRIPT_WINDOW_CHARS:])
    try:
        response = await get_client().chat.completions.create(
            model=settings.scam_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": window},
            ],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
            timeout=8.0,
        )
        result = json.loads(response.choices[0].message.content or "{}")
        verdict = result.get("verdict", "safe")
        if verdict not in VERDICTS:
            verdict = "uncertain"
        criteria = [c for c in result.get("criteria", []) if isinstance(c, int)]
        reasoning = str(result.get("reasoning", ""))
    except Exception as e:
        log.warning("LLM scam analysis failed, deferring to heuristics: %s", e)
        # Regex escalates but never clears: on LLM failure a heuristically
        # loud transcript reads "uncertain", never "safe".
        verdict = "uncertain" if heur.level != "Low" else "safe"
        criteria = []
        reasoning = (
            "Signals: " + ", ".join(heur.signals)
            if heur.signals
            else "Automated analysis unavailable."
        )

    return {
        "verdict": verdict,
        "criteria": criteria,
        "reasoning": reasoning,
        "signals": heur.signals,
    }
