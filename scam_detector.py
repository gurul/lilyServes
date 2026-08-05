"""Two-tier scam risk scoring.

Tier 1 (scam_heuristics) runs inline and is effectively free, so every
transcript line ships to the dashboard with a provisional level instantly.
Tier 2 (this module) refines with an LLM off the hot path: one analysis in
flight per call, coalescing new lines that arrive mid-flight, over a rolling
transcript window so cost and latency stay flat on long calls.
"""
from __future__ import annotations

import json
import logging

from config import settings
from scam_heuristics import max_level, score_text
from summarizer import get_client

log = logging.getLogger("lily.scam")

# Rolling window keeps prompt size (and tail latency) flat on long calls.
TRANSCRIPT_WINDOW_CHARS = 4000
DEEPFAKE_THRESHOLD = 0.5

SYSTEM_PROMPT = """You are a phone call scam detection system protecting an older adult. \
Analyze the phone call transcript (the caller's side only) and assess scam risk.

Rate the call as one of:
- "Low" - A normal, everyday conversation with no suspicious elements.
- "Medium" - Some concerning elements such as urgency, claims to be a family member, or involvement of money.
- "High" - Highly suspect: urgency combined with money requested via gift cards, wire transfer, cryptocurrency, remote access to a computer, or requests for verification codes / personal credentials.

Respond with JSON: {"scam_level": "Low"|"Medium"|"High", "reasoning": "one short sentence"}"""


async def analyze_scam(transcript: str, deepfake_score: float | None = None) -> dict:
    """LLM assessment over a rolling window, floored by heuristics and deepfake.

    Returns {"scam_level": ..., "reasoning": ..., "signals": [...]}.
    Never raises; falls back to the heuristic level on API failure.
    """
    heur = score_text(transcript)
    if not transcript.strip():
        return {"scam_level": "Low", "reasoning": "No transcript yet.", "signals": []}

    window = transcript[-TRANSCRIPT_WINDOW_CHARS:]
    try:
        response = await get_client().chat.completions.create(
            model=settings.scam_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": window},
            ],
            temperature=0,
            max_tokens=120,
            response_format={"type": "json_object"},
            timeout=8.0,
        )
        result = json.loads(response.choices[0].message.content or "{}")
        ai_level = result.get("scam_level", "Low")
        reasoning = str(result.get("reasoning", ""))
    except Exception as e:
        log.warning("LLM scam analysis failed, using heuristics: %s", e)
        ai_level = "Low"
        reasoning = "Signals: " + ", ".join(heur.signals) if heur.signals else "Automated analysis unavailable."

    # The LLM can raise but never lower the heuristic floor.
    level = max_level(ai_level, heur.level)
    if deepfake_score is not None and deepfake_score >= DEEPFAKE_THRESHOLD:
        level = max_level(level, "Medium")
        reasoning = (reasoning + " Synthetic voice detected.").strip()

    return {"scam_level": level, "reasoning": reasoning, "signals": heur.signals}
