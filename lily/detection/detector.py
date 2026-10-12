"""LLM tier of the two-tier scam scorer.

The regex tier (heuristics) reacts instantly; this tier is the authoritative
judgment. Sources and design choices (links in README):

- The prompt is an explicit criteria rubric. arXiv:2506.06180 builds 11
  voice-phishing evaluation criteria into the prompt. On its adversarial test
  set (non-scam transcripts that use scam vocabulary), GPT-4o with criteria
  scored 90.18-99.11% accuracy, a criteria-prompted fine-tuned Llama3-8B
  87.50-94.64%, and a KoBERT classifier 56.25% (98.52% on the normal set).
  Our rubric adapts those criteria and adds grandparent-scam patterns.
- arXiv:2507.16291 reports that LLM rewrites of real vishing scripts, which
  keep the meaning, cut ML classifier accuracy by up to 30.96%. So the regex
  tier is easy to evade, and the LLM tier decides negatives.
- The verdict is three-way: "scam" / "uncertain" / "safe" (our design
  choice). UNCERTAIN keeps watching without alerting.
- Malformed model output never reads as "safe": any parse failure, non-object
  JSON, or unknown verdict gives "uncertain" (our design choice).

Transcripts are PII-sanitized (redact) before leaving the process.
arXiv:2510.18493 (MASK) proposes sanitizing call transcripts because they
"frequently contain sensitive personal information" that third-party LLM
providers may see.
"""
from __future__ import annotations

import json
import logging

from lily.config import settings
from lily.detection.heuristics import score_text
from lily.memory.operational import redact
from lily.summarizer import get_client

log = logging.getLogger("lily.scam")

# Rolling window keeps prompt size (and tail latency) flat on long calls.
TRANSCRIPT_WINDOW_CHARS = 4000

VERDICTS = ("scam", "uncertain", "safe")

# Criteria rubric adapted from the 11 voice-phishing evaluation criteria of
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


def parse_verdict(content: object) -> tuple[str, list[int], str]:
    """Parse the model's JSON reply into (verdict, criteria, reasoning).

    The verdict is read first, case-insensitively. Anything that is not a
    JSON object with a known verdict gives "uncertain", never "safe".
    """
    try:
        result = json.loads(content) if isinstance(content, (str, bytes)) else None
    except ValueError:
        result = None
    if not isinstance(result, dict):
        log.warning("LLM scam output was not a JSON object; treating as uncertain")
        return "uncertain", [], "Automated analysis returned unreadable output."

    raw_verdict = result.get("verdict")
    verdict = raw_verdict.strip().lower() if isinstance(raw_verdict, str) else ""
    if verdict not in VERDICTS:
        verdict = "uncertain"

    raw_criteria = result.get("criteria") or []
    if not isinstance(raw_criteria, list):
        raw_criteria = []
    criteria = [c for c in raw_criteria if isinstance(c, int) and not isinstance(c, bool)]

    raw_reasoning = result.get("reasoning")
    reasoning = raw_reasoning if isinstance(raw_reasoning, str) else ""
    return verdict, criteria, reasoning


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

    else:
        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            content = None
        verdict, criteria, reasoning = parse_verdict(content)

    return {
        "verdict": verdict,
        "criteria": criteria,
        "reasoning": reasoning,
        "signals": heur.signals,
    }
