"""Event capture: turn spoken commitments into structured reminders.

"Main St. Pharmacy confirmed pickup for Friday @ 4:00 PM" — the Active
Assistance / Memory Sync feature. To keep cost and latency near zero, an LLM
extraction only runs when a cheap trigger says the sentence plausibly
contains a commitment (a time, a weekday, an appointment word).
"""
from __future__ import annotations

import json
import logging
import re

from lily.config import settings
from lily.summarizer import get_client

log = logging.getLogger("lily.events")

_TRIGGER = re.compile(
    r"\b(?:appointment|schedule[d]?|pick\s*up|pickup|confirm(?:ed|ing)?|"
    r"reservation|visit|meeting|delivery|prescription|refill|see\s+you)\b"
    r"|\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight)\b"
    r"|\b\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.m\.|p\.m\.|o'?clock)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You extract concrete commitments from one sentence of a phone call.
A commitment is a specific appointment, pickup, visit, delivery, or task with a who/what and ideally a when.
Respond with JSON: {"events": [{"title": "short plain description", "when": "the stated time, or empty string"}]}
Return {"events": []} if the sentence contains no concrete commitment. Never invent details."""


def has_event_trigger(text: str) -> bool:
    """Sub-millisecond gate deciding whether extraction is worth an LLM call."""
    return bool(text and _TRIGGER.search(text))


async def extract_events(sentence: str) -> list[dict]:
    """Extract {"title", "when"} commitments from a final transcript line.

    Callers should gate on has_event_trigger() first. Never raises.
    """
    if not sentence.strip():
        return []
    try:
        response = await get_client().chat.completions.create(
            model=settings.event_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sentence},
            ],
            temperature=0,
            max_tokens=150,
            response_format={"type": "json_object"},
            timeout=8.0,
        )
        data = json.loads(response.choices[0].message.content or "{}")
        events = data.get("events", [])
        return [
            {"title": str(e.get("title", "")).strip(), "when": str(e.get("when", "")).strip()}
            for e in events
            if isinstance(e, dict) and str(e.get("title", "")).strip()
        ]
    except Exception as e:
        log.warning("event extraction failed: %s", e)
        return []
