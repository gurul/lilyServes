"""Shared OpenAI client and the post-call structured summary."""
from __future__ import annotations

import json
import logging
import os

import openai

from lily.config import settings

log = logging.getLogger("lily.summarizer")

_client: openai.AsyncOpenAI | None = None


def get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


SYSTEM_PROMPT = """You are Lily, a gentle call companion for an older adult. \
Given a phone call transcript (the caller's side only), produce JSON with:
- summary: 2-3 plain, simple sentences an older adult can read at a glance, emphasizing what matters
- scam_indicators: list of specific phrases/behaviors suggesting a scam (empty if none)
- risk_level: "low", "medium", or "high"
- recommended_action: one short sentence (e.g. "Hang up if they call back", "This appears legitimate")
- events: list of concrete commitments mentioned, each {"title": str, "when": str} (empty if none)
- facts: list of durable things learned about people or preferences, each \
{"kind": "person"|"preference"|"win", "content": one short sentence, "entities": [names]}. \
Use "person" for who someone is or how they relate, "preference" for how the user \
likes things done, "win" for a genuinely positive moment worth celebrating. \
Empty if nothing durable was learned.

Respond with valid JSON only."""

_FALLBACK = {
    "summary": "No transcript available.",
    "scam_indicators": [],
    "risk_level": "low",
    "recommended_action": "N/A",
    "events": [],
    "facts": [],
}


def sanitize_summary(result: dict) -> dict:
    """Enforce the summary contract on model output. json_object mode doesn't
    enforce a schema — explicit nulls and wrong-typed lists must not crash
    the finalize path."""
    for key, default in _FALLBACK.items():
        value = result.get(key)
        if value is None or not isinstance(value, type(default)):
            result[key] = default
    return result


async def generate_summary(transcript_lines: list[str]) -> dict:
    """Structured post-call analysis. Never raises; returns a fallback dict."""
    if not transcript_lines:
        return dict(_FALLBACK)

    try:
        response = await get_client().chat.completions.create(
            model=settings.summary_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(transcript_lines)},
            ],
            temperature=0.2,
            max_tokens=500,
            response_format={"type": "json_object"},
            timeout=20.0,
        )
        result = json.loads(response.choices[0].message.content or "{}")
        return sanitize_summary(result)
    except Exception as e:
        log.warning("summary generation failed: %s", e)
        fallback = dict(_FALLBACK)
        fallback["summary"] = "Summary unavailable — review the call transcript."
        fallback["risk_level"] = "unknown"
        fallback["recommended_action"] = "Review transcript manually"
        return fallback
