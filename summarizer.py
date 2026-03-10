import os

import openai

_client: openai.AsyncOpenAI | None = None


def get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


SYSTEM_PROMPT = """You are a call analysis assistant specializing in scam detection.
Given a phone call transcript, produce a structured JSON response with these fields:
- summary: A 2-3 sentence summary of what was discussed
- scam_indicators: A list of specific phrases or behaviors that suggest this is a scam (empty list if none)
- risk_level: "low", "medium", or "high"
- recommended_action: What the recipient should do (e.g. "Hang up immediately", "This appears legitimate")

Respond with valid JSON only, no markdown fences."""


async def generate_summary(transcript_lines: list[str]) -> str:
    """
    Generate a structured scam call analysis from the full transcript.
    Returns a JSON string. Returns a fallback string on error.
    """
    if not transcript_lines:
        return '{"summary": "No transcript available.", "scam_indicators": [], "risk_level": "low", "recommended_action": "N/A"}'

    full_text = "\n".join(transcript_lines)
    client = get_client()

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_text},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[summarizer] GPT-4o error: {e}")
        return f'{{"summary": "Error generating summary: {e}", "scam_indicators": [], "risk_level": "unknown", "recommended_action": "Review transcript manually"}}'
