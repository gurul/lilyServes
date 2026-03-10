import json

from summarizer import get_client

SYSTEM_PROMPT = """You are a phone call scam detection system. Analyze the following phone call transcript and assess the scam risk level.

Rate the call as one of:
- "Low" - A normal, everyday conversation with no suspicious elements.
- "Medium" - Some concerning elements such as: urgency, claims to be a family member, or involvement of money.
- "High" - Highly suspect of being a scam: high urgency combined with money being requested via obscure sources (gift cards, wire transfers, cryptocurrency, etc.).

Respond with valid JSON only, no markdown fences:
{"scam_level": "Low" or "Medium" or "High", "reasoning": "brief explanation"}"""


DEEPFAKE_THRESHOLD = 0.5


async def analyze_scam(transcript: str, deepfake_score: float = 0.0) -> dict:
    """
    Ask OpenAI to rate the transcript, then factor in deepfake detection.

    Returns:
        {"scam_level": "Low"|"Medium"|"High", "reasoning": str}
    """
    if not transcript.strip():
        return {"scam_level": "Low", "reasoning": "No transcript available."}

    client = get_client()

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        ai_level = result.get("scam_level", "Low")
        reasoning = result.get("reasoning", "")
    except Exception as e:
        print(f"[scam_detector] OpenAI error: {e}")
        ai_level = "Low"
        reasoning = f"Analysis failed: {e}"

    # Factor in deepfake detection: if synthetic voice detected, bump to at least Medium
    final_level = ai_level
    if deepfake_score >= DEEPFAKE_THRESHOLD and final_level == "Low":
        final_level = "Medium"
        reasoning += " [Elevated: synthetic voice detected]"

    return {"scam_level": final_level, "reasoning": reasoning}
