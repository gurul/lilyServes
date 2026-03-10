import audioop
import base64
import io
import os
import wave

import httpx

HIVE_API_URL = (
    "https://api.thehive.ai/api/v3/hive/"
    "ai-generated-and-deepfake-content-detection"
)
HIVE_API_SECRET = os.environ.get("HIVE_API_SECRET", "")
TIMEOUT = 30.0  # seconds – generous for 10s of audio processing
DEEPFAKE_THRESHOLD = 0.5


def mulaw_to_wav(mulaw_bytes: bytes) -> bytes:
    """Convert raw mulaw audio (8 kHz, mono) to a PCM WAV file in memory."""
    pcm_data = audioop.ulaw2lin(mulaw_bytes, 2)  # 16-bit PCM
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(8000)
        wf.writeframes(pcm_data)
    return buf.getvalue()


async def check_deepfake(mulaw_bytes: bytes) -> dict:
    """
    Send audio to HiveAI and return deepfake detection result.

    Returns:
        {"score": float (0.0-1.0), "is_deepfake": bool}
        or {"score": None, "is_deepfake": False} on error.
    """
    if not HIVE_API_SECRET:
        print("[deepfake] HIVE_API_SECRET not set, skipping")
        return {"score": None, "is_deepfake": False}

    try:
        wav_bytes = mulaw_to_wav(mulaw_bytes)
        wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
        data_uri = f"data:audio/wav;base64,{wav_b64}"

        payload = {
            "input": [{"media_base64": data_uri}],
        }
        headers = {
            "Authorization": f"Bearer {HIVE_API_SECRET}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(HIVE_API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Response: {"output": [{"classes": [{"class": "ai_generated_audio", "value": 0.02}, ...]}]}
        classes = data["output"][0]["classes"]
        ai_score = next(
            (c["value"] for c in classes if c["class"] == "ai_generated_audio"),
            0.0,
        )
        return {
            "score": round(ai_score, 4),
            "is_deepfake": ai_score >= DEEPFAKE_THRESHOLD,
        }
    except Exception as e:
        print(f"[deepfake] HiveAI error: {e}")
        return {"score": None, "is_deepfake": False}
