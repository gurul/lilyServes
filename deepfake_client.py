"""Hive AI deepfake detection on the first seconds of caller audio."""
from __future__ import annotations

import base64
import logging

import httpx

from audio import mulaw_to_wav
from config import settings

log = logging.getLogger("lily.deepfake")

HIVE_API_URL = (
    "https://api.thehive.ai/api/v3/hive/"
    "ai-generated-and-deepfake-content-detection"
)
TIMEOUT = 30.0  # generous for ~10s of audio processing
DEEPFAKE_THRESHOLD = 0.5

# Shared pooled client: keeps the TLS connection warm across calls so the
# handshake isn't paid inside a live call.
_http: httpx.AsyncClient | None = None


def _get_http() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(
            timeout=TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=2, max_connections=4),
        )
    return _http


async def close_http() -> None:
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None


async def check_deepfake(mulaw_bytes: bytes) -> dict:
    """Send audio to Hive AI; returns {"score": float|None, "is_deepfake": bool}."""
    if not settings.hive_api_secret:
        log.info("HIVE_API_SECRET not set, skipping deepfake check")
        return {"score": None, "is_deepfake": False}

    try:
        wav_b64 = base64.b64encode(mulaw_to_wav(mulaw_bytes)).decode("ascii")
        payload = {"input": [{"media_base64": f"data:audio/wav;base64,{wav_b64}"}]}
        headers = {"Authorization": f"Bearer {settings.hive_api_secret}"}

        resp = await _get_http().post(HIVE_API_URL, headers=headers, json=payload)
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
        log.warning("Hive AI error: %s", e)
        return {"score": None, "is_deepfake": False}
