import os

import httpx

DEEPFAKE_ENDPOINT = os.environ.get("DEEPFAKE_ENDPOINT", "http://localhost:3000/api/deepfake")
TIMEOUT = 15.0  # seconds


async def send_to_deepfake(wav_bytes: bytes) -> dict:
    """
    Forward a WAV audio chunk to the deepfake detection endpoint.

    Expects the endpoint to accept multipart/form-data with an "audio" file field
    and return JSON with a "probability" field (0-100).

    Returns {"probability": float} or {"probability": None} on error.
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.post(
                DEEPFAKE_ENDPOINT,
                files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
            )
            response.raise_for_status()
            data = response.json()
            return {"probability": data.get("probability")}
    except Exception as e:
        print(f"[deepfake_client] Error: {e}")
        return {"probability": None}
