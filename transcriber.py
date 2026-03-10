import io
import os

import openai

_client: openai.AsyncOpenAI | None = None


def get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


async def transcribe_chunk(wav_bytes: bytes) -> str:
    """
    Send a WAV audio chunk to OpenAI Whisper and return the transcript text.
    Returns an empty string if transcription fails or produces no text.
    """
    client = get_client()
    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = "audio.wav"

    try:
        result = await client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en",
        )
        return result.text.strip()
    except Exception as e:
        print(f"[transcriber] Whisper error: {e}")
        return ""
