import io
import os

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech as cloud_speech_types

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")


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
