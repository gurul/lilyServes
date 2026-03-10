import audioop
import io
import wave


MIN_CHUNK_BYTES = 4000  # ~0.5s of mulaw audio at 8kHz


def mulaw_to_wav(mulaw_bytes: bytes, ratecv_state=None) -> tuple[bytes, object]:
    """
    Convert raw mulaw (u-law) bytes from Twilio to a 16kHz mono WAV.

    Twilio streams audio at 8kHz, mulaw-encoded. Whisper expects 16kHz PCM.
    Uses stdlib audioop only — no extra dependencies.

    Returns:
        (wav_bytes, new_ratecv_state)
        Pass the returned state back on the next call to maintain continuity
        across chunks and avoid click artifacts at chunk boundaries.
    """
    # 1. mulaw → 16-bit linear PCM at 8kHz
    pcm_8k = audioop.ulaw2lin(mulaw_bytes, 2)

    # 2. Upsample 8kHz → 16kHz, preserving state across chunks
    pcm_16k, new_state = audioop.ratecv(pcm_8k, 2, 1, 8000, 16000, ratecv_state)

    # 3. Wrap as WAV
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm_16k)

    return buf.getvalue(), new_state


def is_chunk_too_small(mulaw_bytes: bytes) -> bool:
    return len(mulaw_bytes) < MIN_CHUNK_BYTES
