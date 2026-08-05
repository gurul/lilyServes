"""Audio conversion helpers.

Pure-Python mulaw decode fallback so the service runs on Python 3.13+,
where the stdlib `audioop` module was removed. On <=3.12 the C
implementation is used for speed.
"""
from __future__ import annotations

import io
import struct
import wave

_BIAS = 0x84


def _decode_sample(mu: int) -> int:
    mu = ~mu & 0xFF
    sign = mu & 0x80
    exponent = (mu >> 4) & 0x07
    mantissa = mu & 0x0F
    sample = ((mantissa << 3) + _BIAS) << exponent
    sample -= _BIAS
    return -sample if sign else sample


_TABLE = [_decode_sample(i) for i in range(256)]

try:
    import audioop  # removed in Python 3.13

    def _ulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
        return audioop.ulaw2lin(mulaw_bytes, 2)

except ImportError:  # pragma: no cover - exercised only on 3.13+

    def _ulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
        return struct.pack("<%dh" % len(mulaw_bytes), *(_TABLE[b] for b in mulaw_bytes))


def mulaw_to_wav(mulaw_bytes: bytes, sample_rate: int = 8000) -> bytes:
    """Convert raw mulaw audio (mono) to a 16-bit PCM WAV file in memory."""
    pcm_data = _ulaw_to_pcm16(mulaw_bytes)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()
