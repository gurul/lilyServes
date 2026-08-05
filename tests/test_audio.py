import io
import wave

from lily.audio import mulaw_to_wav


def test_mulaw_to_wav_shape():
    # 8000 mulaw bytes = 1 second at 8 kHz mono
    wav = mulaw_to_wav(bytes(range(256)) * 32)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 8000
        assert wf.getnframes() == 256 * 32


def test_mulaw_silence_decodes_near_zero():
    # 0xFF is mulaw digital silence (decodes to 0)
    wav = mulaw_to_wav(b"\xff" * 100)
    with wave.open(io.BytesIO(wav), "rb") as wf:
        frames = wf.readframes(100)
    assert set(frames) == {0}


def test_pure_python_fallback_matches_audioop():
    try:
        import audioop
    except ImportError:
        return  # 3.13+: nothing to compare against
    import struct

    from lily.audio import _TABLE

    sample_bytes = bytes(range(256))
    expected = audioop.ulaw2lin(sample_bytes, 2)
    got = struct.pack("<256h", *(_TABLE[b] for b in sample_bytes))
    assert got == expected
