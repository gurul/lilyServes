"""Unit regressions for the hub, STT session, and call registry."""
from __future__ import annotations

import asyncio

import pytest

import lily.hub as hub_mod
import lily.transcription as tr
from lily.calls import CallExistsError, CallRegistry
from lily.hub import ClientHub


class _Healthy:
    def __init__(self):
        self.got = 0

    async def send_text(self, _t):
        self.got += 1


class _Stalled:
    """A TCP peer that stopped reading: send waits for drain forever."""

    async def send_text(self, _t):
        await asyncio.Event().wait()


def test_hub_send_timeout_is_two_seconds():
    assert hub_mod.SEND_TIMEOUT_SECONDS == 2.0


def test_stalled_client_does_not_block_broadcast(monkeypatch):
    monkeypatch.setattr(hub_mod, "SEND_TIMEOUT_SECONDS", 0.05)

    async def main():
        hub = ClientHub()
        good, stalled = _Healthy(), _Stalled()
        await hub.register(good)
        await hub.register(stalled)
        await asyncio.wait_for(hub.broadcast({"event": "x"}, hub_mod.URGENT), timeout=1.0)
        assert good.got == 1
        assert stalled not in hub._clients
        assert hub.client_count == 1
        await asyncio.wait_for(hub.broadcast({"event": "y"}), timeout=1.0)
        assert good.got == 2

    asyncio.run(main())


def test_stt_client_init_failure_retries_and_stop_returns(monkeypatch):
    attempts = []

    class Boom:
        def __init__(self):
            attempts.append(1)
            raise RuntimeError("no ADC")

    monkeypatch.setattr(tr, "SpeechAsyncClient", Boom)

    async def main():
        s = tr.GoogleStreamingSession(project_id="p")
        await s.start()
        await asyncio.sleep(0.05)
        assert not s._stream_task.done()  # the loop survives the failure
        return await s.stop()

    assert asyncio.run(main()) == []
    assert attempts


def test_stt_stop_swallows_stream_task_exception():
    async def main():
        s = tr.GoogleStreamingSession(project_id="p")

        async def broken():
            raise RuntimeError("stream died")

        s._is_running = True
        s._stream_task = asyncio.create_task(broken())
        await asyncio.sleep(0)
        return await s.stop()

    assert asyncio.run(main()) == []


def test_registry_refuses_existing_call_sid():
    reg = CallRegistry()
    first = reg.create("CA1", "+15550000001")
    with pytest.raises(CallExistsError):
        reg.create("CA1", "+15550000002")
    assert reg.get("CA1") is first
    reg.pop("CA1")
    assert reg.create("CA1").call_sid == "CA1"
