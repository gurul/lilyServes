import asyncio

import pytest

from lily.memory import MemoryService, MemoryType


class FakeEmbedder:
    """Deterministic bag-of-words embedder over a tiny vocabulary."""

    VOCAB = ["pharmacy", "pickup", "friday", "scam", "gift", "cards", "doctor",
             "appointment", "grandson", "lunch", "susan", "ride"]
    dimensions = len(VOCAB)

    async def embed(self, texts, timeout=5.0):
        out = []
        for text in texts:
            lower = text.lower()
            out.append([1.0 if word in lower else 0.0 for word in self.VOCAB])
        return out


class FailingEmbedder:
    dimensions = 4

    async def embed(self, texts, timeout=5.0):
        return None


@pytest.fixture()
def svc(tmp_path):
    service = MemoryService(str(tmp_path), embedder=FakeEmbedder())
    asyncio.run(service.open())
    yield service
    asyncio.run(service.close())


def _drain(service):
    """Let background embedding tasks finish inside the running loop."""
    async def wait():
        while service._embed_tasks:
            await asyncio.gather(*list(service._embed_tasks), return_exceptions=True)
    return wait


def test_remember_and_semantic_recall(svc):
    async def run():
        await svc.remember("Pharmacy pickup on Friday at 4 PM", MemoryType.COMMITMENT,
                           caller="+15551230000")
        await svc.remember("Grandson called about lunch next week", MemoryType.EPISODE)
        await svc.remember("Caller pushed gift cards — flagged as scam", MemoryType.SAFETY,
                           importance=0.9)
        await _drain(svc)()

        results = await svc.recall("gift cards scam call")
        assert results and results[0]["memory_type"] == MemoryType.SAFETY

        results = await svc.recall("friday pharmacy pickup")
        assert results[0]["memory_type"] == MemoryType.COMMITMENT
    asyncio.run(run())


def test_recall_tracks_access(svc):
    async def run():
        m = await svc.remember("Doctor appointment on Friday", MemoryType.COMMITMENT)
        await _drain(svc)()
        await svc.recall("doctor appointment")
        got = await svc.db.get(m.id)
        assert got.recall_count == 1
        assert got.last_recalled > 0
    asyncio.run(run())


def test_caller_filter(svc):
    async def run():
        await svc.remember("Susan from pharmacy arranged a ride", MemoryType.EPISODE,
                           caller="+15551110000")
        await svc.remember("Different caller mentioned pharmacy too", MemoryType.EPISODE,
                           caller="+15552220000")
        await _drain(svc)()
        results = await svc.recall("pharmacy ride susan", caller="+15551110000")
        assert results and all(
            m["caller"] in ("", "+15551110000") for m in results
        )
    asyncio.run(run())


def test_forget_removes_from_search(svc):
    async def run():
        m = await svc.remember("Pharmacy pickup Friday", MemoryType.COMMITMENT)
        await _drain(svc)()
        assert await svc.forget(m.id) is True
        assert await svc.forget(m.id) is False  # already gone
        assert await svc.recall("pharmacy pickup friday") == []
        assert await svc.count() == 0
    asyncio.run(run())


def test_redaction_applied_on_write(svc):
    async def run():
        m = await svc.remember("Caller asked for the code 482913 and card 4111 1111 1111 1111")
        assert "482913" not in m.content
        assert "4111" not in m.content
    asyncio.run(run())


def test_lexical_fallback_when_embeddings_fail(tmp_path):
    service = MemoryService(str(tmp_path), embedder=FailingEmbedder())

    async def run():
        await service.open()
        await service.remember("Pharmacy pickup on Friday afternoon", MemoryType.COMMITMENT)
        await _drain(service)()
        results = await service.recall("pharmacy friday")
        assert results and results[0]["memory_type"] == MemoryType.COMMITMENT
        await service.close()
    asyncio.run(run())


def test_remember_call_maps_risk_to_type_and_importance(svc):
    async def run():
        low = await svc.remember_call("CA1", "+15551", "Nice chat about the garden.", "Low")
        high = await svc.remember_call("CA2", "+15552", "Caller demanded gift cards.", "High")
        assert low.memory_type == MemoryType.EPISODE
        assert high.memory_type == MemoryType.SAFETY
        assert high.importance > low.importance
        assert await svc.remember_call("CA3", "+15553", "   ", "Low") is None
    asyncio.run(run())


def test_persistence_across_reopen(tmp_path):
    async def run():
        s1 = MemoryService(str(tmp_path), embedder=FakeEmbedder())
        await s1.open()
        await s1.remember("Pharmacy pickup Friday", MemoryType.COMMITMENT)
        await _drain(s1)()
        await s1.close()

        s2 = MemoryService(str(tmp_path), embedder=FakeEmbedder())
        await s2.open()
        results = await s2.recall("pharmacy friday pickup")
        assert results and results[0]["memory_type"] == MemoryType.COMMITMENT
        await s2.close()
    asyncio.run(run())
