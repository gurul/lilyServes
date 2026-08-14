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
    """Let background tasks (embeddings, recall marks) finish inside the loop."""
    async def wait():
        while service._bg_tasks:
            await asyncio.gather(*list(service._bg_tasks), return_exceptions=True)
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
        await _drain(svc)()  # mark_recalled is fire-and-forget now
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


def test_forget_during_embedding_leaves_no_ghost_vector(tmp_path):
    gate = asyncio.Event()

    class GatedEmbedder(FakeEmbedder):
        async def embed(self, texts, timeout=5.0):
            await gate.wait()
            return await super().embed(texts, timeout)

    service = MemoryService(str(tmp_path), embedder=GatedEmbedder())

    async def run():
        await service.open()
        m = await service.remember("Pharmacy pickup Friday", MemoryType.COMMITMENT)
        assert await service.forget(m.id) is True  # forget wins the race
        gate.set()
        await _drain(service)()
        assert m.id not in service.db._vec_ids
        hits = await service.db.vector_search([1.0] * FakeEmbedder.dimensions, 5)
        assert all(mid != m.id for mid, _ in hits)
        await service.close()
    asyncio.run(run())


def test_set_embedding_twice_keeps_single_index_row(svc):
    async def run():
        m = await svc.remember("Pharmacy pickup Friday", MemoryType.COMMITMENT)
        await _drain(svc)()
        await svc.db.set_embedding(m.id, [1.0] * FakeEmbedder.dimensions)
        assert svc.db._vec_ids.count(m.id) == 1
        assert svc.db._vectors.shape[0] == 1
    asyncio.run(run())


def test_lexical_score_absolute_for_single_result(svc):
    async def run():
        await svc.remember("Pharmacy pickup on Friday at 4 PM", MemoryType.COMMITMENT)
        await _drain(svc)()
        hits = await svc.db.lexical_search("pharmacy pickup friday")
        assert hits and hits[0][1] > 0.4  # min-max would have scored this 0.0
    asyncio.run(run())


def test_like_fallback_matches_any_term(svc):
    async def run():
        await svc.remember("Pharmacy pickup on Friday", MemoryType.COMMITMENT)
        await _drain(svc)()
        svc.db.fts_enabled = False
        hits = await svc.db.lexical_search("hospital pharmacy")
        assert hits  # old fallback only searched the first term
    asyncio.run(run())


def test_importance_cannot_buy_relevance(svc):
    async def run():
        await svc.remember("Grandson called about lunch next week", MemoryType.EPISODE,
                           importance=0.9)
        await _drain(svc)()
        results = await svc.recall("pharmacy pickup friday", min_score=0.35)
        assert results == []  # irrelevant memory must not clear the floor
    asyncio.run(run())


def test_corrupt_embedding_blob_heals_on_open(tmp_path):
    async def run():
        s1 = MemoryService(str(tmp_path), embedder=FakeEmbedder())
        await s1.open()
        good = await s1.remember("Pharmacy pickup Friday", MemoryType.COMMITMENT)
        bad = await s1.remember("Doctor appointment Monday", MemoryType.COMMITMENT)
        await _drain(s1)()

        def corrupt(conn):
            conn.execute("UPDATE memories SET embedding = ? WHERE id = ?",
                         (b"\x01\x02\x03", bad.id))
            conn.commit()
        await s1.db._run(corrupt)
        await s1.close()

        s2 = MemoryService(str(tmp_path), embedder=FakeEmbedder())
        await s2.open()  # must not crash; corrupt row re-embeds in background
        await _drain(s2)()
        assert good.id in s2.db._vec_ids
        assert bad.id in s2.db._vec_ids  # healed via missing_embedding_ids
        await s2.close()
    asyncio.run(run())


def test_recall_after_close_raises_cleanly(tmp_path):
    service = MemoryService(str(tmp_path), embedder=FakeEmbedder())

    async def run():
        await service.open()
        await service.close()
        with pytest.raises(RuntimeError):
            await service.db.get("nope")
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
