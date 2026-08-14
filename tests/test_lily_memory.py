import asyncio
import sqlite3
import time

import pytest

from lily.memory import Memory, MemoryService, MemoryStatus, MemoryType, lifecycle


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


def test_caller_scope_survives_a_crowded_corpus(svc):
    async def run():
        for i in range(40):
            await svc.remember(f"Pharmacy pickup chatter number {i}", MemoryType.EPISODE,
                               caller=f"+1555000{i:04d}")
        await svc.remember("Pharmacy pickup on Friday for Susan", MemoryType.COMMITMENT,
                           caller="+15551110000")
        await _drain(svc)()
        results = await svc.recall("pharmacy pickup", caller="+15551110000")
        assert results  # pre-pushdown, the global top-k starved this scope
        assert all(m["caller"] in ("", "+15551110000") for m in results)
    asyncio.run(run())


def test_type_filter_applies_inside_vector_search(svc):
    async def run():
        await svc.remember("Pharmacy pickup on Friday", MemoryType.COMMITMENT)
        await svc.remember("Chatted about the pharmacy visit", MemoryType.EPISODE)
        await _drain(svc)()
        results = await svc.recall("pharmacy friday pickup",
                                   memory_types=[MemoryType.COMMITMENT])
        assert results and all(m["memory_type"] == MemoryType.COMMITMENT for m in results)
    asyncio.run(run())


def test_entity_hits_outrank_content_mentions(svc):
    async def run():
        ride = await svc.remember("Arranged a ride for the appointment",
                                  MemoryType.EPISODE, entities=["Susan"])
        await svc.remember("susan was mentioned in passing today", MemoryType.EPISODE)
        # Fillers so the term isn't in every document (BM25 IDF needs contrast).
        await svc.remember("Grandson called about lunch", MemoryType.EPISODE)
        await svc.remember("Doctor appointment moved", MemoryType.EPISODE)
        await svc.remember("Gift cards are a scam sign", MemoryType.SAFETY)
        await _drain(svc)()
        hits = await svc.db.lexical_search("susan")
        assert hits[0][0] == ride.id  # entity column outweighs prose mention
    asyncio.run(run())


def test_recency_is_type_aware():
    import time as _time

    from lily.memory.service import MemoryService as MS
    now = _time.time()
    old = now - 180 * 86400
    person = Memory(content="Susan is her daughter", memory_type=MemoryType.PERSON,
                    created_at=old)
    episode = Memory(content="Chatted about the garden", memory_type=MemoryType.EPISODE,
                     created_at=old)
    commitment_old = Memory(content="Pickup", memory_type=MemoryType.COMMITMENT,
                            created_at=now - 21 * 86400)
    commitment_new = Memory(content="Pickup", memory_type=MemoryType.COMMITMENT,
                            created_at=now)
    assert MS._recency(person, now) == 1.0          # stable facts never fade
    assert MS._recency(episode, now) < 0.1          # six-month-old chatter sinks
    assert MS._recency(commitment_new, now) > MS._recency(commitment_old, now)
    # Frequent recall keeps a memory warm.
    episode_recalled = Memory(content="x", memory_type=MemoryType.EPISODE,
                              created_at=old, last_recalled=now - 86400)
    assert MS._recency(episode_recalled, now) > MS._recency(episode, now)


def test_recall_deduplicates_near_identical_memories(svc):
    async def run():
        for _ in range(5):
            await svc.remember("Tom stopped by to check in", MemoryType.EPISODE)
        await svc.remember("Pharmacy pickup on Friday", MemoryType.COMMITMENT)
        await _drain(svc)()
        results = await svc.recall("tom check in pharmacy friday pickup", limit=5)
        contents = [m["content"] for m in results]
        assert contents.count("Tom stopped by to check in") == 1
        assert any("Pharmacy pickup" in c for c in contents)
    asyncio.run(run())


# ----- lifecycle -------------------------------------------------------------


_OLD_SCHEMA = """
CREATE TABLE memories (
    id TEXT PRIMARY KEY, memory_type TEXT NOT NULL, content TEXT NOT NULL,
    entities TEXT DEFAULT '[]', topics TEXT DEFAULT '[]',
    importance REAL DEFAULT 0.5, confidence REAL DEFAULT 0.8,
    caller TEXT DEFAULT '', call_sid TEXT DEFAULT '',
    created_at REAL, last_recalled REAL DEFAULT 0,
    recall_count INTEGER DEFAULT 0, embedding BLOB
);
"""


def test_migration_from_pre_lifecycle_schema(tmp_path):
    db_path = tmp_path / "lily_memory.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(_OLD_SCHEMA)
    conn.execute(
        "INSERT INTO memories (id, memory_type, content, created_at) "
        "VALUES ('legacy1', 'episode', 'Pharmacy pickup Friday', ?)",
        (time.time(),),
    )
    conn.commit()
    conn.close()

    service = MemoryService(str(tmp_path), embedder=FakeEmbedder())

    async def run():
        await service.open()  # guarded ALTERs must upgrade in place
        got = await service.db.get("legacy1")
        assert got is not None and got.status == MemoryStatus.ACTIVE
        assert got.source_count == 1 and got.expires_at == 0
        results = await service.recall("pharmacy pickup friday")
        assert results and results[0]["id"] == "legacy1"
        await service.close()
    asyncio.run(run())


def test_superseded_memory_hidden_but_auditable(svc):
    async def run():
        old = await svc.remember("Lives on Maple Street", MemoryType.PERSON,
                                 caller="+15551110000")
        await _drain(svc)()
        await svc.db.set_status(old.id, MemoryStatus.SUPERSEDED, "newid")
        assert await svc.recall("maple street") == []
        got = await svc.db.get(old.id)  # audit reads still see everything
        assert got.status == MemoryStatus.SUPERSEDED
        assert got.superseded_by == "newid"
        assert await svc.forget(old.id) is True  # privacy promise unchanged
    asyncio.run(run())


def test_duplicate_reinforces_instead_of_duplicating(svc):
    async def run():
        first = await svc.remember("Grandson called about lunch on Friday",
                                   MemoryType.EPISODE, importance=0.5)
        await _drain(svc)()
        await svc.remember("Grandson called about lunch on Friday",
                           MemoryType.EPISODE, importance=0.5)
        await _drain(svc)()
        assert await svc.count() == 1
        got = await svc.db.get(first.id)
        assert got.source_count == 2
        assert got.importance > 0.5
    asyncio.run(run())


def test_duplicate_reinforces_lexically_without_embeddings(tmp_path):
    service = MemoryService(str(tmp_path), embedder=FailingEmbedder())

    async def run():
        await service.open()
        await service.remember("Grandson called about lunch on Friday")
        await _drain(service)()
        await service.remember("Grandson called about lunch on Friday")
        await _drain(service)()
        assert await service.count() == 1
        await service.close()
    asyncio.run(run())


def test_reschedule_supersedes_old_commitment(svc):
    async def run():
        old = await svc.remember_commitment("CA1", "+15551110000",
                                            "Pharmacy pickup", "Friday at 4 pm")
        await _drain(svc)()
        new = await svc.remember_commitment("CA2", "+15551110000",
                                            "Pharmacy pickup", "Saturday at 10 am")
        await _drain(svc)()
        old_row = await svc.db.get(old.id)
        assert old_row.status == MemoryStatus.SUPERSEDED
        assert old_row.superseded_by == new.id
        results = await svc.recall("pharmacy pickup", caller="+15551110000")
        assert [m["id"] for m in results] == [new.id]
    asyncio.run(run())


def test_retrust_replaces_person_fact(svc):
    async def run():
        await svc.remember_person("+15551110000", "Susan")
        await _drain(svc)()
        new = await svc.remember_person("+15551110000", "Susan",
                                        "daughter, calls every Sunday")
        await _drain(svc)()
        results = await svc.recall("susan contact", caller="+15551110000")
        ids = [m["id"] for m in results]
        assert new.id in ids and len(ids) == 1
    asyncio.run(run())


def test_resolve_when():
    ref = time.mktime((2026, 8, 12, 12, 0, 0, -1, -1, -1))  # a Wednesday noon
    friday = lifecycle.resolve_when("Friday at 4 pm", ref)
    assert time.localtime(friday).tm_wday == 4
    assert time.localtime(friday).tm_hour == 16
    tomorrow = lifecycle.resolve_when("tomorrow", ref)
    assert time.localtime(tomorrow).tm_mday == 13
    march = lifecycle.resolve_when("March 5", ref)
    assert time.localtime(march).tm_year == 2027  # already passed → rolls over
    numeric = lifecycle.resolve_when("9/1 at 10 am", ref)
    assert time.localtime(numeric).tm_mon == 9
    assert lifecycle.resolve_when("whenever works", ref) is None
    assert lifecycle.resolve_when("", ref) is None


def test_expired_commitment_leaves_recall(svc):
    async def run():
        m = await svc.remember("Pharmacy pickup", MemoryType.COMMITMENT,
                               expires_at=time.time() - 3600)
        await _drain(svc)()
        await svc._sweep(time.time())
        assert await svc.recall("pharmacy pickup") == []
        got = await svc.db.get(m.id)
        assert got.status == MemoryStatus.EXPIRED
        # Audit view still shows it.
        audit = await svc.recent(include_inactive=True)
        assert any(d["id"] == m.id for d in audit)
    asyncio.run(run())


def test_purge_respects_retention_and_safety_exemption(svc):
    async def run():
        ep = await svc.remember("Old expired chatter", MemoryType.EPISODE)
        sf = await svc.remember("Gift card scam attempt", MemoryType.SAFETY)
        await _drain(svc)()
        long_ago = time.time() - 200 * 86400

        def backdate(conn):
            conn.execute(
                "UPDATE memories SET status = 'expired', updated_at = ?", (long_ago,)
            )
            conn.commit()
        await svc.db._run(backdate)

        await svc._sweep(time.time())
        assert await svc.db.get(ep.id) is None       # purged from disk
        assert await svc.db.get(sf.id) is not None   # scam history is immortal
    asyncio.run(run())


def test_due_soon_commitment_outranks_distant_one():
    now = time.time()
    soon = Memory(content="Pickup", memory_type=MemoryType.COMMITMENT,
                  created_at=now - 86400, expires_at=now + 86400)
    distant = Memory(content="Pickup", memory_type=MemoryType.COMMITMENT,
                     created_at=now, expires_at=now + 21 * 86400)
    assert MemoryService._recency(soon, now) > MemoryService._recency(distant, now)


def test_completed_commitment_expires(svc):
    async def run():
        m = await svc.remember_commitment("CA1", "+15551", "Pharmacy pickup", "Friday")
        await _drain(svc)()
        assert await svc.complete_commitment(m.id) is True
        assert await svc.recall("pharmacy pickup") == []
    asyncio.run(run())


# ----- profile card ----------------------------------------------------------


def test_profile_card_sections(svc):
    async def run():
        caller = "+15551110000"
        await svc.remember_person(caller, "Susan", "daughter, calls Sundays")
        await svc.remember("Prefers morning calls", MemoryType.PREFERENCE)
        await svc.remember_commitment("CA1", caller, "Pharmacy pickup", "tomorrow at 4 pm")
        await svc.remember("Chatted about the garden", MemoryType.EPISODE, caller=caller)
        await svc.remember("Pushed gift cards — scam", MemoryType.SAFETY,
                           importance=0.9, caller=caller)
        await _drain(svc)()

        profile = await svc.profile(caller, "Susan")
        assert any("Susan" in f["content"] for f in profile["facts"])
        assert any("morning" in f["content"].lower() for f in profile["facts"])  # global prefs included
        assert profile["open_commitments"][0]["content"].startswith("Pharmacy pickup")
        assert profile["safety"]["count"] == 1
        assert profile["recent"]

        text = MemoryService.render_profile(profile)
        assert len(text) <= 600
        assert "Susan" in text and "Pharmacy" in text
    asyncio.run(run())


def test_profile_excludes_retired_rows(svc):
    async def run():
        caller = "+15551110000"
        done = await svc.remember_commitment("CA1", caller, "Pharmacy pickup", "Friday")
        await _drain(svc)()
        await svc.complete_commitment(done.id)
        profile = await svc.profile(caller)
        assert profile["open_commitments"] == []
    asyncio.run(run())


def test_profile_for_unknown_caller_is_minimal(svc):
    async def run():
        profile = await svc.profile("+19998887777")
        assert profile["facts"] == [] and profile["open_commitments"] == []
        assert profile["safety"]["count"] == 0
        assert MemoryService.render_profile(profile) != ""  # still renders the caller line
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
