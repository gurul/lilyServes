"""lilyMemory facade — what the rest of lilyServes talks to.

remember() writes instantly and computes the embedding in the background;
recall() fuses semantic similarity, lexical match, importance, and recency
into one ranked list. Both are safe on the call hot path: no network wait
on write, bounded (and optional) embedding wait on read.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time

from lilyMemory.embeddings import NullEmbedder, OpenAIEmbedder
from lilyMemory.models import Memory, MemoryType
from lilyMemory.store import MemoryDB
from memory_store import redact

log = logging.getLogger("lily.memory")

# Rank fusion weights — semantic first, lexical second, then salience:
# how much a memory matters and how fresh it is.
W_SEMANTIC = 0.50
W_LEXICAL = 0.25
W_IMPORTANCE = 0.15
W_RECENCY = 0.10
RECENCY_HALF_LIFE_DAYS = 30.0

# A memory recalled for a caller-context card must clear this floor so the
# dashboard shows genuinely related context, not the least-bad match.
MIN_SCORE = 0.35


class MemoryService:
    def __init__(self, data_dir: str, embedder=None) -> None:
        self.db = MemoryDB(data_dir)
        self.embedder = embedder if embedder is not None else OpenAIEmbedder()
        self._embed_tasks: set[asyncio.Task] = set()

    @classmethod
    def lexical_only(cls, data_dir: str) -> MemoryService:
        return cls(data_dir, embedder=NullEmbedder())

    async def open(self) -> None:
        await self.db.open()

    async def close(self) -> None:
        if self._embed_tasks:
            await asyncio.gather(*self._embed_tasks, return_exceptions=True)
        await self.db.close()

    # ----- write path --------------------------------------------------------

    async def remember(
        self,
        content: str,
        memory_type: str = MemoryType.EPISODE,
        entities: list[str] | None = None,
        topics: list[str] | None = None,
        importance: float = 0.5,
        confidence: float = 0.8,
        caller: str = "",
        call_sid: str = "",
    ) -> Memory:
        """Store a memory. Returns immediately; the embedding lands async."""
        memory = Memory(
            content=redact(content.strip()),
            memory_type=memory_type if memory_type in MemoryType.ALL else MemoryType.EPISODE,
            entities=[e for e in (entities or []) if e],
            topics=[t for t in (topics or []) if t],
            importance=max(0.0, min(1.0, importance)),
            confidence=max(0.0, min(1.0, confidence)),
            caller=caller,
            call_sid=call_sid,
        )
        await self.db.insert(memory)
        task = asyncio.get_running_loop().create_task(self._embed_and_store(memory))
        self._embed_tasks.add(task)
        task.add_done_callback(self._embed_tasks.discard)
        log.info("remembered %s memory %s", memory.memory_type, memory.id[:8])
        return memory

    async def _embed_and_store(self, memory: Memory) -> None:
        vectors = await self.embedder.embed([memory.content])
        if vectors:
            await self.db.set_embedding(memory.id, vectors[0])

    async def forget(self, memory_id: str) -> bool:
        """Delete a memory — "You own your data. Always." """
        return await self.db.delete(memory_id)

    # ----- read path ---------------------------------------------------------

    async def recall(
        self,
        query: str,
        limit: int = 5,
        memory_types: list[str] | None = None,
        caller: str = "",
        min_score: float = 0.0,
    ) -> list[dict]:
        """Hybrid recall: cosine + BM25 + importance + recency, fused."""
        if not query.strip():
            return []
        candidate_limit = max(limit * 5, 25)

        query_vec = None
        vectors = await self.embedder.embed([query], timeout=3.0)
        if vectors:
            query_vec = vectors[0]

        lexical = dict(await self.db.lexical_search(query, candidate_limit))
        semantic: dict[str, float] = {}
        if query_vec is not None:
            semantic = dict(await self.db.vector_search(query_vec, candidate_limit))

        candidates = set(lexical) | set(semantic)
        if not candidates:
            return []

        memories = await self.db.get_many(list(candidates))
        now = time.time()
        ranked: list[tuple[float, Memory]] = []
        for mid, memory in memories.items():
            if memory_types and memory.memory_type not in memory_types:
                continue
            if caller and memory.caller and memory.caller != caller:
                continue
            age_days = max(0.0, (now - memory.created_at) / 86400.0)
            recency = math.exp(-math.log(2) * age_days / RECENCY_HALF_LIFE_DAYS)
            score = (
                W_SEMANTIC * semantic.get(mid, 0.0)
                + W_LEXICAL * lexical.get(mid, 0.0)
                + W_IMPORTANCE * memory.importance
                + W_RECENCY * recency
            )
            if score >= min_score:
                ranked.append((score, memory))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        top = ranked[:limit]
        await self.db.mark_recalled([m.id for _, m in top], now)
        return [m.to_dict(score=s) for s, m in top]

    async def recent(self, limit: int = 50, memory_type: str = "", caller: str = "") -> list[dict]:
        return [m.to_dict() for m in await self.db.recent(limit, memory_type, caller)]

    async def count(self) -> int:
        return await self.db.count()

    # ----- product-shaped helpers -------------------------------------------

    async def caller_context(self, caller: str, name: str = "") -> list[dict]:
        """Memories to surface when this caller rings — Cognitive Continuity."""
        query = " ".join(filter(None, [name, caller, "calls commitments history"]))
        return await self.recall(query, limit=5, caller=caller, min_score=MIN_SCORE)

    async def remember_call(
        self, call_sid: str, caller: str, summary: str, scam_level: str
    ) -> Memory | None:
        if not summary.strip():
            return None
        importance = {"Low": 0.4, "Medium": 0.7, "High": 0.9}.get(scam_level, 0.5)
        memory_type = MemoryType.SAFETY if scam_level in ("Medium", "High") else MemoryType.EPISODE
        return await self.remember(
            summary,
            memory_type=memory_type,
            entities=[caller] if caller and caller != "unknown" else [],
            topics=["phone_call"] + (["scam_risk"] if scam_level != "Low" else []),
            importance=importance,
            caller=caller,
            call_sid=call_sid,
        )

    async def remember_commitment(
        self, call_sid: str, caller: str, title: str, when: str
    ) -> Memory:
        content = title + (f" — {when}" if when else "")
        return await self.remember(
            content,
            memory_type=MemoryType.COMMITMENT,
            entities=[caller] if caller and caller != "unknown" else [],
            topics=["commitment"],
            importance=0.8,
            caller=caller,
            call_sid=call_sid,
        )

    async def remember_person(self, number: str, name: str, note: str = "") -> Memory:
        content = f"{name} ({number})" + (f": {note}" if note else " is a trusted contact.")
        return await self.remember(
            content,
            memory_type=MemoryType.PERSON,
            entities=[name, number],
            topics=["contact"],
            importance=0.9,
            confidence=1.0,
            caller=number,
        )
