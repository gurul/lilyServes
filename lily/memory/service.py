"""lily.memory facade — what the rest of lilyServes talks to.

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

from lily.memory import lifecycle
from lily.memory.embeddings import NullEmbedder, OpenAIEmbedder
from lily.memory.models import Memory, MemoryStatus, MemoryType
from lily.memory.operational import redact
from lily.memory.store import MemoryDB

log = logging.getLogger("lily.memory")

# Rank fusion — relevance first, salience as a bounded multiplier.
# Semantic evidence outweighs lexical 2:1; importance and recency can shade a
# score but can never buy an irrelevant memory past the floor.
W_SEMANTIC = 0.50
W_LEXICAL = 0.25

# Recency half-life per memory type, in days. None = stable facts never fade.
# COMMITMENT is short (what matters is what's coming up), SAFETY is long
# because scammers recycle numbers months later.
RECENCY_HALF_LIFE = {
    MemoryType.EPISODE: 30.0,
    MemoryType.COMMITMENT: 10.0,
    MemoryType.WIN: 90.0,
    MemoryType.SAFETY: 365.0,
    MemoryType.PERSON: None,
    MemoryType.PREFERENCE: None,
}
RECENCY_DEFAULT_HALF_LIFE = 30.0
RECENCY_FLOOR = 0.05

# MMR result diversification: near-duplicates above DUP_SIM never co-occur in
# one result set; below that, relevance trades off against redundancy.
MMR_LAMBDA = 0.75
DUP_SIM = 0.97

# Lazy maintenance sweep cadence and how long retired rows stay auditable
# before they are purged from disk.
SWEEP_INTERVAL = 3600.0
RETENTION_DAYS = 180.0
# A commitment's salience peaks at its due date and falls off over ±3 days.
URGENCY_SCALE_DAYS = 3.0

# Cosine calibration anchors for text-embedding-3-small@512: below the floor
# is noise, above the ceiling is a near-duplicate.
COS_FLOOR = 0.25
COS_CEIL = 0.75
# Minimum relevance evidence before salience is even considered.
REL_FLOOR = 0.15

# A memory recalled for a caller-context card must clear this floor so the
# dashboard shows genuinely related context, not the least-bad match.
MIN_SCORE = 0.35


class MemoryService:
    def __init__(self, data_dir: str, embedder=None) -> None:
        self.db = MemoryDB(data_dir)
        self.embedder = embedder if embedder is not None else OpenAIEmbedder()
        self._bg_tasks: set[asyncio.Task] = set()
        self._closing = False
        self._last_sweep = 0.0

    @classmethod
    def lexical_only(cls, data_dir: str) -> MemoryService:
        return cls(data_dir, embedder=NullEmbedder())

    def _track(self, coro) -> None:
        """Run a background coroutine, tracked so close() can drain it."""
        if self._closing:
            coro.close()
            return
        task = asyncio.get_running_loop().create_task(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def open(self) -> None:
        await self.db.open()
        await self._sweep(time.time())
        # Heal rows written while the embedder was down.
        missing = await self.db.missing_embedding_ids()
        if missing:
            memories = await self.db.get_many(missing)
            for memory in memories.values():
                self._track(self._embed_and_store(memory))

    async def _sweep(self, now: float) -> None:
        """Expire past-due memories and purge long-retired rows."""
        self._last_sweep = now
        expired = await self.db.expire_due(now)
        purged = await self.db.purge_dead(now - RETENTION_DAYS * 86400.0)
        if expired or purged:
            log.info("sweep: expired %d, purged %d", len(expired), purged)

    async def close(self) -> None:
        self._closing = True
        while self._bg_tasks:
            await asyncio.gather(*list(self._bg_tasks), return_exceptions=True)
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
        expires_at: float | None = None,
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
        memory.expires_at = (
            expires_at if expires_at is not None
            else lifecycle.default_expiry(memory.memory_type, memory.importance,
                                          memory.created_at)
        )
        await self.db.insert(memory)
        self._track(self._embed_and_store(memory))
        log.info("remembered %s memory %s", memory.memory_type, memory.id[:8])
        return memory

    async def _embed_and_store(self, memory: Memory) -> None:
        vectors = await self.embedder.embed([memory.content])
        if vectors:
            await self.db.set_embedding(memory.id, vectors[0])
        # Post-write hygiene runs off the hot path: reinforce duplicates,
        # supersede changed facts (lexical-only when embeddings are down).
        try:
            await lifecycle.maintain(self, memory, vectors[0] if vectors else None)
        except Exception:
            log.exception("lifecycle maintenance failed for %s", memory.id[:8])

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
        if time.time() - self._last_sweep > SWEEP_INTERVAL:
            self._last_sweep = time.time()
            self._track(self._sweep(self._last_sweep))
        candidate_limit = max(limit * 5, 25)

        query_vec = None
        vectors = await self.embedder.embed([query], timeout=3.0)
        if vectors:
            query_vec = vectors[0]

        lexical = dict(
            await self.db.lexical_search(query, candidate_limit, caller, memory_types)
        )
        semantic: dict[str, float] = {}
        if query_vec is not None:
            semantic = dict(
                await self.db.vector_search(query_vec, candidate_limit, caller, memory_types)
            )

        candidates = set(lexical) | set(semantic)
        if not candidates:
            return []

        memories = await self.db.get_many(list(candidates))
        now = time.time()
        ranked: list[tuple[float, Memory]] = []
        for mid, memory in memories.items():
            if memory.status != MemoryStatus.ACTIVE:
                continue
            lex = lexical.get(mid, 0.0)
            if query_vec is not None:
                cos = semantic.get(mid, 0.0)
                sem = min(1.0, max(0.0, (cos - COS_FLOOR) / (COS_CEIL - COS_FLOOR)))
                combined = (W_SEMANTIC * sem + W_LEXICAL * lex) / (W_SEMANTIC + W_LEXICAL)
                # Semantic corroboration boosts, but its absence (memory not
                # yet embedded, or embedder down at write time) only mildly
                # discounts a strong lexical match — never annihilates it.
                relevance = max(combined, 0.9 * lex)
            else:
                # Lexical-only mode carries full weight instead of being capped
                # at a fraction of the scale.
                relevance = lex
            if relevance < REL_FLOOR:
                continue
            score = relevance * (
                0.70 + 0.20 * memory.importance + 0.10 * self._recency(memory, now)
            )
            if score >= min_score:
                ranked.append((score, memory))

        ranked.sort(key=lambda pair: pair[0], reverse=True)
        top = await self._diversify(ranked, limit)
        if top:
            # Fire-and-forget: ring-time reads don't wait on this write.
            self._track(self.db.mark_recalled([m.id for _, m in top], now))
        return [m.to_dict(score=s) for s, m in top]

    @staticmethod
    def _recency(memory: Memory, now: float) -> float:
        """Type-aware freshness, anchored to the last recall so memories the
        user keeps coming back to stay warm."""
        if memory.memory_type == MemoryType.COMMITMENT and memory.expires_at > 0:
            # Urgency curve: salience peaks at the due date, not at creation.
            days_out = abs(memory.expires_at - now) / 86400.0
            return max(RECENCY_FLOOR, math.exp(-days_out / URGENCY_SCALE_DAYS))
        half = RECENCY_HALF_LIFE.get(memory.memory_type, RECENCY_DEFAULT_HALF_LIFE)
        if half is None:
            return 1.0
        anchor = max(memory.created_at, memory.last_recalled)
        age_days = max(0.0, (now - anchor) / 86400.0)
        return max(RECENCY_FLOOR, math.exp(-math.log(2) * age_days / half))

    async def _diversify(
        self, ranked: list[tuple[float, Memory]], limit: int
    ) -> list[tuple[float, Memory]]:
        """Greedy MMR over the top of the ranking: drop near-duplicates, favor
        results that add information over ones that repeat it."""
        if len(ranked) <= 1:
            return ranked[:limit]
        pool = ranked[: max(limit * 3, 15)]
        vectors = await self.db.get_vectors([m.id for _, m in pool])

        def sim(a: Memory, b: Memory) -> float:
            va, vb = vectors.get(a.id), vectors.get(b.id)
            if va is not None and vb is not None:
                return float(va @ vb)
            ta = set(a.content.lower().split())
            tb = set(b.content.lower().split())
            return len(ta & tb) / len(ta | tb) if ta and tb else 0.0

        selected: list[tuple[float, Memory]] = []
        remaining = list(pool)
        while remaining and len(selected) < limit:
            best_idx, best_mmr = None, -math.inf
            drop: set[int] = set()
            for i, (score, memory) in enumerate(remaining):
                max_sim = max((sim(memory, s) for _, s in selected), default=0.0)
                if max_sim > DUP_SIM:
                    drop.add(i)
                    continue
                mmr = MMR_LAMBDA * score - (1 - MMR_LAMBDA) * max_sim
                if mmr > best_mmr:
                    best_idx, best_mmr = i, mmr
            if best_idx is None:
                break
            selected.append(remaining[best_idx])
            drop.add(best_idx)
            remaining = [r for i, r in enumerate(remaining) if i not in drop]
        return selected

    async def recent(
        self,
        limit: int = 50,
        memory_type: str = "",
        caller: str = "",
        include_inactive: bool = False,
    ) -> list[dict]:
        return [
            m.to_dict()
            for m in await self.db.recent(limit, memory_type, caller, include_inactive)
        ]

    async def count(self) -> int:
        return await self.db.count()

    # ----- product-shaped helpers -------------------------------------------

    async def profile(self, caller: str, name: str = "") -> dict:
        """Caller profile card: stable facts + open commitments + recent
        history + safety record. One SQL round-trip, no embeddings, no LLM —
        instant on call start and identical under NullEmbedder."""
        now = time.time()
        rows = await self.db.profile_rows(caller, now)
        safety = rows["safety"]
        return {
            "caller": caller,
            "name": name,
            "facts": [m.to_dict() for m in rows["facts"]],
            "open_commitments": [m.to_dict() for m in rows["commitments"]],
            "recent": [
                {**m.to_dict(), "content": m.content[:200]} for m in rows["recent"]
            ],
            "safety": {
                "count": len(safety),
                "last": safety[0].content if safety else "",
                "last_at": safety[0].created_at if safety else 0,
            },
        }

    @staticmethod
    def render_profile(profile: dict, max_chars: int = 600) -> str:
        """Compact plain-text block ready for prompt injection."""
        lines = []
        who = " ".join(filter(None, [profile.get("name"), profile.get("caller")]))
        if who:
            lines.append(f"Caller: {who}")
        if profile["facts"]:
            lines.append("Known: " + "; ".join(f["content"] for f in profile["facts"][:4]))
        if profile["open_commitments"]:
            lines.append(
                "Open: " + "; ".join(c["content"] for c in profile["open_commitments"][:3])
            )
        if profile["recent"]:
            lines.append("Recently: " + profile["recent"][0]["content"])
        if profile["safety"]["count"]:
            lines.append(
                f"Safety: {profile['safety']['count']} past scam-risk call(s); "
                f"last: {profile['safety']['last'][:120]}"
            )
        text = "\n".join(lines)
        return text[:max_chars]

    async def caller_context(self, caller: str, name: str = "") -> list[dict]:
        """Memories to surface when this caller rings — Cognitive Continuity."""
        # Query on real signal only (who is calling); generic keyword padding
        # just drags every query toward the same region of embedding space.
        query = f"{name} {caller}".strip() or caller
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
        due = lifecycle.resolve_when(when)
        return await self.remember(
            content,
            memory_type=MemoryType.COMMITMENT,
            entities=[caller] if caller and caller != "unknown" else [],
            topics=["commitment"],
            importance=0.8,
            caller=caller,
            call_sid=call_sid,
            expires_at=(due + lifecycle.COMMITMENT_GRACE) if due else None,
        )

    async def complete_commitment(self, memory_id: str) -> bool:
        """A done commitment leaves recall and the profile card immediately."""
        return await self.db.set_status(memory_id, MemoryStatus.EXPIRED)

    async def remember_person(self, number: str, name: str, note: str = "") -> Memory:
        content = f"{name} ({number})" + (f": {note}" if note else " is a trusted contact.")
        memory = await self.remember(
            content,
            memory_type=MemoryType.PERSON,
            entities=[name, number],
            topics=["contact"],
            importance=0.9,
            confidence=1.0,
            caller=number,
        )
        # caller + PERSON is an identity key: re-trusting with a new name or
        # note deterministically replaces the previous fact.
        for old in await self.db.recent(20, MemoryType.PERSON, number):
            if old.id != memory.id:
                await self.db.set_status(old.id, MemoryStatus.SUPERSEDED, memory.id)
        return memory
