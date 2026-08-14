"""SQLite persistence for lily.memory.

Same event-loop discipline as the rest of lilyServes: WAL mode, every DB
operation runs on a worker thread behind a lock. FTS5 provides lexical
(BM25) retrieval with an automatic LIKE fallback when the SQLite build
lacks FTS5. Embeddings live in a float32 BLOB column and are mirrored in
an in-process numpy matrix so cosine search never touches the disk.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3

import numpy as np

from lily.memory.embeddings import pack_vector
from lily.memory.models import Memory

log = logging.getLogger("lily.memory.store")

# Saturation constant for absolute BM25 normalization: score = raw / (raw + K).
# Absolute (not min-max) so a single strong hit scores high and scores are
# comparable across queries.
BM25_K = 2.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    memory_type  TEXT NOT NULL,
    content      TEXT NOT NULL,
    entities     TEXT DEFAULT '[]',
    topics       TEXT DEFAULT '[]',
    importance   REAL DEFAULT 0.5,
    confidence   REAL DEFAULT 0.8,
    caller       TEXT DEFAULT '',
    call_sid     TEXT DEFAULT '',
    created_at   REAL,
    last_recalled REAL DEFAULT 0,
    recall_count INTEGER DEFAULT 0,
    embedding    BLOB
);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_caller ON memories(caller);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, entities, topics,
    content='memories', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, entities, topics)
    VALUES (new.rowid, new.content, new.entities, new.topics);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, entities, topics)
    VALUES ('delete', old.rowid, old.content, old.entities, old.topics);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content, entities, topics ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, entities, topics)
    VALUES ('delete', old.rowid, old.content, old.entities, old.topics);
    INSERT INTO memories_fts(rowid, content, entities, topics)
    VALUES (new.rowid, new.content, new.entities, new.topics);
END;
"""


class MemoryDB:
    def __init__(self, data_dir: str) -> None:
        os.makedirs(data_dir, exist_ok=True)
        self._path = os.path.join(data_dir, "lily_memory.db")
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None
        self._closed = False
        self.fts_enabled = False
        # In-process vector index: parallel lists kept in insert order.
        self._vec_ids: list[str] = []
        self._vectors: np.ndarray | None = None  # unit-normalized rows

    # ----- lifecycle ---------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        try:
            conn.executescript(_FTS_SCHEMA)
            self.fts_enabled = True
            # Backfill rows written while FTS was unavailable (external-content
            # tables only index through the triggers, so a downgraded build
            # leaves gaps).
            missing = conn.execute(
                "SELECT (SELECT count(*) FROM memories) - (SELECT count(*) FROM memories_fts)"
            ).fetchone()[0]
            if missing:
                conn.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")
                conn.commit()
                log.info("rebuilt FTS index (%d rows were unindexed)", missing)
        except sqlite3.OperationalError as e:
            log.warning("FTS5 unavailable (%s) — lexical search degrades to LIKE", e)
            self.fts_enabled = False
            # The content triggers reference memories_fts; without FTS5 they
            # would break every write, so drop them (plain DDL, no fts needed).
            for trigger in ("memories_ai", "memories_ad", "memories_au"):
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.commit()
        return conn

    def _load_vectors(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()
        ids, mats, bad = [], [], []
        dim = None
        for row in rows:
            blob = row["embedding"]
            if not isinstance(blob, bytes) or len(blob) == 0 or len(blob) % 4:
                bad.append(row["id"])
                continue
            try:
                vec = np.frombuffer(blob, dtype=np.float32)
            except ValueError:
                bad.append(row["id"])
                continue
            if dim is None:
                dim = vec.shape[0]
            norm = np.linalg.norm(vec)
            if vec.shape[0] != dim or not np.isfinite(norm) or norm == 0:
                bad.append(row["id"])
                continue
            ids.append(row["id"])
            mats.append(vec / norm)
        if bad:
            # NULL the unreadable blobs so the rows lazily re-embed instead of
            # poisoning every future load.
            conn.executemany(
                "UPDATE memories SET embedding = NULL WHERE id = ?", [(b,) for b in bad]
            )
            conn.commit()
            log.warning("dropped %d unreadable embeddings; they will re-embed", len(bad))
        self._vec_ids = ids
        self._vectors = np.vstack(mats) if mats else None
        log.info("loaded %d memory vectors", len(ids))

    async def open(self) -> None:
        def setup():
            conn = self._connect()
            self._load_vectors(conn)
            return conn

        self._conn = await asyncio.to_thread(setup)
        log.info("lily.memory open at %s (fts=%s)", self._path, self.fts_enabled)

    async def close(self) -> None:
        # Take the lock so no query can be mid-flight on the connection.
        async with self._lock:
            if self._conn is not None:
                await asyncio.to_thread(self._conn.close)
                self._conn = None
            self._closed = True

    def _conn_or_raise(self) -> sqlite3.Connection:
        if self._closed or self._conn is None:
            raise RuntimeError("memory store closed")
        return self._conn

    async def _run(self, fn):
        async with self._lock:
            return await asyncio.to_thread(fn, self._conn_or_raise())

    # ----- writes ------------------------------------------------------------

    async def insert(self, memory: Memory) -> None:
        def w(conn: sqlite3.Connection):
            conn.execute(
                "INSERT INTO memories (id, memory_type, content, entities, topics, "
                "importance, confidence, caller, call_sid, created_at, last_recalled, "
                "recall_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                memory.to_row(),
            )
            conn.commit()

        await self._run(w)

    async def set_embedding(self, memory_id: str, vector: list[float]) -> None:
        blob = pack_vector(vector)

        def w(conn: sqlite3.Connection):
            cur = conn.execute(
                "UPDATE memories SET embedding = ? WHERE id = ?", (blob, memory_id)
            )
            conn.commit()
            return cur.rowcount

        vec = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        # DB write and index mutation share one critical section so a
        # concurrent delete() can never leave a ghost vector in RAM.
        async with self._lock:
            updated = await asyncio.to_thread(w, self._conn_or_raise())
            if not updated or norm == 0:
                return  # memory deleted before the embedding landed, or degenerate
            vec = vec / norm
            if self._vectors is not None and vec.shape[0] != self._vectors.shape[1]:
                log.warning(
                    "embedding dim %d != index dim %d — stored to DB only",
                    vec.shape[0], self._vectors.shape[1],
                )
                return
            if memory_id in self._vec_ids:
                # Copy-on-write keeps vector_search's matrix snapshot immutable.
                replaced = self._vectors.copy()
                replaced[self._vec_ids.index(memory_id)] = vec
                self._vectors = replaced
            elif self._vectors is None:
                self._vec_ids.append(memory_id)
                self._vectors = vec.reshape(1, -1)
            else:
                self._vec_ids.append(memory_id)
                self._vectors = np.vstack([self._vectors, vec])

    def _evict_vector_locked(self, memory_id: str) -> None:
        """Remove a memory's row from the in-process index. Caller holds _lock."""
        if memory_id in self._vec_ids:
            idx = self._vec_ids.index(memory_id)
            self._vec_ids.pop(idx)
            if self._vectors is not None:
                self._vectors = np.delete(self._vectors, idx, axis=0)
                if self._vectors.shape[0] == 0:
                    self._vectors = None

    async def delete(self, memory_id: str) -> bool:
        def w(conn: sqlite3.Connection):
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cur.rowcount > 0

        async with self._lock:
            deleted = await asyncio.to_thread(w, self._conn_or_raise())
            if deleted:
                self._evict_vector_locked(memory_id)
        return deleted

    async def mark_recalled(self, memory_ids: list[str], ts: float) -> None:
        if not memory_ids:
            return

        def w(conn: sqlite3.Connection):
            conn.executemany(
                "UPDATE memories SET last_recalled = ?, recall_count = recall_count + 1 "
                "WHERE id = ?",
                [(ts, mid) for mid in memory_ids],
            )
            conn.commit()

        await self._run(w)

    # ----- reads -------------------------------------------------------------

    async def get(self, memory_id: str) -> Memory | None:
        def q(conn: sqlite3.Connection):
            return conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()

        row = await self._run(q)
        return Memory.from_row(row) if row else None

    async def get_many(self, memory_ids: list[str]) -> dict[str, Memory]:
        if not memory_ids:
            return {}

        def q(conn: sqlite3.Connection):
            marks = ",".join("?" * len(memory_ids))
            return conn.execute(
                f"SELECT * FROM memories WHERE id IN ({marks})", memory_ids
            ).fetchall()

        rows = await self._run(q)
        return {row["id"]: Memory.from_row(row) for row in rows}

    async def recent(self, limit: int = 50, memory_type: str = "", caller: str = "") -> list[Memory]:
        def q(conn: sqlite3.Connection):
            sql = "SELECT * FROM memories"
            clauses, params = [], []
            if memory_type:
                clauses.append("memory_type = ?")
                params.append(memory_type)
            if caller:
                clauses.append("caller = ?")
                params.append(caller)
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            return conn.execute(sql, params).fetchall()

        rows = await self._run(q)
        return [Memory.from_row(row) for row in rows]

    async def count(self) -> int:
        def q(conn: sqlite3.Connection):
            return conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]

        return await self._run(q)

    async def missing_embedding_ids(self, limit: int = 50) -> list[str]:
        """Rows written while the embedder was down — candidates for healing."""
        def q(conn: sqlite3.Connection):
            return [
                r["id"] for r in conn.execute(
                    "SELECT id FROM memories WHERE embedding IS NULL LIMIT ?", (limit,)
                ).fetchall()
            ]

        return await self._run(q)

    # ----- lexical search ----------------------------------------------------

    @staticmethod
    def _fts_quote(query: str) -> str:
        """Quote each term so user text can't hit FTS5 query syntax."""
        terms = [t.replace('"', "") for t in query.split()]
        return " OR ".join(f'"{t}"' for t in terms if t)

    async def lexical_search(self, query: str, limit: int = 25) -> list[tuple[str, float]]:
        """Return (memory_id, lexical_score 0..1) — best match first."""
        if not query.strip():
            return []

        if self.fts_enabled:
            match = self._fts_quote(query)
            if not match:
                return []

            def q(conn: sqlite3.Connection):
                return conn.execute(
                    "SELECT m.id, m.content, m.entities, m.topics, "
                    "bm25(memories_fts) AS rank "
                    "FROM memories_fts JOIN memories m ON m.rowid = memories_fts.rowid "
                    "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
                    (match, limit),
                ).fetchall()

            try:
                rows = await self._run(q)
            except sqlite3.OperationalError as e:
                log.warning("FTS query failed (%s)", e)
                return []
            # Absolute scoring, comparable across queries and corpus sizes:
            # term coverage is the anchor (BM25's IDF collapses to ~0 on small
            # corpora where every doc matches), saturated bm25 adds
            # discrimination between matches at scale.
            terms = [t.replace('"', "").lower() for t in query.split() if t]
            out = []
            for row in rows:
                raw = max(0.0, -row["rank"])
                saturated = raw / (raw + BM25_K)
                haystack = " ".join(
                    (row["content"], row["entities"] or "", row["topics"] or "")
                ).lower()
                coverage = sum(1 for t in terms if t in haystack) / len(terms)
                out.append((row["id"], 0.6 * coverage + 0.4 * saturated))
            out.sort(key=lambda pair: pair[1], reverse=True)
            return out

        # LIKE fallback: fraction of query terms present.
        terms = [t.lower() for t in query.split() if t]
        if not terms:
            return []

        def q_like(conn: sqlite3.Connection):
            where = " OR ".join("lower(content) LIKE ?" for _ in terms)
            params = ["%" + t + "%" for t in terms]
            params.append(limit * 4)
            return conn.execute(
                f"SELECT id, content FROM memories WHERE {where} LIMIT ?", params
            ).fetchall()

        rows = await self._run(q_like)
        scored = []
        for row in rows:
            content = row["content"].lower()
            hit = sum(1 for t in terms if t in content) / len(terms)
            if hit > 0:
                scored.append((row["id"], hit))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:limit]

    # ----- semantic search ---------------------------------------------------

    async def vector_search(self, query_vec: list[float], limit: int = 25) -> list[tuple[str, float]]:
        """Return (memory_id, raw cosine -1..1) — best match first."""
        async with self._lock:
            if self._vectors is None:
                return []
            ids = list(self._vec_ids)
            matrix = self._vectors

        vec = np.asarray(query_vec, dtype=np.float32)
        if matrix.shape[1] != vec.shape[0]:
            return []
        norm = np.linalg.norm(vec)
        if norm == 0:
            return []
        sims = matrix @ (vec / norm)  # rows are unit vectors -> cosine
        top = np.argsort(sims)[::-1][:limit]
        return [(ids[i], float(sims[i])) for i in top]
