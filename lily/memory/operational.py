"""SQLite-backed memory: caller history, calls, events, activity feed.

This is the "Cognitive Continuity" layer — Lily remembers who called, what
was said, and what was promised, across restarts.

Design notes:
- WAL mode + a single dedicated writer thread (via asyncio.to_thread and a
  lock) keeps writes off the event loop without a heavyweight dependency.
- Privacy first: transcripts are only persisted when RETAIN_TRANSCRIPTS=true,
  and everything stored passes through redact() to scrub number sequences
  that look like SSNs / card numbers / verification codes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from typing import Any

log = logging.getLogger("lily.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS callers (
    number      TEXT PRIMARY KEY,
    name        TEXT DEFAULT '',
    trusted     INTEGER DEFAULT 0,
    scam_strikes INTEGER DEFAULT 0,
    call_count  INTEGER DEFAULT 0,
    first_seen  REAL,
    last_seen   REAL,
    last_summary TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS calls (
    call_sid    TEXT PRIMARY KEY,
    number      TEXT,
    started_at  REAL,
    ended_at    REAL,
    scam_level  TEXT DEFAULT 'Low',
    deepfake_score REAL,
    screened    INTEGER DEFAULT 0,
    summary_json TEXT DEFAULT '',
    transcript  TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    call_sid    TEXT,
    number      TEXT,
    title       TEXT,
    when_text   TEXT,
    created_at  REAL,
    done        INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS activity (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL,
    kind        TEXT,
    title       TEXT,
    detail      TEXT,
    importance  TEXT,
    call_sid    TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_number ON calls(number);
CREATE INDEX IF NOT EXISTS idx_activity_ts ON activity(ts);
CREATE INDEX IF NOT EXISTS idx_events_number ON events(number);
"""

# Redaction: long digit runs (cards, SSN with/without dashes, codes), plus
# context-keyed short secrets ("my PIN is 1234") that the length-based rules
# are too coarse to catch.
_RE_PIN = re.compile(
    r"(?i)\b(pin|code|otp|cvv|passcode|password|one[- ]time)\b"
    r"(\W{0,3}(?:is|was|:)?\W{0,3})(\d{3,8})\b"
)
_RE_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_RE_SSN = re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b")
_RE_CODE = re.compile(r"\b\d{5,8}\b")


def redact(text: str) -> str:
    """Scrub sequences that look like card numbers, SSNs, or one-time codes."""
    if not text:
        return text
    text = _RE_PIN.sub(lambda m: m.group(1) + m.group(2) + "[redacted code]", text)
    text = _RE_CARD.sub("[redacted number]", text)
    text = _RE_SSN.sub("[redacted number]", text)
    text = _RE_CODE.sub("[redacted code]", text)
    return text


class MemoryStore:
    def __init__(self, data_dir: str, retain_transcripts: bool = False) -> None:
        os.makedirs(data_dir, exist_ok=True)
        self._path = os.path.join(data_dir, "lily.db")
        self._retain_transcripts = retain_transcripts
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None
        self._closed = False

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        # events.memory_id links an event to its COMMITMENT memory twin so
        # completing one closes both. Guarded migration for existing DBs.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(events)")}
        if "memory_id" not in cols:
            conn.execute("ALTER TABLE events ADD COLUMN memory_id TEXT DEFAULT ''")
            conn.commit()
        return conn

    async def open(self) -> None:
        self._conn = await asyncio.to_thread(self._connect)
        log.info("memory store open at %s", self._path)

    async def close(self) -> None:
        # Take the lock so no query can be mid-flight on the connection.
        async with self._lock:
            if self._conn is not None:
                await asyncio.to_thread(self._conn.close)
                self._conn = None
            self._closed = True

    async def _run(self, fn) -> Any:
        """Serialize DB work onto a thread, one operation at a time."""
        async with self._lock:
            if self._closed or self._conn is None:
                raise RuntimeError("memory store closed")
            return await asyncio.to_thread(fn, self._conn)

    # ----- callers -----------------------------------------------------------

    async def caller_context(self, number: str) -> dict:
        """Everything Lily remembers about a caller, for gentle context."""
        def q(conn: sqlite3.Connection):
            caller = conn.execute(
                "SELECT * FROM callers WHERE number = ?", (number,)
            ).fetchone()
            events = conn.execute(
                "SELECT title, when_text FROM events WHERE number = ? AND done = 0 "
                "ORDER BY created_at DESC LIMIT 5",
                (number,),
            ).fetchall()
            return caller, events

        caller, events = await self._run(q)
        if caller is None:
            return {"known": False, "number": number}
        return {
            "known": True,
            "number": number,
            "name": caller["name"],
            "trusted": bool(caller["trusted"]),
            "scam_strikes": caller["scam_strikes"],
            "call_count": caller["call_count"],
            "last_seen": caller["last_seen"],
            "last_summary": caller["last_summary"],
            "open_events": [{"title": e["title"], "when": e["when_text"]} for e in events],
        }

    async def touch_caller(self, number: str) -> None:
        now = time.time()

        def w(conn: sqlite3.Connection):
            conn.execute(
                "INSERT INTO callers (number, first_seen, last_seen, call_count) "
                "VALUES (?, ?, ?, 1) "
                "ON CONFLICT(number) DO UPDATE SET last_seen = ?, call_count = call_count + 1",
                (number, now, now, now),
            )
            conn.commit()

        await self._run(w)

    async def set_trusted(self, number: str, trusted: bool, name: str = "") -> None:
        now = time.time()

        def w(conn: sqlite3.Connection):
            conn.execute(
                "INSERT INTO callers (number, name, trusted, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(number) DO UPDATE SET trusted = ?, name = CASE WHEN ? != '' THEN ? ELSE name END",
                (number, name, int(trusted), now, now, int(trusted), name, name),
            )
            conn.commit()

        await self._run(w)

    async def add_scam_strike(self, number: str) -> int:
        now = time.time()

        def w(conn: sqlite3.Connection):
            conn.execute(
                "INSERT INTO callers (number, first_seen, last_seen, scam_strikes) "
                "VALUES (?, ?, ?, 1) "
                "ON CONFLICT(number) DO UPDATE SET scam_strikes = scam_strikes + 1, last_seen = ?",
                (number, now, now, now),
            )
            conn.commit()
            row = conn.execute(
                "SELECT scam_strikes FROM callers WHERE number = ?", (number,)
            ).fetchone()
            return row["scam_strikes"] if row else 1

        return await self._run(w)

    # ----- calls -------------------------------------------------------------

    async def record_call_start(self, call_sid: str, number: str, screened: bool) -> None:
        def w(conn: sqlite3.Connection):
            conn.execute(
                "INSERT OR REPLACE INTO calls (call_sid, number, started_at, screened) "
                "VALUES (?, ?, ?, ?)",
                (call_sid, number, time.time(), int(screened)),
            )
            conn.commit()

        await self._run(w)

    async def record_call_end(
        self,
        call_sid: str,
        number: str,
        scam_level: str,
        deepfake_score: float | None,
        summary: dict,
        transcript_lines: list[str],
    ) -> None:
        summary_text = redact(json.dumps(summary, separators=(",", ":")))
        transcript = ""
        if self._retain_transcripts:
            transcript = redact("\n".join(transcript_lines))

        def w(conn: sqlite3.Connection):
            conn.execute(
                "UPDATE calls SET ended_at = ?, scam_level = ?, deepfake_score = ?, "
                "summary_json = ?, transcript = ? WHERE call_sid = ?",
                (time.time(), scam_level, deepfake_score, summary_text, transcript, call_sid),
            )
            conn.execute(
                "INSERT INTO callers (number, first_seen, last_seen, last_summary) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(number) DO UPDATE SET last_summary = excluded.last_summary",
                (number, time.time(), time.time(), redact(str(summary.get("summary", "")))),
            )
            conn.commit()

        await self._run(w)

    async def recent_calls(self, limit: int = 20) -> list[dict]:
        def q(conn: sqlite3.Connection):
            rows = conn.execute(
                "SELECT call_sid, number, started_at, ended_at, scam_level, "
                "deepfake_score, screened, summary_json FROM calls "
                "ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

        return await self._run(q)

    # ----- events (Memory Sync / Active Assistance) --------------------------

    async def add_event(
        self, call_sid: str, number: str, title: str, when_text: str, memory_id: str = ""
    ) -> int:
        def w(conn: sqlite3.Connection):
            cur = conn.execute(
                "INSERT INTO events (call_sid, number, title, when_text, created_at, "
                "memory_id) VALUES (?, ?, ?, ?, ?, ?)",
                (call_sid, number, redact(title), redact(when_text), time.time(), memory_id),
            )
            conn.commit()
            return cur.lastrowid

        return await self._run(w)

    async def events_for_call(self, call_sid: str) -> list[dict]:
        def q(conn: sqlite3.Connection):
            rows = conn.execute(
                "SELECT id, title, when_text, memory_id FROM events WHERE call_sid = ?",
                (call_sid,),
            ).fetchall()
            return [dict(r) for r in rows]

        return await self._run(q)

    async def open_events(self, limit: int = 50) -> list[dict]:
        def q(conn: sqlite3.Connection):
            rows = conn.execute(
                "SELECT id, call_sid, number, title, when_text, created_at "
                "FROM events WHERE done = 0 ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

        return await self._run(q)

    async def complete_event(self, event_id: int) -> str:
        """Mark an event done. Returns the linked memory id ('' if none) so
        the caller can retire the COMMITMENT memory twin."""
        def w(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT memory_id FROM events WHERE id = ?", (event_id,)
            ).fetchone()
            conn.execute("UPDATE events SET done = 1 WHERE id = ?", (event_id,))
            conn.commit()
            return row["memory_id"] if row else ""

        return await self._run(w)

    # ----- activity feed -----------------------------------------------------

    async def add_activity(
        self, kind: str, title: str, detail: str, importance: str, call_sid: str | None
    ) -> None:
        def w(conn: sqlite3.Connection):
            conn.execute(
                "INSERT INTO activity (ts, kind, title, detail, importance, call_sid) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (time.time(), kind, redact(title), redact(detail), importance, call_sid),
            )
            conn.commit()

        await self._run(w)

    async def recent_activity(self, limit: int = 50) -> list[dict]:
        def q(conn: sqlite3.Connection):
            rows = conn.execute(
                "SELECT ts, kind, title, detail, importance, call_sid FROM activity "
                "ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

        return await self._run(q)
