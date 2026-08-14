"""Memory lifecycle for lily.memory — the layer that keeps memory honest.

Adapted from what makes SOTA memory engines work, sized for heyLily:
- EXPIRY: "pickup on Friday" is noise on Saturday. Each memory type has a
  deterministic expiry policy; commitment due dates are resolved from their
  when-text with a small regex resolver (no LLM).
- DEDUPE-REINFORCE: a repeated observation strengthens the existing memory
  (importance up, source_count up) instead of piling up near-duplicates.
- SUPERSESSION: "pickup moved to Saturday" retires "pickup on Friday" —
  the newest fact wins, the old one stays auditable but stops surfacing.

Everything here is pure Python + SQLite; it runs at the end of the async
embed task, off the call hot path, and degrades to lexical-only signals
when embeddings are unavailable.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time

from lily.memory.models import Memory, MemoryStatus, MemoryType

log = logging.getLogger("lily.memory.lifecycle")

# Twilio collapses every withheld caller ID onto shared constants; never treat
# those as an identity worth superseding facts across.
ANON_CALLERS = {"unknown", "anonymous"}

DAY = 86400.0

# Expiry after the resolved due time, so "Friday 4 PM" stays visible through
# Friday evening (people run late).
COMMITMENT_GRACE = 1 * DAY
# A commitment whose when-text we can't resolve still shouldn't live forever.
COMMITMENT_FALLBACK_TTL = 14 * DAY
# Low-importance episodic chatter fades out; important episodes are kept.
EPISODE_TTL = 90 * DAY
EPISODE_TTL_IMPORTANCE_FLOOR = 0.5

# Dedupe/supersession bands. Above DUP_COSINE (or DUP_JACCARD lexically) two
# memories say the same thing; the same-subject band below that, combined
# with a shared-entity guard, marks a changed fact.
DUP_COSINE = 0.92
DUP_JACCARD = 0.6
SUPERSEDE_COSINE = 0.70
SUPERSEDE_JACCARD = 0.25
# Types where a newer fact replaces an older one. Histories (EPISODE, WIN,
# SAFETY) are never contradictions — they accumulate.
SUPERSEDABLE = (MemoryType.PERSON, MemoryType.PREFERENCE, MemoryType.COMMITMENT)

_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
_MONTHS = ("january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december")

_RE_TIME = re.compile(r"\bat\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE)
_RE_MONTH_DAY = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})\b", re.IGNORECASE
)
_RE_NUMERIC_DATE = re.compile(r"\b(\d{1,2})/(\d{1,2})\b")


def resolve_when(when_text: str, ref_ts: float | None = None) -> float | None:
    """Resolve informal when-text ("Friday at 4 pm", "tomorrow", "March 5")
    to epoch seconds, relative to ref_ts. None when nothing resolves."""
    if not when_text or not when_text.strip():
        return None
    now = time.localtime(ref_ts if ref_ts is not None else time.time())
    ref = time.mktime(now)
    text = when_text.lower()

    day_ts: float | None = None
    from_weekday = False
    if "today" in text or "tonight" in text:
        day_ts = ref
    elif "tomorrow" in text:
        day_ts = ref + DAY
    elif "next week" in text:
        day_ts = ref + 7 * DAY
    else:
        for i, name in enumerate(_WEEKDAYS):
            if name in text or f" {name[:3]} " in f" {text} ":
                # 0 = today: "Friday at 4 pm" said Friday morning means today;
                # the past-time check below rolls it forward when it's gone by.
                ahead = (i - now.tm_wday) % 7
                day_ts = ref + ahead * DAY
                from_weekday = True
                break
        if day_ts is None:
            m = _RE_MONTH_DAY.search(text)
            if m:
                month, day = _MONTHS.index(m.group(1).lower()) + 1, int(m.group(2))
                day_ts = _next_date(now, month, day)
            else:
                m = _RE_NUMERIC_DATE.search(text)
                if m:
                    month, day = int(m.group(1)), int(m.group(2))
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        day_ts = _next_date(now, month, day)

    if day_ts is None:
        return None

    # Default to end of day; an explicit "at H[:MM] [am/pm]" refines it.
    day = time.localtime(day_ts)
    hour, minute = 23, 59
    m = _RE_TIME.search(text)
    if m:
        hour = int(m.group(1)) % 12 if m.group(3) else int(m.group(1))
        if m.group(3) and m.group(3).lower() == "pm":
            hour += 12
        minute = int(m.group(2) or 0)
    try:
        ts = time.mktime((day.tm_year, day.tm_mon, day.tm_mday,
                          hour, minute, 0, -1, -1, -1))
    except (ValueError, OverflowError):
        return None
    if from_weekday and ts < ref - 300:
        # A same-day weekday mention whose time already passed means next week.
        rolled = time.localtime(day_ts + 7 * DAY)
        try:
            ts = time.mktime((rolled.tm_year, rolled.tm_mon, rolled.tm_mday,
                              hour, minute, 0, -1, -1, -1))
        except (ValueError, OverflowError):
            return None
    return ts


def _next_date(now: time.struct_time, month: int, day: int) -> float | None:
    """The next occurrence of month/day, rolling to next year when passed."""
    for year in (now.tm_year, now.tm_year + 1):
        try:
            ts = time.mktime((year, month, day, 0, 0, 0, -1, -1, -1))
        except (ValueError, OverflowError):
            return None
        if ts >= time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, -1, -1, -1)):
            return ts
    return None


def default_expiry(memory_type: str, importance: float, created_at: float,
                   due_ts: float | None = None) -> float:
    """When this memory stops being true enough to surface. 0 = never."""
    if memory_type == MemoryType.COMMITMENT:
        if due_ts:
            return due_ts + COMMITMENT_GRACE
        return created_at + COMMITMENT_FALLBACK_TTL
    if memory_type == MemoryType.EPISODE and importance < EPISODE_TTL_IMPORTANCE_FLOOR:
        return created_at + EPISODE_TTL
    return 0.0


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = text.lower().split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return set(zip(*(words[i:] for i in range(n))))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _token_jaccard(a: str, b: str) -> float:
    return _jaccard(set(a.lower().split()), set(b.lower().split()))


def _shares_subject(new: Memory, old: Memory) -> bool:
    """Entity overlap (or strong token overlap) — the guard that keeps
    embedding false-positives from superseding unrelated facts."""
    if set(e.lower() for e in new.entities) & set(e.lower() for e in old.entities):
        return True
    return _token_jaccard(new.content, old.content) >= 0.4


def _commitment_titles_match(new: Memory, old: Memory) -> bool:
    """Compare only the title portion (before the em-dash date suffix) so a
    reschedule of the same errand doesn't count as an exact duplicate."""
    title_new = new.content.split("—")[0]
    title_old = old.content.split("—")[0]
    return _token_jaccard(title_new, title_old) >= 0.4


def _commitment_whens_match(new: Memory, old: Memory) -> bool:
    """True when two commitments carry the same schedule — same when-text and
    same resolved expiry. A changed schedule is a reschedule, not a duplicate."""
    when_new = new.content.split("—", 1)[1] if "—" in new.content else ""
    when_old = old.content.split("—", 1)[1] if "—" in old.content else ""
    if when_new.strip().lower() != when_old.strip().lower():
        return False
    return abs(new.expires_at - old.expires_at) < DAY


async def _fire(hook, *args) -> None:
    """Invoke an optional integration hook; sync or async, never raises."""
    if hook is None:
        return
    try:
        result = hook(*args)
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        log.exception("lifecycle hook failed")


async def maintain(service, memory: Memory, vector: list[float] | None) -> None:
    """Post-write hygiene: reinforce duplicates, supersede changed facts.

    Runs after the embedding lands (or immediately, lexical-only, under
    NullEmbedder). `service` is the owning MemoryService.
    """
    fresh = await service.db.get(memory.id)
    if fresh is None or fresh.status != MemoryStatus.ACTIVE:
        return  # deleted or already retired while the embedding was in flight

    # Only strictly older memories are merge/supersede targets: a consistent
    # direction means concurrent maintenance can never mutually annihilate a
    # pair of duplicates.
    candidates = [
        m for m in await service.db.recent(40, memory.memory_type, memory.caller)
        if (m.created_at, m.id) < (memory.created_at, memory.id)
        and (memory.created_at - m.created_at) < 30 * DAY
    ]
    if not candidates:
        return

    cosines: dict[str, float] = {}
    if vector is not None:
        cosines = await service.db.similarity_to([m.id for m in candidates], vector)

    new_shingles = _shingles(memory.content)
    is_commitment = memory.memory_type == MemoryType.COMMITMENT
    for old in candidates:
        cos = cosines.get(old.id, 0.0)
        sh_jac = _jaccard(new_shingles, _shingles(old.content))

        duplicate = cos >= DUP_COSINE or sh_jac >= DUP_JACCARD
        if duplicate and is_commitment:
            if not _commitment_titles_match(memory, old):
                duplicate = False  # similar prose, different errand
            elif not _commitment_whens_match(memory, old):
                # Same errand, new schedule: a reschedule supersedes — merging
                # would silently discard the new due date.
                if await service.db.set_status(old.id, MemoryStatus.SUPERSEDED, memory.id):
                    await _fire(service.on_commitment_retired, old.id)
                continue
        if duplicate:
            # Atomic reinforce+delete; both rows re-checked inside the write.
            if await service.db.merge_into(old.id, memory, time.time()):
                if is_commitment:
                    await _fire(service.on_commitment_merged, memory.id, old.id)
                return  # the new row merged into the old — nothing left to do
            continue

        if memory.memory_type not in SUPERSEDABLE:
            continue
        if memory.caller.lower() in ANON_CALLERS:
            continue  # shared anonymous bucket is not an identity
        same_subject_band = (
            SUPERSEDE_COSINE <= cos < DUP_COSINE
            or SUPERSEDE_JACCARD <= sh_jac < DUP_JACCARD
        )
        if same_subject_band and _shares_subject(memory, old):
            if await service.db.set_status(old.id, MemoryStatus.SUPERSEDED, memory.id):
                if is_commitment:
                    await _fire(service.on_commitment_retired, old.id)
