"""Regression tests for memory-engine defects: withheld-caller supersession,
unredacted entities/topics, redaction misses, forget() leaving the event twin
open, due-date parsing, merge-adopt dropping the vector, forgotten plaintext
on disk, and punctuated lexical queries."""
import asyncio
import glob
import json
import os
import sqlite3
import time

import pytest

from lily.memory import MemoryService, MemoryStatus, MemoryType, lifecycle
from lily.memory.operational import MemoryStore, redact


class TableEmbedder:
    """Fixed vector per text from a table; unknown text -> e0."""

    dimensions = 4

    def __init__(self, table=None):
        self.table = table or {}

    async def embed(self, texts, timeout=5.0):
        return [self.table.get(t, [1.0, 0.0, 0.0, 0.0]) for t in texts]


async def _drain(svc):
    while svc._bg_tasks:
        await asyncio.gather(*list(svc._bg_tasks), return_exceptions=True)


# ----- 1. withheld callers never merge into or supersede others' facts -------


@pytest.mark.parametrize("withheld", ["", "unknown", "Anonymous", "RESTRICTED"])
def test_withheld_caller_fact_does_not_supersede_other_callers_fact(tmp_path, withheld):
    async def run():
        svc = MemoryService.lexical_only(str(tmp_path))
        await svc.open()
        trusted = await svc.remember(
            "Tommy is her grandson and he lives in Ohio with his wife",
            MemoryType.PERSON, entities=["Tommy"], caller="+15550001111")
        await _drain(svc)
        await asyncio.sleep(0.01)
        await svc.remember(
            "Tommy is her grandson and he is in jail needing bail money",
            MemoryType.PERSON, entities=["Tommy"], caller=withheld)
        await _drain(svc)
        old = await svc.db.get(trusted.id)
        card = await svc.profile("+15550001111")
        await svc.close()
        assert old.status == MemoryStatus.ACTIVE and old.superseded_by == ""
        assert any("Ohio" in f["content"] for f in card["facts"])
    asyncio.run(run())


def test_withheld_caller_duplicate_does_not_merge_into_other_caller(tmp_path):
    async def run():
        svc = MemoryService(str(tmp_path), embedder=TableEmbedder())
        await svc.open()
        real = await svc.remember("Grandson Tommy visits on Sunday",
                                  MemoryType.EPISODE, caller="+15550001111")
        await _drain(svc)
        await asyncio.sleep(0.01)
        await svc.remember("Grandson Tommy visits on Sunday", MemoryType.EPISODE, caller="")
        await _drain(svc)
        assert (await svc.db.get(real.id)).source_count == 1
        assert await svc.count() == 2
        await svc.close()
    asyncio.run(run())


def test_is_withheld_caller():
    for c in ("", None, "unknown", "Anonymous", " restricted ", "private"):
        assert lifecycle.is_withheld_caller(c)
    assert not lifecycle.is_withheld_caller("+15550001111")


# ----- 2. entities and topics are redacted on write and merge ---------------


def test_entities_and_topics_redacted_on_write(tmp_path):
    async def run():
        svc = MemoryService.lexical_only(str(tmp_path))
        await svc.open()
        m = await svc.remember("Family note about the bank", MemoryType.PREFERENCE,
                               entities=["SSN 123-45-6789"],
                               topics=["card 4111 1111 1111 1111"])
        await _drain(svc)
        hits = await svc.db.lexical_search("6789", 5)
        await svc.close()
        row = sqlite3.connect(os.path.join(tmp_path, "lily_memory.db")).execute(
            "SELECT entities, topics FROM memories WHERE id=?", (m.id,)).fetchone()
        assert "6789" not in row[0] and "1111" not in row[1]
        assert hits == []
    asyncio.run(run())


def test_merge_redacts_legacy_entities(tmp_path):
    async def run():
        svc = MemoryService(str(tmp_path), embedder=TableEmbedder())
        await svc.open()
        old = await svc.remember("Pharmacy pickup is on Friday", MemoryType.EPISODE,
                                 caller="+15550001111")
        await _drain(svc)
        # A row written before entity redaction existed.
        conn = sqlite3.connect(os.path.join(tmp_path, "lily_memory.db"))
        conn.execute("UPDATE memories SET entities = ? WHERE id = ?",
                     (json.dumps(["card 4111 1111 1111 1111"]), old.id))
        conn.commit()
        conn.close()
        await asyncio.sleep(0.01)
        await svc.remember("Pharmacy pickup is on Friday", MemoryType.EPISODE,
                           caller="+15550001111")
        await _drain(svc)
        got = await svc.db.get(old.id)
        await svc.close()
        assert got.source_count == 2
        assert not any("1111" in e for e in got.entities)
    asyncio.run(run())


# ----- 3. redaction misses and false positives ------------------------------


@pytest.mark.parametrize("text", [
    "your code is 482 913",
    "the code is 482-913",
    "my PIN number is 4821",
    "card 4111.1111.1111.1111",
    "card 4111  1111  1111  1111",
    "social is 123.45.6789",
    "code is 4 8 2 9 1 3",
    "I read him 4 8 2 9 1 3 over the phone",
])
def test_redaction_catches_split_and_spoken_secrets(text):
    out = redact(text)
    assert not any(ch.isdigit() for ch in out), out


@pytest.mark.parametrize("text", [
    "call (206) 555-0100 tomorrow",
    "the appointment is 10/11/2026",
    "meet at 4:00",
    "it cost $12",
    "call 206 555 0100 or 206 555 0101",
])
def test_redaction_leaves_ordinary_numbers_alone(text):
    assert redact(text) == text


def test_separated_card_candidate_needs_luhn():
    # 16 digits, dot-separated, Luhn-invalid: not a card.
    assert redact("ref 1234.5678.1234.5678") == "ref 1234.5678.1234.5678"
    assert "4111" not in redact("ref 4111.1111.1111.1111")  # Luhn-valid


# ----- 4. forget() closes the linked event twin -----------------------------


def test_forget_closes_linked_event(tmp_path):
    async def run():
        svc = MemoryService.lexical_only(str(tmp_path))
        await svc.open()
        store = MemoryStore(str(tmp_path))
        await store.open()
        # Same wiring as lily/app.py.
        svc.on_commitment_retired = store.complete_event_by_memory
        svc.on_commitment_merged = store.remap_event_memory
        await store.touch_caller("+15550002222")
        mem = await svc.remember_commitment("CA1", "+15550002222",
                                            "Oncology appointment Dr Patel", "Friday at 3 pm")
        await store.add_event("CA1", "+15550002222", "Oncology appointment Dr Patel",
                              "Friday at 3 pm", memory_id=mem.id)
        await _drain(svc)
        assert (await store.caller_context("+15550002222"))["open_events"]
        assert await svc.forget(mem.id) is True
        ctx = await store.caller_context("+15550002222")
        events = await store.open_events()
        await store.close()
        await svc.close()
        assert ctx["open_events"] == []
        assert events == []
    asyncio.run(run())


def test_forget_missing_memory_fires_no_hook(tmp_path):
    async def run():
        svc = MemoryService.lexical_only(str(tmp_path))
        await svc.open()
        fired = []
        svc.on_commitment_retired = fired.append
        assert await svc.forget("no-such-id") is False
        await svc.close()
        assert fired == []
    asyncio.run(run())


# ----- 5. due-date parsing --------------------------------------------------


@pytest.fixture()
def new_york_tz():
    prev = os.environ.get("TZ")
    os.environ["TZ"] = "America/New_York"
    time.tzset()
    yield
    if prev is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = prev
    time.tzset()


def _ts(y, mo, d, h, mi):
    return time.mktime((y, mo, d, h, mi, 0, -1, -1, -1))


def _local(ts):
    t = time.localtime(ts)
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min)


def test_friday_next_week(new_york_tz):
    ref = _ts(2026, 10, 12, 10, 0)  # Monday
    due = lifecycle.resolve_when("Friday next week", ref)
    assert _local(due) == (2026, 10, 23, 23, 59)
    assert _local(lifecycle.resolve_when("next week", ref)) == (2026, 10, 19, 23, 59)


@pytest.mark.parametrize("text,expected", [
    ("Friday at 4 p.m.", (2026, 10, 16, 16, 0)),
    ("Friday at 4:00 P.M.", (2026, 10, 16, 16, 0)),
    ("Friday @ 4:00 PM", (2026, 10, 16, 16, 0)),
    ("Friday at 4", (2026, 10, 16, 16, 0)),  # bare 1-7 means afternoon
    ("Friday at 11", (2026, 10, 16, 11, 0)),
    ("Friday at 10 a.m.", (2026, 10, 16, 10, 0)),
    ("Friday at 12 pm", (2026, 10, 16, 12, 0)),
])
def test_time_phrases(new_york_tz, text, expected):
    ref = _ts(2026, 10, 16, 9, 0)  # Friday 09:00
    assert _local(lifecycle.resolve_when(text, ref)) == expected


def test_tomorrow_across_dst_spring_forward(new_york_tz):
    ref = _ts(2026, 3, 7, 23, 30)  # Saturday 23:30; DST starts Sunday Mar 8
    assert _local(lifecycle.resolve_when("tomorrow", ref)) == (2026, 3, 8, 23, 59)
    assert _local(lifecycle.resolve_when("Sunday at 3 pm", ref)) == (2026, 3, 8, 15, 0)


def test_weekday_across_dst_fall_back(new_york_tz):
    ref = _ts(2026, 10, 31, 23, 30)  # Saturday 23:30; DST ends Sunday Nov 1
    assert _local(lifecycle.resolve_when("Monday at 9 am", ref)) == (2026, 11, 2, 9, 0)


# ----- 6. merge-adopt keeps a vector; healer drains every page --------------


def test_merge_adopt_writes_new_vector_to_survivor(tmp_path):
    async def run():
        emb = TableEmbedder({
            "Susan likes tea": [1, 0, 0, 0],
            "Susan likes tea in the afternoon": [1, 0, 0, 0.01],
        })
        svc = MemoryService(str(tmp_path), embedder=emb)
        await svc.open()
        old = await svc.remember("Susan likes tea", MemoryType.PREFERENCE,
                                 caller="+15550003333")
        await _drain(svc)
        await asyncio.sleep(0.01)
        await svc.remember("Susan likes tea in the afternoon", MemoryType.PREFERENCE,
                           caller="+15550003333")
        await _drain(svc)
        row = await svc.db.get(old.id)
        hits = await svc.db.vector_search([1, 0, 0, 0], 10)
        missing = await svc.db.missing_embedding_ids()
        await svc.close()
        assert row.content == "Susan likes tea in the afternoon"
        assert row.source_count == 2
        assert old.id in [i for i, _ in hits]
        assert missing == []
    asyncio.run(run())


def test_open_heals_more_than_one_page(tmp_path):
    async def run():
        svc = MemoryService.lexical_only(str(tmp_path))
        await svc.open()
        for i in range(120):
            await svc.remember(f"distinct note number {i} about item{i}", MemoryType.WIN)
        await _drain(svc)
        await svc.close()
        svc2 = MemoryService(str(tmp_path), embedder=TableEmbedder())
        await svc2.open()
        await _drain(svc2)
        left = await svc2.db.missing_embedding_ids(1000)
        await svc2.close()
        assert left == []
    asyncio.run(run())


def test_heal_terminates_when_embedder_stays_down(tmp_path):
    async def run():
        svc = MemoryService.lexical_only(str(tmp_path))
        await svc.open()
        for i in range(120):
            await svc.remember(f"distinct note number {i} about item{i}", MemoryType.WIN)
        await _drain(svc)
        await svc.close()
        svc2 = MemoryService.lexical_only(str(tmp_path))  # embed() returns None
        await svc2.open()
        await asyncio.wait_for(_drain(svc2), timeout=5)
        await svc2.close()
    asyncio.run(run())


# ----- 7. forgotten text leaves no plaintext on disk ------------------------


def test_forget_leaves_no_plaintext_on_disk(tmp_path):
    async def run():
        svc = MemoryService.lexical_only(str(tmp_path))
        await svc.open()
        keep = await svc.remember("Ordinary note about lunch", MemoryType.EPISODE)
        m = await svc.remember("Zebulonquux secret diagnosis", MemoryType.EPISODE)
        await _drain(svc)
        assert await svc.forget(m.id)
        assert await svc.db.get(keep.id) is not None
        await svc.close()
    asyncio.run(run())
    files = glob.glob(os.path.join(tmp_path, "lily_memory.db*"))
    assert files
    for f in files:
        data = open(f, "rb").read()
        assert b"Zebulonquux" not in data and b"zebulonquux" not in data, f


def test_secure_delete_enabled(tmp_path):
    async def run():
        svc = MemoryService.lexical_only(str(tmp_path))
        await svc.open()
        on = await svc.db._run(lambda c: c.execute("PRAGMA secure_delete").fetchone()[0])
        await svc.close()
        assert on == 1
    asyncio.run(run())


# ----- 8. punctuated lexical queries ----------------------------------------


@pytest.mark.parametrize("query", ["pharmacy?", "pharmacy.", "pickup, pharmacy?",
                                   '"pharmacy"', "pharmacy!!"])
def test_punctuated_query_matches(tmp_path, query):
    async def run():
        svc = MemoryService.lexical_only(str(tmp_path))
        await svc.open()
        await svc.remember("Pharmacy pickup on Friday", MemoryType.COMMITMENT)
        await _drain(svc)
        plain = await svc.db.lexical_search("pharmacy", 5)
        hits = await svc.db.lexical_search(query, 5)
        recalled = await svc.recall(query)
        await svc.close()
        assert hits and hits[0][1] >= plain[0][1]
        assert len(recalled) == 1
    asyncio.run(run())
