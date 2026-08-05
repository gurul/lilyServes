"""Memory model for lilyMemory.

Memory types are shaped around what heyLily promises its users, not around a
generic memory taxonomy:

- EPISODE     what happened on a call — "Cognitive Continuity"
- COMMITMENT  appointments, pickups, follow-ups — "Active Assistance" / "Memory Sync"
- PERSON      who a caller is and how they relate — gentle context for every conversation
- PREFERENCE  how the user likes things done
- WIN         positive moments worth resurfacing — "helping you remember every win"
- SAFETY      scam encounters and screening history
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field


class MemoryType:
    EPISODE = "episode"
    COMMITMENT = "commitment"
    PERSON = "person"
    PREFERENCE = "preference"
    WIN = "win"
    SAFETY = "safety"

    ALL = (EPISODE, COMMITMENT, PERSON, PREFERENCE, WIN, SAFETY)


@dataclass
class Memory:
    content: str
    memory_type: str = MemoryType.EPISODE
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    entities: list[str] = field(default_factory=list)   # people/places referenced
    topics: list[str] = field(default_factory=list)     # free-form tags
    importance: float = 0.5   # 0..1 — how much this matters to surface later
    confidence: float = 0.8   # 0..1 — how sure we are the content is correct
    caller: str = ""          # phone number this memory is tied to, if any
    call_sid: str = ""        # originating call, if any
    created_at: float = field(default_factory=time.time)
    last_recalled: float = 0.0
    recall_count: int = 0

    def to_row(self) -> tuple:
        return (
            self.id, self.memory_type, self.content,
            json.dumps(self.entities), json.dumps(self.topics),
            self.importance, self.confidence,
            self.caller, self.call_sid,
            self.created_at, self.last_recalled, self.recall_count,
        )

    @classmethod
    def from_row(cls, row) -> Memory:
        return cls(
            id=row["id"],
            memory_type=row["memory_type"],
            content=row["content"],
            entities=json.loads(row["entities"] or "[]"),
            topics=json.loads(row["topics"] or "[]"),
            importance=row["importance"],
            confidence=row["confidence"],
            caller=row["caller"] or "",
            call_sid=row["call_sid"] or "",
            created_at=row["created_at"],
            last_recalled=row["last_recalled"],
            recall_count=row["recall_count"],
        )

    def to_dict(self, score: float | None = None) -> dict:
        d = {
            "id": self.id,
            "memory_type": self.memory_type,
            "content": self.content,
            "entities": self.entities,
            "topics": self.topics,
            "importance": self.importance,
            "confidence": self.confidence,
            "caller": self.caller,
            "call_sid": self.call_sid,
            "created_at": self.created_at,
            "recall_count": self.recall_count,
        }
        if score is not None:
            d["score"] = round(score, 4)
        return d
