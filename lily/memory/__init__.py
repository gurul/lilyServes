"""lily.memory — Lily's long-term memory engine.

A ground-up memory system built for heyLily: typed memories with importance
and confidence, entity and topic tagging, hybrid lexical + semantic recall,
and recall tracking — all self-contained on SQLite + FTS5 with float32
embeddings.

Powers Cognitive Continuity (caller context when the phone rings), Active
Assistance (commitments that persist), and the family dashboard's memory
views. Privacy first: everything stored is redacted, and any memory can be
deleted — you own your data, always.
"""
from lily.memory.models import Memory, MemoryStatus, MemoryType
from lily.memory.service import MemoryService

__all__ = ["Memory", "MemoryService", "MemoryStatus", "MemoryType"]
