"""Embedding provider for lily.memory.

Default implementation uses OpenAI (`text-embedding-3-small` at 512 dims —
3x cheaper to store and compare than full width, with negligible quality
loss at this scale). The provider is injectable so tests — and any future
local model — can swap it out. Failures always degrade to lexical-only
search, never to an error the caller sees.
"""
from __future__ import annotations

import logging
import struct

log = logging.getLogger("lily.memory.embed")


def pack_vector(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 512) -> None:
        self.model = model
        self.dimensions = dimensions

    async def embed(self, texts: list[str], timeout: float = 5.0) -> list[list[float]] | None:
        """Embed a batch of texts. Returns None on any failure (degrade, don't raise)."""
        if not texts:
            return []
        try:
            from lily.summarizer import get_client

            response = await get_client().embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimensions,
                timeout=timeout,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            log.warning("embedding failed (%d texts): %s", len(texts), e)
            return None


class NullEmbedder:
    """Disables semantic search; lily.memory runs lexical-only."""

    dimensions = 0

    async def embed(self, texts: list[str], timeout: float = 5.0) -> None:
        return None
