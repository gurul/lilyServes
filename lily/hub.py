"""Fan-out hub for dashboard clients plus smart-notification tagging.

Any number of family/dashboard WebSocket clients may be connected at once.
Payloads are serialized once and sent concurrently; dead sockets are dropped
on send failure. Importance levels drive "Smart Updates": everything reaches
the dashboard feed, but only `important`/`urgent` items are flagged
`notify: true` so clients know to raise a push notification.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

log = logging.getLogger("lily.hub")

# Importance ladder for activity items.
INFO = "info"            # quiet feed item ("All quiet right now")
NOTABLE = "notable"      # EVENT CAPTURED etc.
IMPORTANT = "important"  # Medium scam risk, screening declined a caller
URGENT = "urgent"        # High scam risk, deepfake detected

_NOTIFY_AT = {IMPORTANT, URGENT}


class ClientHub:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def register(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)
        log.info("client connected (%d total)", len(self._clients))

    async def unregister(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        log.info("client disconnected (%d total)", len(self._clients))

    async def broadcast(self, payload: dict[str, Any], importance: str = INFO) -> None:
        """Serialize once, send to every client concurrently."""
        if not self._clients:
            return
        payload.setdefault("importance", importance)
        payload.setdefault("notify", importance in _NOTIFY_AT)
        text = json.dumps(payload, separators=(",", ":"))
        async with self._lock:
            clients = list(self._clients)
        results = await asyncio.gather(
            *(ws.send_text(text) for ws in clients), return_exceptions=True
        )
        dead = [ws for ws, r in zip(clients, results) if isinstance(r, BaseException)]
        if dead:
            async with self._lock:
                for ws in dead:
                    self._clients.discard(ws)
            log.info("dropped %d dead client(s)", len(dead))


hub = ClientHub()


async def push_activity(
    kind: str,
    title: str,
    detail: str = "",
    importance: str = INFO,
    call_sid: str | None = None,
    store: Any = None,
) -> None:
    """Emit a dashboard activity-feed item and (optionally) persist it."""
    payload: dict[str, Any] = {
        "event": "activity",
        "kind": kind,
        "title": title,
        "detail": detail,
    }
    if call_sid:
        payload["call_sid"] = call_sid
    if store is not None:
        try:
            await store.add_activity(kind, title, detail, importance, call_sid)
        except Exception:
            log.exception("failed to persist activity")
    await hub.broadcast(payload, importance)
