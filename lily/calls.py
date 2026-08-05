"""Per-call session state supporting any number of concurrent calls."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from lily.detection.fusion import RiskFusion


@dataclass
class CallSession:
    call_sid: str
    caller: str = "unknown"
    started_at: float = field(default_factory=time.time)
    stream_session: Any = None  # GoogleStreamingSession
    transcript: list[str] = field(default_factory=list)
    scam_level: str = "Low"
    scam_reasoning: str = ""
    heuristic_signals: list[str] = field(default_factory=list)
    fusion: RiskFusion = field(default_factory=RiskFusion)
    deepfake_score: float | None = None
    deepfake_buffer: bytearray = field(default_factory=bytearray)
    deepfake_submitted: bool = False
    screened: bool = False  # went through the Lily screen before bridging
    # Coalesced LLM analysis: one in flight per call; if new lines arrive
    # mid-analysis, mark dirty and rerun once with the latest transcript.
    analysis_in_flight: bool = False
    analysis_dirty: bool = False
    intervened: bool = False

    @property
    def full_transcript(self) -> str:
        return " ".join(self.transcript)


class CallRegistry:
    def __init__(self) -> None:
        self._calls: dict[str, CallSession] = {}

    def create(self, call_sid: str, caller: str = "unknown") -> CallSession:
        session = CallSession(call_sid=call_sid, caller=caller)
        self._calls[call_sid] = session
        return session

    def get(self, call_sid: str) -> CallSession | None:
        return self._calls.get(call_sid)

    def pop(self, call_sid: str) -> CallSession | None:
        return self._calls.pop(call_sid, None)

    def active(self) -> list[CallSession]:
        return list(self._calls.values())

    def __len__(self) -> int:
        return len(self._calls)


registry = CallRegistry()
