"""Instant, zero-latency scam risk scoring.

Pure-Python regex scoring that runs in-line on every transcript sentence so
the dashboard gets a provisional risk level immediately; the LLM refines it
asynchronously. Patterns are compiled once at import.
"""
from __future__ import annotations

import re
from typing import NamedTuple

LEVELS = ("Low", "Medium", "High")

# (name, weight, pattern) — weights accumulate over the whole transcript.
_RULES: list[tuple[str, int, re.Pattern]] = [
    ("gift cards", 5, re.compile(r"\bgift\s*cards?\b|\bitunes\s+cards?\b|\bgoogle\s*play\s+cards?\b", re.IGNORECASE)),
    ("wire transfer", 4, re.compile(r"\bwire\s+(?:transfer|money|funds)\b|\bwestern\s+union\b|\bmoney\s*gram\b", re.IGNORECASE)),
    ("crypto payment", 4, re.compile(r"\bbitcoin\b|\bcrypto(?:currency)?\b|\bbtc\s+atm\b", re.IGNORECASE)),
    ("urgency", 2, re.compile(r"\bright\s+now\b|\bimmediately\b|\burgent(?:ly)?\b|\bbefore\s+it'?s\s+too\s+late\b|\bact\s+fast\b|\btime\s+is\s+running\s+out\b", re.IGNORECASE)),
    ("secrecy", 4, re.compile(r"\bdon'?t\s+tell\s+(?:any(?:one|body)|your)\b|\bkeep\s+(?:this|it)\s+(?:a\s+)?secret\b|\bbetween\s+us\b", re.IGNORECASE)),
    ("family emergency", 3, re.compile(r"\b(?:grandson|granddaughter|grandma|grandpa|nephew|niece)\b.{0,80}\b(?:trouble|jail|arrested|accident|hospital|bail)\b", re.IGNORECASE | re.DOTALL)),
    ("bail/legal threat", 3, re.compile(r"\bbail\s+(?:money|bond)\b|\bwarrant\s+for\s+your\s+arrest\b|\byou\s+will\s+be\s+arrested\b|\blawsuit\s+(?:against|filed)\b", re.IGNORECASE)),
    ("gov impersonation", 3, re.compile(r"\b(?:irs|internal\s+revenue|social\s+security\s+administration|medicare|medicaid)\b", re.IGNORECASE)),
    ("account credentials", 3, re.compile(r"\b(?:social\s+security|account)\s+number\b|\bpin\s+(?:number|code)\b|\bpassword\b|\bmother'?s\s+maiden\s+name\b", re.IGNORECASE)),
    ("verification code", 4, re.compile(r"\b(?:verification|security|one[-\s]?time)\s+code\b|\bcode\s+(?:i|we)\s+(?:just\s+)?(?:sent|texted)\b", re.IGNORECASE)),
    ("remote access", 5, re.compile(r"\bany\s*desk\b|\bteam\s*viewer\b|\bremote\s+(?:access|desktop)\b|\binstall\s+(?:an?\s+)?app\b.{0,40}\bcomputer\b", re.IGNORECASE | re.DOTALL)),
    ("prize/lottery", 3, re.compile(r"\byou'?ve?\s+won\b|\blottery\b|\bsweepstakes\b|\bclaim\s+your\s+(?:prize|reward)\b", re.IGNORECASE)),
    ("tech support", 3, re.compile(r"\b(?:microsoft|windows|apple)\s+(?:support|security)\b|\bvirus\s+(?:on|in)\s+your\s+(?:computer|device)\b|\bcomputer\s+(?:has\s+been\s+)?(?:hacked|infected|compromised)\b", re.IGNORECASE)),
    ("payment demand", 2, re.compile(r"\b(?:owe|pay|payment\s+of)\s+\$?\d[\d,]*\b|\boutstanding\s+(?:balance|debt)\b|\bpast\s+due\b", re.IGNORECASE)),
    ("bank impersonation", 2, re.compile(r"\bfraud\s+department\b|\bsuspicious\s+activity\s+on\s+your\s+account\b|\baccount\s+(?:has\s+been\s+)?(?:locked|suspended|compromised)\b", re.IGNORECASE)),
    ("delivery/refund bait", 2, re.compile(r"\brefund\b.{0,40}\b(?:owed|due|process)\b|\bpackage\s+(?:could\s+not|couldn'?t)\s+be\s+delivered\b", re.IGNORECASE | re.DOTALL)),
]

MEDIUM_THRESHOLD = 3
HIGH_THRESHOLD = 7


class HeuristicResult(NamedTuple):
    level: str
    score: int
    signals: list[str]


def score_text(text: str) -> HeuristicResult:
    """Score a transcript (or message body). Sub-millisecond for call-length text."""
    if not text:
        return HeuristicResult("Low", 0, [])
    score = 0
    signals: list[str] = []
    for name, weight, pattern in _RULES:
        if pattern.search(text):
            score += weight
            signals.append(name)
    if score >= HIGH_THRESHOLD:
        level = "High"
    elif score >= MEDIUM_THRESHOLD:
        level = "Medium"
    else:
        level = "Low"
    return HeuristicResult(level, score, signals)


def max_level(*levels: str) -> str:
    """Return the most severe of the given risk levels."""
    best = 0
    for level in levels:
        try:
            best = max(best, LEVELS.index(level))
        except ValueError:
            continue
    return LEVELS[best]
