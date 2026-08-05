"""Multi-signal scam-risk fusion, grounded in the vishing-detection literature.

Design (citations in README):
- Per-sentence risk is accumulated with an EWMA (phi = 2/(n+1)) behind two
  thresholds — THETA_WARN (soft warn) and THETA_ALERT (hard alert) — rather
  than independent per-sentence verdicts (arXiv:2509.05362).
- Payment-stage markers (gift cards, wire, crypto, codes, remote access,
  secrecy) escalate IMMEDIATELY: scam-progression studies show only 1-2
  conversational turns of lead time before payment once these appear
  (arXiv:2605.12243). The regex tier escalates; it never clears.
- The LLM tier is authoritative for negatives (keyword features are the
  exact surface adversarial rephrasing removes, arXiv:2507.16291) and emits
  a three-way verdict: "scam" / "uncertain" / "safe" — UNCERTAIN keeps
  watching without alerting, trading a little recall for the precision that
  preserves an older adult's trust in alerts (arXiv:2502.03964).
- A synthetic-voice score MULTIPLIES transcript risk instead of alarming on
  its own: telephony's 8 kHz band destroys the high-frequency artifacts
  deepfake detectors depend on, so narrowband scores are weak evidence
  (arXiv:2411.00121).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from scam_heuristics import HeuristicResult

# Dual thresholds on the EWMA-accumulated risk (0..1).
THETA_WARN = 0.35
THETA_ALERT = 0.60

# Heuristic signals that mark the payment/extraction stage of the scam
# kill chain — these bypass accumulation entirely.
PAYMENT_STAGE_SIGNALS = frozenset({
    "gift cards",
    "wire transfer",
    "crypto payment",
    "verification code",
    "remote access",
    "secrecy",
})

# LLM verdict -> transcript-risk contribution.
_VERDICT_RISK = {"scam": 0.90, "uncertain": 0.40, "safe": 0.0}

# Cap on how much a synthetic voice can amplify transcript risk.
DEEPFAKE_GAIN = 0.6

LEVELS = ("Low", "Medium", "High")


@dataclass
class FusionState:
    level: str = "Low"
    score: float = 0.0        # current EWMA risk, 0..1
    stage: str = "contact"    # contact | engagement | payment
    hard_alert: bool = False  # theta-2 fired (payment marker or EWMA)
    verdict: str = "safe"     # latest LLM verdict


@dataclass
class RiskFusion:
    """Per-call fusion of heuristic, LLM, and deepfake signals."""

    sentences: int = 0
    ewma: float = 0.0
    llm_verdict: str = "safe"
    deepfake_score: float = 0.0
    payment_marker_seen: bool = False
    _signals_seen: set = field(default_factory=set)

    # ----- signal inputs -----------------------------------------------------

    def on_sentence(self, heur: HeuristicResult) -> FusionState:
        """Fold one final transcript sentence's heuristics in. Sub-ms."""
        self.sentences += 1
        self._signals_seen.update(heur.signals)
        if PAYMENT_STAGE_SIGNALS & set(heur.signals):
            self.payment_marker_seen = True

        risk = min(1.0, heur.score / 10.0)
        risk = max(risk, _VERDICT_RISK.get(self.llm_verdict, 0.0))
        risk = min(1.0, risk * (1.0 + DEEPFAKE_GAIN * self.deepfake_score))

        phi = 2.0 / (self.sentences + 1.0)
        self.ewma = phi * risk + (1.0 - phi) * self.ewma
        return self.state()

    def on_llm(self, verdict: str) -> FusionState:
        if verdict in _VERDICT_RISK:
            self.llm_verdict = verdict
            # An authoritative scam verdict floors the accumulator so a
            # following stretch of small talk can't wash the alert out.
            if verdict == "scam":
                self.ewma = max(self.ewma, THETA_ALERT)
        return self.state()

    def on_deepfake(self, score: "float | None") -> FusionState:
        if score is not None:
            self.deepfake_score = max(0.0, min(1.0, score))
        return self.state()

    # ----- derived state -----------------------------------------------------

    def stage(self) -> str:
        if self.payment_marker_seen:
            return "payment"
        if self._signals_seen or self.sentences > 3:
            return "engagement"
        return "contact"

    def state(self) -> FusionState:
        hard = (
            self.payment_marker_seen
            or self.ewma >= THETA_ALERT
            or self.llm_verdict == "scam"
        )
        if hard:
            level = "High"
        elif self.ewma >= THETA_WARN or (
            self.llm_verdict == "uncertain" and self._signals_seen
        ):
            level = "Medium"
        else:
            level = "Low"
        return FusionState(
            level=level,
            score=round(self.ewma, 4),
            stage=self.stage(),
            hard_alert=hard,
            verdict=self.llm_verdict,
        )
