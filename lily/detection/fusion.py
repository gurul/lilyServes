"""Multi-signal scam-risk fusion.

Sources and design choices (README "Research grounding" has the links):

- EWMA accumulation follows arXiv:2509.05362, Eq. 2: e_1 = S(m_1),
  e_t = phi * S(m_t) + (1 - phi) * e_{t-1}, with phi = 2/(T+1). That paper
  scores a finished conversation of T messages. A live call has no final T,
  so we fix T at WINDOW_N = 5 sentences (our design choice). A constant phi
  keeps a late-call scam from being diluted by a long benign preamble. With
  phi = 1/3, the newest five sentences carry 1 - (2/3)^5 = 87% of the weight.
- The paper uses one alert threshold (theta_1 = 0.7). Its theta_2 decides
  whether the scam-baiting agent may keep talking, not when to alert. Our two
  thresholds THETA_WARN (Medium) and THETA_ALERT (hard alert) are our design
  choice.
- Payment-stage markers: arXiv:2605.12243 (PreScam) argues a live system
  "cannot always wait for the Termination phase (e.g., the scammer
  explicitly asks for payment)" before it flags a conversation. A single
  payment word is also common in benign calls ("I mailed you a gift card"),
  so a lone marker gives Medium and asks for an immediate LLM check. The
  marker sets hard_alert only when one more distinct heuristic signal, an
  LLM "scam" verdict, or EWMA >= THETA_ALERT corroborates it. An LLM "safe"
  verdict clears an uncorroborated marker. This rule is our design choice.
- The LLM verdict is three-way (scam / uncertain / safe). UNCERTAIN keeps
  watching without alerting. This is our design choice. arXiv:2502.03964
  (real-time LLM phone-scam detection) reports refining its method for
  precision and studies the trade-off between recall and timeliness.
- A synthetic-voice score multiplies transcript risk and never alerts by
  itself (our design choice). arXiv:2411.00121 finds that state-of-the-art
  audio-deepfake detectors often rely on high-frequency features that
  natural corruptions or an attacker can easily manipulate.

Invariants (checked by a formal model; keep these formulas exact):

- Risk input:   r = min(1, score/10), score >= 0
                r = max(r, VERDICT_RISK[llm_verdict])
- Deepfake:     r' = min(1, r * (1 + DEEPFAKE_GAIN * d)), d in [0, 1],
                DEEPFAKE_GAIN = 0.6 >= 0, so r <= r' <= 1 for r in [0, 1].
- EWMA:         phi = 2 / (WINDOW_N + 1) = 1/3 (constant)
                ewma_1 = r'_1;  ewma_t = phi * r'_t + (1 - phi) * ewma_{t-1}
                on_llm("scam"): ewma = max(ewma, THETA_ALERT)
                A convex combination of values in [0, 1], so 0 <= ewma <= 1.
- Hard alert:   corroborated = payment_marker_seen and |signals_seen| >= 2
                hard_alert = corroborated or ewma >= THETA_ALERT (within 1e-9)
                             or llm_verdict == "scam"
                So on_llm("scam") always returns hard_alert = True.
- Lone marker:  lone = payment_marker_seen and not corroborated
                cleared = lone and llm_after_marker and llm_verdict == "safe"
                needs_llm = lone and not llm_after_marker
- Level:        High   if hard_alert
                Medium if ewma >= THETA_WARN (within 1e-9)
                          or (llm_verdict == "uncertain" and signals_seen)
                          or (lone and not cleared)
                Low    otherwise
- state() reads fields only and writes nothing: it is a pure function of
  (sentences, ewma, llm_verdict, payment_marker_seen, llm_after_marker,
  _signals_seen).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from lily.detection.heuristics import HeuristicResult

# Dual thresholds on the EWMA-accumulated risk (0..1). Our design choice.
THETA_WARN = 0.35
THETA_ALERT = 0.60

# Float slack for threshold checks. The EWMA of exact inputs can land one
# rounding step below a threshold it reaches exactly (0.5999999999999999
# for deepfake 0.6 then scores 1, 9, 7 — found by the Lean model in
# proofs/). Erring by 1e-9 toward alerting is the safe side.
_EPS = 1e-9


def _reaches(value: float, threshold: float) -> bool:
    return value >= threshold - _EPS

# EWMA window: phi = 2/(T+1) with T fixed at WINDOW_N sentences.
WINDOW_N = 5
PHI = 2.0 / (WINDOW_N + 1.0)

# Heuristic signals that mark the payment/extraction stage of the scam
# kill chain. A lone marker gives Medium + needs_llm; a corroborated one
# sets hard_alert.
PAYMENT_STAGE_SIGNALS = frozenset({
    "gift cards",
    "wire transfer",
    "crypto payment",
    "verification code",
    "remote access",
    "secrecy",
    "courier/cash pickup",
    "payment app to self",
    "protect-your-money",
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
    hard_alert: bool = False  # corroborated payment marker, EWMA >= THETA_ALERT, or LLM scam
    verdict: str = "safe"     # latest LLM verdict
    needs_llm: bool = False   # lone payment marker waiting for an LLM check


@dataclass
class RiskFusion:
    """Per-call fusion of heuristic, LLM, and deepfake signals."""

    sentences: int = 0
    ewma: float = 0.0
    llm_verdict: str = "safe"
    deepfake_score: float = 0.0
    payment_marker_seen: bool = False
    llm_after_marker: bool = False
    _signals_seen: set = field(default_factory=set)

    # ----- signal inputs -----------------------------------------------------

    def on_sentence(self, heur: HeuristicResult) -> FusionState:
        """Fold one final transcript sentence's heuristics in. Sub-ms."""
        self.sentences += 1
        self._signals_seen.update(heur.signals)
        if not self.payment_marker_seen and PAYMENT_STAGE_SIGNALS & set(heur.signals):
            self.payment_marker_seen = True
            self.llm_after_marker = False

        risk = min(1.0, max(0.0, heur.score / 10.0))
        risk = max(risk, _VERDICT_RISK.get(self.llm_verdict, 0.0))
        risk = min(1.0, risk * (1.0 + DEEPFAKE_GAIN * self.deepfake_score))

        if self.sentences == 1:
            self.ewma = risk
        else:
            self.ewma = PHI * risk + (1.0 - PHI) * self.ewma
        return self.state()

    def on_llm(self, verdict: str) -> FusionState:
        if verdict in _VERDICT_RISK:
            self.llm_verdict = verdict
            if self.payment_marker_seen:
                self.llm_after_marker = True
            # An authoritative scam verdict floors the accumulator so a
            # following stretch of small talk can't wash the alert out.
            if verdict == "scam":
                self.ewma = max(self.ewma, THETA_ALERT)
        return self.state()

    def on_deepfake(self, score: float | None) -> FusionState:
        if score is not None:
            self.deepfake_score = max(0.0, min(1.0, score))
        return self.state()

    # ----- derived state -----------------------------------------------------

    def _corroborated(self) -> bool:
        # The marker plus at least one other distinct heuristic signal.
        return self.payment_marker_seen and len(self._signals_seen) >= 2

    def _lone_marker(self) -> bool:
        return self.payment_marker_seen and not self._corroborated()

    def _marker_cleared(self) -> bool:
        return self._lone_marker() and self.llm_after_marker and self.llm_verdict == "safe"

    def stage(self) -> str:
        if self.payment_marker_seen and not self._marker_cleared():
            return "payment"
        if self._signals_seen or self.sentences > 3:
            return "engagement"
        return "contact"

    def state(self) -> FusionState:
        hard = (
            self._corroborated()
            or _reaches(self.ewma, THETA_ALERT)
            or self.llm_verdict == "scam"
        )
        lone = self._lone_marker()
        if hard:
            level = "High"
        elif (
            _reaches(self.ewma, THETA_WARN)
            or (self.llm_verdict == "uncertain" and self._signals_seen)
            or (lone and not self._marker_cleared())
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
            needs_llm=lone and not self.llm_after_marker,
        )
