import LilyProofs.Model

/-!
# Test vectors from `tests/test_detection_fusion_rules.py`

Each `Heur` below is the exact `score_text(...)` output on main at 243778f
(score, signals), captured with the repo's Python. The `#guard`s replay the
test against the Lean model and check the same assertions.
-/

namespace Lily

def init : Fusion := {}

-- test_first_sentence_initializes_ewma_to_its_risk
-- score_text("This is the IRS. There is a warrant for your arrest.") = (6, [bail/legal threat, gov impersonation])
def irs : Heur := ⟨6, ["bail/legal threat", "gov impersonation"]⟩
#guard (state (onSentence init irs)).ewma == ⟨thetaAlert, 0⟩   -- score == 0.6 exactly
#guard (state (onSentence init irs)).hard == true
#guard (state (onSentence init irs)).level == .high

-- test_lone_marker_then_llm_scam_is_high (+ test_lone_payment_marker_is_medium_and_requests_llm)
-- score_text("Did you see on the news that bitcoin went up again?") = (4, [crypto payment])
def bitcoin : Heur := ⟨4, ["crypto payment"]⟩
#guard (state (onSentence init bitcoin)).hard == false
#guard (state (onSentence init bitcoin)).level == .medium
#guard (state (onSentence init bitcoin)).needsLlm == true
#guard (state (onLLM (onSentence init bitcoin) .scam)).hard == true
#guard (state (onLLM (onSentence init bitcoin) .scam)).level == .high

-- test_corroborated_gift_card_scam_is_high_immediately_without_llm
-- score_text("Go buy gift cards right now and don't tell anyone about this call.") = (11, [gift cards, urgency, secrecy])
def giftCards : Heur := ⟨11, ["gift cards", "urgency", "secrecy"]⟩
#guard (state (onSentence init giftCards)).hard == true
#guard (state (onSentence init giftCards)).level == .high
#guard (state (onSentence init giftCards)).stage == .payment
#guard (state (onSentence init giftCards)).needsLlm == false
#guard (onSentence init giftCards).verdict == .safe

-- test_lone_marker_llm_uncertain_stays_medium_without_new_request
-- score_text("Just between us, the casserole at church was a little dry.") = (4, [secrecy])
def casserole : Heur := ⟨4, ["secrecy"]⟩
#guard (state (onLLM (onSentence init casserole) .uncertain)).level == .medium
#guard (state (onLLM (onSentence init casserole) .uncertain)).hard == false
#guard (state (onLLM (onSentence init casserole) .uncertain)).needsLlm == false

-- test_phi_is_constant_from_fixed_window: 51 x "Hello." (score 0), then
-- "Buy gift cards right now." (score 7); after = PHI * 0.7 + (1 - PHI) * before.
def hello : Heur := ⟨0, []⟩
def buyGift : Heur := ⟨7, ["gift cards", "urgency"]⟩
def phiRun : Fusion := onSentence (run init (List.replicate 51 (.sentence hello))) buyGift
-- before = 0, so after = 0.7 / 3 = 7/30 exactly: num / (U * 3^exp) = 7/30.
#guard phiRun.ewma.num * 30 == 7 * U * 3 ^ phiRun.ewma.exp

/-! ## Float-rounding divergence (found by fuzzing, see README)

Deepfake 0.6, then sentences with scores 1, 9, 7 and no signals. The exact
EWMA is 0.136 -> 0.424 -> 0.6 = THETA_ALERT, so the exact model alerts.
Python computes `0.5999999999999999` and returns `hard_alert=False`,
`level="Medium"`. -/
def floatGap : Fusion :=
  run init [.deepfake (some 6000), .sentence ⟨1, []⟩, .sentence ⟨9, []⟩, .sentence ⟨7, []⟩]
#guard floatGap.ewma.num == thetaAlert * 3 ^ floatGap.ewma.exp   -- exactly 0.6
#guard (state floatGap).hard == true
#guard (state floatGap).level == .high

end Lily
