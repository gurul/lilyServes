"""Fusion fixes: corroborated payment markers (fix 1), fixed EWMA window (fix 5), invariants."""
import pytest

from lily.detection.fusion import PHI, WINDOW_N, RiskFusion
from lily.detection.heuristics import score_text


def _feed(fusion, sentences):
    transcript = ""
    state = fusion.state()
    for sentence in sentences:
        transcript = (transcript + " " + sentence).strip()
        state = fusion.on_sentence(score_text(transcript))
    return state


LONE_MARKER_BENIGN = [
    "Happy birthday Mom! I mailed you a gift card for the bookstore.",
    "Did you see on the news that bitcoin went up again?",
    "Just between us, the casserole at church was a little dry.",
    "We'll text you a verification code to log in to the patient portal.",
]


@pytest.mark.parametrize("sentence", LONE_MARKER_BENIGN)
def test_lone_payment_marker_is_medium_and_requests_llm(sentence):
    fusion = RiskFusion()
    state = _feed(fusion, [sentence])
    assert not state.hard_alert
    assert state.level == "Medium"
    assert state.needs_llm


@pytest.mark.parametrize("sentence", LONE_MARKER_BENIGN)
def test_llm_safe_clears_lone_payment_marker(sentence):
    fusion = RiskFusion()
    _feed(fusion, [sentence])
    state = fusion.on_llm("safe")
    assert not state.hard_alert
    assert not state.needs_llm
    for _ in range(20):
        state = fusion.on_sentence(score_text(sentence + " We love you."))
    assert not state.hard_alert


def test_lone_marker_then_llm_scam_is_high():
    fusion = RiskFusion()
    _feed(fusion, ["Did you see on the news that bitcoin went up again?"])
    state = fusion.on_llm("scam")
    assert state.hard_alert
    assert state.level == "High"


def test_lone_marker_llm_uncertain_stays_medium_without_new_request():
    fusion = RiskFusion()
    _feed(fusion, ["Just between us, the casserole at church was a little dry."])
    state = fusion.on_llm("uncertain")
    assert state.level == "Medium"
    assert not state.hard_alert
    assert not state.needs_llm


def test_corroborated_gift_card_scam_is_high_immediately_without_llm():
    fusion = RiskFusion()
    state = _feed(fusion, ["Go buy gift cards right now and don't tell anyone about this call."])
    assert state.hard_alert
    assert state.level == "High"
    assert state.stage == "payment"
    assert not state.needs_llm
    assert fusion.llm_verdict == "safe"  # no LLM verdict was ever applied


def test_marker_corroborated_later_in_call_becomes_high():
    fusion = RiskFusion()
    state = _feed(fusion, ["Happy birthday Mom! I mailed you a gift card for the bookstore."])
    fusion.on_llm("safe")
    state = _feed(fusion, [
        "Happy birthday Mom! I mailed you a gift card for the bookstore.",
        "Actually, read me the numbers on the back right now.",
    ])
    assert state.hard_alert


IRS_LINES = [
    "This is Officer Brown with the IRS.",
    "There is a warrant for your arrest for unpaid back taxes.",
]


def test_fixed_window_late_call_scam_reaches_high():
    fusion = RiskFusion()
    transcript = ""
    for _ in range(40):
        transcript += " We talked about the garden and the weather."
        fusion.on_sentence(score_text(transcript))
    for _ in range(5):
        for line in IRS_LINES:
            transcript += " " + line
            fusion.on_sentence(score_text(transcript))
    state = fusion.on_llm("uncertain")
    assert state.level == "High"
    assert state.hard_alert


def test_phi_is_constant_from_fixed_window():
    assert PHI == pytest.approx(2.0 / (WINDOW_N + 1.0))
    fusion = RiskFusion()
    fusion.on_sentence(score_text("Hello."))  # e_1 = S(m_1) = 0
    for _ in range(50):
        fusion.on_sentence(score_text("Hello."))
    before = fusion.ewma
    after = fusion.on_sentence(score_text("Buy gift cards right now.")).score
    assert after == pytest.approx(PHI * 0.7 + (1 - PHI) * before, abs=1e-4)


def test_first_sentence_initializes_ewma_to_its_risk():
    fusion = RiskFusion()
    state = fusion.on_sentence(score_text("This is the IRS. There is a warrant for your arrest."))
    assert state.score == pytest.approx(0.6)


# ----- invariants ----------------------------------------------------------

def test_ewma_stays_in_unit_interval():
    fusion = RiskFusion()
    fusion.on_deepfake(1.0)
    loud = "Gift cards wire transfer bitcoin right now don't tell anyone AnyDesk verification code IRS."
    for _ in range(30):
        state = fusion.on_sentence(score_text(loud))
        assert 0.0 <= fusion.ewma <= 1.0
        assert 0.0 <= state.score <= 1.0
    fusion.on_llm("scam")
    assert 0.0 <= fusion.ewma <= 1.0
    for _ in range(30):
        fusion.on_sentence(score_text("Lovely weather."))
        assert 0.0 <= fusion.ewma <= 1.0


@pytest.mark.parametrize("deepfake", [0.0, 0.3, 1.0])
def test_deepfake_never_lowers_risk(deepfake):
    text = "Grandma it's your grandson, I'm in trouble and need money immediately."
    plain = RiskFusion().on_sentence(score_text(text))
    fusion = RiskFusion()
    fusion.on_deepfake(deepfake)
    boosted = fusion.on_sentence(score_text(text))
    assert boosted.score >= plain.score
    assert boosted.score <= 1.0


@pytest.mark.parametrize("prior", ["Hello.", "Did you see on the news that bitcoin went up again?"])
def test_llm_scam_always_sets_hard_alert(prior):
    fusion = RiskFusion()
    fusion.on_sentence(score_text(prior))
    fusion.on_llm("safe")
    assert fusion.on_llm("scam").hard_alert


def test_state_is_pure():
    fusion = RiskFusion()
    _feed(fusion, ["Did you see on the news that bitcoin went up again?"])
    snapshot = dict(vars(fusion))
    snapshot["_signals_seen"] = set(fusion._signals_seen)
    first = fusion.state()
    second = fusion.state()
    assert first == second
    after = dict(vars(fusion))
    assert after == snapshot
