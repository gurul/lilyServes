from risk_fusion import THETA_ALERT, RiskFusion
from scam_heuristics import score_text


def test_benign_call_stays_low():
    fusion = RiskFusion()
    transcript = ""
    for line in [
        "Hi Arthur, it's Susan from the pharmacy.",
        "Your prescription is ready.",
        "See you Friday at 4 PM for the pickup.",
    ]:
        transcript += " " + line
        state = fusion.on_sentence(score_text(transcript))
    assert state.level == "Low"
    assert not state.hard_alert
    assert state.stage in ("contact", "engagement")


def test_payment_marker_fires_immediately():
    fusion = RiskFusion()
    state = fusion.on_sentence(score_text("Go buy three gift cards right now."))
    assert state.hard_alert
    assert state.level == "High"
    assert state.stage == "payment"


def test_llm_scam_verdict_floors_accumulator():
    fusion = RiskFusion()
    fusion.on_sentence(score_text("Hello there."))
    state = fusion.on_llm("scam")
    assert state.level == "High"
    assert state.hard_alert
    assert fusion.ewma >= THETA_ALERT
    # Later small talk cannot wash the alert out.
    for _ in range(10):
        state = fusion.on_sentence(score_text("Lovely weather we're having."))
    assert state.level == "High"


def test_uncertain_verdict_watches_without_alert():
    fusion = RiskFusion()
    fusion.on_sentence(score_text("This is your bank's fraud department."))
    state = fusion.on_llm("uncertain")
    assert state.level == "Medium"
    assert not state.hard_alert


def test_uncertain_without_signals_stays_low():
    fusion = RiskFusion()
    fusion.on_sentence(score_text("Hi, how are you today?"))
    state = fusion.on_llm("uncertain")
    assert state.level == "Low"


def test_deepfake_multiplies_but_never_alarms_alone():
    clean = RiskFusion()
    clean.on_deepfake(0.95)
    state = clean.on_sentence(score_text("Hi Arthur, lovely to chat."))
    assert state.level == "Low"  # synthetic voice + benign words: no alarm

    risky = RiskFusion()
    risky.on_deepfake(0.95)
    transcript = "Grandma it's your grandson, I'm in trouble and need money immediately."
    state_df = risky.on_sentence(score_text(transcript))

    no_df = RiskFusion()
    state_plain = no_df.on_sentence(score_text(transcript))
    assert state_df.score > state_plain.score  # multiplier raised the risk


def test_escalating_scam_crosses_thresholds():
    fusion = RiskFusion()
    transcript = ""
    levels = []
    for line in [
        "This is the IRS calling about your account.",
        "There is a warrant for your arrest unless you pay today.",
        "You owe $4,000 and must act immediately.",
    ]:
        transcript += " " + line
        levels.append(fusion.on_sentence(score_text(transcript)).level)
    assert levels[-1] in ("Medium", "High")
    assert levels[-1] != "Low"


def test_stage_progression():
    fusion = RiskFusion()
    assert fusion.stage() == "contact"
    fusion.on_sentence(score_text("This is Microsoft support about your computer."))
    assert fusion.stage() == "engagement"
    fusion.on_sentence(score_text("Now read me the verification code we texted you."))
    assert fusion.stage() == "payment"
