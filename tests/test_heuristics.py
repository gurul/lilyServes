import pytest

from lily.detection.heuristics import max_level, score_text


def test_benign_conversation_is_low():
    result = score_text(
        "Hi Arthur, it's the pharmacy on Main Street. "
        "Your prescription is ready for pickup on Friday at 4 PM."
    )
    assert result.level == "Low"


def test_empty_is_low():
    assert score_text("").level == "Low"
    assert score_text("").score == 0


def test_gift_card_urgency_is_high():
    result = score_text(
        "You need to act fast. Go buy three gift cards right now "
        "and read me the numbers on the back before it's too late."
    )
    assert result.level == "High"
    assert "gift cards" in result.signals
    assert "urgency" in result.signals


def test_grandparent_scam_is_flagged():
    result = score_text(
        "Grandma, it's me, your grandson. I'm in jail and I need bail money "
        "wired immediately. Please don't tell anyone."
    )
    assert result.level == "High"
    assert "family emergency" in result.signals
    assert "secrecy" in result.signals


def test_irs_threat_is_at_least_medium():
    result = score_text(
        "This is the IRS. There is a warrant for your arrest unless you pay today."
    )
    assert result.level in ("Medium", "High")


def test_remote_access_is_flagged():
    result = score_text(
        "Your computer has been hacked. Install AnyDesk so our Microsoft support "
        "technician can fix the virus on your computer."
    )
    assert result.level == "High"
    assert "remote access" in result.signals


def test_verification_code_request_scores():
    result = score_text("Just read me the verification code we texted you.")
    assert "verification code" in result.signals


def test_max_level():
    assert max_level("Low", "High", "Medium") == "High"
    assert max_level("Low", "Medium") == "Medium"
    assert max_level("Low") == "Low"
    assert max_level("garbage", "Medium") == "Medium"
    assert max_level() == "Low"


# ----- card numbers read back -------------------------------------------------
# Both scam scripts below scored Low before this rule, while the live LLM tier
# called each one a scam (eval: scripts/eval_llm_tier.py).

BANK_SCRIPT = (
    "Hello, this is Daniel with the security team at your bank. "
    "Someone is trying to take money out of your savings as we speak. "
    "To stop them I need you to read me the long number on the front of your card "
    "and the three digits on the back."
)
GRANDSON_SCRIPT = (
    "Grandma it's me, I got into a bad situation with the car and they're holding me. "
    "Please go to the store and get the cards, then read me the numbers on the back."
)


@pytest.mark.parametrize("text", [
    BANK_SCRIPT,
    GRANDSON_SCRIPT,
    "Give me the card number and PIN so I can load the payment.",
    "Tell me the numbers off the back of each gift card.",
    "What are the 3 digits on the back of your card?",
    "Send us the codes from the back of the cards.",
])
def test_card_number_read_back_is_flagged(text):
    result = score_text(text)
    assert "card numbers" in result.signals
    assert result.level in ("Medium", "High")


@pytest.mark.parametrize("text", [
    # The FTC's own advice: hang up and call the number on your card.
    "If you did not make this charge, please call the number on the back of your card.",
    "Give us a call at the number on the back of your card.",
    "Can you read me the number on the front of the envelope?",
    "Read me the numbers on the back of the prescription label.",
    "The phone number is on the back of the bottle.",
])
def test_card_number_rule_ignores_legitimate_calls(text):
    assert "card numbers" not in score_text(text).signals


def test_card_number_request_is_a_payment_stage_marker():
    from lily.detection.fusion import PAYMENT_STAGE_SIGNALS
    from lily.detection.heuristics import CREDENTIAL_SIGNALS

    assert "card numbers" in PAYMENT_STAGE_SIGNALS
    assert "card numbers" in CREDENTIAL_SIGNALS
