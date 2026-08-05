from scam_heuristics import max_level, score_text


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
