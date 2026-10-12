"""Heuristic rule fixes: new scam families (fix 2), secrecy/prize/family regex (fix 6), pharmacy FP (fix 7)."""
import pytest

from lily.detection.heuristics import score_text

# Sentences from the false-negative battery. Each must score Medium or High.
SCAMS = {
    "ssa benefits": "This is the Social Security office. Your social security benefits have been suspended due to suspicious activity in Texas.",
    "jury duty sheriff": "This is Sergeant Miller from the county sheriff's office. You missed jury duty and there is a bench warrant out. You need to pay the fine today to avoid jail.",
    "courier cash": "Ma'am, to protect your savings, withdraw the cash from your bank. A courier will come to your house to pick up the money for safekeeping.",
    "bank zelle": "This is the Chase fraud prevention team. We saw unauthorized charges. To secure your funds, send them to yourself using Zelle.",
    "utility shutoff": "This is Con Edison. Your power will be shut off in thirty minutes unless you pay the overdue bill.",
    "tech support refund": "Hi, I'm calling from Amazon about a charge for an iPhone. We'll need to log in to your computer to process your refund. Go to the website and download this program.",
    "investment": "I have a guaranteed investment opportunity with returns of twenty percent a month. You just need to open an account on our trading platform.",
}


@pytest.mark.parametrize("name", sorted(SCAMS))
def test_new_scam_family_scores_at_least_medium(name):
    assert score_text(SCAMS[name]).level in ("Medium", "High")


@pytest.mark.parametrize("name,signal", [
    ("ssa benefits", "ssa benefits threat"),
    ("jury duty sheriff", "court/jury threat"),
    ("courier cash", "courier/cash pickup"),
    ("courier cash", "protect-your-money"),
    ("bank zelle", "payment app to self"),
    ("bank zelle", "unauthorized charges"),
    ("utility shutoff", "utility shutoff"),
    ("tech support refund", "remote access"),
    ("investment", "guaranteed investment"),
])
def test_new_scam_family_signal(name, signal):
    assert signal in score_text(SCAMS[name]).signals


# Five benign controls per new rule family. Each must stay Low.
BENIGN = {
    "ssa benefits": [
        "My Social Security check came on Wednesday like always.",
        "I went to the Social Security office to update my address.",
        "Did your Social Security benefits go up this year?",
        "Mom's Medicare and Social Security paperwork is in the blue folder.",
        "The library suspended late fees, so you can return the books.",
    ],
    "court/jury": [
        "Jury duty was boring, they sent me home at noon.",
        "I have jury duty next Tuesday at the county courthouse.",
        "The sheriff's office is hosting a pancake breakfast on Sunday.",
        "My nephew just became a deputy sheriff.",
        "We watched a courtroom drama about a warrant last night.",
    ],
    "courier": [
        "The courier dropped off your package this morning.",
        "A courier brought the flowers from the florist.",
        "The bank courier picks up the deposits from the church office.",
        "I gave the money to the cashier at the farmers market.",
        "Grandma's gold necklace is in the jewelry box.",
    ],
    "protect-your-money": [
        "I need to withdraw cash from my account for the farmers market.",
        "I put my savings in a CD at the credit union.",
        "We talked about saving money for the grandkids' college.",
        "Can you take forty dollars out of the ATM for me?",
        "I moved my money to a new bank closer to home.",
    ],
    "payment app": [
        "Grandpa used Venmo to pay me back for lunch.",
        "I'll send you the money for the tickets on Zelle tonight.",
        "Did you get the Cash App payment for the pizza?",
        "I set up Zelle with my bank yesterday.",
        "Treat yourself to something nice with the birthday money.",
    ],
    "unauthorized charges": [
        "I had an unauthorized charge on my card last year and the bank fixed it.",
        "My credit card statement came today and everything looked fine.",
        "The store charged me twice for the milk so I got a refund.",
        "I bought a new iPhone case at the mall.",
        "The bank sent a letter about my new debit card.",
    ],
    "utility": [
        "I paid my power bill online.",
        "The power was shut off during the storm last night.",
        "The water company will shut off the water Tuesday for repairs.",
        "Con Edison sent a crew to fix the line on our street.",
        "Our gas was turned off while they replaced the stove.",
    ],
    "tech support": [
        "Can you help me log in to my email?",
        "The grandkids helped me download the photo app.",
        "I'll call Amazon about my refund for the broken lamp.",
        "My computer is slow, can you look at it on Sunday?",
        "The library has free computer classes on Thursdays.",
    ],
    "investment": [
        "Our CD at the bank pays four percent a year.",
        "We talked about investing in index funds for retirement.",
        "The church raised twenty percent more this year.",
        "My financial advisor said there are no guarantees in the market.",
        "I watched a show about the stock market.",
    ],
}


@pytest.mark.parametrize("sentence", [s for group in BENIGN.values() for s in group])
def test_benign_control_stays_low(sentence):
    result = score_text(sentence)
    assert result.level == "Low", result


# ----- fix 6: secrecy / prize / family emergency ---------------------------

def test_grandparent_reword_is_high():
    result = score_text(
        "Grandma, it's me. I got in a car crash and I'm at the police station. "
        "Please do not tell mom and dad, I need money for a lawyer."
    )
    assert result.level == "High"
    assert {"secrecy", "family emergency"} <= set(result.signals)


def test_do_not_tell_anyone_matches_secrecy():
    result = score_text("Grandpa, I'm in jail. Do not tell anyone, please.")
    assert "secrecy" in result.signals
    assert result.level == "High"


def test_typographic_apostrophe_matches_secrecy():
    result = score_text("Don’t tell anyone about this call.")
    assert "secrecy" in result.signals
    assert result.level == "Medium"


def test_you_have_won_matches_prize():
    result = score_text(
        "Congratulations, you have won a new car from Publishers Clearing House. "
        "You just need to pay the taxes and fees first."
    )
    assert "prize/lottery" in result.signals
    assert result.level in ("Medium", "High")


def test_surprise_party_secrecy_stays_low():
    assert score_text("Don't tell Dad, we're throwing him a surprise party on Saturday.").level == "Low"


def test_six_digit_code_reword_scores():
    result = score_text(
        "I just sent a six digit number to your phone. "
        "Can you read it back to me to verify your identity?"
    )
    assert "verification code" in result.signals


# ----- fix 7: pharmacy false positive / Medicare -----------------------------

PHARMACY = (
    "Hi, this is CVS pharmacy, your prescription is ready. "
    "Medicare covered most of it and you owe $12. "
    "Please pick it up immediately, it needs refrigeration."
)


def test_pharmacy_call_is_below_high():
    result = score_text(PHARMACY)
    assert result.level != "High"
    assert "gov impersonation" not in result.signals
    assert "payment demand" not in result.signals


def test_small_amount_is_not_a_payment_demand():
    assert "payment demand" not in score_text("You owe $12 for the copay.").signals
    assert "payment demand" in score_text("You owe $4,000 in back fees.").signals


def test_medicare_with_number_request_is_impersonation():
    result = score_text(
        "I'm calling about your new Medicare card. "
        "Can you confirm your Medicare number so we can mail it?"
    )
    assert "gov impersonation" in result.signals
    assert result.level in ("Medium", "High")


def test_medicare_with_payment_demand_is_impersonation():
    result = score_text("This is Medicare. You owe $1,200 for your new card.")
    assert "gov impersonation" in result.signals
