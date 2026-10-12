"""Instant, zero-latency scam risk scoring.

Pure-Python regex scoring that runs in-line on every transcript sentence so
the dashboard gets a provisional risk level immediately; the LLM refines it
asynchronously. Patterns are compiled once at import.

Rule sources. Each scam family added for elder-targeted phone fraud cites the
consumer-protection page that describes it. The FBI IC3 2024 Internet Crime
Report (https://www.ic3.gov/AnnualReport/Reports/2024_IC3Report.pdf) lists
Tech Support ($982,440,006) and Government Impersonation ($208,096,366) among
the largest losses reported by people over 60. The weights and thresholds
are our design choices, not values from those sources.
"""
from __future__ import annotations

import re
from typing import NamedTuple

LEVELS = ("Low", "Medium", "High")

_I = re.IGNORECASE
_ID = re.IGNORECASE | re.DOTALL

# (name, weight, pattern) — weights accumulate over the whole transcript.
_RULES: list[tuple[str, int, re.Pattern]] = [
    ("gift cards", 5, re.compile(r"\bgift\s*cards?\b|\bitunes\s+cards?\b|\bgoogle\s*play\s+cards?\b", _I)),
    ("wire transfer", 4, re.compile(r"\bwire\s+(?:transfer|money|funds)\b|\bwestern\s+union\b|\bmoney\s*gram\b", _I)),
    ("crypto payment", 4, re.compile(r"\bbitcoin\b|\bcrypto(?:currency)?\b|\bbtc\s+atm\b", _I)),
    ("urgency", 2, re.compile(r"\bright\s+now\b|\bimmediately\b|\burgent(?:ly)?\b|\bbefore\s+it'?s\s+too\s+late\b|\bact\s+fast\b|\btime\s+is\s+running\s+out\b", _I)),
    # "Don't tell Dad" alone is left out on purpose: it is the surprise-party
    # phrasing. "mom and dad" / "your parents" is the grandparent-scam script.
    ("secrecy", 4, re.compile(
        r"\b(?:don'?t|do\s+not)\s+tell\s+(?:any(?:one|body)|your\b|(?:mom|mum|dad)\s+and\s+(?:mom|mum|dad)\b|(?:my|your)\s+parents\b)"
        r"|\bkeep\s+(?:this|it)\s+(?:a\s+)?secret\b|\bbetween\s+us\b", _I)),
    ("family emergency", 3, re.compile(r"\b(?:grandson|granddaughter|grandma|grandpa|nephew|niece)\b.{0,80}\b(?:trouble|jail|arrested|accident|crash|police|hospital|bail)\b", _ID)),
    ("bail/legal threat", 3, re.compile(r"\bbail\s+(?:money|bond)\b|\bwarrant\s+for\s+your\s+arrest\b|\byou\s+will\s+be\s+arrested\b|\blawsuit\s+(?:against|filed)\b", _I)),
    # Medicare is NOT here: a pharmacy or clinic says "Medicare" in ordinary
    # calls. See "medicare id request" and the combination rule in score_text.
    ("gov impersonation", 3, re.compile(r"\b(?:irs|internal\s+revenue|social\s+security\s+administration)\b", _I)),
    ("account credentials", 3, re.compile(r"\b(?:social\s+security|account|medicare)\s+number\b|\bpin\s+(?:number|code)\b|\bpassword\b|\bmother'?s\s+maiden\s+name\b", _I)),
    ("verification code", 4, re.compile(r"\b(?:verification|security|one[-\s]?time)\s+code\b|\bcode\s+(?:i|we)\s+(?:just\s+)?(?:sent|texted)\b|\b(?:four|five|six|4|5|6)[-\s]digit\s+(?:code|number)\b", _I)),
    # FTC, tech support scams: "If you call, the scammers ask for remote
    # access to your computer."
    # https://consumer.ftc.gov/articles/how-spot-avoid-and-report-tech-support-scams
    ("remote access", 5, re.compile(
        r"\bany\s*desk\b|\bteam\s*viewer\b|\bremote\s+(?:access|desktop)\b|\binstall\s+(?:an?\s+)?app\b.{0,40}\bcomputer\b"
        r"|\b(?:log\s*(?:in|on)\s*(?:to|into)|access|connect\s+to|get\s+into|take\s+control\s+of)\s+your\s+(?:computer|laptop|pc)\b"
        r"|\bdownload\s+(?:this|our|the)\s+(?:program|software)\b", _ID)),
    ("prize/lottery", 3, re.compile(r"\byou(?:'ve|\s+have)\s+(?:just\s+)?won\b|\blottery\b|\bsweepstakes\b|\bclaim\s+your\s+(?:prize|reward)\b", _I)),
    ("tech support", 3, re.compile(r"\b(?:microsoft|windows|apple)\s+(?:support|security)\b|\bvirus\s+(?:on|in)\s+your\s+(?:computer|device)\b|\bcomputer\s+(?:has\s+been\s+)?(?:hacked|infected|compromised)\b", _I)),
    # Amounts under $100 are not a demand by themselves ("you owe $12" at the
    # pharmacy counter). The $100 floor is our design choice.
    # FTC jury duty alert: "the only way to cancel it is to pay a fine."
    # https://consumer.ftc.gov/consumer-alerts/2024/09/did-you-get-call-or-email-saying-you-missed-jury-duty-and-need-pay-its-scam
    ("payment demand", 2, re.compile(
        r"\b(?:owe|pay|payment\s+of)\s+\$?(?:\d{1,3}(?:,\d{3})+|\d{3,})\b|\b(?:owe|pay)\b.{0,20}\b(?:hundred|thousand)\s+dollars\b"
        r"|\boutstanding\s+(?:balance|debt)\b|\bpast\s+due\b|\b(?:unpaid|back)\s+taxes\b|\boverdue\s+(?:bill|balance|payment)\b"
        r"|\bpay\s+(?:the|a|this)\s+fine\b", _I)),
    ("bank impersonation", 2, re.compile(r"\bfraud\s+(?:department|prevention|protection)\b|\bsuspicious\s+activity\s+on\s+your\s+account\b|\baccount\s+(?:has\s+been\s+)?(?:locked|suspended|compromised)\b", _I)),
    ("delivery/refund bait", 2, re.compile(r"\brefund\b.{0,40}\b(?:owed|due|process)\b|\bpackage\s+(?:could\s+not|couldn'?t)\s+be\s+delivered\b", _ID)),
    # FTC: "The real Social Security Administration will never call to
    # threaten your benefits."
    # https://consumer.ftc.gov/consumer-alerts/2019/09/social-security-not-trying-take-your-benefits
    ("ssa benefits threat", 3, re.compile(
        r"\bsocial\s+security\b.{0,60}\b(?:suspended|frozen|cancell?ed|terminated|blocked)\b"
        r"|\bbenefits\s+(?:have|has|will|are)\s+(?:been\s+|be\s+)?(?:suspended|frozen|cancell?ed|terminated|stopped)\b", _ID)),
    # FTC jury duty alert: "They'll tell you there's a warrant out for your
    # arrest", with "a fake sheriff's badge number and case number".
    # https://consumer.ftc.gov/consumer-alerts/2024/09/did-you-get-call-or-email-saying-you-missed-jury-duty-and-need-pay-its-scam
    ("court/jury threat", 3, re.compile(
        r"\b(?:missed|failed\s+to\s+(?:appear|report)\s+for|skipped)\s+(?:your\s+)?jury\s+duty\b|\bbench\s+warrant\b"
        r"|\barrest\s+warrant\b|\bwarrant\s+out\b|\bavoid\s+(?:jail|arrest|being\s+arrested)\b", _I)),
    ("law-enforcement impersonation", 2, re.compile(r"\bsheriff'?s?\s+(?:office|department)\b|\b(?:u\.?s\.?\s+)?marshals?\s+service\b|\bdeputy\s+sheriff\b", _I)),
    # FBI IC3 PSA240129: "the scammers send couriers to retrieve the items at
    # victims' homes or public locations."  https://www.ic3.gov/PSA/2024/PSA240129
    ("courier/cash pickup", 5, re.compile(
        r"\bcourier\b.{0,80}\b(?:cash|money|gold|silver|funds|savings)\b|\b(?:cash|money|gold|silver)\b.{0,80}\bcourier\b"
        r"|\bgold\s+bars?\b", _ID)),
    # FTC: "The FTC will never tell you to move your money to 'protect' it."
    # https://consumer.ftc.gov/consumer-alerts/2024/01/did-someone-tell-you-move-or-transfer-your-money-it-could-be-scam
    ("protect-your-money", 4, re.compile(
        r"\b(?:protect|secure|safeguard)\s+your\s+(?:money|savings|funds|accounts?|assets)\b"
        r"|\bwithdraw\b.{0,30}\bfrom\s+your\s+(?:bank|account|savings)\b|\b(?:safe|secure|protected)\s+account\b", _ID)),
    # FTC payment-app alert: "scammers pose as your bank" and move the money
    # into "a new account in your name" that "really belongs to the scammer."
    # https://consumer.ftc.gov/consumer-alerts/2023/08/do-you-use-payment-apps-venmo-cashapp-or-zelle-read
    ("payment app to self", 4, re.compile(
        r"\b(?:zelle|cash\s*app|venmo)\b.{0,60}\b(?:to\s+yourself|your\s+own\s+account|new\s+account\s+in\s+your\s+name)\b"
        r"|\b(?:to\s+yourself|new\s+account\s+in\s+your\s+name)\b.{0,60}\b(?:zelle|cash\s*app|venmo)\b", _ID)),
    # FTC Data Spotlight: "Scammers send bogus alerts about suspicious
    # activity or unauthorized charges on your account."
    # https://consumer.ftc.gov/consumer-alerts/2024/03/ftc-data-spotlight-new-insights-about-imposter-scams
    ("unauthorized charges", 2, re.compile(
        r"\bunauthori[sz]ed\s+(?:charges?|transactions?|purchases?|withdrawals?)\b|\bsuspicious\s+(?:charges?|purchases?|transactions?|activity)\b"
        r"|\bcharge\s+(?:for|of)\s+(?:an?\s+)?(?:iphone|laptop|subscription|renewal)\b", _I)),
    # FTC: scammers "threaten to shut off your gas, electric, or water service
    # if you don't pay them immediately."
    # https://consumer.ftc.gov/articles/scammers-pretend-be-your-utility-company
    ("utility shutoff", 3, re.compile(
        r"\b(?:shut\s*off|disconnected|cut\s+off|turned\s+off)\b.{0,60}\b(?:unless|if\s+you\s+(?:don'?t|do\s+not))\s+(?:you\s+)?pay\b", _ID)),
    # FTC: "Scammers promise you'll make big money, guaranteed income, or
    # guaranteed profits or returns." https://consumer.ftc.gov/articles/investment-scams
    ("guaranteed investment", 4, re.compile(
        r"\bguaranteed\s+(?:\w+\s+){0,2}?(?:returns?|profits?|investment)\b|\brisk[-\s]free\s+(?:investment|returns?|profits?)\b"
        r"|\breturns?\s+of\s+\w+(?:\s+\w+)?\s+percent\s+(?:a|per|every)\s+(?:day|week|month)\b", _I)),
]

# Medicare counts as government impersonation only next to a demand for the
# Medicare ID or for money (fix for the pharmacy false positive).
_MEDICARE = re.compile(r"\bmedicare\b", _I)
_MEDICARE_ID = re.compile(r"\bmedicare\b.{0,80}\b(?:number|id)\b|\b(?:medicare|beneficiary)\s+(?:number|id)\b", _ID)

MEDIUM_THRESHOLD = 3
HIGH_THRESHOLD = 7

_WEIGHTS = {name: weight for name, weight, _ in _RULES}
_GOV = "gov impersonation"


class HeuristicResult(NamedTuple):
    level: str
    score: int
    signals: list[str]


def _normalize(text: str) -> str:
    # Speech-to-text and pasted text use typographic apostrophes (U+2019,
    # U+2018); the patterns use ASCII "'".
    return text.replace("’", "'").replace("‘", "'")


def score_text(text: str) -> HeuristicResult:
    """Score a transcript (or message body). Sub-millisecond for call-length text."""
    if not text:
        return HeuristicResult("Low", 0, [])
    text = _normalize(text)
    score = 0
    signals: list[str] = []
    for name, weight, pattern in _RULES:
        if pattern.search(text):
            score += weight
            signals.append(name)
    if _GOV not in signals and _MEDICARE.search(text) and (
        "payment demand" in signals or _MEDICARE_ID.search(text)
    ):
        score += _WEIGHTS[_GOV]
        signals.append(_GOV)
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


# Signal families used by call screening (assess_screen_answer) to decline a
# caller whose answer pairs an impersonation claim with a threat, a credential
# request, or a payment request. The grouping is our design choice.
IMPERSONATION_SIGNALS = frozenset({
    "gov impersonation",
    "bank impersonation",
    "law-enforcement impersonation",
    "tech support",
    "utility shutoff",
})
THREAT_SIGNALS = frozenset({
    "bail/legal threat",
    "court/jury threat",
    "ssa benefits threat",
    "unauthorized charges",
})
CREDENTIAL_SIGNALS = frozenset({
    "account credentials",
    "verification code",
    "remote access",
})
PAYMENT_SIGNALS = frozenset({
    "payment demand",
    "gift cards",
    "wire transfer",
    "crypto payment",
    "courier/cash pickup",
    "protect-your-money",
    "payment app to self",
})
