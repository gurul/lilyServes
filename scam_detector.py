SCAM_KEYWORDS = [
    "gift card",
    "irs",
    "internal revenue",
    "social security",
    "wire transfer",
    "western union",
    "moneygram",
    "account number",
    "routing number",
    "verify your",
    "urgent",
    "immediately",
    "suspended",
    "fbi",
    "police",
    "arrest warrant",
    "lawsuit",
    "cryptocurrency",
    "bitcoin",
    "pin number",
    "one-time password",
    "otp",
    "remote access",
    "anydesk",
    "teamviewer",
    "refund",
    "overpayment",
]

POINTS_PER_KEYWORD = 20
ALERT_THRESHOLD = 40


def analyze_scam(text: str) -> dict:
    """
    Scan transcript text for scam indicators.

    Returns:
        {
            "score": int (0-100),
            "keywords": list[str],
            "alert": bool
        }
    """
    text_lower = text.lower()
    found = [kw for kw in SCAM_KEYWORDS if kw in text_lower]
    score = min(len(found) * POINTS_PER_KEYWORD, 100)
    return {
        "score": score,
        "keywords": found,
        "alert": score >= ALERT_THRESHOLD,
    }
