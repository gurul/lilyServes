"""Call screening: Lily answers first when the caller isn't trusted.

Routing:
- trusted caller (env allowlist or DB)          -> pass straight through (still monitored)
- repeat scammer (2+ strikes)                   -> blocked outright
- restricted / anonymous / first-time caller    -> Lily answers, asks who's
  calling, risk-scores the answer, then bridges or politely declines

Every bridged call is still streamed and scored live; screening only decides
whether the phone rings at all.
"""
from __future__ import annotations

import logging
from xml.sax.saxutils import escape

from scam_heuristics import score_text

log = logging.getLogger("lily.screening")

# Twilio presents these for callers withholding caller ID.
_ANONYMOUS = {"", "unknown", "anonymous", "restricted", "private", "unavailable", "+266696687"}

ROUTE_PASS = "pass"
ROUTE_SCREEN = "screen"
ROUTE_BLOCK = "block"

SCREEN_GREETING = (
    "Hello, this is Lily, a call assistant. "
    "May I ask who's calling and what it's regarding?"
)
REJECT_MESSAGE = (
    "Thank you. This number doesn't accept unidentified calls. "
    "If this is important, please leave a message with a family member. Goodbye."
)
BLOCK_MESSAGE = "This number is not accepting your calls. Goodbye."


def is_anonymous(caller: str | None) -> bool:
    return (caller or "").strip().lower() in _ANONYMOUS


def decide_route(
    caller: str | None,
    caller_ctx: dict,
    trusted_numbers: list[str],
    screen_unknown: bool,
) -> str:
    """Choose pass/screen/block for an incoming caller."""
    number = (caller or "").strip()
    if number and number in trusted_numbers:
        return ROUTE_PASS
    if caller_ctx.get("trusted"):
        return ROUTE_PASS
    if caller_ctx.get("scam_strikes", 0) >= 2:
        return ROUTE_BLOCK
    if is_anonymous(number):
        return ROUTE_SCREEN
    if caller_ctx.get("known") and caller_ctx.get("call_count", 0) > 1:
        # A caller Lily has spoken with before and never flagged.
        return ROUTE_PASS
    return ROUTE_SCREEN if screen_unknown else ROUTE_PASS


def assess_screen_answer(speech: str) -> tuple[bool, str]:
    """Decide whether a screened caller's answer earns a bridge.

    Heuristic-only so the caller isn't left hanging on an LLM round-trip:
    High risk -> declined; silence/empty -> declined; otherwise bridged
    (the live call is still transcribed and scored end-to-end).
    Returns (allow, reason).
    """
    text = (speech or "").strip()
    if not text:
        return False, "Caller gave no response to screening"
    result = score_text(text)
    if result.level == "High":
        return False, "Screening answer matched scam patterns: " + ", ".join(result.signals)
    return True, "Screening answer acceptable"


# ----- TwiML builders --------------------------------------------------------

def twiml_pass(ws_url: str, forward_to: str, caller: str = "", screened: bool = False) -> str:
    """Fork caller audio to our WebSocket and bridge the call.

    The caller number and screening flag ride along as Stream custom
    parameters so the media-stream handler doesn't need a DB lookup.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Start>
    <Stream url="{escape(ws_url)}/ws/twilio" track="inbound_track">
      <Parameter name="caller" value="{escape(caller, {'"': '&quot;'})}" />
      <Parameter name="screened" value="{'1' if screened else '0'}" />
    </Stream>
  </Start>
  <Dial>{escape(forward_to)}</Dial>
</Response>"""


def twiml_screen(action_url: str) -> str:
    """Lily answers and asks who's calling."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Gather input="speech" speechTimeout="auto" timeout="6" action="{escape(action_url)}" method="POST">
    <Say voice="Polly.Joanna">{escape(SCREEN_GREETING)}</Say>
  </Gather>
  <Say voice="Polly.Joanna">{escape(REJECT_MESSAGE)}</Say>
  <Hangup/>
</Response>"""


def twiml_reject(blocked: bool = False) -> str:
    message = BLOCK_MESSAGE if blocked else REJECT_MESSAGE
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna">{escape(message)}</Say>
  <Hangup/>
</Response>"""


def twiml_hangup_polite() -> str:
    """Used by auto-intervention to end a High-risk bridged call."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Say voice="Polly.Joanna">This call has been ended for your safety. Goodbye.</Say>
  <Hangup/>
</Response>"""
