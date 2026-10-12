"""Request authentication: Twilio webhook signatures and client tokens."""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import re

from fastapi import HTTPException, Request, WebSocket

from lily.config import settings

log = logging.getLogger("lily.auth")


def compute_twilio_signature(auth_token: str, url: str, params: dict[str, str]) -> str:
    """Twilio's X-Twilio-Signature: HMAC-SHA1 over URL + sorted form params."""
    payload = url + "".join(k + params[k] for k in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode("utf-8"), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


async def require_twilio_signature(request: Request) -> dict[str, str]:
    """Validate the webhook came from Twilio; return the parsed form.

    Enforced whenever TWILIO_AUTH_TOKEN is configured. Without a token we
    can't verify, so we log loudly and accept (dev/demo mode).
    """
    form = {k: v for k, v in (await request.form()).items() if isinstance(v, str)}
    token = settings.twilio_auth_token
    if not token:
        log.warning("TWILIO_AUTH_TOKEN unset — webhook signature NOT verified")
        return form

    signature = request.headers.get("X-Twilio-Signature", "")
    # Twilio signs the public URL it called, not the proxied localhost URL.
    base = settings.public_url or getattr(request.app.state, "public_url", "")
    url = (base or str(request.base_url).rstrip("/")) + request.url.path
    if request.url.query:
        url += "?" + request.url.query

    expected = compute_twilio_signature(token, url, form)
    if not _equal(expected, signature):
        log.warning("rejected webhook with bad Twilio signature for %s", url)
        raise HTTPException(status_code=403, detail="invalid signature")
    return form


def check_client_token(provided: str) -> bool:
    """Constant-time check of the dashboard shared secret.

    If CLIENT_TOKEN is unset, access is open (dev/demo mode) — logged once
    at startup by the server.
    """
    if not settings.client_token:
        return True
    return _equal(settings.client_token, provided or "")


def _equal(a: str, b: str) -> bool:
    """Constant-time compare on UTF-8 bytes.

    hmac.compare_digest raises TypeError on non-ASCII str, which turned a
    bad token into a 500 instead of a 403.
    """
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def stream_url(base: str, path: str) -> str:
    """The wss:// URL Twilio connects to for a media stream at `path`."""
    return base.replace("https://", "wss://").replace("http://", "ws://") + path


def verify_twilio_ws_signature(websocket: WebSocket) -> bool:
    """Check X-Twilio-Signature on the Media Streams WebSocket handshake.

    Twilio signs the handshake like a webhook with no form params, over the
    wss:// URL from the TwiML <Stream url>. The docs note the URL may need a
    trailing "/", so both forms are accepted:
    https://www.twilio.com/docs/voice/media-streams ("You must also configure
    your application to validate the X-Twilio-Signature header.") and
    https://www.twilio.com/docs/usage/security ("For voice WSS handshake
    requests ... try appending a trailing / character to the URL").

    Returns True without a TWILIO_AUTH_TOKEN (dev mode).
    """
    token = settings.twilio_auth_token
    if not token:
        return True
    signature = websocket.headers.get("X-Twilio-Signature", "")
    base = settings.public_url or getattr(websocket.app.state, "public_url", "")
    if not base or not signature:
        return False
    url = stream_url(base, websocket.url.path)
    if websocket.url.query:
        url += "?" + websocket.url.query
    return any(
        _equal(compute_twilio_signature(token, candidate, {}), signature)
        for candidate in (url, url + "/")
    )


def stream_param_sig(auth_token: str, call_sid: str) -> str:
    """HMAC-SHA256(auth_token, CallSid) hex, sent as the Stream "sig" parameter."""
    return hmac.new(auth_token.encode(), call_sid.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_stream_param_sig(call_sid: str, sig: str) -> bool:
    """Check the "sig" custom parameter on a media stream `start` event."""
    token = settings.twilio_auth_token
    if not token:
        return True
    return _equal(stream_param_sig(token, call_sid), sig or "")


_TOKEN_IN_QUERY = re.compile(r"([?&]token=)[^&\s\"]*")


def mask_token(text: str) -> str:
    return _TOKEN_IN_QUERY.sub(r"\1***", text)


class TokenMaskFilter(logging.Filter):
    """Mask ?token=... in uvicorn access-log lines (/ws/client carries it)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(mask_token(a) if isinstance(a, str) else a for a in record.args)
        elif isinstance(record.msg, str):
            record.msg = mask_token(record.msg)
        return True


def install_access_log_filter() -> None:
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, TokenMaskFilter) for f in access.filters):
        access.addFilter(TokenMaskFilter())
