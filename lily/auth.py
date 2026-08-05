"""Request authentication: Twilio webhook signatures and client tokens."""
from __future__ import annotations

import base64
import hashlib
import hmac
import logging

from fastapi import HTTPException, Request

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
    if not hmac.compare_digest(expected, signature):
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
    return hmac.compare_digest(settings.client_token, provided or "")
