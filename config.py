"""Central configuration for lilyServes, loaded once from the environment."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _csv(name: str) -> list[str]:
    raw = os.environ.get(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    port: int = 8000
    forward_to: str = ""
    project_id: str = "heylily"
    stt_location: str = "global"
    stt_model: str = "telephony"
    stt_language: str = "en-US"

    scam_model: str = "gpt-4o-mini"
    summary_model: str = "gpt-4o"
    event_model: str = "gpt-4o-mini"

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    hive_api_secret: str = ""

    # Set PUBLIC_URL for deployed environments; ngrok is only used when unset.
    public_url: str = ""
    ngrok_enabled: bool = True

    # Shared secret required (as ?token=) on /ws/client and dashboard REST.
    client_token: str = ""

    data_dir: str = "data"
    retain_transcripts: bool = False

    # Long-term memory (lilyMemory). Embeddings enrich recall; disabling
    # them keeps memory fully offline (lexical search only).
    memory_embeddings: bool = True

    screen_unknown_callers: bool = True
    trusted_numbers: list[str] = field(default_factory=list)

    # If true, a call that reaches High risk mid-call is redirected to a
    # polite hangup via the Twilio REST API. Off by default: warn, don't act.
    auto_intervene: bool = False

    # Privacy: whether family clients receive raw transcript text.
    share_transcripts: bool = True

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            port=int(os.environ.get("PORT", 8000)),
            forward_to=os.environ.get("FORWARD_TO", ""),
            project_id=os.environ.get("PROJECT_ID", "heylily"),
            stt_location=os.environ.get("STT_LOCATION", "global"),
            stt_model=os.environ.get("STT_MODEL", "telephony"),
            stt_language=os.environ.get("STT_LANGUAGE", "en-US"),
            scam_model=os.environ.get("SCAM_MODEL", "gpt-4o-mini"),
            summary_model=os.environ.get("SUMMARY_MODEL", "gpt-4o"),
            event_model=os.environ.get("EVENT_MODEL", "gpt-4o-mini"),
            twilio_account_sid=os.environ.get("TWILIO_ACCOUNT_SID", ""),
            twilio_auth_token=os.environ.get("TWILIO_AUTH_TOKEN", ""),
            hive_api_secret=os.environ.get("HIVE_API_SECRET", ""),
            public_url=os.environ.get("PUBLIC_URL", "").rstrip("/"),
            ngrok_enabled=_bool("NGROK_ENABLED", True),
            client_token=os.environ.get("CLIENT_TOKEN", ""),
            data_dir=os.environ.get("DATA_DIR", "data"),
            retain_transcripts=_bool("RETAIN_TRANSCRIPTS", False),
            memory_embeddings=_bool("MEMORY_EMBEDDINGS", True),
            screen_unknown_callers=_bool("SCREEN_UNKNOWN_CALLERS", True),
            trusted_numbers=_csv("TRUSTED_NUMBERS"),
            auto_intervene=_bool("AUTO_INTERVENE", False),
            share_transcripts=_bool("SHARE_TRANSCRIPTS", True),
        )


settings = Settings.from_env()
