from lily.auth import check_client_token, compute_twilio_signature
from lily.config import settings


def test_twilio_signature_matches_sdk_reference():
    from twilio.request_validator import RequestValidator

    url = "https://mycompany.com/myapp.php?foo=1&bar=2"
    params = {
        "CallSid": "CA1234567890ABCDE",
        "Caller": "+12349013030",
        "Digits": "1234",
        "From": "+12349013030",
        "To": "+18005551212",
    }
    ours = compute_twilio_signature("12345", url, params)
    assert RequestValidator("12345").validate(url, params, ours)


def test_client_token_open_when_unset(monkeypatch):
    monkeypatch.setattr(settings, "client_token", "")
    assert check_client_token("") is True
    assert check_client_token("anything") is True


def test_client_token_enforced(monkeypatch):
    monkeypatch.setattr(settings, "client_token", "secret123")
    assert check_client_token("secret123") is True
    assert check_client_token("wrong") is False
    assert check_client_token("") is False


def test_client_token_non_ascii_is_rejected_not_raised(monkeypatch):
    monkeypatch.setattr(settings, "client_token", "secret123")
    assert check_client_token("é") is False
    monkeypatch.setattr(settings, "client_token", "sécret")
    assert check_client_token("sécret") is True


def test_stream_param_sig_is_hmac_sha256_hex():
    import hashlib
    import hmac

    from lily.auth import stream_param_sig

    expected = hmac.new(b"tok", b"CA123", hashlib.sha256).hexdigest()
    assert stream_param_sig("tok", "CA123") == expected
