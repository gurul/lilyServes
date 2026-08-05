from auth import check_client_token, compute_twilio_signature
from config import settings


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
