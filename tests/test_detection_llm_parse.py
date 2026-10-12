"""LLM verdict parsing (fix 4): malformed output never reads as "safe"."""
import asyncio
import types

import pytest

import lily.detection.detector as det

TRANSCRIPT = (
    "Grandma, it's me. I got in a car crash and I'm at the police station. "
    "Please do not tell mom and dad, I need money for a lawyer."
)


class _FakeClient:
    def __init__(self, content):
        self._content = content
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        message = types.SimpleNamespace(content=self._content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def _analyze(monkeypatch, content, transcript=TRANSCRIPT):
    monkeypatch.setattr(det, "get_client", lambda: _FakeClient(content))
    return asyncio.run(det.analyze_scam(transcript))


@pytest.mark.parametrize("content,verdict,criteria", [
    ('{"verdict":"scam","criteria":null,"reasoning":"grandparent scam"}', "scam", []),
    ('{"verdict":"scam","criteria":"8, 9","reasoning":"grandparent scam"}', "scam", []),
    ('{"verdict":"Scam","criteria":[8,9],"reasoning":"grandparent scam"}', "scam", [8, 9]),
    ('{"verdict":" SAFE ","criteria":[],"reasoning":"chat"}', "safe", []),
    ('{"verdict":"scam","criteria":[8,true,"9",9.5,9],"reasoning":"x"}', "scam", [8, 9]),
])
def test_verdict_read_first_and_criteria_guarded(monkeypatch, content, verdict, criteria):
    result = _analyze(monkeypatch, content)
    assert result["verdict"] == verdict
    assert result["criteria"] == criteria


TRUNCATED = (
    '{"verdict":"scam","criteria":[8,9],"reasoning":"The caller claims to be a grandson '
    "in jail after a car crash, asks for lawyer money and secrecy from parents, matching "
    'the grandparent scam pattern with urgency and'
)


@pytest.mark.parametrize("content", [
    TRUNCATED,
    '["scam"]',
    '"scam"',
    "42",
    "null",
    "",
    None,
    "not json at all",
    "{}",
    '{"verdict": null}',
    '{"verdict": 1}',
    '{"verdict": "maybe"}',
])
def test_malformed_output_is_uncertain_never_safe(monkeypatch, content):
    result = _analyze(monkeypatch, content, transcript="Hello, how is the garden doing?")
    assert result["verdict"] == "uncertain"
    assert isinstance(result["criteria"], list)
    assert isinstance(result["reasoning"], str)


def test_missing_choices_is_uncertain(monkeypatch):
    class _Empty(_FakeClient):
        async def create(self, **kwargs):
            return types.SimpleNamespace(choices=[])

    monkeypatch.setattr(det, "get_client", lambda: _Empty(None))
    result = asyncio.run(det.analyze_scam("Hello, how is the garden doing?"))
    assert result["verdict"] == "uncertain"


def test_non_string_reasoning_becomes_empty(monkeypatch):
    result = _analyze(monkeypatch, '{"verdict":"safe","criteria":[],"reasoning":{"a":1}}')
    assert result["reasoning"] == ""
