from types import SimpleNamespace

import pytest

from syco.model_registry import ModelSpec, Reasoning
from syco.models import (
    AnthropicAdapter,
    ChatAdapter,
    Conversation,
    OpenAIAdapter,
    _retry,
)


def _adapter(adapter_type, spec, client):
    adapter = object.__new__(adapter_type)
    ChatAdapter.__init__(adapter, spec)
    adapter.client = client
    return adapter


class RecordingResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="provider reply")


class RecordingMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        block = SimpleNamespace(type="text", text="provider reply")
        return SimpleNamespace(stop_reason="end_turn", content=[block])


def test_openai_gpt56_uses_responses_api_without_sampling_parameters():
    spec = ModelSpec(
        alias="GPT-5.6",
        family="gpt",
        ref="gpt-5.6",
        backend="openai",
        generation="gpt-5.6",
        reasoning=Reasoning(True, "effort", True),
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=321,
    )
    responses = RecordingResponses()
    client = SimpleNamespace(responses=responses)
    adapter = _adapter(OpenAIAdapter, spec, client)
    conv = Conversation(
        system="system instruction",
        messages=({"role": "user", "content": "hello"},),
    )

    assert adapter.chat(conv, n=2) == ["provider reply", "provider reply"]
    assert len(responses.calls) == 2
    request = responses.calls[0]
    assert request == {
        "model": "gpt-5.6",
        "input": [{"role": "user", "content": "hello"}],
        "instructions": "system instruction",
        "max_output_tokens": 321,
        "reasoning": {"effort": "none"},
    }
    assert spec.provenance()["temperature"] is None
    assert spec.provenance()["top_p"] is None


def test_anthropic_sonnet5_disables_thinking_without_sampling_parameters():
    spec = ModelSpec(
        alias="Claude-Sonnet-5",
        family="claude",
        ref="claude-sonnet-5",
        backend="anthropic",
        generation="claude-5",
        reasoning=Reasoning(True, "thinking_param", True),
        max_output_tokens=456,
    )
    messages = RecordingMessages()
    client = SimpleNamespace(messages=messages)
    adapter = _adapter(AnthropicAdapter, spec, client)
    conv = Conversation(messages=({"role": "user", "content": "hello"},))

    assert adapter.chat(conv) == ["provider reply"]
    request = messages.calls[0]
    assert request["thinking"] == {"type": "disabled"}
    assert request["output_config"] == {"effort": "low"}
    assert "temperature" not in request
    assert "top_p" not in request
    assert spec.provenance()["temperature"] is None
    assert spec.provenance()["top_p"] is None


def test_anthropic_claude45_off_omits_thinking_and_uses_temperature():
    spec = ModelSpec(
        alias="Claude-Haiku-4.5",
        family="claude",
        ref="claude-haiku-4-5-20251001",
        backend="anthropic",
        generation="claude-4.5",
        reasoning=Reasoning(False, "thinking_param", True),
        temperature=0.4,
    )
    messages = RecordingMessages()
    adapter = _adapter(
        AnthropicAdapter, spec, SimpleNamespace(messages=messages)
    )

    assert adapter.chat(Conversation()) == ["provider reply"]
    request = messages.calls[0]
    assert request["temperature"] == 0.4
    assert "thinking" not in request
    assert "top_p" not in request
    assert "system" not in request


def test_retry_fails_fast_on_non_transient_http_error(monkeypatch):
    calls = 0

    class BadRequest(Exception):
        status_code = 400

    def fail():
        nonlocal calls
        calls += 1
        raise BadRequest("invalid request")

    monkeypatch.setattr("syco.models.time.sleep", lambda _: None)
    with pytest.raises(BadRequest):
        _retry(fail)
    assert calls == 1


def test_retry_retries_rate_limit(monkeypatch):
    calls = 0

    class RateLimit(Exception):
        status_code = 429

    def eventually():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimit("slow down")
        return "ok"

    monkeypatch.setattr("syco.models.time.sleep", lambda _: None)
    assert _retry(eventually) == "ok"
    assert calls == 2


def test_retry_fails_fast_on_local_request_shape_error(monkeypatch):
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        raise TypeError("unexpected keyword")

    monkeypatch.setattr("syco.models.time.sleep", lambda _: None)
    with pytest.raises(TypeError):
        _retry(fail)
    assert calls == 1
