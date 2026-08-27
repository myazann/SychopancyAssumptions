import sys
from types import SimpleNamespace

import pytest

from syco.experiments import load_profile
from syco.model_registry import ModelSpec, Reasoning, load_registry
from syco.models import (
    AnthropicAdapter,
    ChatAdapter,
    Conversation,
    LlamaCppAdapter,
    MockAdapter,
    OpenAIAdapter,
    _retry,
)
from syco.parse import CLEAN, parse_completion, parse_structured
from syco.prompts import ProbeSpec, build


def _adapter(adapter_type, spec, client):
    adapter = object.__new__(adapter_type)
    ChatAdapter.__init__(adapter, spec)
    adapter.client = client
    return adapter


def test_llamacpp_forwards_l40_runtime_options(monkeypatch, tmp_path):
    calls = []

    class RecordingLlama:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    fake_module = SimpleNamespace(Llama=RecordingLlama)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)
    monkeypatch.setattr(
        "syco.models.ChatTemplateRenderer",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    model_path = tmp_path / "model.gguf"
    monkeypatch.setattr(LlamaCppAdapter, "_ensure_local_file", lambda self: model_path)
    spec = ModelSpec(
        alias="local",
        family="llama",
        ref=str(model_path),
        backend="llamacpp",
        tokenizer_id="tokenizer",
        runtime={
            "n_ctx": 8192,
            "n_gpu_layers": -1,
            "flash_attn": True,
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_threads": 8,
            "n_threads_batch": 8,
        },
    )

    adapter = LlamaCppAdapter(spec)

    assert adapter.n_ctx == 8192
    assert calls == [{
        "model_path": str(model_path),
        "verbose": False,
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "n_batch": 2048,
        "n_ubatch": 512,
        "n_threads": 8,
        "n_threads_batch": 8,
        "flash_attn": True,
    }]


def test_llamacpp_rejects_unknown_runtime_option_before_loading(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=object))
    spec = ModelSpec(
        alias="local",
        family="llama",
        ref="model.gguf",
        backend="llamacpp",
        runtime={"flash_attention": True},
    )

    with pytest.raises(ValueError, match="unsupported.*flash_attention"):
        LlamaCppAdapter(spec)


def test_default_profile_models_inherit_l40_runtime_tuning():
    registry = load_registry()
    specs = load_profile("default").select_models(registry)

    assert len(specs) == 5
    for spec in specs:
        assert spec.backend == "llamacpp"
        assert spec.runtime["flash_attn"] is True
        assert spec.runtime["n_batch"] == 2048
        assert spec.runtime["n_ubatch"] == 512
        assert spec.runtime["n_threads"] == 8
        assert spec.runtime["n_threads_batch"] == 8
    assert registry.get("Gemma3-27B").runtime["n_ctx"] == 8192
    assert registry.get("Gemma3-12B").runtime["n_ctx"] == 16384


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


@pytest.mark.parametrize(
    "kind,expected_outputs",
    [("openended", 3), ("4dims", 4), ("supporttypes", 5)],
)
def test_mock_adapter_emits_the_selected_probe_schema(kind, expected_outputs):
    spec = ModelSpec(
        alias="mock",
        family="mock",
        ref="mock",
        backend="mock",
    )
    probe = ProbeSpec(kind=kind)
    messages = build(probe, [], "Was I wrong?")
    raw = MockAdapter(spec).chat(Conversation(messages=tuple(messages)))[0]

    if probe.family == "open-ended":
        parsed = parse_completion(raw, expected_n=probe.n_models)
        assert parsed.status == CLEAN
        assert parsed.n_assumptions == expected_outputs
    else:
        parsed = parse_structured(raw, kind)
        assert parsed.status == CLEAN
        assert parsed.n_dimensions == expected_outputs
        assert parsed.n_scored == expected_outputs
        assert {belief.dimension for belief in parsed.beliefs} == set(probe.dimensions)
    assert parsed.has_response is True
