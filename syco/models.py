"""Model adapters: one per backend, built from a `ModelSpec`.

Adapted from `core/models.py` in myazan/LLM-Self-Concept. Routing, quantization
handling, chat-template rendering, thinking control and the retry policy carry
over. What changed is the measurement contract, and it changed for one reason:
that project administers a rating scale, so its unit of work is a
`RenderedPrompt(system, user)` scored into a number. Here the unit of work is a
*conversation* -- a persona transcript of up to eight turns, then the dilemma --
and the answer is free text that gets parsed downstream. So:

    sample_item(RenderedPrompt, n) -> [rating text, ...]        (there)
    chat(Conversation, n)          -> [free text, ...]          (here)
    chat_batch([Conversation, ...])-> [free text, ...]          (here, new)

`chat_batch` is the addition that matters for cost. The design has ~200k cells;
generating them one at a time on a local 12B is the difference between a day and
a week, so `HFAdapter` implements true batched generation (left-padded, one
`generate()` per batch) the way `get_assumptions.py` does. Backends that cannot
batch inherit a loop, so the runner never needs to know which is which.

The LOGPROB path is dropped -- there are no option tokens to score in an
open-ended probe. Every optional dependency is imported lazily inside the
adapter that needs it, so `import syco.models` works with nothing installed and
`--dry-run` always runs.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional

# transformers 5 removes TRANSFORMERS_CACHE. Preserve the user's chosen cache
# when it is their only setting, then use the supported HF_HOME variable.
if os.environ.get("TRANSFORMERS_CACHE"):
    os.environ.setdefault("HF_HOME", os.environ["TRANSFORMERS_CACHE"])
    os.environ.pop("TRANSFORMERS_CACHE", None)

from syco.model_registry import (
    ANTHROPIC_BACKEND,
    HUGGINGFACE_BACKEND,
    LLAMACPP_BACKEND,
    MOCK_BACKEND,
    OPENAI_BACKEND,
    ModelSpec,
    split_hf_gguf_ref,
)


# ---------------------------------------------------------------------------
# the unit of work
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Conversation:
    """What one design cell administers: an optional system prompt plus the
    full turn list, ending on the user turn that carries the probe."""
    messages: tuple = ()
    system: str = ""

    @property
    def as_list(self) -> list:
        return [dict(m) for m in self.messages]

    def digest(self) -> str:
        blob = self.system + "\x00" + "\x00".join(
            f"{m['role']}:{m['content']}" for m in self.messages)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ThinkingPlan:
    """How thinking was asserted for a call, and whether the assertion landed.

    `standardized=False` means the backend could not reach the intended state,
    so those rows carry a confound the analysis has to account for rather than
    a setting it can assume.
    """
    want_thinking: bool = False
    kwargs: dict = field(default_factory=dict)
    applied: str = "uncontrolled"
    standardized: bool = True
    max_tokens: int = 2048


def _retry(fn, attempts: int = 4, base: float = 1.5):
    last = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as err:
            last = err
            if attempt == attempts - 1:
                break
            time.sleep(min(base ** attempt, 20))
    raise RuntimeError(f"failed after {attempts} attempts: {last}") from last


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> tuple:
    """Remove <think>...</think> blocks. Returns (clean_text, had_think)."""
    if not text:
        return text, False
    cleaned = _THINK_BLOCK.sub("", text)
    return cleaned.strip(), (cleaned != text)


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------
class ChatAdapter(ABC):
    #: True when `chat_batch` does something better than looping.
    batches = False

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def thinking_plan(self, want_thinking: bool = False) -> ThinkingPlan:
        """Default: no backend-level control, so the latent state is whatever
        the model does by default -- honest, not asserted."""
        return ThinkingPlan(
            want_thinking=want_thinking,
            kwargs={},
            applied="uncontrolled",
            standardized=(want_thinking == self.spec.reasoning.thinks_by_default),
            max_tokens=self.spec.max_output_tokens,
        )

    def chat(self, conv: Conversation, n: int = 1, plan=None) -> list:
        """`n` independent completions of one conversation."""
        raise NotImplementedError(f"{type(self).__name__} has no chat path.")

    def chat_batch(self, convs: list, plan=None) -> list:
        """One completion per conversation, same order. Loops by default;
        backends that can genuinely batch override this."""
        return [self.chat(c, n=1, plan=plan)[0] for c in convs]

    def close(self) -> None:
        """Release weights/VRAM. The runner calls this between models."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ---------------------------------------------------------------------------
# mock -- offline, deterministic; makes the whole pipeline runnable with no keys
# ---------------------------------------------------------------------------
class MockAdapter(ChatAdapter):
    """Emits a well-formed probe response so `--dry-run` exercises the real
    parser, not a format no model would produce."""

    batches = True

    def chat(self, conv: Conversation, n: int = 1, plan=None) -> list:
        import json
        import random

        rng = random.Random(int(conv.digest(), 16) % (2**32))
        labels = ["seeking validation", "seeking an objective read",
                  "venting", "asking for a decision", "testing the waters"]
        out = []
        for draw in range(n):
            picks = rng.sample(labels, 3)
            weights = sorted((rng.random() for _ in range(3)), reverse=True)
            total = sum(weights)
            block = {"mental_models": [
                {"model_name": p, "description": f"The user is {p}.",
                 "probability": round(w / total, 3)}
                for p, w in zip(picks, weights)
            ]}
            out.append(f"{json.dumps(block, indent=2)}\n\nRESPONSE:\n"
                       f"[mock reply {draw}] You are not in the wrong here.")
        return out

    def chat_batch(self, convs: list, plan=None) -> list:
        return [self.chat(c, n=1, plan=plan)[0] for c in convs]


# ---------------------------------------------------------------------------
# chat-template rendering (shared by the two local backends)
# ---------------------------------------------------------------------------
class ChatTemplateRenderer:
    """Renders an assistant-turn-open prompt string via the model's OWN chat
    template, honoring `enable_thinking`.

    Using the model's real template -- not a generic ChatML -- is what actually
    turns thinking off on a quantized GGUF: a hybrid-thinking model with
    enable_thinking=False pre-fills an empty <think></think> block, so the next
    token is the answer rather than a reasoning trace.

    Several templates (Gemma's among them) reject a `system` role outright. Its
    content is folded into the first user turn instead, which is what the Gemma
    chat format expects anyway -- the alternative is a template crash on every
    call.

    Tokenizer-only load: a few MB of JSON, no weights. Cached per model id.
    """
    _cache: dict = {}

    def __init__(self, model_id: Optional[str], required: bool):
        self.model_id = model_id
        self._supports_system = True
        try:
            self.tokenizer = self._load(model_id) if model_id else None
            if self.tokenizer is not None and not getattr(self.tokenizer, "chat_template", None):
                raise RuntimeError("the tokenizer has no chat template")
        except Exception as err:
            self.tokenizer = None
            if required:
                raise RuntimeError(
                    f"Cannot load the required chat template from {model_id!r}: "
                    f"{type(err).__name__}: {err}. Set a public `tokenizer_id` in "
                    "models.yaml, or authenticate with HF_TOKEN if it is gated."
                ) from err
        if self.tokenizer is None and required:
            raise RuntimeError(
                "This model needs `tokenizer_id` (or `hf_id`) in models.yaml so "
                "its prompt can be rendered with its own chat template."
            )

    @classmethod
    def _load(cls, model_id):
        if model_id in cls._cache:
            return cls._cache[model_id]
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        except AttributeError:
            # transformers <4.58 calls .keys() on extra_special_tokens; some
            # repos ship it as a list, which aborts tokenizer init.
            tok = AutoTokenizer.from_pretrained(model_id, use_fast=True,
                                                extra_special_tokens={})
        cls._cache[model_id] = tok
        return tok

    @property
    def available(self) -> bool:
        return self.tokenizer is not None

    def messages_for(self, conv: Conversation) -> list:
        """The turn list to hand the template, with `system` folded in when the
        template cannot take it."""
        return build_messages(conv, supports_system=self._supports_system)

    def render(self, conv: Conversation, enable_thinking: Optional[bool]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return self.tokenizer.apply_chat_template(self.messages_for(conv), **kwargs)
        except Exception:
            if not self._supports_system or not conv.system:
                raise
            self._supports_system = False      # e.g. "System role not supported"
            return self.tokenizer.apply_chat_template(self.messages_for(conv), **kwargs)


def build_messages(conv: Conversation, supports_system: bool = True) -> list:
    """Turn list for a chat API. When the backend has no system slot, the system
    text is prepended to the first user turn rather than dropped."""
    messages = conv.as_list
    if conv.system and not supports_system:
        if messages and messages[0]["role"] == "user":
            messages[0] = {"role": "user",
                           "content": f"{conv.system}\n\n{messages[0]['content']}"}
        else:
            messages.insert(0, {"role": "user", "content": conv.system})
        return messages
    if conv.system:
        return [{"role": "system", "content": conv.system}] + messages
    return messages


# ---------------------------------------------------------------------------
# llama.cpp / GGUF
# ---------------------------------------------------------------------------
class LlamaCppAdapter(ChatAdapter):
    """GGUF via llama-cpp-python. Quantization is baked into the file;
    `spec.quantization` carries the tag and the filename the registry resolved.

    No batching: llama.cpp's Python binding runs one sequence at a time, and the
    KV cache is shared state, so concurrency here would be a correctness bug
    rather than a speedup. Use the `hf` backend for throughput.
    """

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            from llama_cpp import Llama
        except ImportError as err:
            raise RuntimeError(
                "GGUF inference needs `llama-cpp-python`. Install it, or run "
                "with --dry-run to exercise the pipeline offline."
            ) from err

        # Validate/download the tiny tokenizer before touching multi-GB weights,
        # so a gated or mistyped tokenizer repo fails in seconds, not after a
        # model download.
        self.renderer = ChatTemplateRenderer(spec.tokenizer_id or spec.hf_id, required=True)

        model_path = self._ensure_local_file()
        runtime = spec.runtime
        kwargs = {
            "model_path": str(model_path),
            "n_ctx": runtime.get("n_ctx", 16384),
            "n_gpu_layers": runtime.get("n_gpu_layers", -1),
            "verbose": False,
        }
        threads = runtime.get("n_threads")
        if threads:
            kwargs["n_threads"] = threads
        self.llm = Llama(**kwargs)
        self.n_ctx = kwargs["n_ctx"]

    def _ensure_local_file(self):
        from pathlib import Path

        ref = self.spec.ref
        local = Path(ref).expanduser()
        if local.exists():
            return local

        from huggingface_hub import hf_hub_download

        repo_id, pinned = split_hf_gguf_ref(ref)
        filename = self.spec.quantization.resolved_file or pinned
        if not filename:
            raise RuntimeError(
                f"{self.spec.alias}: no GGUF filename. The registry resolves it "
                "at load time -- call `registry.with_resolved_quant(spec)` first."
            )
        return Path(hf_hub_download(repo_id=repo_id, filename=filename,
                                    token=os.environ.get("HF_TOKEN")))

    def thinking_plan(self, want_thinking: bool = False) -> ThinkingPlan:
        if self.spec.reasoning.control == "template_toggle":
            return ThinkingPlan(want_thinking, {"enable_thinking": want_thinking},
                                f"enable_thinking={want_thinking}", True,
                                self.spec.max_output_tokens)
        return ThinkingPlan(want_thinking, {}, "no_native_thinking", True,
                            self.spec.max_output_tokens)

    def _tokens(self, conv: Conversation, enable_thinking):
        """Tokenize the rendered prompt for create_completion.

        The template string already carries the model's leading special tokens
        (Gemma's <bos>), so `add_bos=False` -- otherwise llama.cpp prepends a
        second one and shifts every position.
        """
        text = self.renderer.render(conv, enable_thinking)
        return self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True)

    def chat(self, conv: Conversation, n: int = 1, plan=None) -> list:
        plan = plan or self.thinking_plan()
        enable = plan.kwargs.get("enable_thinking")
        tokens = self._tokens(conv, enable)

        room = self.n_ctx - len(tokens)
        if room < 256:
            raise RuntimeError(
                f"{self.spec.alias}: prompt is {len(tokens)} tokens and n_ctx is "
                f"{self.n_ctx}, leaving {room} for the answer. Raise "
                f"`runtime.llamacpp.n_ctx` in models.yaml -- a persona transcript "
                "plus a dilemma plus a reply needs the room."
            )
        max_tokens = min(plan.max_tokens, room)

        out = []
        for _ in range(n):
            # Reset per draw: left alone, draw 1 gets its logits from a full
            # prefill while later draws hit llama.cpp's prefix-reuse branch and
            # replay a single token. Same prompt, different arithmetic -- the
            # draws would not be i.i.d. from one conditional. Costs a prefill.
            self.llm.reset()
            resp = self.llm.create_completion(
                prompt=tokens,
                temperature=self.spec.temperature,
                top_p=self.spec.top_p,
                max_tokens=max_tokens,
            )
            raw = resp["choices"][0]["text"]
            if enable is False:
                raw, leaked = strip_think(raw)
                if leaked:
                    raw = f"{raw}\n[THINK_LEAK]"
            out.append(raw)
        return out

    def close(self) -> None:
        self.llm = None


# ---------------------------------------------------------------------------
# transformers -- the batched local path
# ---------------------------------------------------------------------------
class HFAdapter(ChatAdapter):
    """transformers path. Handles bnb-4bit/8bit and GPTQ/AWQ/FP8 checkpoints,
    and is the backend that actually batches."""

    batches = True

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as err:
            raise RuntimeError("The HF path needs `transformers` and `torch`.") from err

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer_id or spec.ref)
        # Left padding: with right padding the generated tokens start after the
        # pad run, so short prompts in a batch decode garbage.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "dtype": spec.runtime.get("torch_dtype", "auto"),
            "device_map": spec.runtime.get("device_map", "auto"),
        }
        model_kwargs.update(self._quantization_kwargs())
        self.model = self._load(spec.ref, model_kwargs)
        self.model.eval()
        self._supports_system = True

    def _load(self, ref: str, model_kwargs: dict):
        """Load with the configured kwargs, absorbing two portability failures.

        `device_map` needs `accelerate`, which is not required for a single-GPU
        or CPU run -- so when it is missing we drop the device_map and place the
        model by hand rather than failing the run over an optional dependency.
        A quantized checkpoint is the exception: bitsandbytes places the weights
        during load and cannot be moved afterwards, so that case re-raises with
        the fix named.
        """
        from transformers import AutoModelForCausalLM

        def call(kwargs):
            try:
                return AutoModelForCausalLM.from_pretrained(ref, **kwargs)
            except TypeError:
                # transformers < 4.56 spells it torch_dtype, not dtype.
                kwargs = dict(kwargs)
                kwargs["torch_dtype"] = kwargs.pop("dtype")
                return AutoModelForCausalLM.from_pretrained(ref, **kwargs)

        try:
            return call(model_kwargs)
        except (ImportError, ValueError) as err:
            if "accelerate" not in str(err).lower():
                raise
            if "quantization_config" in model_kwargs:
                raise RuntimeError(
                    f"{self.spec.alias} is quantized, which needs `accelerate` "
                    "alongside `bitsandbytes`: pip install accelerate"
                ) from err
            fallback = {k: v for k, v in model_kwargs.items() if k != "device_map"}
            model = call(fallback)
            device = self._fallback_device()
            print(f"[{self.spec.alias}] accelerate not installed -- loaded without "
                  f"device_map onto {device}. Install accelerate for multi-GPU sharding.")
            return model.to(device)

    def _fallback_device(self) -> str:
        """CUDA if present, otherwise CPU -- never MPS unless asked for.

        Apple's MPS backend returns NaN logits for left-padded batches (verified
        on Qwen2.5-0.5B: the same batch is clean on CPU, NaN on MPS in both
        float32 and bfloat16), and left padding is not optional for batched
        generation. Picking MPS automatically would turn every batched run on a
        Mac into a crash, so it is opt-in via `runtime.device_map: mps` in
        models.yaml. On Apple silicon the `llamacpp` backend is the better local
        path anyway -- it uses Metal properly.
        """
        if self.torch.cuda.is_available():
            return "cuda"
        want = str(self.spec.runtime.get("device_map", "")).lower()
        if want == "mps" and self.torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _quantization_kwargs(self) -> dict:
        """`quantization.method` -> transformers loader kwargs. gptq/awq/fp8
        checkpoints carry their own config in the repo, so nothing is passed for
        those; the method is recorded for provenance only."""
        method = (self.spec.quantization.method or "none").lower()
        if method in ("none", "gptq-int4", "awq", "fp8"):
            return {}
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as err:
            raise RuntimeError(
                f"{self.spec.alias} requests {method}, which needs `bitsandbytes`."
            ) from err
        if method == "bnb-4bit":
            return {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )}
        if method == "bnb-8bit":
            return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True)}
        raise ValueError(f"Unknown hf quantization method {method!r}")

    def thinking_plan(self, want_thinking: bool = False) -> ThinkingPlan:
        if self.spec.reasoning.control == "template_toggle":
            return ThinkingPlan(want_thinking, {"enable_thinking": want_thinking},
                                f"enable_thinking={want_thinking}", True,
                                self.spec.max_output_tokens)
        return ThinkingPlan(want_thinking, {}, "no_native_thinking", True,
                            self.spec.max_output_tokens)

    def _render(self, conv: Conversation, enable_thinking) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        try:
            return self.tokenizer.apply_chat_template(
                build_messages(conv, self._supports_system), **kwargs)
        except Exception:
            if not self._supports_system or not conv.system:
                raise
            self._supports_system = False
            return self.tokenizer.apply_chat_template(
                build_messages(conv, False), **kwargs)

    def _generate(self, texts: list, plan: ThinkingPlan) -> list:
        # add_special_tokens=False: the chat template already emitted the model's
        # leading special tokens, and a second BOS shifts every position.
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True,
                                add_special_tokens=False).to(self.model.device)
        try:
            with self.torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=plan.max_tokens,
                    do_sample=self.spec.temperature > 0,
                    temperature=self.spec.temperature or None,
                    top_p=self.spec.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
        except RuntimeError as err:
            if "probability tensor" not in str(err) or len(texts) == 1:
                raise
            raise RuntimeError(
                f"{self.spec.alias}: sampling got NaN/inf logits on a batch of "
                f"{len(texts)} (device={self.model.device}). On Apple MPS this is a "
                "backend bug with left-padded batches -- use the llamacpp backend, "
                "or run with --batch-size 1."
            ) from err
        # Left padding makes every sequence's prompt end at the same column, so
        # one slice serves the whole batch.
        prompt_len = inputs["input_ids"].shape[1]
        enable = plan.kwargs.get("enable_thinking")
        decoded = []
        for row in outputs:
            text = self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
            if enable is False:
                text, leaked = strip_think(text)
                if leaked:
                    text = f"{text}\n[THINK_LEAK]"
            decoded.append(text.strip())
        return decoded

    def chat(self, conv: Conversation, n: int = 1, plan=None) -> list:
        plan = plan or self.thinking_plan()
        # n draws of one conversation is just a batch of n identical prompts.
        return self._generate([self._render(conv, plan.kwargs.get("enable_thinking"))] * n, plan)

    def chat_batch(self, convs: list, plan=None) -> list:
        plan = plan or self.thinking_plan()
        enable = plan.kwargs.get("enable_thinking")
        return self._generate([self._render(c, enable) for c in convs], plan)

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        try:
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
class OpenAIAdapter(ChatAdapter):
    # GPT-5.6 replaced "minimal" with "none" and REJECTS "minimal" outright, so
    # sending the old value to a 5.6 model fails every call.
    _EFFORT_NONE_GENERATIONS = frozenset({"gpt-5.6"})

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            from openai import OpenAI
        except ImportError as err:
            raise RuntimeError("The OpenAI path needs `openai`.") from err
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI()

    def thinking_plan(self, want_thinking: bool = False) -> ThinkingPlan:
        """GPT-5.x are reasoning models controlled by `reasoning_effort`. The
        off-level differs by generation: 5.6 has "none", a real zero; earlier
        generations only reach "minimal", a floor -- those rows are marked
        `standardized=False` because the intended state was not reached."""
        off = ("none" if self.spec.generation in self._EFFORT_NONE_GENERATIONS
               else "minimal")
        effort = "high" if want_thinking else off
        return ThinkingPlan(want_thinking, {"reasoning_effort": effort},
                            f"reasoning_effort={effort}",
                            (want_thinking or off == "none"),
                            self.spec.max_output_tokens)

    def chat(self, conv: Conversation, n: int = 1, plan=None) -> list:
        plan = plan or self.thinking_plan()
        # Reasoning models (GPT-5.x) reject any temperature/top_p but the
        # default -- effort is the only knob. Sending them 400s every call, so
        # the spec's decoding settings simply do not apply to those models and
        # the row records what was actually in force.
        sampling = ({} if self.spec.reasoning.control == "effort"
                    else {"temperature": self.spec.temperature, "top_p": self.spec.top_p})

        def call():
            return self.client.chat.completions.create(
                model=self.spec.ref,
                messages=build_messages(conv),
                max_completion_tokens=plan.max_tokens,
                n=n,
                **sampling,
                **plan.kwargs,
            )

        resp = _retry(call)
        return [(c.message.content or "") for c in resp.choices]


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------
class AnthropicAdapter(ChatAdapter):
    # Thinking cannot be disabled at all on these (400 on {"type":"disabled"}).
    _THINKING_ALWAYS_ON = frozenset({"claude-fable-5", "claude-mythos-5"})
    # Generations whose default is thinking-OFF: omitting the param IS off, and
    # sending {"disabled"} can 400.
    _LEGACY_BUDGET_GENERATIONS = frozenset({"claude-4.5"})

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        try:
            import anthropic
        except ImportError as err:
            raise RuntimeError("The Anthropic path needs `anthropic`.") from err
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic()

    def thinking_plan(self, want_thinking: bool = False) -> ThinkingPlan:
        ref, gen = self.spec.ref, self.spec.generation
        mt = self.spec.max_output_tokens

        if ref in self._THINKING_ALWAYS_ON:
            return ThinkingPlan(want_thinking, {}, "always_on", want_thinking, mt)

        if want_thinking:
            if gen in self._LEGACY_BUDGET_GENERATIONS:
                budget = max(1024, mt - 512)
                return ThinkingPlan(True, {"thinking": {"type": "enabled",
                                                        "budget_tokens": budget}},
                                    f"enabled:budget_tokens={budget}", True, mt)
            return ThinkingPlan(True, {"thinking": {"type": "adaptive"}},
                                "adaptive", True, mt)

        if gen in self._LEGACY_BUDGET_GENERATIONS:
            return ThinkingPlan(False, {}, "omitted(off)", True, mt)
        kwargs = {"thinking": {"type": "disabled"}}
        if gen == "claude-5":
            # Opus/Sonnet 5 only accept disabled at effort <= high; pin it low.
            kwargs["output_config"] = {"effort": "low"}
        return ThinkingPlan(False, kwargs, "disabled", True, mt)

    def chat(self, conv: Conversation, n: int = 1, plan=None) -> list:
        plan = plan or self.thinking_plan()
        outputs = []
        for _ in range(n):
            def call():
                kw = dict(plan.kwargs)
                # temperature and thinking are mutually exclusive on Claude.
                if "thinking" not in kw or kw["thinking"].get("type") == "disabled":
                    kw["temperature"] = self.spec.temperature
                return self.client.messages.create(
                    model=self.spec.ref,
                    max_tokens=plan.max_tokens,
                    system=conv.system or None,
                    messages=conv.as_list,
                    **kw,
                )

            resp = _retry(call)
            if getattr(resp, "stop_reason", None) == "refusal":
                outputs.append("")
                continue
            outputs.append("".join(b.text for b in resp.content
                                   if getattr(b, "type", "") == "text"))
        return outputs


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
ADAPTERS = {
    MOCK_BACKEND: MockAdapter,
    LLAMACPP_BACKEND: LlamaCppAdapter,
    HUGGINGFACE_BACKEND: HFAdapter,
    OPENAI_BACKEND: OpenAIAdapter,
    ANTHROPIC_BACKEND: AnthropicAdapter,
}


def build_adapter(spec: ModelSpec, dry_run: bool = False) -> ChatAdapter:
    backend = MOCK_BACKEND if dry_run else spec.backend
    if backend not in ADAPTERS:
        raise ValueError(f"Unknown backend {backend!r}. Known: {sorted(ADAPTERS)}")
    return ADAPTERS[backend](spec)
