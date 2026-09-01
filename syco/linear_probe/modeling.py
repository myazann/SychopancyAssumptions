"""Hugging Face target loading and exact decoder-block resolution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

KNOWN_BLOCK_PATHS = (
    "model.layers",                         # Llama, Mistral, Qwen causal LM
    "model.language_model.layers",          # Gemma 3 conditional wrapper
    "language_model.model.layers",
    "language_model.layers",
    "transformer.h",                        # GPT-like
    "gpt_neox.layers",
)


def _get_path(root, path: str):
    value = root
    for part in path.split("."):
        if not hasattr(value, part):
            return None
        value = getattr(value, part)
    return value


def _is_block_sequence(value) -> bool:
    if value is None or isinstance(value, (str, bytes)):
        return False
    try:
        length = len(value)
    except TypeError:
        return False
    return length >= 2 and all(hasattr(value[i], "register_forward_hook")
                               for i in range(min(length, 3)))


def resolve_decoder_blocks(model):
    """Return the canonical sequence of zero-based transformer blocks.

    Extraction and steering both call this function and hook the returned
    modules. This intentionally avoids ``output_hidden_states`` indexing,
    whose embedding entry and final normalization create off-by-one ambiguity.
    """
    for path in KNOWN_BLOCK_PATHS:
        value = _get_path(model, path)
        if _is_block_sequence(value):
            return value, path

    candidates = []
    for name, module in model.named_modules():
        if name.endswith((".layers", ".h")) and _is_block_sequence(module):
            candidates.append((name, module))
    if len(candidates) == 1:
        return candidates[0][1], candidates[0][0]
    names = [name for name, _ in candidates]
    raise RuntimeError(
        "could not identify one decoder-block sequence; "
        f"known paths failed and fallback candidates were {names}"
    )


def candidate_block_indices(n_blocks: int, layer_config) -> tuple[int, ...]:
    if n_blocks <= 0:
        raise ValueError("model has no decoder blocks")
    if layer_config.explicit:
        values = tuple(int(value) for value in layer_config.explicit)
    else:
        values = tuple(round(float(fraction) * (n_blocks - 1))
                       for fraction in layer_config.fractions)
    if any(value < 0 or value >= n_blocks for value in values):
        raise ValueError(
            f"candidate blocks {values} are outside a {n_blocks}-block model"
        )
    return tuple(dict.fromkeys(values))


def pick_input_device(model):
    import torch

    try:
        embedding = model.get_input_embeddings()
        if embedding is not None and embedding.weight.device.type != "meta":
            return embedding.weight.device
    except (AttributeError, RuntimeError):
        pass
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _dtype(torch, name: str):
    normalized = name.lower()
    values = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in values:
        raise ValueError(f"unsupported target dtype {name!r}")
    if (normalized == "bfloat16" and torch.cuda.is_available()
            and not torch.cuda.is_bf16_supported()):
        raise RuntimeError("target requests bfloat16 but this CUDA device lacks it")
    return values[normalized]


def load_tokenizer(target):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        target.tokenizer_ref or target.hf_ref,
        revision=target.revision,
        use_fast=True,
        trust_remote_code=target.trust_remote_code,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("target tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_target_model(target):
    """Load the differentiable HF checkpoint required by probes and hooks."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(
        target.hf_ref,
        revision=target.revision,
        trust_remote_code=target.trust_remote_code,
    )
    # Some recent model configs auto-enable tensor parallelism; device_map is
    # the pipeline's explicit placement policy.
    if hasattr(config, "tp_plan"):
        config.tp_plan = None
    kwargs = {
        "config": config,
        "revision": target.revision,
        "trust_remote_code": target.trust_remote_code,
        "low_cpu_mem_usage": True,
        "device_map": target.device_map,
        "dtype": _dtype(torch, target.dtype),
    }
    if target.attention_implementation:
        kwargs["attn_implementation"] = target.attention_implementation
    if target.quantization != "none":
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("bnb target quantization needs bitsandbytes") from exc
        if target.quantization == "bnb-4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(target.hf_ref, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(target.hf_ref, **kwargs)
    model.eval()
    return model


def target_fingerprint(model, tokenizer, target, block_path: str,
                       block_indices: Sequence[int]) -> dict:
    config_dict = model.config.to_dict()
    compact = {
        "model_alias": target.model,
        "hf_ref": target.hf_ref,
        "revision_requested": target.revision,
        "resolved_commit": getattr(model.config, "_commit_hash", None),
        "architecture": list(getattr(model.config, "architectures", []) or []),
        "model_type": getattr(model.config, "model_type", None),
        "hidden_size": config_dict.get("hidden_size") or
            (config_dict.get("text_config") or {}).get("hidden_size"),
        "dtype": target.dtype,
        "quantization": target.quantization,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "tokenizer_resolved_commit": (
            getattr(tokenizer, "init_kwargs", {}) or {}
        ).get("_commit_hash"),
        "vocab_size": len(tokenizer),
        "chat_template_sha256": hashlib.sha256(
            str(tokenizer.chat_template or "").encode()
        ).hexdigest(),
        "decoder_block_path": block_path,
        "decoder_block_count": config_dict.get("num_hidden_layers") or
            (config_dict.get("text_config") or {}).get("num_hidden_layers"),
        "candidate_block_indices": list(block_indices),
    }
    compact["fingerprint"] = hashlib.sha256(json.dumps(
        compact, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return compact
