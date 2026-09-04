"""The two prompt views used by labels and target-model activations."""

from __future__ import annotations

from syco.data import transcript_to_text
from syco.models import build_messages
from syco.prompts import (
    FOUR_DIMS_PROMPT_EXPLICIT_V2,
    build_prompt_4dims,
    build_prompt_supporttypes,
)

_RESPONSE_INSTRUCTION = (
    "\n\nThen, on a new line, write your actual reply to User A under the heading:\n\n"
    "RESPONSE:"
)


def build_label_prompt(
    instrument: str,
    persona_messages,
    post_text: str,
    *,
    four_dims_prompt_version: str = FOUR_DIMS_PROMPT_EXPLICIT_V2,
) -> str:
    """Paper structured instrument with only its annotation portion.

    Removing the reply request isolates the teacher as a labeller and prevents
    its advice text from becoming accidental supervision. New 4dims datasets
    use the explicit 0-1 prompt; callers can request paper-v1 only to reproduce
    an already-frozen dataset.
    """
    history = transcript_to_text(list(persona_messages))
    if instrument == "4dims":
        full = build_prompt_4dims(
            history, post_text, prompt_version=four_dims_prompt_version
        )
    elif instrument == "supporttypes":
        full = build_prompt_supporttypes(history, post_text)
    else:
        raise ValueError(f"unsupported label instrument {instrument!r}")
    if not full.endswith(_RESPONSE_INSTRUCTION):
        raise RuntimeError("paper prompt suffix changed; update label-only derivation")
    return full[:-len(_RESPONSE_INSTRUCTION)].rstrip()


def deployment_messages(cell, answer_instruction: str) -> list[dict]:
    """Actual persona chat followed by the constrained sycophancy question."""
    messages = [dict(message) for message in cell.persona.messages]
    current = deployment_user_text(cell, answer_instruction)
    if messages and messages[-1]["role"] == "user":
        # Some recovered histories end mid-user-turn. Keep a valid alternating
        # chat without losing either piece of context.
        messages[-1] = {
            "role": "user",
            "content": f"{messages[-1]['content'].rstrip()}\n\n{current}",
        }
    else:
        messages.append({"role": "user", "content": current})
    return messages


def deployment_user_text(cell, answer_instruction: str) -> str:
    return f"{cell.prompt.text.rstrip()}\n\n{answer_instruction.strip()}"


def render_target_prompt(tokenizer, cell, target) -> str:
    """Render the exact assistant-turn-open prompt used in steering."""
    messages = deployment_messages(cell, target.answer_instruction)
    conv = type("ConversationLike", (), {
        "as_list": [dict(m) for m in messages],
        "system": target.system_prompt,
    })()
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return tokenizer.apply_chat_template(
            build_messages(conv, supports_system=True), **kwargs
        )
    except Exception:  # noqa: BLE001 -- templates raise model-specific exception types
        return tokenizer.apply_chat_template(
            build_messages(conv, supports_system=False), **kwargs
        )
